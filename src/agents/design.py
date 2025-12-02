"""
Design agent nodes: simulation_designer_node, design_reviewer_node.

These nodes handle simulation design and review before code generation.

State Keys
----------
simulation_designer_node:
    READS: current_stage_id, plan, paper_text, paper_domain, validated_materials,
           assumptions, design_revision_count, design_feedback
    WRITES: workflow_phase, design_description, design_parameters, design_feedback,
            ask_user_trigger, pending_user_questions, awaiting_user_input

design_reviewer_node:
    READS: current_stage_id, design_description, plan, paper_text, paper_domain,
           design_revision_count, design_feedback, runtime_config
    WRITES: workflow_phase, last_design_review_verdict, design_feedback,
            design_revision_count, ask_user_trigger, pending_user_questions,
            awaiting_user_input
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from schemas.state import (
    ReproState,
    get_plan_stage,
    get_stage_design_spec,
    MAX_DESIGN_REVISIONS,
)
from src.prompts import build_agent_prompt
from src.llm_client import (
    call_agent_with_metrics,
    build_user_content_for_designer,
)

from .helpers.context import check_context_or_escalate
from .helpers.metrics import log_agent_call


def simulation_designer_node(state: ReproState) -> dict:
    """
    SimulationDesignerAgent: Design simulation setup for current stage.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls, so it must check context first.
    """
    start_time = datetime.now(timezone.utc)
    logger = logging.getLogger(__name__)
    
    # Context check
    escalation = check_context_or_escalate(state, "design")
    if escalation is not None:
        if escalation.get("awaiting_user_input"):
            return escalation
        # Just state updates (e.g., metrics) - merge into state and continue
        state = {**state, **escalation}
    
    # Validate current_stage_id
    current_stage_id = state.get("current_stage_id")
    if not current_stage_id:
        logger.error(
            "current_stage_id is None - cannot design without a selected stage. "
            "This indicates select_stage_node did not run or returned None."
        )
        return {
            "workflow_phase": "design",
            "ask_user_trigger": "missing_stage_id",
            "pending_user_questions": [
                "ERROR: No stage selected for design. This indicates a workflow error. "
                "Please check stage selection or restart the workflow."
            ],
            "awaiting_user_input": True,
        }
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("simulation_designer", state)
    
    # Inject complexity class from plan
    complexity_class = get_stage_design_spec(state, current_stage_id, "complexity_class", "unknown")
    
    # Build user content for designer
    user_content = build_user_content_for_designer(state)
    
    # Inject complexity class info
    system_prompt += f"\n\nComplexity class for this stage: {complexity_class}"
    
    # Add revision feedback if any
    feedback = state.get("reviewer_feedback", "")
    if feedback:
        system_prompt += f"\n\nREVISION FEEDBACK: {feedback}"
    
    # Call LLM for design
    try:
        agent_output = call_agent_with_metrics(
            agent_name="simulation_designer",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
    except Exception as e:
        logger.error(f"Simulation designer LLM call failed: {e}")
        return {
            "workflow_phase": "design",
            "ask_user_trigger": "llm_error",
            "pending_user_questions": [
                f"Design generation failed: {str(e)[:500]}. Please check API and try again."
            ],
            "awaiting_user_input": True,
        }
    
    # Extract design description
    result: Dict[str, Any] = {
        "workflow_phase": "design",
        "design_description": agent_output,
    }
    
    # Add new assumptions if any
    new_assumptions = agent_output.get("new_assumptions", [])
    if new_assumptions:
        existing = state.get("assumptions", {})
        global_assumptions = existing.get("global_assumptions", [])
        global_assumptions.extend(new_assumptions)
        result["assumptions"] = {**existing, "global_assumptions": global_assumptions}
    
    log_agent_call("SimulationDesignerAgent", "design", start_time)(state, result)
    return result


def design_reviewer_node(state: ReproState) -> dict:
    """
    DesignReviewerAgent: Review simulation design before code generation.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_design_review_verdict` state field.
    - Increments `design_revision_count` when verdict is "needs_revision".
    """
    logger = logging.getLogger(__name__)
    
    # Context check
    context_update = check_context_or_escalate(state, "design_review")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("design_reviewer", state)
    
    # Build user content with design and stage info
    design = state.get("design_description", {})
    stage_id = state.get("current_stage_id", "unknown")
    
    user_content = f"# DESIGN TO REVIEW\n\nStage: {stage_id}\n\n"
    if isinstance(design, dict):
        user_content += f"```json\n{json.dumps(design, indent=2)}\n```"
    else:
        user_content += f"```\n{design}\n```"
    
    # Add plan stage spec for reference
    plan_stage = get_plan_stage(state, stage_id)
    if plan_stage:
        user_content += f"\n\n# PLAN STAGE SPEC\n\n```json\n{json.dumps(plan_stage, indent=2)}\n```"
    
    # Add previous feedback if revision
    feedback = state.get("reviewer_feedback", "")
    if feedback:
        user_content += f"\n\n# REVISION FEEDBACK\n\n{feedback}"
    
    # Call LLM for design review
    try:
        agent_output = call_agent_with_metrics(
            agent_name="design_reviewer",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
    except Exception as e:
        logger.error(f"Design reviewer LLM call failed: {e}")
        agent_output = {
            "verdict": "approve",
            "issues": [{"severity": "minor", "description": f"LLM review unavailable: {str(e)[:200]}"}],
            "summary": "Design auto-approved due to LLM unavailability",
        }
    
    result: Dict[str, Any] = {
        "workflow_phase": "design_review",
        "last_design_review_verdict": agent_output["verdict"],
        "reviewer_issues": agent_output.get("issues", []),
    }
    
    # Increment design revision counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        current_count = state.get("design_revision_count", 0)
        runtime_config = state.get("runtime_config", {})
        max_revisions = runtime_config.get("max_design_revisions", MAX_DESIGN_REVISIONS)
        if current_count < max_revisions:
            result["design_revision_count"] = current_count + 1
        else:
            result["design_revision_count"] = current_count
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result

