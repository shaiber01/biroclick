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
from .base import (
    with_context_check,
    increment_counter_with_max,
    create_llm_error_auto_approve,
    create_llm_error_escalation,
)


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
        return create_llm_error_escalation("simulation_designer", "design", e)
    
    # Extract design description
    result: Dict[str, Any] = {
        "workflow_phase": "design",
        "design_description": agent_output,
    }
    
    # Add new assumptions if any
    # Handle case where agent_output might not be a dict (e.g., string, None)
    if isinstance(agent_output, dict):
        new_assumptions = agent_output.get("new_assumptions", [])
        # Validate that new_assumptions is a list before extending
        if new_assumptions and isinstance(new_assumptions, list):
            existing = state.get("assumptions") or {}
            global_assumptions = list(existing.get("global_assumptions", []))
            global_assumptions.extend(new_assumptions)
            result["assumptions"] = {**existing, "global_assumptions": global_assumptions}
        elif new_assumptions:
            # Log warning if new_assumptions exists but is not a list
            logger.warning(
                f"simulation_designer_node: new_assumptions is not a list (type: {type(new_assumptions)}). "
                "Ignoring invalid assumptions."
            )
    
    log_agent_call("SimulationDesignerAgent", "design", start_time)(state, result)
    return result


@with_context_check("design_review")
def design_reviewer_node(state: ReproState) -> dict:
    """
    DesignReviewerAgent: Review simulation design before code generation.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_design_review_verdict` state field.
    - Increments `design_revision_count` when verdict is "needs_revision".
    
    Note: Context check is handled by @with_context_check decorator.
    """
    logger = logging.getLogger(__name__)

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
        agent_output = create_llm_error_auto_approve("design_reviewer", e)
    
    # Normalize verdict to allowed values
    raw_verdict = agent_output.get("verdict")
    if not raw_verdict:
        logger.warning("Design reviewer output missing 'verdict'. Defaulting to 'approve'.")
        verdict = "approve"
    # Normalize common variations: "pass" -> "approve", "reject" -> "needs_revision"
    elif raw_verdict in ["pass", "approved", "accept"]:
        verdict = "approve"
    elif raw_verdict in ["reject", "revision_needed", "needs_work"]:
        verdict = "needs_revision"
    elif raw_verdict in ["approve", "needs_revision"]:
        verdict = raw_verdict
    else:
        # Unknown verdict - log warning and default to approve
        logger.warning(
            f"Design reviewer returned unexpected verdict '{raw_verdict}'. "
            "Normalizing to 'approve'. Allowed values: 'approve', 'needs_revision'."
        )
        verdict = "approve"
        
    result: Dict[str, Any] = {
        "workflow_phase": "design_review",
        "last_design_review_verdict": verdict,
        "reviewer_issues": agent_output.get("issues") or [],
        "design_revision_count": state.get("design_revision_count", 0),  # Always include current count
    }
    
    # Increment design revision counter if needs_revision
    if verdict == "needs_revision":
        new_count, _ = increment_counter_with_max(
            state, "design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS
        )
        result["design_revision_count"] = new_count
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
        
        # If we hit the max revision budget, escalate to ask_user immediately.
        # BUG FIX: Check if new_count >= max (not just if increment failed).
        # The routing also checks this, but we need to set the trigger/questions.
        runtime_config = state.get("runtime_config", {})
        max_revs = runtime_config.get("max_design_revisions", MAX_DESIGN_REVISIONS)
        if new_count >= max_revs:
            stage_id = state.get("current_stage_id", "unknown")
            question = (
                "Design review limit reached.\n\n"
                f"- Stage: {stage_id}\n"
                f"- Attempts: {new_count}/{max_revs}\n"
                "- Latest reviewer feedback:\n"
                f"  {result.get('reviewer_feedback', 'No feedback available')}\n\n"
                "Please respond with PROVIDE_HINT (include guidance for next attempt), "
                "SKIP to bypass this stage, or STOP to end the workflow."
            )
            result.update({
                "ask_user_trigger": "design_review_limit",
                "pending_user_questions": [question],
                "awaiting_user_input": True,
                "last_node_before_ask_user": "design_review",
            })
    
    return result

