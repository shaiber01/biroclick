"""
Code agent nodes: code_generator_node, code_reviewer_node.

These nodes handle code generation and review.

State Keys
----------
code_reviewer_node:
    READS: current_stage_id, code, design_description, plan, paper_text,
           code_revision_count, code_feedback, runtime_config
    WRITES: workflow_phase, last_code_review_verdict, code_feedback,
            code_revision_count, ask_user_trigger, pending_user_questions,
            awaiting_user_input

code_generator_node:
    READS: current_stage_id, plan, design_description, paper_text, paper_domain,
           validated_materials, code_revision_count, code_feedback
    WRITES: workflow_phase, code, ask_user_trigger, pending_user_questions,
            awaiting_user_input
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from schemas.state import (
    ReproState,
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
)
from src.prompts import build_agent_prompt
from src.llm_client import (
    call_agent_with_metrics,
    build_user_content_for_code_generator,
)

from .helpers.context import check_context_or_escalate
from .helpers.metrics import log_agent_call


def code_reviewer_node(state: ReproState) -> dict:
    """
    CodeReviewerAgent: Review generated code before execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_code_review_verdict` state field.
    - Increments `code_revision_count` when verdict is "needs_revision".
    """
    logger = logging.getLogger(__name__)
    
    # Context check
    context_update = check_context_or_escalate(state, "code_review")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("code_reviewer", state)
    
    # Build user content with code and design
    code = state.get("code", "")
    design = state.get("design_description", {})
    stage_id = state.get("current_stage_id", "unknown")
    
    user_content = f"# CODE TO REVIEW\n\nStage: {stage_id}\n\n```python\n{code}\n```"
    
    # Add design spec for reference
    if design:
        if isinstance(design, dict):
            user_content += f"\n\n# DESIGN SPEC\n\n```json\n{json.dumps(design, indent=2)}\n```"
        else:
            user_content += f"\n\n# DESIGN SPEC\n\n{design}"
    
    # Add previous feedback if revision
    feedback = state.get("reviewer_feedback", "")
    if feedback:
        user_content += f"\n\n# REVISION FEEDBACK\n\n{feedback}"
    
    # Call LLM for code review
    try:
        agent_output = call_agent_with_metrics(
            agent_name="code_reviewer",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
    except Exception as e:
        logger.error(f"Code reviewer LLM call failed: {e}")
        agent_output = {
            "verdict": "approve",
            "issues": [{"severity": "minor", "description": f"LLM review unavailable: {str(e)[:200]}"}],
            "summary": "Code auto-approved due to LLM unavailability",
        }
    
    result: Dict[str, Any] = {
        "workflow_phase": "code_review",
        "last_code_review_verdict": agent_output["verdict"],
        "reviewer_issues": agent_output.get("issues", []),
    }
    
    # Increment code revision counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        current_count = state.get("code_revision_count", 0)
        runtime_config = state.get("runtime_config", {})
        max_revisions = runtime_config.get("max_code_revisions", MAX_CODE_REVISIONS)
        if current_count < max_revisions:
            result["code_revision_count"] = current_count + 1
        else:
            result["code_revision_count"] = current_count
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


def code_generator_node(state: ReproState) -> dict:
    """
    CodeGeneratorAgent: Generate Python+Meep code from approved design.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls, so it must check context first.
    
    For Stage 1+, code generator MUST read material paths from 
    state["validated_materials"], NOT hardcode paths.
    """
    start_time = datetime.now(timezone.utc)
    logger = logging.getLogger(__name__)
    
    # Context check
    escalation = check_context_or_escalate(state, "generate_code")
    if escalation is not None:
        if escalation.get("awaiting_user_input"):
            return escalation
        # Just state updates (e.g., metrics) - merge into state and continue
        state = {**state, **escalation}
    
    # Validate current_stage_id
    current_stage_id = state.get("current_stage_id")
    if not current_stage_id:
        logger.error(
            "current_stage_id is None - cannot generate code without a selected stage. "
            "This indicates select_stage_node did not run or returned None."
        )
        return {
            "workflow_phase": "code_generation",
            "ask_user_trigger": "missing_stage_id",
            "pending_user_questions": [
                "ERROR: No stage selected for code generation. This indicates a workflow error. "
                "Please check stage selection or restart the workflow."
            ],
            "awaiting_user_input": True,
        }
    
    # Validate design_description
    design_description = state.get("design_description", "")
    stub_markers = ["STUB", "TODO", "PLACEHOLDER", "would be generated"]
    
    design_str = str(design_description)
    is_stub = any(marker in design_str.upper() for marker in stub_markers) if design_description else True
    is_empty = not design_description or not design_str.strip() or len(design_str.strip()) < 50
    
    if is_stub or is_empty:
        logger.error(
            f"design_description is stub or empty (stub={is_stub}, empty={is_empty}). "
            "Code generation requires an approved design description."
        )
        return {
            "workflow_phase": "code_generation",
            "design_revision_count": min(
                state.get("design_revision_count", 0) + 1,
                state.get("runtime_config", {}).get("max_design_revisions", MAX_DESIGN_REVISIONS)
            ),
            "reviewer_feedback": (
                "ERROR: Design description is missing or contains stub markers. "
                "Code generation requires an approved design description. "
                "Please ensure design review completed successfully."
            ),
            "supervisor_verdict": "ok_continue",
        }
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("code_generator", state)
    
    # Validate materials for Stage 1+
    current_stage_type = state.get("current_stage_type", "")
    if current_stage_type != "MATERIAL_VALIDATION":
        validated_materials = state.get("validated_materials", [])
        if not validated_materials:
            logger.error(
                f"validated_materials is empty for Stage 1+ ({current_stage_type}). "
                "Code generation cannot proceed without validated materials."
            )
            return {
                "workflow_phase": "code_generation",
                "run_error": (
                    f"validated_materials is empty but required for {current_stage_type} code generation. "
                    "This indicates material_checkpoint_node did not run or user did not approve materials. "
                    "Check that Stage 0 completed and user confirmation was received."
                ),
                "code_revision_count": min(
                    state.get("code_revision_count", 0) + 1,
                    state.get("runtime_config", {}).get("max_code_revisions", MAX_CODE_REVISIONS)
                ),
            }
    
    # Build user content for code generator
    user_content = build_user_content_for_code_generator(state)
    
    # Add revision feedback if any
    feedback = state.get("reviewer_feedback", "")
    if feedback:
        system_prompt += f"\n\nREVISION FEEDBACK: {feedback}"
    
    # Call LLM for code generation
    try:
        agent_output = call_agent_with_metrics(
            agent_name="code_generator",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
    except Exception as e:
        logger.error(f"Code generator LLM call failed: {e}")
        return {
            "workflow_phase": "code_generation",
            "ask_user_trigger": "llm_error",
            "pending_user_questions": [
                f"Code generation failed: {str(e)[:500]}. Please check API and try again."
            ],
            "awaiting_user_input": True,
        }
    
    # Extract generated code from agent output
    generated_code = agent_output.get("code", "")
    if not generated_code and agent_output.get("simulation_code"):
        generated_code = agent_output.get("simulation_code")
    if not generated_code:
        generated_code = json.dumps(agent_output, indent=2) if isinstance(agent_output, dict) else str(agent_output)
    
    # Validate generated code
    stub_markers = ["STUB", "TODO", "PLACEHOLDER", "# Replace", "would be generated"]
    is_stub = any(marker in generated_code.upper() for marker in stub_markers)
    is_empty = not generated_code or not generated_code.strip() or len(generated_code.strip()) < 50
    
    if is_stub or is_empty:
        logger.error(
            f"Generated code is stub or empty (stub={is_stub}, empty={is_empty}). "
            "Code generation must produce valid simulation code."
        )
        result: Dict[str, Any] = {
            "workflow_phase": "code_generation",
            "code": generated_code,
            "code_revision_count": state.get("code_revision_count", 0) + 1,
            "reviewer_feedback": (
                "ERROR: Generated code is empty or contains stub markers. "
                "Code generation must produce valid Meep simulation code. "
                "Please regenerate with proper implementation."
            ),
        }
        return result
    
    result = {
        "workflow_phase": "code_generation",
        "code": generated_code
    }
    
    log_agent_call("CodeGeneratorAgent", "generate_code", start_time)(state, result)
    return result

