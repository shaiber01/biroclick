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
from .base import (
    with_context_check,
    increment_counter_with_max,
    create_llm_error_auto_approve,
    create_llm_error_escalation,
)


@with_context_check("code_review")
def code_reviewer_node(state: ReproState) -> dict:
    """
    CodeReviewerAgent: Review generated code before execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_code_review_verdict` state field.
    - Increments `code_revision_count` when verdict is "needs_revision".
    
    Note: Context check is handled by @with_context_check decorator.
    """
    logger = logging.getLogger(__name__)

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
        agent_output = create_llm_error_auto_approve("code_reviewer", e)
    
    # Normalize verdict to allowed values
    raw_verdict = agent_output.get("verdict", "needs_revision")
    # Normalize common variations: "pass" -> "approve", "reject" -> "needs_revision"
    if raw_verdict in ["pass", "approved", "accept"]:
        verdict = "approve"
    elif raw_verdict in ["reject", "revision_needed", "needs_work"]:
        verdict = "needs_revision"
    elif raw_verdict in ["approve", "needs_revision"]:
        verdict = raw_verdict
    else:
        # Unknown verdict - log warning and default to needs_revision (safer for code)
        logger.warning(
            f"Code reviewer returned unexpected verdict '{raw_verdict}'. "
            "Normalizing to 'needs_revision'. Allowed values: 'approve', 'needs_revision'."
        )
        verdict = "needs_revision"
    
    result: Dict[str, Any] = {
        "workflow_phase": "code_review",
        "last_code_review_verdict": verdict,
        "reviewer_issues": agent_output.get("issues", []),
        "code_revision_count": state.get("code_revision_count", 0),  # Always include current count
    }
    
    # Increment code revision counter if needs_revision
    if result["last_code_review_verdict"] == "needs_revision":
        new_count, incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", MAX_CODE_REVISIONS
        )
        result["code_revision_count"] = new_count
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", "Missing verdict or feedback in review"))
        
        # If we hit the max revision budget, escalate to ask_user immediately.
        if not incremented:
            runtime_config = state.get("runtime_config", {})
            max_revs = runtime_config.get("max_code_revisions", MAX_CODE_REVISIONS)
            stage_id = state.get("current_stage_id", "unknown")
            question = (
                "Code review limit reached.\n\n"
                f"- Stage: {stage_id}\n"
                f"- Attempts: {new_count}/{max_revs}\n"
                "- Latest reviewer feedback:\n"
                f"  {result.get('reviewer_feedback', 'No feedback available')}\n\n"
                "Please respond with PROVIDE_HINT (include guidance for next attempt), "
                "SKIP to bypass this stage, or STOP to end the workflow."
            )
            result.update({
                "ask_user_trigger": "code_review_limit",
                "pending_user_questions": [question],
                "awaiting_user_input": True,
                "last_node_before_ask_user": "code_review",
            })
    
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
        return create_llm_error_escalation("code_generator", "code_generation", e)
    
    # Extract generated code from agent output
    generated_code = ""
    if isinstance(agent_output, dict):
        # Check if "code" key exists (even if empty string)
        if "code" in agent_output:
            generated_code = agent_output["code"]
        elif "simulation_code" in agent_output:
            generated_code = agent_output["simulation_code"]
        else:
            # Fall back to JSON dump only if neither key exists
            generated_code = json.dumps(agent_output, indent=2)
    else:
        generated_code = str(agent_output)

    # Validate generated code
    # Stub markers indicate incomplete code when they appear at the start
    # or when code is very short. TODO in comments of otherwise valid code is OK.
    stub_markers = ["STUB", "TODO", "PLACEHOLDER", "# Replace", "would be generated"]
    code_stripped = generated_code.strip()
    code_stripped_upper = code_stripped.upper()
    
    # Only flag as stub if:
    # 1. Code is very short (< 100 chars) and contains stub markers, OR
    # 2. Stub markers appear at the very start of the code (first line)
    is_stub = False
    if len(code_stripped) < 100:
        # Short code with stub markers is likely a stub
        is_stub = any(marker in code_stripped_upper for marker in stub_markers)
    else:
        # For longer code, only flag if stub markers appear at the start
        first_line = code_stripped.split('\n')[0].upper()
        is_stub = any(
            first_line.startswith(marker) or 
            first_line.startswith(f"# {marker}") or
            first_line.startswith(f"// {marker}")
            for marker in stub_markers
        )
    
    is_empty = not generated_code or not generated_code.strip() or len(generated_code.strip()) < 50
    
    if is_stub or is_empty:
        logger.error(
            f"Generated code is stub or empty (stub={is_stub}, empty={is_empty}). "
            "Code generation must produce valid simulation code."
        )
        # Calculate new revision count respecting max
        new_count, _ = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", MAX_CODE_REVISIONS
        )
        
        result: Dict[str, Any] = {
            "workflow_phase": "code_generation",
            "code": generated_code,
            "code_revision_count": new_count,
            "reviewer_feedback": (
                "ERROR: Generated code is empty or contains stub markers. "
                "Code generation must produce valid Meep simulation code. "
                "Please regenerate with proper implementation."
            ),
        }
        return result
    
    result = {
        "workflow_phase": "code_generation",
        "code": generated_code,
        "expected_outputs": agent_output.get("expected_outputs", []) if isinstance(agent_output, dict) else [],
    }
    
    log_agent_call("CodeGeneratorAgent", "generate_code", start_time)(state, result)
    return result

