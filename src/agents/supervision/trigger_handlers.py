"""
Trigger handlers for supervisor ask_user responses.

Each handler processes a specific ask_user_trigger type and updates
the result dict appropriately.

All handlers follow the signature:
    handler(state, result, user_responses, current_stage_id, **kwargs) -> None

They mutate the result dict in place and return None.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

from schemas.state import (
    ReproState,
    update_progress_stage_status,
    archive_stage_outputs_to_progress,
)
from src.agents.base import parse_user_response, check_keywords

logger = logging.getLogger(__name__)


# Common keyword sets for response parsing
APPROVAL_KEYWORDS = ["APPROVE", "YES", "CORRECT", "OK", "ACCEPT", "VALID", "PROCEED"]
REJECTION_KEYWORDS = ["REJECT", "NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]


def _archive_with_error_handling(
    state: ReproState,
    result: Dict[str, Any],
    stage_id: str,
) -> None:
    """Archive stage outputs with error handling."""
    try:
        archive_stage_outputs_to_progress(state, stage_id)
    except Exception as e:
        logger.error(f"Failed to archive outputs for stage {stage_id}: {e}")
        archive_errors = result.get("archive_errors", state.get("archive_errors", []))
        archive_errors = archive_errors + [{
            "stage_id": stage_id,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
        result["archive_errors"] = archive_errors


def _update_progress_with_error_handling(
    state: ReproState,
    result: Dict[str, Any],
    stage_id: str,
    status: str,
    summary: Optional[str] = None,
    invalidation_reason: Optional[str] = None,
) -> None:
    """Update progress stage status with error handling."""
    try:
        # Only pass kwargs that are provided (not None)
        kwargs = {}
        if summary is not None:
            kwargs["summary"] = summary
        if invalidation_reason is not None:
            kwargs["invalidation_reason"] = invalidation_reason
        update_progress_stage_status(state, stage_id, status, **kwargs)
    except Exception as e:
        logger.error(f"Failed to update progress for stage {stage_id}: {e}")
        # Continue execution - progress update failure shouldn't block workflow


def handle_material_checkpoint(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle material_checkpoint trigger response.
    
    User can:
    - APPROVE: Accept pending materials and continue
    - REJECT + DATABASE: Request database change
    - REJECT + MATERIAL: Request material change
    - NEED_HELP: Get more guidance
    """
    response_text = parse_user_response(user_responses)
    
    is_approval = check_keywords(response_text, APPROVAL_KEYWORDS)
    is_rejection = check_keywords(response_text, REJECTION_KEYWORDS)
    
    # Check for CHANGE_DATABASE/CHANGE_MATERIAL FIRST (before approval)
    # These take precedence even if approval keywords are present
    if "CHANGE_DATABASE" in response_text or (is_rejection and "DATABASE" in response_text):
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User rejected material validation and requested database change: {response_text}."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "needs_rerun",
                invalidation_reason="User requested material change"
            )
        result["pending_validated_materials"] = []
        result["validated_materials"] = []
    
    elif "CHANGE_MATERIAL" in response_text or (is_rejection and "MATERIAL" in response_text):
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User indicated wrong material: {response_text}. Please update plan."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "needs_rerun",
                invalidation_reason="User rejected material"
            )
        result["pending_validated_materials"] = []
        result["validated_materials"] = []
    
    elif is_approval and not is_rejection:
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Material validation approved by user."
        pending_materials = state.get("pending_validated_materials", [])
        
        if pending_materials:
            result["validated_materials"] = pending_materials
            result["pending_validated_materials"] = []
        else:
            result["supervisor_verdict"] = "ask_user"
            result["pending_user_questions"] = [
                "ERROR: No materials were extracted for validation. "
                "Please specify materials manually using CHANGE_MATERIAL or CHANGE_DATABASE."
            ]
            # Set both material lists to empty even when error
            result["validated_materials"] = []
            result["pending_validated_materials"] = []
            return
        
        if current_stage_id:
            _archive_with_error_handling(state, result, current_stage_id)
            _update_progress_with_error_handling(state, result, current_stage_id, "completed_success")
    
    elif "NEED_HELP" in response_text or "HELP" in response_text:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please provide more details about the material issue."
        ]
    
    elif is_rejection:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "You indicated rejection but didn't specify what to change."
        ]
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            f"Your response '{response_text[:100]}' is unclear."
        ]


def handle_code_review_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle code_review_limit trigger response.
    
    User can:
    - PROVIDE_HINT: Reset counter and retry with user guidance
    - SKIP: Skip this stage
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "PROVIDE_HINT" in response_text or "HINT" in response_text:
        result["code_revision_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["reviewer_feedback"] = f"User hint: {raw_response}"
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying code generation with user hint."
    
    elif "SKIP" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to code review issues"
            )
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"
        ]


def handle_design_review_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle design_review_limit trigger response.
    
    User can:
    - PROVIDE_HINT: Reset counter and retry with user guidance
    - SKIP: Skip this stage
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "PROVIDE_HINT" in response_text or "HINT" in response_text:
        result["design_revision_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["reviewer_feedback"] = f"User hint: {raw_response}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif "SKIP" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to design review issues"
            )
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"
        ]


def handle_execution_failure_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle execution_failure_limit trigger response.
    
    User can:
    - RETRY_WITH_GUIDANCE: Reset counter and retry with guidance
    - SKIP: Skip this stage
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "RETRY" in response_text or "GUIDANCE" in response_text:
        result["execution_failure_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["supervisor_feedback"] = f"User guidance: {raw_response}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif "SKIP" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            update_progress_stage_status(
                state, current_stage_id, "blocked",
                summary="Skipped by user due to execution failures"
            )
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: RETRY_WITH_GUIDANCE, SKIP_STAGE, or STOP?"
        ]


def handle_physics_failure_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle physics_failure_limit trigger response.
    
    User can:
    - RETRY: Reset counter and retry
    - ACCEPT_PARTIAL: Mark stage as partial success
    - SKIP: Skip this stage
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "RETRY" in response_text:
        result["physics_failure_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["supervisor_feedback"] = f"User guidance: {raw_response}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif "ACCEPT" in response_text or "PARTIAL" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )
    
    elif "SKIP" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            update_progress_stage_status(
                state, current_stage_id, "blocked",
                summary="Skipped by user due to physics check failures"
            )
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, or STOP?"
        ]


def handle_context_overflow(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle context_overflow trigger response.
    
    User can:
    - SUMMARIZE: Apply feedback summarization
    - TRUNCATE: Truncate paper text
    - SKIP: Skip this stage
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "SUMMARIZE" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Applying feedback summarization for context management."
    
    elif "TRUNCATE" in response_text:
        current_text = state.get("paper_text", "")
        # Truncation marker adds 39 chars, so truncated length = 15000 + 39 + 5000 = 20039
        # Only truncate if original is longer than truncated result would be
        truncation_marker = "\n\n... [TRUNCATED BY USER REQUEST] ...\n\n"
        truncated_length = 15000 + len(truncation_marker) + 5000
        if len(current_text) > truncated_length:
            truncated_text = (
                current_text[:15000] +
                truncation_marker +
                current_text[-5000:]
            )
            result["paper_text"] = truncated_text
            result["supervisor_feedback"] = "Truncating paper to first 15k and last 5k chars."
        else:
            # Preserve paper_text even when not truncating
            result["paper_text"] = current_text
            result["supervisor_feedback"] = "Paper already short enough, proceeding."
        result["supervisor_verdict"] = "ok_continue"
    
    elif "SKIP" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            update_progress_stage_status(
                state, current_stage_id, "blocked",
                summary="Skipped due to context overflow"
            )
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: SUMMARIZE, TRUNCATE, SKIP_STAGE, or STOP?"
        ]


def handle_replan_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle replan_limit trigger response.
    
    User can:
    - FORCE/ACCEPT: Force-accept current plan
    - GUIDANCE: Reset counter and retry with guidance
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "FORCE" in response_text or "ACCEPT" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Plan force-accepted by user."
    
    elif "GUIDANCE" in response_text:
        result["replan_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        # Strip "GUIDANCE:" prefix (case-insensitive) from the raw response
        guidance_text = raw_response
        if guidance_text:
            # Remove "GUIDANCE:" prefix if present (case-insensitive)
            guidance_upper = guidance_text.upper().strip()
            if guidance_upper.startswith("GUIDANCE"):
                # Find the colon after GUIDANCE and strip everything before it (including colon and whitespace)
                colon_idx = guidance_text.find(":")
                if colon_idx != -1:
                    guidance_text = guidance_text[colon_idx + 1:].strip()
                else:
                    # No colon, just remove "GUIDANCE" keyword
                    guidance_text = guidance_text[len("GUIDANCE"):].strip()
        result["planner_feedback"] = f"User guidance: {guidance_text}"
        result["supervisor_verdict"] = "replan_needed"
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: FORCE_ACCEPT, GUIDANCE, or STOP?"
        ]


def handle_backtrack_approval(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
    get_dependent_stages_fn: Optional[Callable] = None,
) -> None:
    """
    Handle backtrack_approval trigger response.
    
    User can:
    - APPROVE: Proceed with backtrack
    - REJECT: Cancel backtrack
    """
    response_text = parse_user_response(user_responses)
    
    is_approval = check_keywords(response_text, APPROVAL_KEYWORDS)
    is_rejection = check_keywords(response_text, REJECTION_KEYWORDS)
    
    if is_approval and not is_rejection:
        result["supervisor_verdict"] = "backtrack_to_stage"
        decision = state.get("backtrack_decision", {})
        if decision:
            if get_dependent_stages_fn:
                target = decision.get("target_stage_id")
                if target:
                    dependent = get_dependent_stages_fn(state.get("plan", {}), target)
                    # Ensure stages_to_invalidate is always a list, never None
                    decision["stages_to_invalidate"] = dependent if dependent is not None else []
            else:
                # When no function provided, set empty list for stages_to_invalidate
                decision["stages_to_invalidate"] = []
            result["backtrack_decision"] = decision
    
    elif is_rejection:
        result["backtrack_suggestion"] = None
        result["supervisor_verdict"] = "ok_continue"
    
    else:
        result["supervisor_verdict"] = "ok_continue"


def handle_deadlock_detected(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle deadlock_detected trigger response.
    
    User can:
    - GENERATE_REPORT: Generate report with current progress
    - REPLAN: Request replanning
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "GENERATE_REPORT" in response_text or "REPORT" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    elif "REPLAN" in response_text:
        result["supervisor_verdict"] = "replan_needed"
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["planner_feedback"] = f"User requested replan due to deadlock: {raw_response}."
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: GENERATE_REPORT, REPLAN, or STOP?"
        ]


def handle_llm_error(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle llm_error trigger response.
    
    User can:
    - RETRY: Retry the LLM call
    - SKIP: Skip this stage
    - STOP: Stop workflow
    """
    response_text = parse_user_response(user_responses)
    
    if "RETRY" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying after user acknowledged LLM error."
    
    elif "SKIP" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user after LLM error"
            )
    
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: RETRY, SKIP_STAGE, or STOP?"
        ]


def handle_clarification(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle clarification trigger response.
    
    Free-form response from user to clarify ambiguity.
    Adds clarification to assumptions/feedback and continues.
    """
    raw_response = list(user_responses.values())[-1] if user_responses else ""
    
    if raw_response:
        # Append clarification to assumptions or feedback
        # Here we append to supervisor_feedback to ensure it reaches the relevant agent
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = f"User clarification provided: {raw_response}"
    else:
        # If user didn't provide anything, ask again? or just continue?
        # Usually 'ask_user' requires input, so empty might mean "no clarification"
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "No clarification provided by user."


def handle_critical_error_retry(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Generic handler for critical errors where user can RETRY or STOP.
    
    Used for:
    - missing_paper_text
    - missing_stage_id
    - progress_init_failed
    """
    response_text = parse_user_response(user_responses)
    
    if "RETRY" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying after critical error acknowledged by user."
        
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
        
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: RETRY or STOP?"
        ]


def handle_planning_error_retry(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Generic handler for planning errors where user can REPLAN or STOP.
    
    Used for:
    - no_stages_available
    - invalid_backtrack_target
    - backtrack_target_not_found
    """
    response_text = parse_user_response(user_responses)
    
    if "REPLAN" in response_text:
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User requested replan after error: {response_text}"
        
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
        
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: REPLAN or STOP?"
        ]


def handle_backtrack_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle backtrack_limit trigger response.
    
    User can:
    - STOP: Stop workflow
    - FORCE_CONTINUE: Ignore limit and continue
    """
    response_text = parse_user_response(user_responses)
    
    if "FORCE" in response_text or "CONTINUE" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Continuing despite backtrack limit as per user request."
        # Reset or increment limit? Logic usually handles count, we just unblock verdict.
        
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
        
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: FORCE_CONTINUE or STOP?"
        ]


def handle_invalid_backtrack_decision(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle invalid_backtrack_decision trigger response.
    
    User can:
    - STOP: Stop workflow
    - CONTINUE: Continue normally (ignoring invalid backtrack)
    """
    response_text = parse_user_response(user_responses)
    
    if "CONTINUE" in response_text:
        result["supervisor_verdict"] = "ok_continue"
        result["backtrack_decision"] = None  # Clear invalid decision
        
    elif "STOP" in response_text:
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
        
    else:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please clarify: CONTINUE or STOP?"
        ]


# Registry of trigger handlers
TRIGGER_HANDLERS: Dict[str, Callable] = {
    "material_checkpoint": handle_material_checkpoint,
    "code_review_limit": handle_code_review_limit,
    "design_review_limit": handle_design_review_limit,
    "execution_failure_limit": handle_execution_failure_limit,
    "physics_failure_limit": handle_physics_failure_limit,
    "context_overflow": handle_context_overflow,
    "replan_limit": handle_replan_limit,
    "backtrack_approval": handle_backtrack_approval,
    "deadlock_detected": handle_deadlock_detected,
    "llm_error": handle_llm_error,
    "clarification": handle_clarification,
    
    # Critical Errors (Retry/Stop)
    "missing_paper_text": handle_critical_error_retry,
    "missing_stage_id": handle_critical_error_retry,
    "progress_init_failed": handle_critical_error_retry,
    
    # Planning Errors (Replan/Stop)
    "no_stages_available": handle_planning_error_retry,
    "invalid_backtrack_target": handle_planning_error_retry,
    "backtrack_target_not_found": handle_planning_error_retry,
    
    # Specific Backtrack Errors
    "backtrack_limit": handle_backtrack_limit,
    "invalid_backtrack_decision": handle_invalid_backtrack_decision,
}


def handle_trigger(
    trigger: str,
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
    get_dependent_stages_fn: Optional[Callable] = None,
) -> None:
    """
    Dispatch to appropriate trigger handler.
    
    Args:
        trigger: The ask_user_trigger value
        state: Current workflow state
        result: Result dict to update (mutated in place)
        user_responses: User's responses to questions
        current_stage_id: Current stage ID if any
        get_dependent_stages_fn: Function to get dependent stages (for backtrack)
        
    Returns:
        None (result dict is mutated in place)
    """
    handler = TRIGGER_HANDLERS.get(trigger)
    
    if handler:
        # backtrack_approval needs the dependent stages function
        if trigger == "backtrack_approval":
            handler(state, result, user_responses, current_stage_id, get_dependent_stages_fn)
        else:
            handler(state, result, user_responses, current_stage_id)
    else:
        # Unknown trigger - default to continue
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = f"Handled unknown trigger: {trigger}"
