"""
Trigger handlers for supervisor ask_user responses.

Each handler processes a specific ask_user_trigger type and updates
the result dict appropriately.

All handlers follow the signature:
    handler(state, result, user_responses, current_stage_id, **kwargs) -> None

They mutate the result dict in place and return None.

NOTE: User options are defined in src/agents/user_options.py as the single
source of truth for what options are shown to users and how they are matched.
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
from src.agents.user_options import (
    match_user_response,
    get_clarification_message,
    extract_guidance_text,
)

logger = logging.getLogger(__name__)


# Common keyword sets for response parsing (kept for material_checkpoint and backtrack_approval)
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
    
    Options defined in user_options.py: APPROVE, CHANGE_DATABASE, CHANGE_MATERIAL, NEED_HELP, STOP
    
    Note: This handler has special logic for combined rejection+keyword patterns
    (e.g., "REJECT DATABASE" -> CHANGE_DATABASE). Rejection patterns take precedence
    over approval keywords when both are present.
    """
    response_text = parse_user_response(user_responses)
    
    # Check for rejection/change keywords FIRST - these take precedence
    is_rejection = check_keywords(response_text, REJECTION_KEYWORDS)
    is_approval = check_keywords(response_text, APPROVAL_KEYWORDS)
    
    # Check for database-related keywords (CHANGE_DATABASE, DATABASE, or rejection + database)
    wants_database_change = (
        check_keywords(response_text, ["CHANGE_DATABASE"]) or 
        (is_rejection and check_keywords(response_text, ["DATABASE"]))
    )
    
    # Check for material-related keywords (CHANGE_MATERIAL, MATERIAL, or rejection + material)
    wants_material_change = (
        check_keywords(response_text, ["CHANGE_MATERIAL"]) or
        (is_rejection and check_keywords(response_text, ["MATERIAL"]))
    )
    
    # Database change requested
    if wants_database_change:
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User rejected material validation and requested database change: {response_text}."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "needs_rerun",
                invalidation_reason="User requested material change"
            )
        result["pending_validated_materials"] = []
        result["validated_materials"] = []
        return
    
    # Material change requested
    if wants_material_change:
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User indicated wrong material: {response_text}. Please update plan."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "needs_rerun",
                invalidation_reason="User rejected material"
            )
        result["pending_validated_materials"] = []
        result["validated_materials"] = []
        return
    
    # If both approval AND rejection without specific target, ask for clarification
    # (e.g., "YES but I don't like it")
    if is_approval and is_rejection:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Your response contains both approval and rejection indicators. "
            "Please be more specific: APPROVE, CHANGE_DATABASE, or CHANGE_MATERIAL?"
        ]
        return
    
    # If rejection alone without DATABASE/MATERIAL, ask for clarification
    if is_rejection:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "You indicated rejection but didn't specify what to change. "
            f"{get_clarification_message('material_checkpoint')}"
        ]
        return
    
    # Now try standard matching
    matched = match_user_response("material_checkpoint", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            f"Your response '{response_text[:100]}' is unclear. "
            f"{get_clarification_message('material_checkpoint')}"
        ]
        return
    
    if matched.action == "change_database":
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User rejected material validation and requested database change: {response_text}."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "needs_rerun",
                invalidation_reason="User requested material change"
            )
        result["pending_validated_materials"] = []
        result["validated_materials"] = []
    
    elif matched.action == "change_material":
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User indicated wrong material: {response_text}. Please update plan."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "needs_rerun",
                invalidation_reason="User rejected material"
            )
        result["pending_validated_materials"] = []
        result["validated_materials"] = []
    
    elif matched.action == "approve":
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
            result["validated_materials"] = []
            result["pending_validated_materials"] = []
            return
        
        if current_stage_id:
            _archive_with_error_handling(state, result, current_stage_id)
            _update_progress_with_error_handling(state, result, current_stage_id, "completed_success")
    
    elif matched.action == "need_help":
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Please provide more details about the material issue."
        ]
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_code_review_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle code_review_limit trigger response.
    
    Options defined in user_options.py: PROVIDE_HINT, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("code_review_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("code_review_limit")]
        return
    
    if matched.action == "provide_hint":
        result["code_revision_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["reviewer_feedback"] = f"User hint: {extract_guidance_text(raw_response)}"
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying code generation with user hint."
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to code review issues"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_design_review_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle design_review_limit trigger response.
    
    Options defined in user_options.py: PROVIDE_HINT, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("design_review_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("design_review_limit")]
        return
    
    if matched.action == "provide_hint":
        result["design_revision_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["reviewer_feedback"] = f"User hint: {extract_guidance_text(raw_response)}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to design review issues"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_execution_failure_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle execution_failure_limit trigger response.
    
    Options defined in user_options.py: RETRY_WITH_GUIDANCE, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("execution_failure_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("execution_failure_limit")]
        return
    
    if matched.action == "retry_with_guidance":
        result["execution_failure_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["supervisor_feedback"] = f"User guidance: {extract_guidance_text(raw_response)}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to execution failures"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_physics_failure_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle physics_failure_limit trigger response.
    
    Options defined in user_options.py: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("physics_failure_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("physics_failure_limit")]
        return
    
    if matched.action == "retry_with_guidance":
        result["physics_failure_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["supervisor_feedback"] = f"User guidance: {extract_guidance_text(raw_response)}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif matched.action == "accept_partial":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to physics check failures"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_context_overflow(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Handle context_overflow trigger response.
    
    Options defined in user_options.py: SUMMARIZE, TRUNCATE, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("context_overflow", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("context_overflow")]
        return
    
    if matched.action == "summarize":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Applying feedback summarization for context management."
    
    elif matched.action == "truncate":
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
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped due to context overflow"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_replan_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle replan_limit trigger response.
    
    Options defined in user_options.py: APPROVE_PLAN, GUIDANCE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("replan_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("replan_limit")]
        return
    
    if matched.action == "force_accept":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Plan force-accepted by user."
    
    elif matched.action == "replan_with_guidance":
        result["replan_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        guidance_text = extract_guidance_text(raw_response)
        result["planner_feedback"] = f"User guidance: {guidance_text}"
        result["supervisor_verdict"] = "replan_with_guidance"
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


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
    
    Options defined in user_options.py: GENERATE_REPORT, REPLAN, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("deadlock_detected", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("deadlock_detected")]
        return
    
    if matched.action == "generate_report":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
    
    elif matched.action == "replan":
        result["supervisor_verdict"] = "replan_needed"
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["planner_feedback"] = f"User requested replan due to deadlock: {raw_response}."
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_llm_error(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle llm_error trigger response.
    
    Options defined in user_options.py: RETRY, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("llm_error", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("llm_error")]
        return
    
    if matched.action == "retry":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying after user acknowledged LLM error."
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user after LLM error"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


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
    
    Used for: missing_paper_text, missing_stage_id, progress_init_failed
    Options defined in user_options.py (via "critical_error" alias): RETRY, STOP
    """
    response_text = parse_user_response(user_responses)
    # Use "critical_error" as the trigger - aliases resolve this in user_options
    matched = match_user_response("critical_error", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("critical_error")]
        return
    
    if matched.action == "retry":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying after critical error acknowledged by user."
        
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_planning_error_retry(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Generic handler for planning errors where user can REPLAN or STOP.
    
    Used for: no_stages_available, invalid_backtrack_target, backtrack_target_not_found
    Options defined in user_options.py (via "planning_error" alias): REPLAN, STOP
    """
    response_text = parse_user_response(user_responses)
    # Use "planning_error" as the trigger - aliases resolve this in user_options
    matched = match_user_response("planning_error", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("planning_error")]
        return
    
    if matched.action == "replan":
        result["supervisor_verdict"] = "replan_needed"
        result["planner_feedback"] = f"User requested replan after error: {response_text}"
        
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_backtrack_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle backtrack_limit trigger response.
    
    Options defined in user_options.py: FORCE_CONTINUE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("backtrack_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("backtrack_limit")]
        return
    
    if matched.action == "force_continue":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Continuing despite backtrack limit as per user request."
        
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_invalid_backtrack_decision(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle invalid_backtrack_decision trigger response.
    
    Options defined in user_options.py: CONTINUE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("invalid_backtrack_decision", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("invalid_backtrack_decision")]
        return
    
    if matched.action == "continue":
        result["supervisor_verdict"] = "ok_continue"
        result["backtrack_decision"] = None  # Clear invalid decision
        
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_analysis_limit(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle analysis_limit trigger response.
    
    Options defined in user_options.py: ACCEPT_PARTIAL, PROVIDE_HINT, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("analysis_limit", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("analysis_limit")]
        return
    
    if matched.action == "accept_partial":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "completed_partial",
                summary="Accepted as partial match by user"
            )
    
    elif matched.action == "provide_hint":
        result["analysis_revision_count"] = 0
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        result["analysis_feedback"] = f"User hint: {extract_guidance_text(raw_response)}"
        result["supervisor_verdict"] = "ok_continue"
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_supervisor_error(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle supervisor_error trigger response.
    
    Options defined in user_options.py: RETRY, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("supervisor_error", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("supervisor_error")]
        return
    
    if matched.action == "retry":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying after supervisor error acknowledged by user."
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_missing_design(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle missing_design trigger response.
    
    Options defined in user_options.py: RETRY, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("missing_design", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("missing_design")]
        return
    
    if matched.action == "retry":
        # Reset to design phase
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Returning to design phase as requested."
        result["design_revision_count"] = 0
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to missing design"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


def handle_unknown_escalation(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle unknown_escalation trigger response.
    
    This is a generic fallback for unexpected workflow errors.
    Options defined in user_options.py: RETRY, SKIP_STAGE, STOP
    """
    response_text = parse_user_response(user_responses)
    matched = match_user_response("unknown_escalation", response_text)
    
    if matched is None:
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("unknown_escalation")]
        return
    
    if matched.action == "retry":
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "Retrying after unknown error acknowledged by user."
    
    elif matched.action == "skip":
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to unexpected error"
            )
    
    elif matched.action == "stop":
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True


# Registry of trigger handlers
TRIGGER_HANDLERS: Dict[str, Callable] = {
    "material_checkpoint": handle_material_checkpoint,
    "code_review_limit": handle_code_review_limit,
    "design_review_limit": handle_design_review_limit,
    "execution_failure_limit": handle_execution_failure_limit,
    "physics_failure_limit": handle_physics_failure_limit,
    "analysis_limit": handle_analysis_limit,
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
    "supervisor_error": handle_supervisor_error,
    
    # Missing Requirements
    "missing_design": handle_missing_design,
    
    # Planning Errors (Replan/Stop)
    "no_stages_available": handle_planning_error_retry,
    "invalid_backtrack_target": handle_planning_error_retry,
    "backtrack_target_not_found": handle_planning_error_retry,
    
    # Specific Backtrack Errors
    "backtrack_limit": handle_backtrack_limit,
    "invalid_backtrack_decision": handle_invalid_backtrack_decision,
    
    # Generic Fallback (must be last resort)
    "unknown_escalation": handle_unknown_escalation,
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
