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

import json
import logging
import re
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
from src.prompts import build_agent_prompt
from src.llm_client import call_agent_with_metrics

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
        result["supervisor_verdict"] = "retry_generate_code"
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
        result["supervisor_verdict"] = "retry_design"
    
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
        user_guidance = extract_guidance_text(raw_response)
        # Combine original execution feedback with user guidance so code generator gets full context
        original_fb = state.get("execution_feedback", "")
        if original_fb:
            result["execution_feedback"] = f"{original_fb}\n\nUSER GUIDANCE: {user_guidance}"
        else:
            result["execution_feedback"] = f"USER GUIDANCE: {user_guidance}"
        result["supervisor_feedback"] = f"User guidance applied to execution_feedback"
        result["supervisor_verdict"] = "retry_generate_code"
    
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
        user_guidance = extract_guidance_text(raw_response)
        original_fb = state.get("physics_feedback", "")
        combined_fb = f"{original_fb}\n\nUSER GUIDANCE: {user_guidance}" if original_fb else f"USER GUIDANCE: {user_guidance}"
        
        # Route based on where the physics failure originated
        # Write to the feedback field that the destination node actually reads
        last_node = state.get("last_node_before_ask_user", "")
        if last_node == "design_review" or "design" in last_node.lower():
            # Designer reads reviewer_feedback, not physics_feedback
            result["reviewer_feedback"] = combined_fb
            result["physics_feedback"] = None  # Clear stale data
            result["supervisor_feedback"] = "User guidance applied to reviewer_feedback for designer"
            result["supervisor_verdict"] = "retry_design"
        else:
            # Code generator reads physics_feedback (labeled as PHYSICS VALIDATION FEEDBACK)
            result["physics_feedback"] = combined_fb
            result["supervisor_feedback"] = "User guidance applied to physics_feedback for code generator"
            result["supervisor_verdict"] = "retry_generate_code"
    
    elif matched.action == "accept_partial":
        # User accepts partial results - proceed to analysis phase
        result["supervisor_verdict"] = "retry_analyze"
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
    
    # Debug logging to trace match results
    logger.debug(
        f"handle_analysis_limit: user_responses={user_responses}, "
        f"response_text='{response_text}', matched={matched}"
    )
    
    if matched is None:
        logger.info(f"handle_analysis_limit: No match found for '{response_text}', asking for clarification")
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [get_clarification_message("analysis_limit")]
        return
    
    if matched.action == "accept_partial":
        logger.info(f"handle_analysis_limit: Matched 'accept_partial', setting verdict=ok_continue")
        result["supervisor_verdict"] = "ok_continue"
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "completed_partial",
                summary="Accepted as partial match by user"
            )
    
    elif matched.action == "provide_hint":
        raw_response = list(user_responses.values())[-1] if user_responses else ""
        hint_text = extract_guidance_text(raw_response)
        logger.info(f"handle_analysis_limit: Matched 'provide_hint', setting verdict=retry_analyze, hint='{hint_text[:100]}...'")
        result["analysis_revision_count"] = 0
        result["analysis_feedback"] = f"User hint: {hint_text}"
        result["supervisor_verdict"] = "retry_analyze"
    
    elif matched.action == "stop":
        logger.info(f"handle_analysis_limit: Matched 'stop', setting verdict=all_complete")
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
        result["supervisor_verdict"] = "retry_design"
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


def _build_routing_context(
    state: ReproState,
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> str:
    """
    Build comprehensive context for LLM-based routing decision.
    
    Provides the supervisor LLM with all the information needed to decide
    where to route after user provides guidance.
    """
    trigger = state.get("ask_user_trigger", "unknown")
    last_node = state.get("last_node_before_ask_user", "unknown")
    stage_type = state.get("current_stage_type", "unknown")
    pending_questions = state.get("pending_user_questions", [])
    response_text = parse_user_response(user_responses)
    
    # Get progress summary
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    completed = [s.get("stage_id") for s in stages if s.get("status", "").startswith("completed")]
    in_progress = [s.get("stage_id") for s in stages if s.get("status") == "in_progress"]
    
    context = f"""# USER RESPONSE ROUTING DECISION

## Context
- **Trigger:** {trigger}
- **Source Node:** {last_node}
- **Current Stage:** {current_stage_id or "None"}
- **Stage Type:** {stage_type}

## Progress
- **Completed stages:** {", ".join(completed) if completed else "None"}
- **In progress:** {", ".join(in_progress) if in_progress else "None"}

## Original Question
{pending_questions[0] if pending_questions else "N/A"}

## User Response
{response_text}

## Available Routing Options

Based on what the user's response REQUIRES (not where you came from), choose the appropriate verdict:

| Verdict | Routes To | Use When |
|---------|-----------|----------|
| `ok_continue` | Next stage | User approves current state, no changes needed |
| `retry_generate_code` | Code generator | User provides parameter fix, value correction, algorithm change |
| `retry_design` | Designer | User requests design-level change (geometry, model type, simulation approach) |
| `retry_code_review` | Code reviewer | User answers a code reviewer question (re-run review with answer) |
| `retry_design_review` | Design reviewer | User answers a design reviewer question |
| `retry_plan_review` | Plan reviewer | User answers a plan reviewer question |
| `retry_analyze` | Analyzer | User provides analysis/comparison hint |
| `replan_needed` | Planner | System decides replan needed (fundamental issue) |
| `replan_with_guidance` | Planner | User explicitly requests plan changes with specific guidance |
| `backtrack_to_stage` | Backtrack handler | Need to redo an earlier stage (set backtrack_decision.target_stage_id) |
| `ask_user` | Ask user again | Response unclear, need clarification |
| `all_complete` | End workflow | User wants to stop |

## Key Insight
The source node tells you CONTEXT, not DESTINATION. Route based on what the user's response REQUIRES.

Examples:
- physics_check + "fix gamma to 66 meV" -> `retry_generate_code` (code needs parameter change)
- physics_check + "use Drude-Lorentz model" -> `retry_design` (design-level model change)
- physics_check + "looks acceptable" -> `ok_continue` (user approves)
- code_review + "reduce resolution to 10" -> `retry_generate_code` (code parameter)
- any + "I'm not sure what to do" -> `ask_user` (need clarification)

## Your Task
Decide the best verdict and provide feedback for the target node.
"""
    return context


def _route_with_llm(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """
    Use supervisor LLM to decide routing based on user response.
    
    The LLM has full flexibility to route to any node based on what
    the user's response requires.
    """
    response_text = parse_user_response(user_responses)
    guidance_text = extract_guidance_text(
        list(user_responses.values())[-1] if user_responses else ""
    )
    
    # Build context for LLM
    user_content = _build_routing_context(state, user_responses, current_stage_id)
    
    try:
        system_prompt = build_agent_prompt("supervisor", state)
        agent_output = call_agent_with_metrics(
            agent_name="supervisor",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
        
        verdict = agent_output.get("verdict", "ok_continue")
        summary = agent_output.get("summary", "")
        
        result["supervisor_verdict"] = verdict
        result["supervisor_feedback"] = summary
        
        # Map verdict to correct feedback field
        # Different nodes read different feedback fields
        if verdict in ["retry_generate_code", "retry_design", "retry_code_review", "retry_design_review"]:
            result["reviewer_feedback"] = f"User guidance: {guidance_text}"
        elif verdict in ["replan_needed", "replan_with_guidance", "retry_plan_review"]:
            result["planner_feedback"] = f"User guidance: {guidance_text}"
        elif verdict == "retry_analyze":
            result["analysis_feedback"] = f"User guidance: {guidance_text}"
        elif verdict == "backtrack_to_stage":
            # LLM should have provided backtrack_decision with target_stage_id
            backtrack_decision = agent_output.get("backtrack_decision", {})
            target_stage = backtrack_decision.get("target_stage_id", "") if backtrack_decision else ""
            
            if target_stage:
                # Valid backtrack decision
                result["backtrack_decision"] = {
                    "target_stage_id": target_stage,
                    "reason": backtrack_decision.get("reason", guidance_text),
                    "accepted": True,
                }
            else:
                # LLM wants to backtrack but didn't specify target - ask user
                logger.warning(
                    "LLM returned backtrack_to_stage without target_stage_id. "
                    "Asking user for clarification."
                )
                # Get list of completed stages for user reference
                progress = state.get("progress", {})
                stages = progress.get("stages", [])
                completed_stages = [
                    s.get("stage_id") for s in stages 
                    if s.get("status", "").startswith("completed")
                ]
                stage_list = ", ".join(completed_stages) if completed_stages else "No stages completed yet"
                
                # Override verdict to ask_user
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    f"You mentioned going back to an earlier stage. Which stage should we backtrack to?\n\n"
                    f"**Completed stages:** {stage_list}\n\n"
                    f"Please respond with the stage ID (e.g., 'stage0', 'stage1') or 'CANCEL' to continue without backtracking."
                ]
                result["awaiting_user_input"] = True
        elif verdict == "ask_user":
            # LLM needs clarification - set the question for the user
            user_question = agent_output.get("user_question", "")
            if user_question:
                result["pending_user_questions"] = [user_question]
            else:
                # Fallback question if LLM didn't provide one
                result["pending_user_questions"] = [
                    f"Your response was unclear. Could you please clarify?\n\n"
                    f"Original response: {guidance_text[:200]}{'...' if len(guidance_text) > 200 else ''}"
                ]
            result["awaiting_user_input"] = True
        # For ok_continue, all_complete - no special feedback field needed
        
        logger.info(
            f"LLM routing decision: verdict={verdict}, "
            f"source={state.get('last_node_before_ask_user', 'unknown')}"
        )
        
    except Exception as e:
        logger.warning(f"LLM routing failed: {e}. Defaulting to ok_continue with guidance.")
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = f"LLM routing unavailable: {str(e)[:100]}"
        result["reviewer_feedback"] = f"User guidance: {guidance_text}"


def handle_reviewer_escalation(
    state: ReproState,
    result: Dict[str, Any],
    user_responses: Dict[str, str],
    current_stage_id: Optional[str] = None,
) -> None:
    """
    Handle reviewer_escalation trigger response.
    
    Uses a hybrid approach:
    1. FAST PATH: Simple keyword matching for obvious choices (STOP, SKIP, APPROVE)
    2. SMART PATH: LLM decides routing for everything else
    
    The LLM has full flexibility to route to any node based on what
    the user's response requires, not restricted by the source node.
    """
    response_text = parse_user_response(user_responses)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FAST PATH: Explicit keywords (no LLM needed)
    # ═══════════════════════════════════════════════════════════════════════
    
    # STOP - user wants to end the workflow
    if check_keywords(response_text, ["STOP", "QUIT", "EXIT", "ABORT", "END"]):
        result["supervisor_verdict"] = "all_complete"
        result["should_stop"] = True
        result["supervisor_feedback"] = "User requested workflow stop."
        return
    
    # SKIP - user wants to skip this stage
    if check_keywords(response_text, ["SKIP", "SKIP_STAGE"]):
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "User skipped stage."
        if current_stage_id:
            _update_progress_with_error_handling(
                state, result, current_stage_id, "blocked",
                summary="Skipped by user due to reviewer escalation"
            )
        return
    
    # APPROVE - explicit approval, no changes needed
    # Require keyword at START of response to avoid false matches like "I accept that..."
    # Matches: "APPROVE", "APPROVE:", "APPROVE: some note", etc.
    approval_pattern = r"^\s*(APPROVE|PROCEED|ACCEPT|LOOKS_GOOD|OK_CONTINUE)\s*:?"
    if re.match(approval_pattern, response_text.strip(), re.IGNORECASE):
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = "User explicitly approved, continuing."
        return
    
    # Empty response - ask for clarification
    if not response_text.strip():
        result["supervisor_verdict"] = "ask_user"
        result["pending_user_questions"] = [
            "Your response was empty. Please provide your answer to the question, "
            "or type SKIP_STAGE to skip this stage, or STOP to end the workflow."
        ]
        return
    
    # ═══════════════════════════════════════════════════════════════════════
    # SMART PATH: LLM decides routing (full flexibility)
    # ═══════════════════════════════════════════════════════════════════════
    _route_with_llm(state, result, user_responses, current_stage_id)


# Registry of trigger handlers
TRIGGER_HANDLERS: Dict[str, Callable] = {
    "material_checkpoint": handle_material_checkpoint,
    "code_review_limit": handle_code_review_limit,
    "design_review_limit": handle_design_review_limit,
    "design_flaw_limit": handle_design_review_limit,  # Same as design_review_limit (from physics_check)
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
    
    # Reviewer Escalation (explicit escalate_to_user from reviewer LLM)
    "reviewer_escalation": handle_reviewer_escalation,
    
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
