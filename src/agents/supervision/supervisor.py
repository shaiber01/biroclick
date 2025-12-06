"""
Supervisor agent node: supervisor_node.

The supervisor handles big-picture assessment and decisions,
including handling user responses and routing decisions.

State Keys
----------
supervisor_node:
    READS: current_stage_id, plan, progress, stage_outputs, analysis_reports,
           stage_comparisons, validated_materials, pending_validated_materials,
           ask_user_trigger, user_responses, pending_user_questions,
           supervisor_call_count, backtrack_count, runtime_config, metrics,
           _trigger_persistence_count, _last_seen_trigger, _trigger_first_seen_time
    WRITES: workflow_phase, supervisor_verdict, supervisor_feedback, progress,
            stage_outputs, validated_materials, pending_validated_materials,
            current_stage_id, user_responses, pending_user_questions,
            ask_user_trigger, supervisor_call_count, backtrack_count, 
            backtrack_decision, replan_count, design_feedback,
            code_feedback, design_revision_count, code_revision_count,
            _trigger_persistence_count, _last_seen_trigger, _trigger_first_seen_time,
            _last_stuck_trigger_recovery
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from schemas.state import (
    ReproState,
    get_validation_hierarchy,
    update_progress_stage_status,
    archive_stage_outputs_to_progress,
    MAX_REPLANS,
)
from src.prompts import build_agent_prompt
from src.llm_client import call_agent_with_metrics

from src.agents.helpers.context import check_context_or_escalate
from src.agents.helpers.validation import (
    stage_comparisons_for_stage,
    breakdown_comparison_classifications,
)
from .trigger_handlers import handle_trigger
from src.agents.user_options import get_options_prompt


def _get_dependent_stages(plan: dict, target_stage_id: str) -> list:
    """
    Identify all stages that depend on the target stage (transitively).
    
    Args:
        plan: The plan dictionary containing stages and dependencies
        target_stage_id: The ID of the stage being backtracked to
        
    Returns:
        List of stage_ids that depend on target_stage_id
    """
    stages = plan.get("stages", [])
    # Handle None case: if stages is None, treat as empty list
    if stages is None:
        stages = []
    
    # Build dependents map, skipping stages without stage_id
    dependents_map = {}
    for s in stages:
        if isinstance(s, dict) and "stage_id" in s:
            dependents_map[s["stage_id"]] = []
    
    # Build dependency relationships
    for stage in stages:
        if not isinstance(stage, dict) or "stage_id" not in stage:
            continue
        
        stage_id = stage["stage_id"]
        dependencies = stage.get("dependencies", [])
        # Handle None case: if dependencies is None, treat as empty list
        if dependencies is None:
            dependencies = []
        
        # Ensure dependencies is iterable
        if not isinstance(dependencies, (list, tuple)):
            continue
        
        for dep in dependencies:
            if dep in dependents_map:
                dependents_map[dep].append(stage_id)
                
    invalidated = set()
    queue = [target_stage_id]
    
    while queue:
        current = queue.pop(0)
        if current in dependents_map:
            for dep in dependents_map[current]:
                if dep not in invalidated:
                    invalidated.add(dep)
                    queue.append(dep)
                    
    return list(invalidated)


def _derive_stage_completion_outcome(
    state: ReproState,
    stage_id: Optional[str],
) -> Tuple[str, str]:
    """
    Determine stage completion status and summary from analysis results.
    
    Returns:
        Tuple of (status, summary_text)
    """
    classification = (state.get("analysis_overall_classification") or "").upper()
    comparison_verdict = state.get("comparison_verdict")
    physics_verdict = state.get("physics_verdict")
    comparisons = stage_comparisons_for_stage(state, stage_id)
    comparison_breakdown = breakdown_comparison_classifications(comparisons)
    
    classification_map = {
        "FAILED": "completed_failed",
        "POOR_MATCH": "completed_failed",
        "PARTIAL_MATCH": "completed_partial",
        "ACCEPTABLE_MATCH": "completed_success",
        "EXCELLENT_MATCH": "completed_success",
        "NO_TARGETS": "completed_success",
    }
    status = classification_map.get(classification, "completed_success")
    
    if comparison_breakdown["missing"]:
        status = "completed_failed"
    elif comparison_breakdown["pending"] and status == "completed_success":
        status = "completed_partial"
    
    if comparison_verdict == "needs_revision" and status == "completed_success":
        status = "completed_partial"
    if physics_verdict == "warning" and status == "completed_success":
        status = "completed_partial"
    if physics_verdict == "fail" or classification == "":
        if physics_verdict == "fail":
            status = "completed_failed"
    
    summary_data = state.get("analysis_summary")
    if comparison_breakdown["missing"]:
        summary_text = f"Missing outputs for: {', '.join(comparison_breakdown['missing'])}"
    elif comparison_breakdown["pending"]:
        summary_text = f"Comparisons pending for: {', '.join(comparison_breakdown['pending'])}"
    elif isinstance(summary_data, dict):
        totals = summary_data.get("totals", {})
        summary_text = summary_data.get("notes") or \
            f"{totals.get('matches', 0)}/{totals.get('targets', 0)} targets matched"
    else:
        summary_text = summary_data or \
            f"Stage classified as {classification or comparison_verdict or 'OK_CONTINUE'}"
    
    return status, summary_text


def _retry_archive_errors(
    state: ReproState,
    result: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Retry any failed archive operations from previous runs."""
    archive_errors = state.get("archive_errors", [])
    
    # Handle non-list archive_errors gracefully
    if not isinstance(archive_errors, list):
        logger.warning(
            f"archive_errors has invalid type {type(archive_errors)}, expected list. "
            "Skipping archive error retry."
        )
        result["archive_errors"] = []
        return
    
    if archive_errors:
        logger.info(f"Retrying {len(archive_errors)} failed archive operations...")
        
        retried_errors = []
        for error_entry in archive_errors:
            # Skip entries that aren't dicts
            if not isinstance(error_entry, dict):
                logger.warning(f"Skipping invalid archive error entry: {error_entry}")
                # Keep invalid entries in retried_errors so they're preserved
                retried_errors.append(error_entry)
                continue
            
            stage_id = error_entry.get("stage_id")
            if stage_id:
                try:
                    archive_stage_outputs_to_progress(state, stage_id)
                    logger.info(f"Successfully archived outputs for stage {stage_id} on retry.")
                except Exception as e:
                    logger.warning(
                        f"Archive retry failed for stage {stage_id}: {e}. "
                        "Will retry again on next supervisor call."
                    )
                    retried_errors.append(error_entry)
            else:
                # Keep errors without stage_id (can't retry them)
                retried_errors.append(error_entry)
        
        result["archive_errors"] = retried_errors if retried_errors else []
    else:
        result["archive_errors"] = []


def _run_normal_supervision(
    state: ReproState,
    result: Dict[str, Any],
    system_prompt: str,
    current_stage_id: Optional[str],
    logger: logging.Logger,
) -> None:
    """Run normal supervision (when not handling user response)."""
    validation_hierarchy = get_validation_hierarchy(state)
    
    user_content = f"# CURRENT WORKFLOW STATUS\n\n"
    user_content += f"Current Stage: {current_stage_id or 'None'}\n"
    user_content += f"Workflow Phase: {state.get('workflow_phase', 'unknown')}\n\n"
    
    analysis_summary = state.get("analysis_summary", {})
    if analysis_summary:
        user_content += f"## Analysis Summary\n```json\n{json.dumps(analysis_summary, indent=2, default=str)}\n```\n\n"
    
    user_content += f"## Validation Hierarchy\n```json\n{json.dumps(validation_hierarchy, indent=2)}\n```\n\n"
    
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    completed_stages = [s for s in stages if s.get("status", "").startswith("completed")]
    pending_stages = [s for s in stages if s.get("status") in ["not_started", "in_progress"]]
    
    user_content += f"## Progress\nCompleted: {len(completed_stages)}, Pending: {len(pending_stages)}, Total: {len(stages)}\n"
    
    try:
        agent_output = call_agent_with_metrics(
            agent_name="supervisor",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
        
        result["supervisor_verdict"] = agent_output.get("verdict", "ok_continue")
        result["supervisor_feedback"] = agent_output.get("summary", "")
        
        # Propagate should_stop if present
        if "should_stop" in agent_output:
            result["should_stop"] = agent_output.get("should_stop", False)
        
        # Handle backtrack decision - schema uses nested backtrack_decision object
        backtrack_decision = agent_output.get("backtrack_decision", {})
        if isinstance(backtrack_decision, dict) and backtrack_decision.get("target_stage_id"):
            result["backtrack_decision"] = {
                "target_stage_id": backtrack_decision.get("target_stage_id", ""),
                "reason": backtrack_decision.get("reason", ""),
            }
            
    except Exception as e:
        logger.warning(f"Supervisor LLM call failed: {e}. Defaulting to ok_continue.")
        result["supervisor_verdict"] = "ok_continue"
        result["supervisor_feedback"] = f"LLM unavailable: {str(e)[:200]}"
    
    # Archive current stage outputs
    if current_stage_id:
        try:
            archive_stage_outputs_to_progress(state, current_stage_id)
        except Exception as e:
            logger.error(f"Failed to archive outputs for stage {current_stage_id}: {e}")
            archive_errors = result.get("archive_errors", state.get("archive_errors", []))
            archive_errors = archive_errors + [{
                "stage_id": current_stage_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
            result["archive_errors"] = archive_errors
        
        status, summary_text = _derive_stage_completion_outcome(state, current_stage_id)
        update_progress_stage_status(state, current_stage_id, status, summary=summary_text)


def _log_user_interaction(
    state: ReproState,
    result: Dict[str, Any],
    ask_user_trigger: str,
    user_responses: Dict[str, str],
    current_stage_id: Optional[str],
) -> None:
    """Log user interaction to progress."""
    # Use `or {}` to handle both missing key AND None value
    progress = state.get("progress") or {}
    user_interactions = progress.get("user_interactions") or []
    
    # Safely get question text (may be empty after ask_user clears it)
    pending_questions = state.get("pending_user_questions", [])
    question_text = str(pending_questions[0]) if pending_questions else "(question cleared)"
    
    interaction_entry = {
        "id": f"U{len(user_interactions) + 1}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_type": ask_user_trigger,
        "context": {
            "stage_id": current_stage_id,
            "agent": "SupervisorAgent",
            "reason": ask_user_trigger
        },
        "question": question_text,
        "user_response": str(list(user_responses.values())[-1]) if user_responses else "",
        "impact": result.get("supervisor_feedback", "User decision processed"),
        "alternatives_considered": []
    }
    
    updated_interactions = user_interactions + [interaction_entry]
    result["progress"] = {
        **progress,
        "user_interactions": updated_interactions
    }


def supervisor_node(state: ReproState) -> dict:
    """
    SupervisorAgent: Big-picture assessment and decisions.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT implementation notes:
    
    1. CHECK ask_user_trigger to understand what user was responding to
    2. HANDLE EACH TRIGGER TYPE appropriately (via trigger_handlers)
    3. RESET COUNTERS on user intervention
    4. USE get_validation_hierarchy() - never store directly
    """
    logger = logging.getLogger(__name__)
    
    # Initialize result dict
    result: Dict[str, Any] = {
        "workflow_phase": "supervision",
    }
    
    # Archive error recovery
    _retry_archive_errors(state, result, logger)
    
    # Context check
    context_update = check_context_or_escalate(state, "supervisor")
    if context_update:
        if context_update.get("ask_user_trigger"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("supervisor", state)
    
    # Get trigger info
    ask_user_trigger = state.get("ask_user_trigger")
    user_responses_raw = state.get("user_responses", {})
    
    if not isinstance(user_responses_raw, dict):
        logger.warning(
            f"user_responses has invalid type {type(user_responses_raw)}, expected dict. "
            "Defaulting to empty dict."
        )
        user_responses = {}
    else:
        user_responses = user_responses_raw
    
    current_stage_id = state.get("current_stage_id")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STUCK TRIGGER DETECTION
    # Track how many times we've seen the same trigger without resolution.
    # This catches bugs where triggers aren't being cleared properly.
    # ═══════════════════════════════════════════════════════════════════════
    
    # Configurable thresholds (can be overridden via runtime_config)
    runtime_config = state.get("runtime_config") or {}
    WARN_THRESHOLD = runtime_config.get("stuck_trigger_warn_threshold", 3)
    ERROR_THRESHOLD = runtime_config.get("stuck_trigger_error_threshold", 5)
    AUTO_CLEAR_THRESHOLD = runtime_config.get("stuck_trigger_auto_clear", 7)
    
    # Get current tracking state (handle None values explicitly)
    last_seen_trigger = state.get("_last_seen_trigger")
    persistence_count = state.get("_trigger_persistence_count") or 0  # Treat None as 0
    first_seen_time = state.get("_trigger_first_seen_time")
    
    if ask_user_trigger:
        # Track trigger persistence
        if ask_user_trigger == last_seen_trigger:
            # Same trigger as before - increment counter
            persistence_count += 1
        else:
            # New or different trigger - reset counter
            persistence_count = 1
            first_seen_time = datetime.now(timezone.utc).isoformat()
        
        # Calculate duration string for logging
        duration_str = ""
        if first_seen_time:
            try:
                first_seen = datetime.fromisoformat(first_seen_time.replace('Z', '+00:00'))
                duration = datetime.now(timezone.utc) - first_seen
                duration_str = f" (stuck for {duration.total_seconds():.1f}s)"
            except (ValueError, TypeError):
                pass
        
        # Log based on threshold
        if persistence_count >= ERROR_THRESHOLD:
            logger.error(
                f"STUCK TRIGGER DETECTED: '{ask_user_trigger}' has persisted for "
                f"{persistence_count} supervisor cycles{duration_str}. "
                f"This indicates a bug - the trigger is not being cleared. "
                f"User responses: {user_responses}"
            )
        elif persistence_count >= WARN_THRESHOLD:
            logger.warning(
                f"Possible stuck trigger: '{ask_user_trigger}' seen {persistence_count} times{duration_str}. "
                f"If this continues, auto-recovery may activate at {AUTO_CLEAR_THRESHOLD} cycles."
            )
        
        # Auto-recovery (if enabled and threshold reached)
        if AUTO_CLEAR_THRESHOLD > 0 and persistence_count >= AUTO_CLEAR_THRESHOLD:
            logger.error(
                f"AUTO-RECOVERY ACTIVATED: Force-clearing stuck trigger '{ask_user_trigger}' "
                f"after {persistence_count} cycles{duration_str}. "
                f"Workflow will attempt to continue with 'ok_continue' verdict. "
                f"Please investigate the root cause."
            )
            
            # Save diagnostic information before clearing
            stuck_trigger_info = {
                "trigger": ask_user_trigger,
                "persistence_count": persistence_count,
                "first_seen_time": first_seen_time,
                "cleared_at": datetime.now(timezone.utc).isoformat(),
                "user_responses_at_clear": dict(user_responses) if user_responses else {},
                "pending_questions_at_clear": list(state.get("pending_user_questions", [])),
                "current_stage_id": current_stage_id,
            }
            
            # Return early with auto-recovery state
            return {
                "workflow_phase": "supervision",
                "ask_user_trigger": None,
                "pending_user_questions": [],
                "user_responses": {},
                "supervisor_verdict": "ok_continue",
                "supervisor_feedback": (
                    f"Auto-recovered from stuck trigger '{ask_user_trigger}'. "
                    f"Trigger persisted for {persistence_count} cycles. "
                    f"Continuing workflow - please check logs for root cause."
                ),
                # Clear tracking state
                "_last_seen_trigger": None,
                "_trigger_persistence_count": 0,
                "_trigger_first_seen_time": None,
                # Save diagnostic info
                "_last_stuck_trigger_recovery": stuck_trigger_info,
            }
        
        # Update tracking in result (will be applied after normal processing)
        result["_last_seen_trigger"] = ask_user_trigger
        result["_trigger_persistence_count"] = persistence_count
        result["_trigger_first_seen_time"] = first_seen_time
    else:
        # No trigger - clear tracking state
        result["_last_seen_trigger"] = None
        result["_trigger_persistence_count"] = 0
        result["_trigger_first_seen_time"] = None
    
    # Post ask_user handling
    if ask_user_trigger:
        # Dispatch to appropriate trigger handler
        handle_trigger(
            trigger=ask_user_trigger,
            state=state,
            result=result,
            user_responses=user_responses,
            current_stage_id=current_stage_id,
            get_dependent_stages_fn=_get_dependent_stages,
        )
        
        # Only clear trigger if handler didn't request more user input.
        # If handler set supervisor_verdict="ask_user" (needs clarification),
        # preserve the trigger so the next ask_user→supervisor cycle uses
        # the correct handler instead of falling back to "unknown_escalation".
        if result.get("supervisor_verdict") != "ask_user":
            result["ask_user_trigger"] = None
            result["pending_user_questions"] = []
            # Clear user_responses to prevent stale responses from affecting
            # future ask_user cycles. Each cycle should start fresh.
            result["user_responses"] = {}
            # Clear stuck trigger tracking since trigger was successfully handled
            result["_last_seen_trigger"] = None
            result["_trigger_persistence_count"] = 0
            result["_trigger_first_seen_time"] = None
        else:
            result["ask_user_trigger"] = ask_user_trigger
            # Trigger is preserved for next ask_user cycle
        
        # After handling a trigger, we skip the LLM call because:
        # 1. handle_trigger() already set the appropriate verdict
        # 2. Calling LLM would risk overriding that decision
        # 3. The router uses supervisor_verdict to route next steps
    
    # Normal supervision (not post-ask_user)
    else:
        _run_normal_supervision(state, result, system_prompt, current_stage_id, logger)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECK FOR REPLAN LIMIT: If supervisor returns replan_needed but we're
    # already at the replan limit, set up ask_user state so the routing
    # function doesn't send us to ask_user with empty questions.
    # ═══════════════════════════════════════════════════════════════════════
    verdict = result.get("supervisor_verdict")
    if verdict == "replan_needed":
        runtime_config = state.get("runtime_config") or {}
        max_replans = runtime_config.get("max_replans", MAX_REPLANS)
        current_replan_count = state.get("replan_count", 0)
        
        if current_replan_count >= max_replans:
            logger.warning(
                f"Supervisor returned replan_needed but replan_count ({current_replan_count}) "
                f">= max_replans ({max_replans}). Setting up ask_user state."
            )
            feedback = result.get("supervisor_feedback", "No feedback available")
            question = (
                f"Supervisor requested replanning but replan limit reached.\n\n"
                f"- Replan attempts: {current_replan_count}/{max_replans}\n"
                f"- Supervisor feedback: {feedback}\n\n"
                f"{get_options_prompt('replan_limit')}"
            )
            result["ask_user_trigger"] = "replan_limit"
            result["supervisor_verdict"] = "ask_user"
            result["pending_user_questions"] = [question]
            result["last_node_before_ask_user"] = "supervisor"
    
    # Log user interaction if one just happened
    # Log even if user_responses is empty (to track that user was asked)
    if ask_user_trigger:
        _log_user_interaction(
            state, result, ask_user_trigger, user_responses, current_stage_id
        )
    
    # Log supervisor decision
    verdict = result.get("supervisor_verdict", "unknown")
    feedback = result.get("supervisor_feedback", "")
    stage_info = f"stage={current_stage_id}" if current_stage_id else "no stage"
    emoji = "✅" if verdict in ["ok_continue", "all_complete"] else "🔄" if verdict in ["replan_needed", "change_priority"] else "⏪" if verdict == "backtrack_to_stage" else "❓" if verdict == "ask_user" else "🔍"
    logger.info(f"{emoji} supervisor: {stage_info}, verdict={verdict}" + (f" ({feedback})" if feedback else ""))
    
    return result


