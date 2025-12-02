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
           supervisor_call_count, backtrack_count, runtime_config, metrics
    WRITES: workflow_phase, supervisor_verdict, supervisor_feedback, progress,
            stage_outputs, validated_materials, pending_validated_materials,
            current_stage_id, user_responses, pending_user_questions,
            awaiting_user_input, ask_user_trigger, supervisor_call_count,
            backtrack_count, backtrack_decision, replan_count, design_feedback,
            code_feedback, design_revision_count, code_revision_count
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
)
from src.prompts import build_agent_prompt
from src.llm_client import call_agent_with_metrics

from src.agents.helpers.context import check_context_or_escalate
from src.agents.helpers.validation import (
    stage_comparisons_for_stage,
    breakdown_comparison_classifications,
)
from .trigger_handlers import handle_trigger


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
    dependents_map = {s["stage_id"]: [] for s in stages}
    
    for stage in stages:
        for dep in stage.get("dependencies", []):
            if dep in dependents_map:
                dependents_map[dep].append(stage["stage_id"])
                
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
    
    if comparison_verdict == "needs_revision":
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
    
    if archive_errors:
        logger.info(f"Retrying {len(archive_errors)} failed archive operations...")
        
        retried_errors = []
        for error_entry in archive_errors:
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
        result["supervisor_feedback"] = agent_output.get("reasoning", "")
        
        if agent_output.get("verdict") == "backtrack_to_stage" and agent_output.get("backtrack_target"):
            result["backtrack_decision"] = {
                "target_stage_id": agent_output["backtrack_target"],
                "reason": agent_output.get("reasoning", ""),
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
    progress = state.get("progress", {})
    user_interactions = progress.get("user_interactions", [])
    
    interaction_entry = {
        "id": f"U{len(user_interactions) + 1}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_type": ask_user_trigger,
        "context": {
            "stage_id": current_stage_id,
            "agent": "SupervisorAgent",
            "reason": ask_user_trigger
        },
        "question": str(state.get("pending_user_questions", [""])[0]),
        "user_response": str(list(user_responses.values())[-1]),
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
        if context_update.get("awaiting_user_input"):
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
    
    # Post ask_user handling
    if ask_user_trigger:
        result["ask_user_trigger"] = None  # Clear trigger
        
        # Dispatch to appropriate trigger handler
        handle_trigger(
            trigger=ask_user_trigger,
            state=state,
            result=result,
            user_responses=user_responses,
            current_stage_id=current_stage_id,
            get_dependent_stages_fn=_get_dependent_stages,
        )
    
    # Normal supervision (not post-ask_user)
    else:
        _run_normal_supervision(state, result, system_prompt, current_stage_id, logger)
    
    # Log user interaction if one just happened
    if ask_user_trigger and user_responses:
        _log_user_interaction(
            state, result, ask_user_trigger, user_responses, current_stage_id
        )
    
    return result


