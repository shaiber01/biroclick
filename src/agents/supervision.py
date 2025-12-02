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

from .helpers.context import check_context_or_escalate
from .helpers.validation import (
    stage_comparisons_for_stage,
    breakdown_comparison_classifications,
)


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


def supervisor_node(state: ReproState) -> dict:
    """
    SupervisorAgent: Big-picture assessment and decisions.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT implementation notes:
    
    1. CHECK ask_user_trigger to understand what user was responding to
    2. HANDLE EACH TRIGGER TYPE appropriately
    3. RESET COUNTERS on user intervention
    4. USE get_validation_hierarchy() - never store directly
    """
    logger = logging.getLogger(__name__)
    
    # Initialize result dict
    result: Dict[str, Any] = {
        "workflow_phase": "supervision",
    }
    
    # Archive error recovery
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
        
        if retried_errors:
            result["archive_errors"] = retried_errors
        else:
            result["archive_errors"] = []
    else:
        result["archive_errors"] = []
    
    # Context check
    context_update = check_context_or_escalate(state, "supervisor")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    def _derive_stage_completion_outcome(current_state: ReproState, stage_id: Optional[str]) -> Tuple[str, str]:
        classification = (current_state.get("analysis_overall_classification") or "").upper()
        comparison_verdict = current_state.get("comparison_verdict")
        physics_verdict = current_state.get("physics_verdict")
        comparisons = stage_comparisons_for_stage(current_state, stage_id)
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
        
        summary_data = current_state.get("analysis_summary")
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
        
        # Material checkpoint
        if ask_user_trigger == "material_checkpoint":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            approval_keywords = ["APPROVE", "YES", "CORRECT", "OK", "ACCEPT", "VALID", "PROCEED"]
            rejection_keywords = ["REJECT", "NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]
            
            is_approval = any(kw in response_text for kw in approval_keywords)
            is_rejection = any(kw in response_text for kw in rejection_keywords)
            
            if is_approval and not is_rejection:
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
                    return result
                
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
                    update_progress_stage_status(state, current_stage_id, "completed_success")
                    
            elif "CHANGE_DATABASE" in response_text or (is_rejection and "DATABASE" in response_text):
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = f"User rejected material validation and requested database change: {response_text}."
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "needs_rerun",
                                                invalidation_reason="User requested material change")
                result["pending_validated_materials"] = []
                result["validated_materials"] = []
            elif "CHANGE_MATERIAL" in response_text or (is_rejection and "MATERIAL" in response_text):
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = f"User indicated wrong material: {response_text}. Please update plan."
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "needs_rerun",
                                                invalidation_reason="User rejected material")
                result["pending_validated_materials"] = []
                result["validated_materials"] = []
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
        
        # Code review limit
        elif ask_user_trigger == "code_review_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "PROVIDE_HINT" in response_text or "HINT" in response_text:
                result["code_revision_count"] = 0
                result["reviewer_feedback"] = f"User hint: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Retrying code generation with user hint."
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to code review issues")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = ["Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"]
        
        # Design review limit
        elif ask_user_trigger == "design_review_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "PROVIDE_HINT" in response_text or "HINT" in response_text:
                result["design_revision_count"] = 0
                result["reviewer_feedback"] = f"User hint: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to design review issues")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = ["Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"]
        
        # Execution failure limit
        elif ask_user_trigger == "execution_failure_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "RETRY" in response_text or "GUIDANCE" in response_text:
                result["execution_failure_count"] = 0
                result["supervisor_feedback"] = f"User guidance: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to execution failures")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = ["Please clarify: RETRY_WITH_GUIDANCE, SKIP_STAGE, or STOP?"]
        
        # Physics failure limit
        elif ask_user_trigger == "physics_failure_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "RETRY" in response_text:
                result["physics_failure_count"] = 0
                result["supervisor_feedback"] = f"User guidance: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
            elif "ACCEPT" in response_text or "PARTIAL" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "completed_partial",
                                                summary="Accepted as partial by user despite physics issues")
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to physics check failures")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = ["Please clarify: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, or STOP?"]
        
        # Context overflow
        elif ask_user_trigger == "context_overflow":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "SUMMARIZE" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Applying feedback summarization for context management."
            elif "TRUNCATE" in response_text:
                current_text = state.get("paper_text", "")
                if len(current_text) > 20000:
                    truncated_text = current_text[:15000] + "\n\n... [TRUNCATED BY USER REQUEST] ...\n\n" + current_text[-5000:]
                    result["paper_text"] = truncated_text
                    result["supervisor_feedback"] = "Truncating paper to first 15k and last 5k chars."
                else:
                    result["supervisor_feedback"] = "Paper already short enough, proceeding."
                result["supervisor_verdict"] = "ok_continue"
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped due to context overflow")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # Replan limit
        elif ask_user_trigger == "replan_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "FORCE" in response_text or "ACCEPT" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Plan force-accepted by user."
            elif "GUIDANCE" in response_text:
                result["replan_count"] = 0
                result["planner_feedback"] = f"User guidance: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "replan_needed"
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # Backtrack approval
        elif ask_user_trigger == "backtrack_approval":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "APPROVE" in response_text:
                result["supervisor_verdict"] = "backtrack_to_stage"
                decision = state.get("backtrack_decision", {})
                if decision:
                    target = decision.get("target_stage_id")
                    if target:
                        dependent = _get_dependent_stages(state.get("plan", {}), target)
                        decision["stages_to_invalidate"] = dependent
                        result["backtrack_decision"] = decision
            elif "REJECT" in response_text:
                result["backtrack_suggestion"] = None
                result["supervisor_verdict"] = "ok_continue"
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # Handle other triggers similarly...
        elif ask_user_trigger == "deadlock_detected":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "GENERATE_REPORT" in response_text or "REPORT" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            elif "REPLAN" in response_text:
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = f"User requested replan due to deadlock: {response_text}."
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = ["Please clarify: GENERATE_REPORT, REPLAN, or STOP?"]
        
        # Default handler
        else:
            result["supervisor_verdict"] = "ok_continue"
            result["supervisor_feedback"] = f"Handled unknown trigger: {ask_user_trigger}"
    
    # Normal supervision (not post-ask_user)
    else:
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
    
    # Log user interaction if one just happened
    if ask_user_trigger and user_responses:
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
    
    return result

