"""
Stage selection agent node.

Handles the selection of the next stage to execute based on
dependencies, validation hierarchy, and stage status.

State Keys
----------
select_stage_node:
    READS: plan, progress, current_stage_id, validated_materials
    WRITES: workflow_phase, current_stage_id, progress, design_revision_count,
            code_revision_count, execution_failure_count, physics_failure_count,
            analysis_revision_count, ask_user_trigger, pending_user_questions,
            awaiting_user_input
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

from schemas.state import (
    ReproState,
    get_validation_hierarchy,
    STAGE_TYPE_TO_HIERARCHY_KEY,
    STAGE_TYPE_ORDER,
    initialize_progress_from_plan,
    get_progress_stage,
    get_plan_stage,
    update_progress_stage_status,
)


def select_stage_node(state: ReproState) -> dict:
    """
    Select next stage based on dependencies and validation hierarchy.
    
    Priority order:
    1. Stages with status "needs_rerun" (highest priority - backtrack targets)
    2. Stages with status "not_started" whose dependencies are satisfied
    3. No more stages to run (return None for current_stage_id)
    
    Skips stages with status:
    - "completed_success" / "completed_partial" / "completed_failed" - already done
    - "invalidated" - waiting for dependency to complete
    - "in_progress" - shouldn't happen, but skip
    - "blocked" - dependencies not met or budget exceeded
    
    Returns:
        Dict with state updates (LangGraph merges this into state)
    """
    logger = logging.getLogger(__name__)
    
    progress = state.get("progress") or {}  # Handle None gracefully
    stages = progress.get("stages", [])
    plan = state.get("plan") or {}  # Handle None gracefully
    plan_stages = plan.get("stages", [])
    
    if not stages and not plan_stages:
        logger.error(
            "select_stage_node called but no stages exist in plan or progress. "
            "This indicates plan_review failed to validate or plan initialization failed."
        )
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": None,
            "current_stage_type": None,
            "ask_user_trigger": "no_stages_available",
            "pending_user_questions": [
                "ERROR: No stages available to execute. The plan appears to be empty. "
                "Please check the plan and replan if necessary."
            ],
            "awaiting_user_input": True,
        }
    
    # Ensure progress initialized
    if not stages and plan_stages:
        logger.warning(
            "Progress stages not initialized. Initializing from plan stages on-demand."
        )
        try:
            state = initialize_progress_from_plan(state)
            progress = state.get("progress", {})
            stages = progress.get("stages", [])
        except Exception as e:
            logger.error(
                f"Failed to initialize progress from plan: {e}. "
                "Cannot proceed with stage selection."
            )
            return {
                "workflow_phase": "stage_selection",
                "current_stage_id": None,
                "current_stage_type": None,
                "ask_user_trigger": "progress_init_failed",
                "pending_user_questions": [
                    f"ERROR: Failed to initialize progress stages: {e}. "
                    "Please check plan structure or restart workflow."
                ],
                "awaiting_user_input": True,
            }
    
    if not stages or len(stages) == 0:
        logger.error("Stages list is empty after initialization - this should not happen")
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": None,
            "current_stage_type": None,
            "ask_user_trigger": "no_stages_available",
            "pending_user_questions": [
                "ERROR: Stages list is empty. Cannot proceed with reproduction."
            ],
            "awaiting_user_input": True,
        }
    
    # Get current validation hierarchy
    hierarchy = get_validation_hierarchy(state)
    
    # Priority 1: Find stages that need re-run (backtrack targets)
    for stage in stages:
        if stage.get("status") == "needs_rerun":
            # Guard against race condition
            # Get dependencies from plan stage (design spec) first, fall back to progress stage
            stage_id = stage.get("stage_id")
            plan_stage = get_plan_stage(state, stage_id) if stage_id else None
            dependencies = plan_stage.get("dependencies") if plan_stage else None
            # Fall back to progress stage dependencies if plan doesn't have them
            if dependencies is None:
                dependencies = stage.get("dependencies")
            # Handle None dependencies - treat as empty list
            if dependencies is None:
                dependencies = []
            elif not isinstance(dependencies, list):
                # If dependencies is not a list, treat as empty
                dependencies = []
            
            has_blocking_deps = False
            
            for dep_id in dependencies:
                dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
                if dep_stage:
                    dep_status = dep_stage.get("status")
                    # Only allow dependencies that are completed_success or completed_partial
                    # All other statuses (invalidated, needs_rerun, not_started, in_progress, blocked, completed_failed) are blocking
                    if dep_status not in ["completed_success", "completed_partial"]:
                        has_blocking_deps = True
                        break
                else:
                    # Missing dependency - this is blocking
                    has_blocking_deps = True
                    break
            
            # For needs_rerun stages, if explicit dependencies aren't satisfied,
            # check if validation hierarchy allows the stage type to run
            if has_blocking_deps:
                stage_type = stage.get("stage_type")
                if stage_type:
                    MAT_VAL_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["MATERIAL_VALIDATION"]
                    SINGLE_STRUCT_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["SINGLE_STRUCTURE"]
                    ARRAY_SYS_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["ARRAY_SYSTEM"]
                    PARAM_SWEEP_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["PARAMETER_SWEEP"]
                    
                    required_level_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
                    
                    # Check if validation hierarchy allows this stage type
                    hierarchy_allows = False
                    if stage_type == "SINGLE_STRUCTURE":
                        hierarchy_allows = hierarchy.get(MAT_VAL_KEY) in ["passed", "partial"]
                    elif stage_type == "ARRAY_SYSTEM":
                        hierarchy_allows = hierarchy.get(SINGLE_STRUCT_KEY) in ["passed", "partial"]
                    elif stage_type == "PARAMETER_SWEEP":
                        hierarchy_allows = hierarchy.get(SINGLE_STRUCT_KEY) in ["passed", "partial"]
                    elif stage_type == "COMPLEX_PHYSICS":
                        hierarchy_allows = (hierarchy.get(PARAM_SWEEP_KEY) in ["passed", "partial"] or
                                          hierarchy.get(ARRAY_SYS_KEY) in ["passed", "partial"])
                    elif stage_type == "MATERIAL_VALIDATION":
                        hierarchy_allows = True  # Material validation has no prerequisites
                    
                    if not hierarchy_allows:
                        continue
                else:
                    # No stage_type and dependencies not satisfied - skip
                    continue
            
            selected_stage_id = stage.get("stage_id")
            current_stage_id = state.get("current_stage_id")
            reset_counters = (selected_stage_id != current_stage_id) or (stage.get("status") == "needs_rerun")
            
            result_updates: Dict[str, Any] = {
                "workflow_phase": "stage_selection",
                "current_stage_id": selected_stage_id,
                "current_stage_type": stage.get("stage_type"),
                "stage_outputs": {},
                "run_error": None,
                "analysis_summary": None,
                "analysis_overall_classification": None,
            }
            
            if reset_counters:
                result_updates.update({
                    "design_revision_count": 0,
                    "code_revision_count": 0,
                    "execution_failure_count": 0,
                    "physics_failure_count": 0,
                    "analysis_revision_count": 0,
                    # Clear all feedback fields to prevent stale feedback leaking between stages
                    "reviewer_feedback": None,
                    "physics_feedback": None,
                    "execution_feedback": None,
                    "analysis_feedback": None,
                    "design_feedback": None,
                    "comparison_feedback": None,
                })
            
            # Log stage selection
            logger.info(f"🎯 select_stage: {selected_stage_id} ({stage.get('stage_type')}, needs_rerun)")
            
            return result_updates
    
    # Priority 2: Find not_started stages with satisfied dependencies
    for stage in stages:
        stage_id = stage.get("stage_id")
        status = stage.get("status", "not_started")
        
        if status in ["completed_success", "completed_partial", "completed_failed", "in_progress", "invalidated"]:
            continue
        
        # Re-check blocked stages
        if status == "blocked":
            # Before trying to unblock, check if stage has valid stage_type
            # Stages blocked due to missing/unknown stage_type should not be unblocked
            stage_type = stage.get("stage_type")
            if not stage_type:
                # Stage is blocked due to missing stage_type - don't try to unblock
                continue
            
            # Check if stage_type is valid (not unknown)
            required_level_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
            if stage_type and stage_type not in ["COMPLEX_PHYSICS"] and not required_level_key:
                # Stage is blocked due to unknown stage_type - don't try to unblock
                continue
            
            # Stage has valid stage_type, check if dependencies are now satisfied
            # Get dependencies from plan stage (design spec), fall back to progress stage
            plan_stage_for_deps = get_plan_stage(state, stage.get("stage_id"))
            dependencies = plan_stage_for_deps.get("dependencies") if plan_stage_for_deps else None
            if dependencies is None:
                dependencies = stage.get("dependencies", [])
            # Only unblock if stage has dependencies that are now satisfied
            # Stages blocked for other reasons (budget, etc.) should remain blocked
            if not dependencies:
                # No dependencies - blocked for non-dependency reason, skip
                continue
            
            deps_satisfied = True
            missing_deps = []
            
            stage_ids = {s.get("stage_id") for s in stages if s.get("stage_id")}
            
            for dep_id in dependencies:
                if dep_id not in stage_ids:
                    missing_deps.append(dep_id)
                    deps_satisfied = False
                    continue
                
                dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
                if dep_stage:
                    dep_status = dep_stage.get("status", "not_started")
                    if dep_status not in ["completed_success", "completed_partial"]:
                        deps_satisfied = False
                        break
                else:
                    missing_deps.append(dep_id)
                    deps_satisfied = False
            
            if deps_satisfied:
                update_progress_stage_status(
                    state,
                    stage.get("stage_id"),
                    "not_started",
                    summary="Unblocked: Dependencies now satisfied"
                )
                logger.info(
                    f"Stage '{stage.get('stage_id')}' unblocked - dependencies now satisfied."
                )
            else:
                continue
        
        # Check dependencies - get from plan stage (design spec), fall back to progress stage
        plan_stage_for_deps = get_plan_stage(state, stage.get("stage_id"))
        dependencies = plan_stage_for_deps.get("dependencies") if plan_stage_for_deps else None
        if dependencies is None:
            dependencies = stage.get("dependencies", [])
        
        if not dependencies:
            deps_satisfied = True
            missing_deps = []
        else:
            deps_satisfied = True
            missing_deps = []
            
            stage_ids = {s.get("stage_id") for s in stages if s.get("stage_id")}
            
            for dep_id in dependencies:
                if dep_id not in stage_ids:
                    missing_deps.append(dep_id)
                    deps_satisfied = False
                    continue
            
            dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
            if dep_stage:
                dep_status = dep_stage.get("status", "not_started")
                if dep_status not in ["completed_success", "completed_partial"]:
                    deps_satisfied = False
            else:
                missing_deps.append(dep_id)
                deps_satisfied = False
        
        if missing_deps:
            logger.error(
                f"Stage '{stage.get('stage_id')}' has missing dependencies: {missing_deps}. "
                "This indicates a plan inconsistency. Marking stage as blocked."
            )
            progress_stage = get_progress_stage(state, stage.get("stage_id"))
            if progress_stage and progress_stage.get("status") != "blocked":
                update_progress_stage_status(
                    state,
                    stage.get("stage_id"),
                    "blocked",
                    summary=f"Blocked: Missing dependencies {missing_deps}"
                )
            continue
        
        if not deps_satisfied:
            continue
        
        # Validate stage_type
        stage_type = stage.get("stage_type")
        
        if not stage_type:
            logger.error(
                f"Stage '{stage.get('stage_id')}' has no stage_type defined. "
                "Cannot determine validation hierarchy. Marking as blocked."
            )
            progress_stage = get_progress_stage(state, stage.get("stage_id"))
            if progress_stage and progress_stage.get("status") != "blocked":
                update_progress_stage_status(
                    state,
                    stage.get("stage_id"),
                    "blocked",
                    summary="Blocked: Missing stage_type field"
                )
            continue
        
        hierarchy = get_validation_hierarchy(state)
        
        MAT_VAL_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["MATERIAL_VALIDATION"]
        SINGLE_STRUCT_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["SINGLE_STRUCTURE"]
        ARRAY_SYS_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["ARRAY_SYSTEM"]
        PARAM_SWEEP_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["PARAMETER_SWEEP"]
        
        required_level_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
        
        if stage_type and stage_type not in ["COMPLEX_PHYSICS"] and not required_level_key:
            logger.warning(
                f"Unknown stage type '{stage_type}' for stage '{stage.get('stage_id')}'. "
                "Stage type not recognized in validation hierarchy. Marking as blocked."
            )
            progress_stage = get_progress_stage(state, stage.get("stage_id"))
            if progress_stage and progress_stage.get("status") != "blocked":
                update_progress_stage_status(
                    state,
                    stage.get("stage_id"),
                    "blocked",
                    summary=f"Blocked: Unknown stage type '{stage_type}'"
                )
            continue
        
        # Helper to check if a stage type exists in the plan and can potentially complete
        def stage_type_can_complete(target_type: str) -> bool:
            """Check if at least one stage of the target type exists and is not permanently blocked."""
            for s in stages:
                if s.get("stage_type") == target_type:
                    s_status = s.get("status", "not_started")
                    # A stage can potentially complete if it's not already blocked or failed
                    if s_status not in ["blocked", "completed_failed"]:
                        return True
            return False
        
        # Validation hierarchy checks with permanent blocking if prerequisite type can't complete
        if required_level_key:
            if stage_type == "SINGLE_STRUCTURE":
                if hierarchy.get(MAT_VAL_KEY) not in ["passed", "partial"]:
                    # Check if MATERIAL_VALIDATION stages can complete
                    if not stage_type_can_complete("MATERIAL_VALIDATION"):
                        logger.error(
                            f"Stage '{stage.get('stage_id')}' (SINGLE_STRUCTURE) requires MATERIAL_VALIDATION "
                            "but no completable MATERIAL_VALIDATION stage exists. Marking as blocked."
                        )
                        progress_stage = get_progress_stage(state, stage.get("stage_id"))
                        if progress_stage and progress_stage.get("status") != "blocked":
                            update_progress_stage_status(
                                state,
                                stage.get("stage_id"),
                                "blocked",
                                summary="Blocked: No completable MATERIAL_VALIDATION stage exists in plan"
                            )
                    continue
            elif stage_type == "ARRAY_SYSTEM":
                if hierarchy.get(SINGLE_STRUCT_KEY) not in ["passed", "partial"]:
                    # Check if SINGLE_STRUCTURE stages can complete
                    if not stage_type_can_complete("SINGLE_STRUCTURE"):
                        logger.error(
                            f"Stage '{stage.get('stage_id')}' (ARRAY_SYSTEM) requires SINGLE_STRUCTURE "
                            "but no completable SINGLE_STRUCTURE stage exists. Marking as blocked."
                        )
                        progress_stage = get_progress_stage(state, stage.get("stage_id"))
                        if progress_stage and progress_stage.get("status") != "blocked":
                            update_progress_stage_status(
                                state,
                                stage.get("stage_id"),
                                "blocked",
                                summary="Blocked: No completable SINGLE_STRUCTURE stage exists in plan"
                            )
                    continue
            elif stage_type == "PARAMETER_SWEEP":
                if hierarchy.get(SINGLE_STRUCT_KEY) not in ["passed", "partial"]:
                    # Check if SINGLE_STRUCTURE stages can complete
                    if not stage_type_can_complete("SINGLE_STRUCTURE"):
                        logger.error(
                            f"Stage '{stage.get('stage_id')}' (PARAMETER_SWEEP) requires SINGLE_STRUCTURE "
                            "but no completable SINGLE_STRUCTURE stage exists. Marking as blocked."
                        )
                        progress_stage = get_progress_stage(state, stage.get("stage_id"))
                        if progress_stage and progress_stage.get("status") != "blocked":
                            update_progress_stage_status(
                                state,
                                stage.get("stage_id"),
                                "blocked",
                                summary="Blocked: No completable SINGLE_STRUCTURE stage exists in plan"
                            )
                    continue
        elif stage_type == "COMPLEX_PHYSICS":
            if hierarchy.get(PARAM_SWEEP_KEY) not in ["passed", "partial"] and \
               hierarchy.get(ARRAY_SYS_KEY) not in ["passed", "partial"]:
                # Check if either PARAMETER_SWEEP or ARRAY_SYSTEM can complete
                can_param_sweep = stage_type_can_complete("PARAMETER_SWEEP")
                can_array_sys = stage_type_can_complete("ARRAY_SYSTEM")
                if not can_param_sweep and not can_array_sys:
                    logger.error(
                        f"Stage '{stage.get('stage_id')}' (COMPLEX_PHYSICS) requires PARAMETER_SWEEP or "
                        "ARRAY_SYSTEM but neither can complete. Marking as blocked."
                    )
                    progress_stage = get_progress_stage(state, stage.get("stage_id"))
                    if progress_stage and progress_stage.get("status") != "blocked":
                        update_progress_stage_status(
                            state,
                            stage.get("stage_id"),
                            "blocked",
                            summary="Blocked: No completable PARAMETER_SWEEP or ARRAY_SYSTEM stage exists in plan"
                        )
                continue
        
        # Type order enforcement
        # NOTE: For type order purposes, completed_failed counts as "done" (just unsuccessfully).
        # This allows the workflow to proceed to alternative paths when one path fails.
        if stage_type and stage_type in STAGE_TYPE_ORDER:
            current_type_index = STAGE_TYPE_ORDER.index(stage_type)
            skip_stage = False
            for lower_type_index in range(current_type_index):
                lower_type = STAGE_TYPE_ORDER[lower_type_index]
                has_completed_lower_type = False
                for other_stage in stages:
                    if other_stage.get("stage_type") == lower_type:
                        other_status = other_stage.get("status", "not_started")
                        # Include completed_failed - the stage is done, just not successfully
                        # This allows alternative paths to be tried
                        if other_status in ["completed_success", "completed_partial", "completed_failed"]:
                            has_completed_lower_type = True
                            break
                if not has_completed_lower_type:
                    lower_type_exists = any(
                        s.get("stage_type") == lower_type for s in stages
                    )
                    if lower_type_exists:
                        logger.warning(
                            f"Stage '{stage.get('stage_id')}' (type {stage_type}) cannot be selected "
                            f"because lower-order type '{lower_type}' exists but is not completed. "
                            "This enforces STAGE_TYPE_ORDER validation hierarchy."
                        )
                        skip_stage = True
                        break
            if skip_stage:
                continue
        
        # Stage is eligible
        selected_stage_id = stage.get("stage_id")
        current_stage_id = state.get("current_stage_id")
        reset_counters = (selected_stage_id != current_stage_id) or (status == "needs_rerun")
        
        result_updates = {
            "workflow_phase": "stage_selection",
            "current_stage_id": selected_stage_id,
            "current_stage_type": stage_type,
            "stage_start_time": datetime.now(timezone.utc).isoformat(),
            "stage_outputs": {},
            "run_error": None,
        }
        
        if reset_counters:
            result_updates.update({
                "design_revision_count": 0,
                "code_revision_count": 0,
                "execution_failure_count": 0,
                "physics_failure_count": 0,
                "analysis_revision_count": 0,
                # Clear all feedback fields to prevent stale feedback leaking between stages
                "reviewer_feedback": None,
                "physics_feedback": None,
                "execution_feedback": None,
                "analysis_feedback": None,
                "design_feedback": None,
                "comparison_feedback": None,
            })
        
        # Log stage selection
        logger.info(f"🎯 select_stage: {selected_stage_id} ({stage_type}, ready)")
        
        return result_updates
    
    # Deadlock detection - comprehensive check including transitive dependencies and hierarchy
    remaining_stages = [
        s for s in stages
        if s.get("status") not in ["completed_success", "completed_partial"]
    ]
    
    if remaining_stages:
        potentially_runnable = []
        permanently_blocked = []
        
        # Build set of completed stage IDs for dependency checking
        completed_stage_ids = {
            s.get("stage_id") for s in stages
            if s.get("status") in ["completed_success", "completed_partial"]
        }
        
        # Helper to recursively check if a stage can ever complete
        def can_stage_ever_complete(stage_id: str, visited: set = None) -> bool:
            """Check if a stage can ever complete, considering transitive dependencies."""
            if visited is None:
                visited = set()
            
            if stage_id in visited:
                return False  # Circular dependency
            visited.add(stage_id)
            
            stage = next((s for s in stages if s.get("stage_id") == stage_id), None)
            if not stage:
                return False  # Missing stage
            
            status = stage.get("status", "not_started")
            
            # Already completed
            if status in ["completed_success", "completed_partial"]:
                return True
            
            # Permanently blocked or failed
            if status in ["blocked", "completed_failed"]:
                return False
            
            # Check dependencies
            plan_stage_for_deps = get_plan_stage(state, stage_id) if stage_id else None
            dependencies = plan_stage_for_deps.get("dependencies") if plan_stage_for_deps else None
            if dependencies is None:
                dependencies = stage.get("dependencies", [])
            
            for dep_id in dependencies:
                if not can_stage_ever_complete(dep_id, visited.copy()):
                    return False
            
            return True
        
        # Get validation hierarchy once for checking
        hierarchy = get_validation_hierarchy(state)
        MAT_VAL_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["MATERIAL_VALIDATION"]
        SINGLE_STRUCT_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["SINGLE_STRUCTURE"]
        ARRAY_SYS_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["ARRAY_SYSTEM"]
        PARAM_SWEEP_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["PARAMETER_SWEEP"]
        
        for stage in remaining_stages:
            status = stage.get("status", "not_started")
            stage_id = stage.get("stage_id")
            stage_type = stage.get("stage_type")
            
            if status in ["blocked", "completed_failed"]:
                permanently_blocked.append(stage_id)
            elif status in ["not_started", "invalidated", "needs_rerun"]:
                # Check if this stage's dependencies can ever be satisfied (recursively)
                # Get dependencies from plan stage (design spec), fall back to progress stage
                plan_stage_for_deps = get_plan_stage(state, stage_id) if stage_id else None
                dependencies = plan_stage_for_deps.get("dependencies") if plan_stage_for_deps else None
                if dependencies is None:
                    dependencies = stage.get("dependencies", [])
                has_blocking_deps = False
                
                for dep_id in dependencies:
                    if not can_stage_ever_complete(dep_id):
                        has_blocking_deps = True
                        break
                
                # Also check validation hierarchy constraints
                if not has_blocking_deps and stage_type:
                    hierarchy_blocks = False
                    if stage_type == "SINGLE_STRUCTURE":
                        if hierarchy.get(MAT_VAL_KEY) == "failed":
                            hierarchy_blocks = True
                    elif stage_type == "ARRAY_SYSTEM":
                        if hierarchy.get(SINGLE_STRUCT_KEY) == "failed":
                            hierarchy_blocks = True
                    elif stage_type == "PARAMETER_SWEEP":
                        if hierarchy.get(SINGLE_STRUCT_KEY) == "failed":
                            hierarchy_blocks = True
                    elif stage_type == "COMPLEX_PHYSICS":
                        if (hierarchy.get(PARAM_SWEEP_KEY) == "failed" and 
                            hierarchy.get(ARRAY_SYS_KEY) == "failed"):
                            hierarchy_blocks = True
                    
                    if hierarchy_blocks:
                        has_blocking_deps = True
                
                if has_blocking_deps:
                    permanently_blocked.append(stage_id)
                else:
                    potentially_runnable.append(stage_id)
        
        if not potentially_runnable and permanently_blocked:
            logger.warning(
                f"Deadlock detected: All remaining stages are permanently blocked or failed. "
                f"Blocked stages: {permanently_blocked}. Cannot proceed with reproduction."
            )
            return {
                "workflow_phase": "stage_selection",
                "current_stage_id": None,
                "current_stage_type": None,
                "ask_user_trigger": "deadlock_detected",
                "pending_user_questions": [
                    f"Deadlock detected: All remaining stages ({len(permanently_blocked)}) are permanently blocked or failed. "
                    f"Blocked stages: {', '.join(permanently_blocked[:5])}{'...' if len(permanently_blocked) > 5 else ''}. "
                    "Options: 1) Generate report with current results, 2) Replan to fix blocked stages, 3) Stop."
                ],
                "awaiting_user_input": True,
            }
    
    # No more stages to run - normal completion
    logger.info("🎯 select_stage: all stages complete → generate_report")
    return {
        "workflow_phase": "stage_selection",
        "current_stage_id": None,
        "current_stage_type": None,
    }

