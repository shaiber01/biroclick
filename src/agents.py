"""
Agent Node Implementations - Stubs for LangGraph Workflow

Each function is a LangGraph node that:
1. Receives the current ReproState
2. Performs agent-specific processing (TODO: implement with LLM calls)
3. Updates relevant state fields
4. Returns the updated state

These are currently stubs showing expected state mutations.
See prompts/*.md for the corresponding agent system prompts.
"""

import os
import signal
from typing import Dict, Any, Optional

from schemas.state import (
    ReproState, 
    save_checkpoint, 
    check_context_before_node,
    initialize_progress_from_plan,
    sync_extracted_parameters,
    validate_state_for_node,
)
from src.prompts import build_agent_prompt
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════
# Metrics Logging
# ═══════════════════════════════════════════════════════════════════════

def log_agent_call(agent_name: str, node_name: str, start_time: datetime):
    """
    Decorator to log agent calls to state['metrics'].
    
    Note: This is a simplified version. Ideally this would be a proper decorator,
    but for state-passing functions, we can just call a helper at the end.
    """
    def record_metric(state: ReproState, result_dict: Dict[str, Any] = None):
        if "metrics" not in state:
            state["metrics"] = {"agent_calls": [], "stage_metrics": []}
            
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        metric = {
            "agent": agent_name,
            "node": node_name,
            "stage_id": state.get("current_stage_id"),
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "verdict": result_dict.get("execution_verdict") or 
                       result_dict.get("physics_verdict") or 
                       result_dict.get("supervisor_verdict") or
                       result_dict.get("last_plan_review_verdict") or
                       None,
            "error": result_dict.get("run_error") if result_dict else None
        }
        
        if "agent_calls" not in state["metrics"]:
            state["metrics"]["agent_calls"] = []
            
        state["metrics"]["agent_calls"].append(metric)
        
    return record_metric


# ═══════════════════════════════════════════════════════════════════════
# Context Window Management
# ═══════════════════════════════════════════════════════════════════════

def _check_context_or_escalate(state: ReproState, node_name: str) -> Optional[Dict[str, Any]]:
    """
    Check context before LLM call. Returns state updates if escalation needed, None otherwise.
    
    This is called explicitly at the start of each agent node that makes LLM calls.
    If context is critical and cannot be auto-recovered, prepares escalation to user.
    
    Args:
        state: Current ReproState
        node_name: Name of the node about to make LLM call
        
    Returns:
        None if safe to proceed (or with minor auto-recovery applied)
        Dict with state updates if escalation to user is needed
    """
    check = check_context_before_node(state, node_name, auto_recover=True)
    
    if check["ok"]:
        # Safe to proceed, possibly with state updates from auto-recovery
        return check.get("state_updates") if check.get("state_updates") else None
    
    if check["escalate"]:
        # Must ask user - return state updates to trigger ask_user
        return {
            "pending_user_questions": [check["user_question"]],
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
            "last_node_before_ask_user": node_name,
        }
    
    # Shouldn't reach here, but fallback to escalation
    return {
        "pending_user_questions": [f"Context overflow in {node_name}. How should we proceed?"],
        "awaiting_user_input": True,
        "ask_user_trigger": "context_overflow",
        "last_node_before_ask_user": node_name,
    }


def _validate_state_or_warn(state: ReproState, node_name: str) -> list:
    """
    Validate state for a node and return list of issues.
    
    This wraps validate_state_for_node() to provide consistent validation
    across agent nodes. Returns empty list if state is valid.
    
    Args:
        state: Current ReproState
        node_name: Name of the node about to execute
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = validate_state_for_node(state, node_name)
    if issues:
        import logging
        logger = logging.getLogger(__name__)
        for issue in issues:
            logger.warning(f"State validation issue for {node_name}: {issue}")
    return issues


def adapt_prompts_node(state: ReproState) -> ReproState:
    """PromptAdaptorAgent: Customize prompts for paper-specific needs."""
    state["workflow_phase"] = "adapting_prompts"
    # TODO: Implement prompt adaptation logic
    # - Analyze paper domain and techniques
    # - Generate prompt modifications
    # - Store in state["prompt_adaptations"]
    return state


def plan_node(state: ReproState) -> dict:
    """
    PlannerAgent: Analyze paper and create reproduction plan.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls with full paper text, so it must 
    check context first. The planner receives the largest context of any node.
    """
    start_time = datetime.now(timezone.utc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: CRITICAL for planner - receives full paper text
    # ═══════════════════════════════════════════════════════════════════════
    escalation = _check_context_or_escalate(state, "plan")
    if escalation is not None:
        # Context overflow - return escalation state updates
        return escalation
    
    # TODO: Implement planning logic
    # - Extract parameters from paper
    # - Classify figures
    # - Design staged reproduction plan
    # - Initialize assumptions
    # - Call LLM with planner_agent.md prompt
    # - Parse agent output per planner_output_schema.json
    
    # Note: We need to fetch the prompt to use adaptations
    system_prompt = build_agent_prompt("planner", state)
    
    # STUB: Replace with actual LLM call that populates plan, assumptions, etc.
    result = {
        "workflow_phase": "planning",
        "plan": {
            "reproducible_figure_ids": [], # Initialize empty list to satisfy schema
            # ... other plan fields would be populated by LLM ...
        }
        # state["plan"] = generated_plan
        # state["assumptions"] = generated_assumptions
        # state["planned_materials"] = extracted_materials
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # MANDATORY: Initialize progress stages from plan (after plan is set)
    # This converts plan stages into progress stages with status tracking.
    # Must be called before select_stage_node runs.
    # ═══════════════════════════════════════════════════════════════════════
    if result.get("plan") and result["plan"].get("stages"):
        state_with_plan = {**state, **result}
        state_with_plan = initialize_progress_from_plan(state_with_plan)
        state_with_plan = sync_extracted_parameters(state_with_plan)
        result["progress"] = state_with_plan.get("progress")
        result["extracted_parameters"] = state_with_plan.get("extracted_parameters")
    
    # Log metrics
    log_agent_call("PlannerAgent", "plan", start_time)(state, result)
    
    return result


def plan_reviewer_node(state: ReproState) -> dict:
    """
    PlanReviewerAgent: Review reproduction plan before stage execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_plan_review_verdict` state field.
    - Increments `replan_count` when verdict is "needs_revision".
    The routing function `route_after_plan_review` reads these fields.
    - Validates plan precision requirements (excellent targets need digitized data).
    """
    # ═══════════════════════════════════════════════════════════════════════
    # STATE VALIDATION: Check plan meets requirements before review
    # ═══════════════════════════════════════════════════════════════════════
    validation_issues = _validate_state_or_warn(state, "plan_review")
    
    # Separate blocking issues (PLAN_ISSUE) from other validation warnings
    blocking_issues = [i for i in validation_issues if i.startswith("PLAN_ISSUE:")]
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("plan_reviewer", state)
    
    # TODO: Implement plan review logic using prompts/plan_reviewer_agent.md
    # - Check coverage of reproducible figures
    # - Verify Stage 0 and Stage 1 present
    # - Validate parameter extraction
    # - Check assumptions quality
    # - Verify runtime estimates
    # - Call LLM with plan_reviewer_agent.md prompt
    # - Parse agent output per plan_reviewer_output_schema.json
    
    # If there are blocking plan issues (e.g., excellent precision without digitized data),
    # automatically flag for revision
    if blocking_issues:
        agent_output = {
            "verdict": "needs_revision",
            "issues": [{"severity": "blocking", "description": issue} for issue in blocking_issues],
            "summary": f"Plan has {len(blocking_issues)} blocking issue(s) requiring revision",
            "feedback": "The following issues must be resolved:\n" + "\n".join(blocking_issues),
        }
    else:
        # STUB: Replace with actual LLM call
        agent_output = {
            "verdict": "approve",  # "approve" | "needs_revision"
            "issues": [],
            "summary": "Plan review stub - implement with LLM call",
        }
    
    result = {
        "workflow_phase": "plan_review",
        "last_plan_review_verdict": agent_output["verdict"],
    }
    
    # Increment replan counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        result["replan_count"] = state.get("replan_count", 0) + 1
        result["planner_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


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
    from schemas.state import get_validation_hierarchy, STAGE_TYPE_TO_HIERARCHY_KEY
    
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    plan = state.get("plan", {})
    plan_stages = plan.get("stages", [])
    
    if not stages and not plan_stages:
        # No stages defined yet
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": None,
            "current_stage_type": None,
        }
    
    # Use plan stages if progress stages aren't initialized
    if not stages:
        stages = plan_stages
    
    # Get current validation hierarchy
    hierarchy = get_validation_hierarchy(state)
    
    # Priority 1: Find stages that need re-run (backtrack targets)
    for stage in stages:
        if stage.get("status") == "needs_rerun":
            return {
                "workflow_phase": "stage_selection",
                "current_stage_id": stage.get("stage_id"),
                "current_stage_type": stage.get("stage_type"),
                # Reset per-stage counters when entering a new stage
                "design_revision_count": 0,
                "code_revision_count": 0,
                "execution_failure_count": 0,
                "physics_failure_count": 0,
                "analysis_revision_count": 0,
                # Reset per-stage outputs (prevent stale data from previous stage)
                "stage_outputs": {},
                "run_error": None,
            }
    
    # Priority 2: Find not_started stages with satisfied dependencies
    for stage in stages:
        status = stage.get("status", "not_started")
        
        # Skip completed, in_progress, blocked, or invalidated stages
        if status in ["completed_success", "completed_partial", "completed_failed", 
                      "in_progress", "blocked", "invalidated"]:
            continue
        
        # Check if dependencies are satisfied
        dependencies = stage.get("dependencies", [])
        deps_satisfied = True
        
        for dep_id in dependencies:
            # Find the dependency stage
            dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
            if dep_stage:
                dep_status = dep_stage.get("status", "not_started")
                # Dependency must be completed (success or partial)
                if dep_status not in ["completed_success", "completed_partial"]:
                    deps_satisfied = False
                    break
        
        if not deps_satisfied:
            continue
        
    # Check validation hierarchy for stage type
    stage_type = stage.get("stage_type")
    
    # Use consistent hierarchy keys from state.py
    # Note: get_validation_hierarchy returns keys like 'stage0_material_validation' (specific)
    # or abstract keys. The current implementation of get_validation_hierarchy in state.py
    # returns a dict with keys corresponding to the hierarchy levels.
    # We need to re-fetch hierarchy here as it might change if we processed multiple stages.
    hierarchy = get_validation_hierarchy(state)
    
    # Use schema-defined keys for robustness
    MAT_VAL_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["MATERIAL_VALIDATION"]
    SINGLE_STRUCT_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["SINGLE_STRUCTURE"]
    ARRAY_SYS_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["ARRAY_SYSTEM"]
    PARAM_SWEEP_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["PARAMETER_SWEEP"]
    
    # Enforce hierarchy using STAGE_TYPE_TO_HIERARCHY_KEY mapping
    # This ensures robustness against schema changes
    required_level_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
    
    if required_level_key:
        # Map current stage type to its prerequisite level in the hierarchy
        # e.g., SINGLE_STRUCTURE needs 'material_validation' to be passed
        if stage_type == "SINGLE_STRUCTURE":
            if hierarchy.get(MAT_VAL_KEY) not in ["passed", "partial"]:
                continue
        elif stage_type == "ARRAY_SYSTEM":
            if hierarchy.get(SINGLE_STRUCT_KEY) not in ["passed", "partial"]:
                continue
        elif stage_type == "PARAMETER_SWEEP":
            # Parameter sweeps typically need at least single structure
            if hierarchy.get(SINGLE_STRUCT_KEY) not in ["passed", "partial"]:
                continue
        elif stage_type == "COMPLEX_PHYSICS":
             if hierarchy.get(PARAM_SWEEP_KEY) not in ["passed", "partial"] and \
                hierarchy.get(ARRAY_SYS_KEY) not in ["passed", "partial"]:
                continue

        # This stage is eligible
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": stage.get("stage_id"),
            "current_stage_type": stage_type,
            # Reset per-stage counters when entering a new stage
            "design_revision_count": 0,
            "code_revision_count": 0,
            "execution_failure_count": 0,
            "physics_failure_count": 0,
            "analysis_revision_count": 0,
            # Reset per-stage outputs (prevent stale data from previous stage)
            "stage_outputs": {},
            "run_error": None,
        }
    
    # No more stages to run
    return {
        "workflow_phase": "stage_selection",
        "current_stage_id": None,
        "current_stage_type": None,
    }


def simulation_designer_node(state: ReproState) -> dict:
    """
    SimulationDesignerAgent: Design simulation setup for current stage.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls, so it must check context first.
    """
    start_time = datetime.now(timezone.utc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Required for all nodes that make LLM calls
    # ═══════════════════════════════════════════════════════════════════════
    escalation = _check_context_or_escalate(state, "design")
    if escalation is not None:
        # Context overflow - return escalation state updates
        return escalation
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("simulation_designer", state)
    
    # TODO: Implement design logic
    # - Interpret geometry from plan
    # - Select materials from validated_materials (Stage 1+) or planned_materials (Stage 0)
    # - Configure sources, BCs, monitors
    # - Estimate performance
    # - Call LLM with simulation_designer_agent.md prompt
    # - Parse agent output per simulation_designer_output_schema.json
    
    # STUB: Replace with actual LLM call
    result = {
        "workflow_phase": "design",
        # agent_output fields would go here
    }
    
    log_agent_call("SimulationDesignerAgent", "design", start_time)(state, result)
    return result


def design_reviewer_node(state: ReproState) -> dict:
    """
    DesignReviewerAgent: Review simulation design before code generation.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_design_review_verdict` state field.
    - Increments `design_revision_count` when verdict is "needs_revision".
    The routing function `route_after_design_review` reads these fields.
    """
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("design_reviewer", state)
    
    # TODO: Implement design review logic using prompts/design_reviewer_agent.md
    # - Check geometry matches paper
    # - Verify physics setup is correct
    # - Validate material choices
    # - Check unit system (a_unit)
    # - Verify source/excitation setup
    # - Call LLM with design_reviewer_agent.md prompt
    # - Parse agent output per design_reviewer_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "approve",  # "approve" | "needs_revision"
        "issues": [],
        "summary": "Design review stub - implement with LLM call",
    }
    
    result = {
        "workflow_phase": "design_review",
        "last_design_review_verdict": agent_output["verdict"],
        "reviewer_issues": agent_output.get("issues", []),
    }
    
    # Increment design revision counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        result["design_revision_count"] = state.get("design_revision_count", 0) + 1
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


def code_reviewer_node(state: ReproState) -> dict:
    """
    CodeReviewerAgent: Review generated code before execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_code_review_verdict` state field.
    - Increments `code_revision_count` when verdict is "needs_revision".
    The routing function `route_after_code_review` reads these fields.
    """
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("code_reviewer", state)
    
    # TODO: Implement code review logic using prompts/code_reviewer_agent.md
    # - Verify a_unit matches design
    # - Check Meep API usage
    # - Validate numerics implementation
    # - Check code quality (no plt.show, etc.)
    # - Call LLM with code_reviewer_agent.md prompt
    # - Parse agent output per code_reviewer_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "approve",  # "approve" | "needs_revision"
        "issues": [],
        "summary": "Code review stub - implement with LLM call",
    }
    
    result = {
        "workflow_phase": "code_review",
        "last_code_review_verdict": agent_output["verdict"],
        "reviewer_issues": agent_output.get("issues", []),
    }
    
    # Increment code revision counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        result["code_revision_count"] = state.get("code_revision_count", 0) + 1
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


def code_generator_node(state: ReproState) -> dict:
    """
    CodeGeneratorAgent: Generate Python+Meep code from approved design.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls, so it must check context first.
    
    For Stage 1+, code generator MUST read material paths from 
    state["validated_materials"], NOT hardcode paths. validated_materials 
    is populated after Stage 0 + user approval.
    """
    start_time = datetime.now(timezone.utc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Required for all nodes that make LLM calls
    # ═══════════════════════════════════════════════════════════════════════
    escalation = _check_context_or_escalate(state, "generate_code")
    if escalation is not None:
        # Context overflow - return escalation state updates
        return escalation
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("code_generator", state)
    
    # TODO: Implement code generation logic
    # - Convert design to Meep code
    # - For Stage 1+: Read material paths from state["validated_materials"]
    # - Include progress prints
    # - Set expected outputs per stage specification
    # - Call LLM with code_generator_agent.md prompt
    # - Parse agent output per code_generator_output_schema.json
    
    # STUB: Replace with actual LLM call
    result = {
        "workflow_phase": "code_generation",
        # agent_output fields would go here (simulation_code, etc.)
    }
    
    log_agent_call("CodeGeneratorAgent", "generate_code", start_time)(state, result)
    return result


def execution_validator_node(state: ReproState) -> dict:
    """
    ExecutionValidatorAgent: Validate simulation ran correctly.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `execution_verdict` state field from agent output's `verdict`.
    - Increments `execution_failure_count` when verdict is "fail".
    - Increments `total_execution_failures` for metrics tracking.
    The routing function `route_after_execution_check` reads these fields.
    """
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("execution_validator", state)
    
    # TODO: Implement execution validation logic
    # - Check completion status
    # - Verify output files exist
    # - Check for NaN/Inf in data
    # - Check for TIMEOUT_ERROR in run_error
    # - Call LLM with execution_validator_agent.md prompt
    # - Parse agent output per execution_validator_output_schema.json
    
    run_error = state.get("run_error")
    if run_error and "TIMEOUT_ERROR" in run_error:
        # Auto-fail on timeout if not handled by LLM
        agent_output = {
            "verdict": "fail",
            "stage_id": state.get("current_stage_id"),
            "summary": f"Execution timed out: {run_error}",
        }
    else:
        # STUB: Replace with actual LLM call
        agent_output = {
            "verdict": "pass",  # "pass" | "warning" | "fail"
            "stage_id": state.get("current_stage_id"),
            "summary": "Execution validation stub - implement with LLM call",
        }
    
    result = {
        "workflow_phase": "execution_validation",
        # Copy verdict to type-specific state field for routing
        "execution_verdict": agent_output["verdict"],
    }
    
    # Increment failure counters if verdict is "fail"
    # This happens BEFORE routing function reads the count
    if agent_output["verdict"] == "fail":
        result["execution_failure_count"] = state.get("execution_failure_count", 0) + 1
        result["total_execution_failures"] = state.get("total_execution_failures", 0) + 1
    
    return result


def physics_sanity_node(state: ReproState) -> dict:
    """
    PhysicsSanityAgent: Validate physics of results.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `physics_verdict` state field from agent output's `verdict`.
    - Increments `physics_failure_count` when verdict is "fail".
    - Increments `design_revision_count` when verdict is "design_flaw".
    The routing function `route_after_physics_check` reads these fields.
    
    Verdict options:
    - "pass": Physics looks good, proceed to analysis
    - "warning": Minor concerns but proceed
    - "fail": Code/numerics issue, route to code generator
    - "design_flaw": Fundamental design problem, route to simulation designer
    """
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("physics_sanity", state)
    
    # TODO: Implement physics validation logic
    # - Check conservation laws (T + R + A ≈ 1)
    # - Verify value ranges
    # - Check numerical quality
    # - Call LLM with physics_sanity_agent.md prompt
    # - Parse agent output per physics_sanity_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "pass",  # "pass" | "warning" | "fail" | "design_flaw"
        "stage_id": state.get("current_stage_id"),
        "summary": "Physics validation stub - implement with LLM call",
        "backtrack_suggestion": {"suggest_backtrack": False},
    }
    
    result = {
        "workflow_phase": "physics_validation",
        # Copy verdict to type-specific state field for routing
        "physics_verdict": agent_output["verdict"],
    }
    
    # Increment failure counters based on verdict type
    # This happens BEFORE routing function reads the count
    if agent_output["verdict"] == "fail":
        result["physics_failure_count"] = state.get("physics_failure_count", 0) + 1
    elif agent_output["verdict"] == "design_flaw":
        # design_flaw routes to design node, use design_revision_count
        result["design_revision_count"] = state.get("design_revision_count", 0) + 1
    
    # If agent suggests backtrack, populate backtrack_suggestion for supervisor
    if agent_output.get("backtrack_suggestion", {}).get("suggest_backtrack"):
        result["backtrack_suggestion"] = agent_output["backtrack_suggestion"]
    
    return result


def results_analyzer_node(state: ReproState) -> ReproState:
    """ResultsAnalyzerAgent: Compare results to paper figures."""
    state["workflow_phase"] = "analysis"
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("results_analyzer", state)
    
    # TODO: Implement analysis logic
    # - Compare simulation outputs to paper figures
    # - Compute discrepancies
    # - Classify reproduction quality
    return state


def comparison_validator_node(state: ReproState) -> dict:
    """
    ComparisonValidatorAgent: Validate comparison accuracy.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `comparison_verdict` state field from agent output's `verdict`.
    - Increments `analysis_revision_count` when verdict is "needs_revision".
    The routing function `route_after_comparison_check` reads these fields.
    """
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("comparison_validator", state)
    
    # TODO: Implement comparison validation logic
    # - Verify math is correct
    # - Check classifications match numbers
    # - Validate discrepancy documentation
    # - Call LLM with comparison_validator_agent.md prompt
    # - Parse agent output per comparison_validator_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "approve",  # "approve" | "needs_revision"
        "stage_id": state.get("current_stage_id"),
        "summary": "Comparison validation stub - implement with LLM call",
    }
    
    result = {
        "workflow_phase": "comparison_validation",
        # Copy verdict to type-specific state field for routing
        "comparison_verdict": agent_output["verdict"],
    }
    
    # Increment analysis revision counter if needs_revision
    # This happens BEFORE routing function reads the count
    if agent_output["verdict"] == "needs_revision":
        result["analysis_revision_count"] = state.get("analysis_revision_count", 0) + 1
    
    return result


def supervisor_node(state: ReproState) -> dict:
    """
    SupervisorAgent: Big-picture assessment and decisions.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT implementation notes:
    
    1. CHECK ask_user_trigger:
       When this node is called after ask_user, check state["ask_user_trigger"]
       to understand what the user was responding to. See src/prompts.py
       ASK_USER_TRIGGERS for full documentation of all trigger types.
       
    2. HANDLE EACH TRIGGER TYPE:
       Each trigger has specific handling requirements documented in
       src/prompts.py:ASK_USER_TRIGGERS. Key triggers:
       - "material_checkpoint": Mandatory Stage 0 validation
       - "code_review_limit": User guidance on stuck code generation
       - "design_review_limit": User guidance on stuck design
       - "execution_failure_limit": Simulation runtime failures
       - "physics_failure_limit": Physics sanity check failures
       - "context_overflow": LLM context management
       - "replan_limit": Planning iteration limit
       - "backtrack_approval": Cross-stage backtrack confirmation
       
    3. RESET COUNTERS on user intervention:
       When user provides guidance that resolves an issue, reset relevant
       counters to prevent limit exhaustion:
       - code_revision_count = 0 if routing back to generate_code
       - design_revision_count = 0 if routing back to design
       - execution_failure_count = 0 if retrying execution
       - physics_failure_count = 0 if retrying physics check
       
    4. USE get_validation_hierarchy():
       Always use get_validation_hierarchy(state) to check hierarchy status.
       Never store validation_hierarchy in state directly - it's computed.
    """
    from schemas.state import (
        get_validation_hierarchy, 
        update_progress_stage_status,
        archive_stage_outputs_to_progress
    )
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("supervisor", state)
    
    # TODO: Implement full supervision logic using prompts/supervisor_agent.md
    # - Call LLM with supervisor_agent.md prompt
    # - Parse agent output per supervisor_output_schema.json
    
    # Get trigger info
    ask_user_trigger = state.get("ask_user_trigger")
    user_responses = state.get("user_responses", {})
    last_node = state.get("last_node_before_ask_user")
    current_stage_id = state.get("current_stage_id")
    
    result: Dict[str, Any] = {
        "workflow_phase": "supervision",
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # POST-ASK_USER HANDLING: Route based on trigger type
    # ═══════════════════════════════════════════════════════════════════════
    
    if ask_user_trigger:
        result["ask_user_trigger"] = None  # Clear trigger after handling
        
        # ─── MATERIAL CHECKPOINT ──────────────────────────────────────────────
        if ask_user_trigger == "material_checkpoint":
            # Parse user response from any question (get last response)
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "APPROVE" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Material validation approved by user."
                # CRITICAL: Move pending materials to validated_materials on approval
                pending_materials = state.get("pending_validated_materials", [])
                if pending_materials:
                    result["validated_materials"] = pending_materials
                    result["pending_validated_materials"] = []  # Clear pending
                
                # Archive outputs before moving on
                if current_stage_id:
                    archive_stage_outputs_to_progress(state, current_stage_id)
                    update_progress_stage_status(state, current_stage_id, "completed_success")
                    
            elif "CHANGE_DATABASE" in response_text:
                # User requested database change - requires REPLAN to update assumptions
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = (
                    f"User rejected material validation and requested database change: {response_text}. "
                    "Please update the plan and assumptions to use the specified database/material, "
                    "then re-run Stage 0."
                )
                # Mark stage 0 as invalid
                if current_stage_id:
                    update_progress_stage_status(
                        state, 
                        current_stage_id, 
                        "invalidated", 
                        invalidation_reason="User requested material change"
                    )
            elif "CHANGE_MATERIAL" in response_text:
                # Need to replan with different material
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = f"User indicated wrong material: {response_text}. Please update plan."
            elif "NEED_HELP" in response_text:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "Please provide more details about the material issue. "
                    "What specific aspect of the optical constants looks incorrect?"
                ]
            else:
                # Default: assume approval if unclear
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = f"Proceeding with user response: {response_text[:100]}"
        
        # ─── CODE REVIEW LIMIT ────────────────────────────────────────────────
        elif ask_user_trigger == "code_review_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "PROVIDE_HINT" in response_text or "HINT" in response_text:
                # Reset counter and route back to code generation with hint
                result["code_revision_count"] = 0
                result["reviewer_feedback"] = f"User hint: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"  # Will route to select_stage, then design->code
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
                result["pending_user_questions"] = [
                    "Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"
                ]
        
        # ─── DESIGN REVIEW LIMIT ──────────────────────────────────────────────
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
                result["pending_user_questions"] = [
                    "Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"
                ]
        
        # ─── EXECUTION FAILURE LIMIT ──────────────────────────────────────────
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
                result["pending_user_questions"] = [
                    "Please clarify: RETRY_WITH_GUIDANCE (with guidance), SKIP_STAGE, or STOP?"
                ]
        
        # ─── PHYSICS FAILURE LIMIT ────────────────────────────────────────────
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
                result["pending_user_questions"] = [
                    "Please clarify: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, or STOP?"
                ]
        
        # ─── CONTEXT OVERFLOW ─────────────────────────────────────────────────
        elif ask_user_trigger == "context_overflow":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "SUMMARIZE" in response_text:
                # Apply feedback summarization (handled elsewhere)
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Applying feedback summarization for context management."
            elif "TRUNCATE" in response_text:
                # Actually truncate the paper text to resolve the loop
                current_text = state.get("paper_text", "")
                # Keep first 15k chars and last 5k as a simple heuristic
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
        
        # ─── REPLAN LIMIT ─────────────────────────────────────────────────────
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
        
        # ─── BACKTRACK APPROVAL ───────────────────────────────────────────────
        elif ask_user_trigger == "backtrack_approval":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "APPROVE" in response_text:
                result["supervisor_verdict"] = "backtrack_to_stage"
                # backtrack_decision should already be set
            elif "REJECT" in response_text:
                result["backtrack_suggestion"] = None
                result["supervisor_verdict"] = "ok_continue"
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # ─── UNKNOWN/DEFAULT ──────────────────────────────────────────────────
        else:
            # Unknown trigger - default to continue
            result["supervisor_verdict"] = "ok_continue"
            result["supervisor_feedback"] = f"Handled unknown trigger: {ask_user_trigger}"
    
    # ═══════════════════════════════════════════════════════════════════════
    # NORMAL SUPERVISION (not post-ask_user)
    # ═══════════════════════════════════════════════════════════════════════
    else:
        # TODO: Implement actual LLM-based supervision logic
        # For now, default to continue
        result["supervisor_verdict"] = "ok_continue"
        
        # If completing a stage, archive outputs
        if current_stage_id:
            archive_stage_outputs_to_progress(state, current_stage_id)
            # NOTE: In full implementation, LLM decides status. 
            # For stub, assume success if we got here without issues.
            update_progress_stage_status(state, current_stage_id, "completed_success")
    
    # Log user interaction if one just happened
    if ask_user_trigger and user_responses:
        # Record structured interaction log
        interaction_entry = {
            "id": f"U{len(state.get('progress', {}).get('user_interactions', [])) + 1}",
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
            "alternatives_considered": [] # Would be populated by LLM
        }
        
        # Ensure progress structure exists
        if "progress" not in state:
            state["progress"] = {}
        if "user_interactions" not in state["progress"]:
            state["progress"]["user_interactions"] = []
            
        state["progress"]["user_interactions"].append(interaction_entry)
    
    return result


def ask_user_node(state: ReproState) -> Dict[str, Any]:
    """
    CLI-based user interaction node.
    
    Prompts user in terminal for input. If user doesn't respond within timeout
    (Ctrl+C or timeout), saves checkpoint and exits gracefully.
    
    Environment variables:
        REPROLAB_USER_TIMEOUT_SECONDS: Override default timeout (default: 86400 = 24h)
        REPROLAB_NON_INTERACTIVE: If "1", immediately save checkpoint and exit
        
    Returns:
        Dict with state updates (user_responses, cleared pending questions)
    """
    timeout_seconds = int(os.environ.get("REPROLAB_USER_TIMEOUT_SECONDS", "86400"))
    non_interactive = os.environ.get("REPROLAB_NON_INTERACTIVE", "0") == "1"
    
    questions = state.get("pending_user_questions", [])
    trigger = state.get("ask_user_trigger", "unknown")
    paper_id = state.get("paper_id", "unknown")
    
    if not questions:
        return {
            "awaiting_user_input": False,
            "workflow_phase": "awaiting_user",
        }
    
    # Non-interactive mode: save and exit immediately
    if non_interactive:
        print("\n" + "=" * 60)
        print("USER INPUT REQUIRED (non-interactive mode)")
        print("=" * 60)
        print(f"\nPaper: {paper_id}")
        print(f"Trigger: {trigger}")
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:\n{q}")
        
        checkpoint_path = save_checkpoint(state, f"awaiting_user_{trigger}")
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    # Interactive mode: prompt user
    print("\n" + "=" * 60)
    print("USER INPUT REQUIRED")
    print("=" * 60)
    print(f"Paper: {paper_id}")
    print(f"Trigger: {trigger}")
    print("(Press Ctrl+C to save checkpoint and exit)")
    print("=" * 60)
    
    responses = {}
    
    def timeout_handler(signum, frame):
        raise TimeoutError("User response timeout")
    
    try:
        # Set timeout (Unix only - SIGALRM not available on Windows)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            print(f"\n{question}")
            print("-" * 40)
            
            # Multi-line input support: empty line ends input
            print("(Enter your response, then press Enter twice to submit)")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            response = "\n".join(lines).strip()
            if not response:
                response = input("Your response (single line): ").strip()
            
            responses[question] = response
            print(f"✓ Response recorded")
            
            # Log interaction immediately
            if "progress" not in state:
                state["progress"] = {}
            if "user_interactions" not in state["progress"]:
                state["progress"]["user_interactions"] = []
                
            state["progress"]["user_interactions"].append({
                "id": f"U{len(state['progress']['user_interactions']) + 1}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "interaction_type": state.get("ask_user_trigger", "unknown"),
                "context": {
                    "stage_id": state.get("current_stage_id"),
                    "agent": "AskUserNode",
                    "reason": "Direct user input"
                },
                "question": question,
                "user_response": response,
                "impact": "Response recorded",
                "alternatives_considered": []
            })
        
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        print("\n" + "=" * 60)
        print("✓ All responses collected")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user (Ctrl+C)")
        checkpoint_path = save_checkpoint(state, f"interrupted_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except TimeoutError:
        print(f"\n\n⚠ User response timeout ({timeout_seconds}s)")
        checkpoint_path = save_checkpoint(state, f"timeout_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except EOFError:
        print(f"\n\n⚠ End of input (EOF)")
        checkpoint_path = save_checkpoint(state, f"eof_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    return {
        "user_responses": {**state.get("user_responses", {}), **responses},
        "pending_user_questions": [],
        "awaiting_user_input": False,
        "workflow_phase": "awaiting_user",
    }


def material_checkpoint_node(state: ReproState) -> dict:
    """
    Mandatory material validation checkpoint after Stage 0.
    
    This node ALWAYS routes to ask_user to require user confirmation
    of material validation results before proceeding to Stage 1+.
    
    IMPORTANT: This node stores extracted materials in `pending_validated_materials`,
    NOT in `validated_materials`. The supervisor_node will move them to 
    `validated_materials` ONLY when the user approves.
    
    Per global_rules.md RULE 0A:
    "After Stage 0 completes, you MUST pause and ask the user to confirm
    the material optical constants are correct before proceeding."
    
    Returns:
        Dict with state updates including pending_user_questions and pending_validated_materials
    """
    from schemas.state import get_validation_hierarchy
    
    # Get material validation results from progress
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    # Find Stage 0 (material validation) results
    stage0_info = None
    for stage in stages:
        if stage.get("stage_type") == "MATERIAL_VALIDATION":
            stage0_info = stage
            break
    
    # Get output files from stage_outputs
    stage_outputs = state.get("stage_outputs", {})
    output_files = stage_outputs.get("files", [])
    plot_files = [f for f in output_files if f.endswith(('.png', '.pdf', '.jpg'))]
    
    # Extract materials from plan parameters - stored as PENDING until user approves
    pending_materials = _extract_validated_materials(state)
    
    # Build the checkpoint question per global_rules.md RULE 0A format
    question = _format_material_checkpoint_question(state, stage0_info, plot_files, pending_materials)
    
    return {
        "workflow_phase": "material_checkpoint",
        "pending_user_questions": [question],
        "awaiting_user_input": True,
        "ask_user_trigger": "material_checkpoint",
        "last_node_before_ask_user": "material_checkpoint",
        # Store as PENDING - will be moved to validated_materials on user approval
        "pending_validated_materials": pending_materials,
    }


def _extract_validated_materials(state: ReproState) -> list:
    """
    Extract material information from plan and build validated_materials list.
    
    Uses materials/index.json as the authoritative source for material data.
    Scans extracted_parameters for material-related entries and matches them
    against material_id values in the index.
    
    Returns:
        List of dicts with material_id, data_file, drude_lorentz_fit, etc.
    """
    import json
    import os
    
    plan = state.get("plan", {})
    extracted_params = plan.get("extracted_parameters", [])
    assumptions = state.get("assumptions", {})
    
    # Load material database
    material_db = _load_material_database()
    if not material_db:
        return []
    
    # Build lookup tables from material database
    material_lookup = {}
    for mat in material_db.get("materials", []):
        mat_id = mat.get("material_id", "")
        material_lookup[mat_id] = mat
        
        # Also index by simple name (e.g., "silver" -> "palik_silver")
        # Extract simple name from material_id (e.g., "palik_silver" -> "silver")
        parts = mat_id.split("_")
        if len(parts) >= 2:
            simple_name = parts[-1]  # e.g., "silver", "gold", "aluminum"
            if simple_name not in material_lookup:
                material_lookup[simple_name] = mat
    
    validated_materials = []
    seen_material_ids = set()
    
    # Scan extracted parameters for material info
    for param in extracted_params:
        name = param.get("name", "").lower()
        value = str(param.get("value", "")).lower()
        
        # Look for material-related parameters
        if "material" in name:
            matched_material = _match_material_from_text(value, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(_format_validated_material(
                    matched_material, 
                    from_source=f"parameter: {param.get('name')}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    # Also check assumptions for material choices
    global_assumptions = assumptions.get("global_assumptions", {})
    material_assumptions = global_assumptions.get("materials", [])
    
    for assumption in material_assumptions:
        if isinstance(assumption, dict):
            desc = assumption.get("description", "").lower()
            matched_material = _match_material_from_text(desc, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(_format_validated_material(
                    matched_material,
                    from_source=f"assumption: {assumption.get('description', '')[:50]}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    return validated_materials


def _load_material_database() -> dict:
    """Load materials/index.json database."""
    import json
    import os
    
    index_path = os.path.join(os.path.dirname(__file__), "..", "materials", "index.json")
    if not os.path.exists(index_path):
        # Try relative to current working directory
        index_path = "materials/index.json"
    
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load materials/index.json: {e}")
        return {}


def _match_material_from_text(text: str, material_lookup: dict) -> dict:
    """
    Match material name from text against material database.
    
    Returns the best matching material entry or None.
    """
    import re
    text_lower = text.lower()
    
    # Priority 1: Exact material_id match (e.g., "palik_gold")
    for mat_id, mat_entry in material_lookup.items():
        if mat_id in text_lower:
            return mat_entry
    
    # Priority 2: Word-boundary match for simple names
    # This avoids "golden" matching "gold" or "usage" matching "ag"
    simple_names = ["gold", "silver", "aluminum", "silicon", "sio2", "glass", "water", "air", "ag", "au", "al", "si"]
    
    # Map common chemical symbols to full names for lookup
    symbol_map = {
        "ag": "silver",
        "au": "gold",
        "al": "aluminum",
        "si": "silicon"
    }
    
    for name in simple_names:
        # Use regex to match whole words only
        if re.search(r'\b' + re.escape(name) + r'\b', text_lower):
            lookup_name = symbol_map.get(name, name)
            
            # Find the best match in lookup (prefer entries with csv_available=true)
            candidates = [v for k, v in material_lookup.items() if lookup_name in k]
            csv_available = [c for c in candidates if c.get("csv_available", False)]
            
            if csv_available:
                return csv_available[0]
            elif candidates:
                return candidates[0]
    
    return None


def _format_validated_material(mat_entry: dict, from_source: str) -> dict:
    """Format a material database entry for validated_materials list."""
    data_file = mat_entry.get("data_file")
    path = f"materials/{data_file}" if data_file else None
    
    return {
        "material_id": mat_entry.get("material_id"),
        "name": mat_entry.get("name"),
        "source": mat_entry.get("source"),
        "path": path,
        "csv_available": mat_entry.get("csv_available", False),
        "drude_lorentz_fit": mat_entry.get("drude_lorentz_fit"),
        "wavelength_range_nm": mat_entry.get("wavelength_range_nm"),
        "from": from_source,
    }


def _format_material_checkpoint_question(
    state: ReproState, 
    stage0_info: dict, 
    plot_files: list,
    validated_materials: list
) -> str:
    """Format the material checkpoint question per global_rules.md RULE 0A."""
    paper_id = state.get("paper_id", "unknown")
    
    # Format validated materials
    if validated_materials:
        materials_info = []
        for mat in validated_materials:
            mat_name = mat.get('name') or mat.get('material_id', 'unknown')
            materials_info.append(
                f"- {mat_name.upper()}: source={mat.get('source', 'unknown')}, file={mat.get('path', 'N/A')}"
            )
    else:
        materials_info = ["- No materials automatically detected"]
    
    # Format plot files list
    plots_text = "\n".join(f"- {f}" for f in plot_files) if plot_files else "- No plots generated"
    
    question = f"""
═══════════════════════════════════════════════════════════════════════
MANDATORY MATERIAL VALIDATION CHECKPOINT
═══════════════════════════════════════════════════════════════════════

Stage 0 (Material Validation) has completed for paper: {paper_id}

**Validated materials (will be used for all subsequent stages):**
{chr(10).join(materials_info)}

**Generated plots:**
{plots_text}

Please review the material optical constants comparison plots above.

**Required confirmation:**

Do the simulated optical constants (n, k, ε) match the paper's data 
within acceptable tolerance?

Options:
1. APPROVE - Material validation looks correct, proceed to Stage 1
2. CHANGE_DATABASE - Use different material database (specify which)
3. CHANGE_MATERIAL - Paper uses different material than assumed (specify which)
4. NEED_HELP - Unclear how to validate, need guidance

Note: If you APPROVE, the validated_materials list above will be passed
to Code Generator for all subsequent stages.

Please respond with your choice and any notes.
═══════════════════════════════════════════════════════════════════════
"""
    return question


def generate_report_node(state: ReproState) -> ReproState:
    """Generate final reproduction report."""
    state["workflow_phase"] = "reporting"
    # TODO: Implement report generation logic
    # - Compile figure comparisons
    # - Document assumptions
    # - Generate REPRODUCTION_REPORT.md
    return state


def handle_backtrack_node(state: ReproState) -> dict:
    """
    Process cross-stage backtracking.
    
    When SupervisorAgent decides to backtrack (verdict="backtrack_to_stage"),
    this node:
    1. Reads the backtrack_decision from state
    2. Marks the target stage as "needs_rerun"
    3. Marks all dependent stages as "invalidated"
    4. Increments the backtrack_count
    5. Clears working data to prepare for re-run
    
    Returns:
        Dict with state updates (LangGraph merges this into state)
    """
    import copy
    
    decision = state.get("backtrack_decision", {})
    if not decision or not decision.get("accepted"):
        # No valid backtrack decision - shouldn't happen, but handle gracefully
        return {"workflow_phase": "backtracking"}
    
    target_id = decision.get("target_stage_id", "")
    stages_to_invalidate = decision.get("stages_to_invalidate", [])
    
    # Deep copy progress to avoid mutating original
    progress = copy.deepcopy(state.get("progress", {}))
    stages = progress.get("stages", [])
    
    # Update stage statuses
    for stage in stages:
        stage_id = stage.get("stage_id", "")
        if stage_id == target_id:
            # Target stage: mark for re-run
            stage["status"] = "needs_rerun"
        elif stage_id in stages_to_invalidate:
            # Dependent stages: mark as invalidated
            stage["status"] = "invalidated"
    
    # Build return dict with all state updates
    return {
        "workflow_phase": "backtracking",
        "progress": progress,
        "current_stage_id": target_id,
        "backtrack_count": state.get("backtrack_count", 0) + 1,
        # Clear working data to prepare for re-run
        "code": None,
        "design_description": None,
        "stage_outputs": {},
        "run_error": None,
        "analysis_summary": None,
        # Track invalidated stages
        "invalidated_stages": stages_to_invalidate,
        # Clear verdicts
        "last_design_review_verdict": None,
        "last_code_review_verdict": None,
        "supervisor_verdict": None,
        # Reset per-stage counters
        "design_revision_count": 0,
        "code_revision_count": 0,
        "execution_failure_count": 0,
        "physics_failure_count": 0,
        "analysis_revision_count": 0,
    }

