"""
Execution agent nodes: execution_validator_node, physics_sanity_node.

These nodes handle execution validation and physics sanity checking.

State Keys
----------
execution_validator_node:
    READS: execution_result, current_stage_id, code, design_description,
           execution_failure_count, runtime_config
    WRITES: workflow_phase, execution_verdict, execution_feedback,
            execution_failure_count, ask_user_trigger, pending_user_questions,
            awaiting_user_input

physics_sanity_node:
    READS: execution_result, current_stage_id, design_description, plan,
           physics_failure_count, runtime_config
    WRITES: workflow_phase, physics_verdict, physics_feedback, physics_failure_count,
            design_revision_count, design_feedback, ask_user_trigger,
            pending_user_questions, awaiting_user_input
"""

import json
import logging
from typing import Dict, Any

from schemas.state import (
    ReproState,
    get_stage_design_spec,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_DESIGN_REVISIONS,
)
from src.prompts import build_agent_prompt
from src.llm_client import call_agent_with_metrics

from .helpers.context import check_context_or_escalate
from .base import (
    with_context_check,
    increment_counter_with_max,
    create_llm_error_auto_approve,
)


@with_context_check("execution_check")
def execution_validator_node(state: ReproState) -> dict:
    """
    ExecutionValidatorAgent: Validate simulation ran correctly.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `execution_verdict` state field from agent output's `verdict`.
    - Increments `execution_failure_count` when verdict is "fail".
    - Increments `total_execution_failures` for metrics tracking.
    
    Note: Context check is handled by @with_context_check decorator.
    """
    logger = logging.getLogger(__name__)

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("execution_validator", state)
    
    # Inject run_error into prompt if present
    run_error = state.get("run_error")
    if run_error:
        system_prompt += f"\n\nCONTEXT: The previous execution failed with error:\n{run_error}\nAnalyze this error and suggest fixes."
    
    # Check fallback strategy if we are about to fail
    fallback = get_stage_design_spec(state, state.get("current_stage_id"), "fallback_strategy", "ask_user")
    
    # Check for timeout (via flag or string match)
    stage_outputs = state.get("stage_outputs") or {}
    is_timeout = stage_outputs.get("timeout_exceeded", False)
    if not is_timeout and run_error:
        # Legacy/fallback check for string pattern
        err_str = str(run_error).lower()
        is_timeout = "exceeded timeout" in err_str or "timeout_error" in err_str
    
    if is_timeout:
        if fallback == "skip_with_warning":
            agent_output = {
                "verdict": "pass",
                "stage_id": state.get("current_stage_id"),
                "summary": f"Execution timed out (skip_with_warning): {run_error}",
            }
        else:
            agent_output = {
                "verdict": "fail",
                "stage_id": state.get("current_stage_id"),
                "summary": f"Execution timed out: {run_error}",
            }
    else:
        # Build user content for execution validation
        # stage_outputs already retrieved above
        stage_id = state.get("current_stage_id", "unknown")
        
        user_content = f"# EXECUTION RESULTS FOR STAGE: {stage_id}\n\n"
        user_content += f"## Stage Outputs\n```json\n{json.dumps(stage_outputs, indent=2, default=str)}\n```"
        
        if run_error:
            user_content += f"\n\n## Run Error\n\n```\n{run_error}\n```"
        
        # Call LLM for execution validation
        try:
            agent_output = call_agent_with_metrics(
                agent_name="execution_validator",
                system_prompt=system_prompt,
                user_content=user_content,
                state=state,
            )
        except Exception as e:
            logger.error(f"Execution validator LLM call failed: {e}")
            agent_output = create_llm_error_auto_approve("execution_validator", e, default_verdict="pass")
            agent_output["summary"] = f"Auto-approved due to LLM error: {e}"
        
        # Handle missing verdict gracefully (separate from LLM unavailability)
        if "verdict" not in agent_output:
            logger.warning("Execution validator output missing 'verdict'. Defaulting to 'pass'.")
            agent_output["verdict"] = "pass"
            # Always indicate missing verdict in summary, even if summary was provided
            missing_verdict_msg = "Missing verdict in LLM response, defaulting to pass."
            if "summary" in agent_output:
                agent_output["summary"] = f"{missing_verdict_msg} Original summary: {agent_output['summary']}"
            else:
                agent_output["summary"] = missing_verdict_msg
        agent_output["stage_id"] = stage_id
    
    result: Dict[str, Any] = {
        "workflow_phase": "execution_validation",
        "execution_verdict": agent_output["verdict"],
        "execution_feedback": agent_output.get("summary", "No feedback provided."),
    }
    
    # Increment failure counters if verdict is "fail"
    if agent_output["verdict"] == "fail":
        new_count, was_incremented = increment_counter_with_max(
            state, "execution_failure_count", "max_execution_failures", MAX_EXECUTION_FAILURES
        )
        result["execution_failure_count"] = new_count
        # Handle None value for total_execution_failures
        current_total = state.get("total_execution_failures", 0)
        if current_total is None:
            current_total = 0
        result["total_execution_failures"] = current_total + 1
        
        runtime_config = state.get("runtime_config", {})
        max_failures = runtime_config.get("max_execution_failures", MAX_EXECUTION_FAILURES)
        
        if not was_incremented:
            result["ask_user_trigger"] = "execution_failure_limit"
            result["pending_user_questions"] = [
                f"Execution failed {max_failures} times. Last error: {run_error or 'Unknown'}. "
                "Options: RETRY_WITH_GUIDANCE (provide hint), SKIP_STAGE, or STOP?"
            ]
    
    return result


@with_context_check("physics_check")
def physics_sanity_node(state: ReproState) -> dict:
    """
    PhysicsSanityAgent: Validate physics of results.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `physics_verdict` state field from agent output's `verdict`.
    - Increments `physics_failure_count` when verdict is "fail".
    - Increments `design_revision_count` when verdict is "design_flaw".
    
    Verdict options:
    - "pass": Physics looks good, proceed to analysis
    - "warning": Minor concerns but proceed
    - "fail": Code/numerics issue, route to code generator
    - "design_flaw": Fundamental design problem, route to simulation designer
    
    Note: Context check is handled by @with_context_check decorator.
    """
    logger = logging.getLogger(__name__)

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("physics_sanity", state)
    
    # Build user content for physics sanity check
    stage_outputs = state.get("stage_outputs") or {}
    stage_id = state.get("current_stage_id", "unknown")
    design = state.get("design_description", {})
    
    user_content = f"# PHYSICS SANITY CHECK FOR STAGE: {stage_id}\n\n"
    user_content += f"## Stage Outputs\n```json\n{json.dumps(stage_outputs, indent=2, default=str)}\n```"
    
    # Add design spec for physics reference
    # Include design_description even if empty dict (falsy but should be shown)
    if design is not None:
        if isinstance(design, dict):
            user_content += f"\n\n## Design Spec\n```json\n{json.dumps(design, indent=2)}\n```"
        else:
            user_content += f"\n\n## Design Spec\n{design}"
    
    # Call LLM for physics sanity check
    try:
        agent_output = call_agent_with_metrics(
            agent_name="physics_sanity",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
    except Exception as e:
        logger.error(f"Physics sanity LLM call failed: {e}")
        agent_output = create_llm_error_auto_approve("physics_sanity", e, default_verdict="pass")
        agent_output["summary"] = f"Auto-approved due to LLM error: {e}"
    
    # Handle missing verdict gracefully (separate from LLM unavailability)
    if "verdict" not in agent_output:
        logger.warning("Physics sanity output missing 'verdict'. Defaulting to 'pass'.")
        agent_output["verdict"] = "pass"
        # Always indicate missing verdict in summary, even if summary was provided
        missing_verdict_msg = "Missing verdict in LLM response, defaulting to pass."
        if "summary" in agent_output:
            agent_output["summary"] = f"{missing_verdict_msg} Original summary: {agent_output['summary']}"
        else:
            agent_output["summary"] = missing_verdict_msg
    
    agent_output["stage_id"] = stage_id
    if "backtrack_suggestion" not in agent_output:
        agent_output["backtrack_suggestion"] = {"suggest_backtrack": False}
    
    result: Dict[str, Any] = {
        "workflow_phase": "physics_validation",
        "physics_verdict": agent_output["verdict"],
        "physics_feedback": agent_output.get("summary", "No feedback provided."),
    }
    
    # Increment failure counters based on verdict type
    if agent_output["verdict"] == "fail":
        new_count, _ = increment_counter_with_max(
            state, "physics_failure_count", "max_physics_failures", MAX_PHYSICS_FAILURES
        )
        result["physics_failure_count"] = new_count
    elif agent_output["verdict"] == "design_flaw":
        new_count, _ = increment_counter_with_max(
            state, "design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS
        )
        result["design_revision_count"] = new_count
        result["design_feedback"] = agent_output.get("summary", "Design flaw detected.")
    
    # If agent suggests backtrack, populate backtrack_suggestion for supervisor
    backtrack = agent_output.get("backtrack_suggestion", {})
    if isinstance(backtrack, dict) and backtrack.get("suggest_backtrack"):
        result["backtrack_suggestion"] = backtrack
    
    return result

