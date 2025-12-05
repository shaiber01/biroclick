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
from .user_options import get_options_prompt


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
                agent_output["summary"] = f"{missing_verdict_msg} Original summary: {agent_output.get('summary', '')}"
            else:
                agent_output["summary"] = missing_verdict_msg
        agent_output["stage_id"] = stage_id
    
    # Extract warnings array from agent output (schema field: "warnings")
    warnings_list = agent_output.get("warnings", [])
    if not isinstance(warnings_list, list):
        warnings_list = [str(warnings_list)] if warnings_list else []
    
    # Extract structured data from agent output (preserves rich LLM response)
    execution_status = agent_output.get("execution_status")
    execution_files_check = agent_output.get("files_check")
    
    result: Dict[str, Any] = {
        "workflow_phase": "execution_validation",
        "execution_verdict": agent_output.get("verdict", "pass"),
        "execution_feedback": agent_output.get("summary", "No feedback provided."),
        "execution_warnings": warnings_list,
        # Preserve structured data from LLM output
        "execution_status": execution_status,
        "execution_files_check": execution_files_check,
    }
    
    # Increment failure counters if verdict is "fail"
    if agent_output.get("verdict") == "fail":
        new_count, _ = increment_counter_with_max(
            state, "execution_failure_count", "max_execution_failures", MAX_EXECUTION_FAILURES
        )
        result["execution_failure_count"] = new_count
        # Handle None value for total_execution_failures
        current_total = state.get("total_execution_failures", 0)
        if current_total is None:
            current_total = 0
        result["total_execution_failures"] = current_total + 1
        
        # If we hit the max failure limit, escalate to ask_user immediately.
        # BUG FIX: Check if new_count >= max (not just if increment failed).
        runtime_config = state.get("runtime_config", {})
        max_failures = runtime_config.get("max_execution_failures", MAX_EXECUTION_FAILURES)
        
        if new_count >= max_failures:
            result["ask_user_trigger"] = "execution_failure_limit"
            result["pending_user_questions"] = [
                f"Execution failed {new_count}/{max_failures} times. Last error: {run_error or 'Unknown'}.\n\n"
                f"{get_options_prompt('execution_failure_limit')}"
            ]
            result["awaiting_user_input"] = True
            result["last_node_before_ask_user"] = "execution_check"
    
    # Log execution validation result using structured data
    verdict = agent_output.get("verdict", "pass")
    stage_id = state.get("current_stage_id", "unknown")
    emoji = "✅" if verdict == "pass" else "⚠️" if verdict == "warning" else "❌"
    
    # Build concise log message from structured data
    status = execution_status or {}
    files = execution_files_check or {}
    exit_code = status.get("exit_code", "?")
    runtime = status.get("runtime_seconds", 0)
    files_found = len(files.get("found_files", []))
    files_expected = len(files.get("expected_files", []))
    
    logger.info(
        f"{emoji} execution_check: stage={stage_id}, verdict={verdict}, "
        f"exit={exit_code}, runtime={runtime:.1f}s, files={files_found}/{files_expected}"
    )
    
    # Log full summary at DEBUG level for detailed troubleshooting
    summary = agent_output.get("summary", "")
    if summary:
        logger.debug(f"   execution_check details: {summary[:300]}{'...' if len(summary) > 300 else ''}")
    
    # Log warnings separately for visibility
    if warnings_list:
        logger.info(f"   ⚠️ {len(warnings_list)} warning(s) from execution validation:")
        for i, warning in enumerate(warnings_list, 1):
            logger.info(f"      {i}. {warning}")
    
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
    # Only include if design has actual content (empty dict/string provides no value)
    if design:
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
    
    # Check for explicit escalation to user (before processing verdict)
    escalate = agent_output.get("escalate_to_user")
    if escalate and isinstance(escalate, str) and escalate.strip():
        return {
            "workflow_phase": "physics_validation",
            "ask_user_trigger": "reviewer_escalation",
            "pending_user_questions": [f"{escalate}\n\n{get_options_prompt('reviewer_escalation')}"],
            "awaiting_user_input": True,
            "last_node_before_ask_user": "physics_check",
            "reviewer_escalation_source": "physics_sanity",
        }
    
    # Handle missing verdict gracefully (separate from LLM unavailability)
    if "verdict" not in agent_output:
        logger.warning("Physics sanity output missing 'verdict'. Defaulting to 'pass'.")
        agent_output["verdict"] = "pass"
        # Always indicate missing verdict in summary, even if summary was provided
        missing_verdict_msg = "Missing verdict in LLM response, defaulting to pass."
        if "summary" in agent_output:
            agent_output["summary"] = f"{missing_verdict_msg} Original summary: {agent_output.get('summary', '')}"
        else:
            agent_output["summary"] = missing_verdict_msg
    
    agent_output["stage_id"] = stage_id
    if "backtrack_suggestion" not in agent_output:
        agent_output["backtrack_suggestion"] = {"suggest_backtrack": False}
    
    # Extract concerns from agent output (schema field: "concerns")
    # Transform concern objects into warning strings for state
    concerns_list = agent_output.get("concerns", [])
    physics_warnings = []
    if isinstance(concerns_list, list):
        for concern in concerns_list:
            if isinstance(concern, dict):
                concern_text = concern.get("concern", "")
                severity = concern.get("severity", "")
                if concern_text:
                    physics_warnings.append(f"[{severity}] {concern_text}" if severity else concern_text)
            elif isinstance(concern, str):
                physics_warnings.append(concern)
    
    # Extract structured data from agent output (preserves rich LLM response)
    conservation_checks = agent_output.get("conservation_checks")
    value_range_checks = agent_output.get("value_range_checks")
    
    result: Dict[str, Any] = {
        "workflow_phase": "physics_validation",
        "physics_verdict": agent_output.get("verdict", "pass"),
        "physics_feedback": agent_output.get("summary", "No feedback provided."),
        "physics_warnings": physics_warnings,
        # Preserve structured data from LLM output
        "physics_conservation_checks": conservation_checks,
        "physics_value_range_checks": value_range_checks,
    }
    
    # Increment failure counters based on verdict type
    if agent_output.get("verdict") == "fail":
        new_count, _ = increment_counter_with_max(
            state, "physics_failure_count", "max_physics_failures", MAX_PHYSICS_FAILURES
        )
        result["physics_failure_count"] = new_count
        
        # If we hit the max physics failure limit, escalate to ask_user immediately.
        # BUG FIX: Check if new_count >= max (not just if increment failed).
        runtime_config = state.get("runtime_config", {})
        max_failures = runtime_config.get("max_physics_failures", MAX_PHYSICS_FAILURES)
        stage_id = state.get("current_stage_id", "unknown")
        
        if new_count >= max_failures:
            result["ask_user_trigger"] = "physics_failure_limit"
            result["pending_user_questions"] = [
                f"Physics sanity check failed {new_count}/{max_failures} times.\n\n"
                f"- Stage: {stage_id}\n"
                f"- Latest feedback: {result.get('physics_feedback', 'No feedback available')}\n\n"
                f"{get_options_prompt('physics_failure_limit')}"
            ]
            result["awaiting_user_input"] = True
            result["last_node_before_ask_user"] = "physics_check"
    elif agent_output.get("verdict") == "design_flaw":
        new_count, _ = increment_counter_with_max(
            state, "design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS
        )
        result["design_revision_count"] = new_count
        result["design_feedback"] = agent_output.get("summary", "Design flaw detected.")
        
        # If we hit the max design revision limit, escalate to ask_user immediately.
        # This mirrors the physics_failure_limit logic above.
        runtime_config = state.get("runtime_config", {})
        max_revisions = runtime_config.get("max_design_revisions", MAX_DESIGN_REVISIONS)
        stage_id = state.get("current_stage_id", "unknown")
        
        if new_count >= max_revisions:
            result["ask_user_trigger"] = "design_flaw_limit"
            result["pending_user_questions"] = [
                f"Physics check detected design flaws {new_count}/{max_revisions} times.\n\n"
                f"- Stage: {stage_id}\n"
                f"- Latest feedback: {result.get('design_feedback', 'No feedback available')}\n\n"
                f"{get_options_prompt('design_flaw_limit')}"
            ]
            result["awaiting_user_input"] = True
            result["last_node_before_ask_user"] = "physics_check"
    
    # If agent suggests backtrack, populate backtrack_suggestion for supervisor
    backtrack = agent_output.get("backtrack_suggestion", {})
    if isinstance(backtrack, dict) and backtrack.get("suggest_backtrack"):
        result["backtrack_suggestion"] = backtrack
    
    # Log physics sanity result using structured data
    verdict = agent_output.get("verdict", "pass")
    emoji = "✅" if verdict == "pass" else "⚠️" if verdict == "warning" else "🔧" if verdict == "design_flaw" else "❌"
    
    # Build concise log message from structured data
    n_conservation = len(conservation_checks) if conservation_checks else 0
    n_range = len(value_range_checks) if value_range_checks else 0
    n_concerns = len(physics_warnings)
    
    # Count passed vs failed checks
    conservation_passed = sum(1 for c in (conservation_checks or []) if c.get("status") == "pass")
    range_passed = sum(1 for c in (value_range_checks or []) if c.get("status") == "pass")
    
    logger.info(
        f"{emoji} physics_check: stage={stage_id}, verdict={verdict}, "
        f"conservation={conservation_passed}/{n_conservation}, ranges={range_passed}/{n_range}, concerns={n_concerns}"
    )
    
    # Log full summary at DEBUG level for detailed troubleshooting
    summary = agent_output.get("summary", "")
    if summary:
        logger.debug(f"   physics_check details: {summary[:300]}{'...' if len(summary) > 300 else ''}")
    
    # Log concerns/warnings separately for visibility
    if physics_warnings:
        logger.info(f"   ⚠️ {len(physics_warnings)} concern(s) from physics validation:")
        for i, warning in enumerate(physics_warnings, 1):
            logger.info(f"      {i}. {warning}")
    
    return result

