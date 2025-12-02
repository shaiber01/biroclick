"""
Base utilities and decorators for agent nodes.

Provides common patterns used across all agent node implementations:
- Context checking decorator
- Counter incrementing with max bounds
- LLM error handling (auto-approve, escalation)
"""

import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple

from schemas.state import ReproState
from .helpers.context import check_context_or_escalate

# Project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Common logger for base utilities
_logger = logging.getLogger(__name__)


def with_context_check(node_name: str):
    """
    Decorator that handles context checking boilerplate.
    
    Wraps an agent node function to automatically check context
    before execution and handle escalation if needed.
    
    Usage:
        @with_context_check("design")
        def simulation_designer_node(state: ReproState) -> dict:
            # ... actual logic, no boilerplate ...
    """
    def decorator(func: Callable[[ReproState], Dict[str, Any]]):
        @wraps(func)
        def wrapper(state: ReproState) -> Dict[str, Any]:
            escalation = check_context_or_escalate(state, node_name)
            if escalation is not None:
                if escalation.get("awaiting_user_input"):
                    return escalation
                state = {**state, **escalation}
            return func(state)
        return wrapper
    return decorator


def bounded_increment(current: int, max_value: int) -> int:
    """
    Increment counter without exceeding maximum.
    
    Args:
        current: Current counter value
        max_value: Maximum allowed value
        
    Returns:
        min(current + 1, max_value)
    """
    return min(current + 1, max_value)


def parse_user_response(user_responses: Dict[str, str]) -> str:
    """
    Extract and normalize the last user response.
    
    Args:
        user_responses: Dict mapping question -> response
        
    Returns:
        Uppercased last response string, or empty string if no responses
    """
    if not user_responses:
        return ""
    last_response = list(user_responses.values())[-1]
    return last_response.upper() if isinstance(last_response, str) else str(last_response)


def check_keywords(response: str, keywords: list) -> bool:
    """
    Check if response contains any of the keywords.
    
    Args:
        response: Response string to check (should be uppercased)
        keywords: List of keywords to look for
        
    Returns:
        True if any keyword is found
    """
    return any(kw in response for kw in keywords)


def increment_counter_with_max(
    state: ReproState,
    counter_name: str,
    config_key: str,
    default_max: int,
) -> Tuple[int, bool]:
    """
    Increment a counter up to a maximum value from runtime config.
    
    Common pattern used for revision/failure counters that need to respect
    configurable limits while tracking how many attempts have been made.
    
    Args:
        state: Current workflow state
        counter_name: Name of counter in state (e.g., "code_revision_count")
        config_key: Key in runtime_config for max value (e.g., "max_code_revisions")
        default_max: Default max if not in config
        
    Returns:
        Tuple of (new_count, was_incremented)
        - new_count: The new counter value to set in result
        - was_incremented: True if counter was incremented, False if at max
        
    Example:
        new_count, incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", MAX_CODE_REVISIONS
        )
        result["code_revision_count"] = new_count
        if not incremented:
            # Handle max reached
    """
    current_count = state.get(counter_name, 0)
    runtime_config = state.get("runtime_config", {})
    max_value = runtime_config.get(config_key, default_max)
    
    if current_count < max_value:
        return current_count + 1, True
    return current_count, False


def create_llm_error_auto_approve(
    agent_name: str,
    error: Exception,
    default_verdict: str = "approve",
    error_truncate_len: int = 200,
) -> Dict[str, Any]:
    """
    Create auto-approve response when reviewer LLM call fails.
    
    Used by reviewer nodes (code_reviewer, design_reviewer, plan_reviewer, etc.)
    that should auto-approve and continue when LLM is unavailable, rather than
    blocking the workflow.
    
    Args:
        agent_name: Name of the agent for logging/messaging
        error: The exception that was raised
        default_verdict: Verdict to return ("approve" for reviewers, "pass" for validators)
        error_truncate_len: Max length for error message in summary
        
    Returns:
        Dict with verdict, issues, and summary suitable for agent_output
        
    Example:
        try:
            agent_output = call_agent_with_metrics(...)
        except Exception as e:
            logger.error(f"Code reviewer LLM call failed: {e}")
            agent_output = create_llm_error_auto_approve("code_reviewer", e)
    """
    _logger.warning(f"{agent_name} LLM call failed: {error}. Auto-{default_verdict}ing.")
    
    error_msg = str(error)[:error_truncate_len]
    agent_label = agent_name.replace("_", " ").title()
    
    return {
        "verdict": default_verdict,
        "issues": [{"severity": "minor", "description": f"LLM review unavailable: {error_msg}"}],
        "summary": f"{agent_label} auto-{default_verdict}ed due to LLM unavailability",
    }


def create_llm_error_escalation(
    agent_name: str,
    workflow_phase: str,
    error: Exception,
    error_truncate_len: int = 500,
) -> Dict[str, Any]:
    """
    Create user escalation response when critical LLM call fails.
    
    Used by generator/planner nodes (code_generator, simulation_designer, plan_node)
    where LLM failure requires user intervention since the node cannot produce
    meaningful output without the LLM.
    
    Args:
        agent_name: Name of the agent for logging/messaging
        workflow_phase: Current workflow phase to preserve
        error: The exception that was raised
        error_truncate_len: Max length for error message in question
        
    Returns:
        Dict with workflow_phase, ask_user_trigger, pending_user_questions,
        and awaiting_user_input
        
    Example:
        try:
            agent_output = call_agent_with_metrics(...)
        except Exception as e:
            logger.error(f"Code generator LLM call failed: {e}")
            return create_llm_error_escalation("code_generator", "code_generation", e)
    """
    _logger.error(f"{agent_name} LLM call failed: {error}. Escalating to user.")
    
    error_msg = str(error)[:error_truncate_len]
    agent_label = agent_name.replace("_", " ").title()
    
    return {
        "workflow_phase": workflow_phase,
        "ask_user_trigger": "llm_error",
        "pending_user_questions": [
            f"{agent_label} failed: {error_msg}. Please check API and try again."
        ],
        "awaiting_user_input": True,
    }


def create_llm_error_fallback(
    agent_name: str,
    default_verdict: str,
    feedback_msg: Optional[str] = None,
    error_truncate_len: int = 200,
) -> Callable[[Exception], Dict[str, Any]]:
    """
    Create a fallback handler for non-critical LLM failures.
    
    Used by nodes where LLM failure should not block workflow but should
    be noted (e.g., supervisor defaulting to ok_continue).
    
    Args:
        agent_name: Name of the agent for logging
        default_verdict: Default verdict/value to use
        feedback_msg: Optional custom feedback message format (use {error} placeholder)
        error_truncate_len: Max length for error in feedback
        
    Returns:
        A callable that takes an exception and returns appropriate dict
        
    Example:
        handle_error = create_llm_error_fallback("supervisor", "ok_continue")
        try:
            agent_output = call_agent_with_metrics(...)
        except Exception as e:
            fallback = handle_error(e)
            result.update(fallback)
    """
    def handler(error: Exception) -> Dict[str, Any]:
        _logger.warning(f"{agent_name} LLM call failed: {error}. Using default: {default_verdict}")
        
        error_msg = str(error)[:error_truncate_len]
        feedback = feedback_msg.format(error=error_msg) if feedback_msg else f"LLM unavailable: {error_msg}"
        
        return {
            f"{agent_name}_verdict": default_verdict,
            f"{agent_name}_feedback": feedback,
        }
    
    return handler

