"""
Base utilities and decorators for agent nodes.

Provides common patterns used across all agent node implementations:
- Context checking decorator
- Counter incrementing with max bounds
- LLM error handling (auto-approve, escalation)
"""

import logging
import re
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
    def decorator(func: Callable[..., Dict[str, Any]]):
        @wraps(func)
        def wrapper(state: ReproState, *args, **kwargs) -> Dict[str, Any]:
            escalation = check_context_or_escalate(state, node_name)
            if escalation is not None:
                if escalation.get("awaiting_user_input"):
                    return escalation
                state = {**state, **escalation}
            return func(state, *args, **kwargs)
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
    if user_responses is None:
        return ""
    if not isinstance(user_responses, dict):
        raise TypeError(f"user_responses must be a dict, got {type(user_responses).__name__}")
    if not user_responses:
        return ""
    last_response = list(user_responses.values())[-1]
    return str(last_response).strip().upper()


def check_keywords(response: str, keywords: list) -> bool:
    """
    Check if response contains any of the keywords.
    
    Args:
        response: Response string to check (case-insensitive)
        keywords: List of keywords to look for
        
    Returns:
        True if any keyword is found as a distinct word
    """
    if response is None:
        return False
    if keywords is None:
        raise TypeError("keywords must be a list, got None")
    if not isinstance(keywords, list):
        raise TypeError(f"keywords must be a list, got {type(keywords).__name__}")
    if not response or not keywords:
        return False
        
    response_upper = response.upper()
    
    for kw in keywords:
        if not kw:
            continue
        # Convert keyword to string and uppercase for case-insensitive matching
        kw_str = str(kw).upper()
        # Escape regex special characters
        kw_escaped = re.escape(kw_str)
        # Match whole word only to avoid false positives (e.g. "DISAPPROVE" matching "APPROVE")
        # Use word boundaries, but handle keywords with non-word characters
        # For keywords with only word characters, use \b boundaries
        # For keywords with non-word characters, use lookahead/lookbehind for boundaries
        if re.search(r'^[a-zA-Z0-9_]+$', kw_str):
            # Simple word-only keyword: use word boundaries
            pattern = rf"\b{kw_escaped}\b"
        else:
            # Keyword contains non-word characters: match as literal with boundaries
            # Match start of string or non-word char before, and end of string or non-word char after
            pattern = rf"(?<!\w){kw_escaped}(?!\w)"
        if re.search(pattern, response_upper):
            return True
            
    return False


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
    if state is None:
        raise TypeError("state must be a dict, got None")
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state).__name__}")
    
    current_count = state.get(counter_name, 0)
    # Handle case where counter exists but is None
    if current_count is None:
        current_count = 0
    # Validate counter value is an integer
    if not isinstance(current_count, int):
        raise TypeError(f"counter '{counter_name}' must be an int, got {type(current_count).__name__}: {current_count}")
    
    runtime_config = state.get("runtime_config") or {}
    if not isinstance(runtime_config, dict):
        raise TypeError(f"runtime_config must be a dict, got {type(runtime_config).__name__}")
    
    max_value = runtime_config.get(config_key, default_max)
    # Validate max_value is an integer
    if max_value is not None and not isinstance(max_value, int):
        raise TypeError(f"max value '{config_key}' must be an int, got {type(max_value).__name__}: {max_value}")
    if max_value is None:
        max_value = default_max
    
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
    verb_suffix = "d" if default_verdict.endswith("e") else "ed"
    # If default_verdict is "pass", -> "passed". If "approve" -> "approved".
    
    # Handle case where __str__ might return None
    try:
        error_str = str(error)
        if error_str is None:
            error_str = ""
    except (TypeError, ValueError):
        error_str = ""
    
    _logger.warning(f"{agent_name} LLM call failed: {error_str or repr(error)}. Auto-{default_verdict}ing.")
    
    error_msg = error_str[:error_truncate_len] if error_str else ""
    agent_label = agent_name.replace("_", " ").title()
    
    return {
        "verdict": default_verdict,
        "issues": [{"severity": "minor", "description": f"LLM review unavailable: {error_msg}"}],
        "summary": f"{agent_label} auto-{default_verdict}{verb_suffix} due to LLM unavailability",
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
    # Handle case where __str__ might return None
    try:
        error_str = str(error)
        if error_str is None:
            error_str = ""
    except (TypeError, ValueError):
        error_str = ""
    
    _logger.error(f"{agent_name} LLM call failed: {error_str or repr(error)}. Escalating to user.")
    
    error_msg = error_str[:error_truncate_len] if error_str else ""
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
        # Handle case where __str__ might return None
        try:
            error_str = str(error)
            if error_str is None:
                error_str = ""
        except (TypeError, ValueError):
            error_str = ""
        
        _logger.warning(f"{agent_name} LLM call failed: {error_str or repr(error)}. Using default: {default_verdict}")
        
        error_msg = error_str[:error_truncate_len] if error_str else ""
        feedback = feedback_msg.format(error=error_msg) if feedback_msg else f"LLM unavailable: {error_msg}"
        
        return {
            f"{agent_name}_verdict": default_verdict,
            f"{agent_name}_feedback": feedback,
        }
    
    return handler
