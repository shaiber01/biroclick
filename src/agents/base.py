"""
Base utilities and decorators for agent nodes.

Provides common patterns used across all agent node implementations.
"""

from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Any

from schemas.state import ReproState
from .helpers.context import check_context_or_escalate

# Project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


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

