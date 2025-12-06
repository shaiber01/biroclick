"""
Context window management and state validation utilities.
"""

import logging
from typing import Dict, Any, Optional, List

from schemas.state import (
    ReproState,
    check_context_before_node,
    validate_state_for_node,
)
from src.agents.user_options import get_options_for_trigger, get_clarification_message


def check_context_or_escalate(state: ReproState, node_name: str) -> Optional[Dict[str, Any]]:
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
    
    if check.get("escalate", False):
        # Must ask user - return state updates to trigger ask_user
        return {
            "pending_user_questions": [check.get("user_question", f"Context overflow in {node_name}. How should we proceed?")],
            "ask_user_trigger": "context_overflow",
            "last_node_before_ask_user": node_name,
        }
    
    # Shouldn't reach here, but fallback to escalation
    return {
        "pending_user_questions": [f"Context overflow in {node_name}. How should we proceed?"],
        "ask_user_trigger": "context_overflow",
        "last_node_before_ask_user": node_name,
    }


def _contains_keyword(text: str, keyword: str) -> bool:
    """
    Check if text contains keyword as a whole word or substring.
    For short keywords like "NO" and "YES", check for whole word matches to avoid false positives.
    """
    # For very short keywords that are substrings of common words, use word boundary matching
    short_keywords = {"NO", "YES"}
    if keyword in short_keywords:
        # Check for whole word match (surrounded by spaces or at start/end)
        normalized_text = f" {text} "
        return f" {keyword} " in normalized_text
    else:
        # For longer keywords, substring matching is fine
        return keyword in text


def validate_user_responses(trigger: str, responses: Dict[str, str], questions: List[str]) -> List[str]:
    """
    Validate user responses against expected format for the trigger type.
    
    Uses the centralized USER_OPTIONS from user_options.py as the single source of truth.
    
    Args:
        trigger: The ask_user_trigger value (e.g., "material_checkpoint")
        responses: Dict mapping question -> response
        questions: List of questions that were asked
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not responses:
        errors.append("No responses provided")
        return errors
    
    # Get all response text (combined)
    # Normalize spaces/underscores for keyword matching
    all_responses = " ".join(str(r).upper() for r in responses.values())
    # Replace underscores with spaces for matching (user might type "change material" instead of "CHANGE_MATERIAL")
    all_responses_normalized = all_responses.replace("_", " ")
    
    # Get options from centralized configuration
    options = get_options_for_trigger(trigger)
    
    if not options:
        # Unknown trigger - just check that response is not empty
        if not all_responses.strip():
            errors.append("Response cannot be empty")
        return errors
    
    # Build keyword list from all options (display + aliases)
    valid_keywords = []
    for opt in options:
        # Add display keyword (normalized - underscore to space)
        valid_keywords.append(opt.display.replace("_", " "))
        # Add all aliases (normalized)
        for alias in opt.aliases:
            valid_keywords.append(alias.replace("_", " "))
    
    # Check if any keyword matches (accounting for short keywords)
    matched = False
    for kw in valid_keywords:
        if _contains_keyword(all_responses_normalized, kw):
            matched = True
            break
    
    if not matched:
        # Use the centralized clarification message
        clarification = get_clarification_message(trigger)
        errors.append(clarification)
    
    return errors


def validate_state_or_warn(state: ReproState, node_name: str) -> list:
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
        logger = logging.getLogger(__name__)
        for issue in issues:
            logger.warning(f"State validation issue for {node_name}: {issue}")
    return issues



