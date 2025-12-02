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


def validate_user_responses(trigger: str, responses: Dict[str, str], questions: List[str]) -> List[str]:
    """
    Validate user responses against expected format for the trigger type.
    
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
    all_responses = " ".join(str(r).upper() for r in responses.values())
    
    if trigger == "material_checkpoint":
        # Must contain one of: APPROVE, CHANGE_MATERIAL, CHANGE_DATABASE, NEED_HELP
        valid_keywords = ["APPROVE", "CHANGE_MATERIAL", "CHANGE_DATABASE", "NEED_HELP", "HELP", 
                         "YES", "NO", "REJECT", "CORRECT", "WRONG"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: APPROVE, CHANGE_MATERIAL, CHANGE_DATABASE, or NEED_HELP"
            )
    
    elif trigger == "code_review_limit":
        valid_keywords = ["PROVIDE_HINT", "HINT", "SKIP", "STOP", "RETRY"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: PROVIDE_HINT, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "design_review_limit":
        valid_keywords = ["PROVIDE_HINT", "HINT", "SKIP", "STOP", "RETRY"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: PROVIDE_HINT, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "execution_failure_limit":
        valid_keywords = ["RETRY", "GUIDANCE", "SKIP", "STOP"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: RETRY_WITH_GUIDANCE, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "physics_failure_limit":
        valid_keywords = ["RETRY", "ACCEPT", "PARTIAL", "SKIP", "STOP"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "backtrack_approval":
        valid_keywords = ["APPROVE", "REJECT", "YES", "NO"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: APPROVE or REJECT"
            )
    
    elif trigger == "replan_limit":
        valid_keywords = ["FORCE", "ACCEPT", "GUIDANCE", "STOP"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: FORCE_ACCEPT, PROVIDE_GUIDANCE, or STOP"
            )
    
    # For unknown triggers, just check that response is not empty
    elif trigger not in ["context_overflow", "backtrack_limit"]:
        if not all_responses.strip():
            errors.append("Response cannot be empty")
    
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



