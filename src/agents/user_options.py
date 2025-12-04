"""
Single source of truth for user interaction options.

This module defines all valid user responses for each trigger type,
eliminating the possibility of mismatch between what's shown to users
and what's recognized by handlers.

Usage:
    from src.agents.user_options import (
        match_user_response,
        get_options_prompt,
        get_clarification_message,
        validate_no_collisions,
    )
    
    # In trigger handler:
    matched = match_user_response("replan_limit", response_text)
    if matched:
        if matched.action == "force_accept":
            # Handle force accept
    else:
        # No match - ask for clarification
        result["pending_user_questions"] = [get_clarification_message("replan_limit")]
    
    # In agent node (prompt generation):
    result["pending_user_questions"] = [
        f"Some message here.\n\n{get_options_prompt('replan_limit')}"
    ]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UserOption:
    """
    A user interaction option.
    
    Attributes:
        display: The keyword shown to and expected from user (e.g., "APPROVE_PLAN")
        description: Explanation of what the option does
        action: Internal action identifier used by handlers
        aliases: Additional keywords that match this option (optional)
    """
    display: str
    description: str
    action: str
    aliases: List[str] = field(default_factory=list)
    
    @property
    def keywords(self) -> List[str]:
        """All keywords that match this option (display + aliases), uppercase."""
        return [self.display.upper()] + [a.upper() for a in self.aliases]


# ═══════════════════════════════════════════════════════════════════════
# USER OPTIONS BY TRIGGER TYPE
# ═══════════════════════════════════════════════════════════════════════
# Each trigger type maps to a list of UserOption objects.
# Options are checked in order, so put more specific options first.

USER_OPTIONS: Dict[str, List[UserOption]] = {
    # ───────────────────────────────────────────────────────────────────
    # PLANNING TRIGGERS
    # ───────────────────────────────────────────────────────────────────
    "replan_limit": [
        UserOption(
            display="APPROVE_PLAN",
            description="Force-accept the current plan despite issues",
            action="force_accept",
            aliases=["APPROVE", "ACCEPT", "FORCE", "FORCE_ACCEPT"],
        ),
        UserOption(
            display="GUIDANCE",
            description="Provide specific guidance for replanning (include guidance after keyword)",
            action="replan_with_guidance",
            aliases=["GUIDE", "HINT"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # REVIEW LIMIT TRIGGERS
    # ───────────────────────────────────────────────────────────────────
    "code_review_limit": [
        UserOption(
            display="PROVIDE_HINT",
            description="Reset counter and retry with your guidance",
            action="provide_hint",
            aliases=["HINT"],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    "design_review_limit": [
        UserOption(
            display="PROVIDE_HINT",
            description="Reset counter and retry with your guidance",
            action="provide_hint",
            aliases=["HINT"],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # EXECUTION TRIGGERS
    # ───────────────────────────────────────────────────────────────────
    "execution_failure_limit": [
        UserOption(
            display="RETRY_WITH_GUIDANCE",
            description="Reset counter and retry with guidance",
            action="retry_with_guidance",
            aliases=["RETRY", "TRY_AGAIN", "GUIDANCE"],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    "physics_failure_limit": [
        UserOption(
            display="RETRY_WITH_GUIDANCE",
            description="Reset counter and retry with guidance",
            action="retry_with_guidance",
            aliases=["RETRY", "TRY_AGAIN"],
        ),
        UserOption(
            display="ACCEPT_PARTIAL",
            description="Accept results despite physics issues",
            action="accept_partial",
            aliases=["ACCEPT", "PARTIAL"],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # ANALYSIS TRIGGERS
    # ───────────────────────────────────────────────────────────────────
    "analysis_limit": [
        UserOption(
            display="ACCEPT_PARTIAL",
            description="Mark stage as partial success and continue",
            action="accept_partial",
            aliases=["ACCEPT", "PARTIAL"],
        ),
        UserOption(
            display="PROVIDE_HINT",
            description="Reset counter and retry with guidance",
            action="provide_hint",
            aliases=["HINT"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # MATERIAL CHECKPOINT
    # ───────────────────────────────────────────────────────────────────
    "material_checkpoint": [
        UserOption(
            display="APPROVE",
            description="Accept materials and continue",
            action="approve",
            aliases=["YES", "OK", "ACCEPT", "PROCEED", "CORRECT", "VALID"],
        ),
        UserOption(
            display="CHANGE_DATABASE",
            description="Request a different material database",
            action="change_database",
            aliases=["DATABASE"],
        ),
        UserOption(
            display="CHANGE_MATERIAL",
            description="Request a different material",
            action="change_material",
            aliases=["MATERIAL"],
        ),
        UserOption(
            display="NEED_HELP",
            description="Get more guidance on materials",
            action="need_help",
            aliases=["HELP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # CONTEXT OVERFLOW
    # ───────────────────────────────────────────────────────────────────
    "context_overflow": [
        UserOption(
            display="SUMMARIZE",
            description="Apply feedback summarization for context management",
            action="summarize",
            aliases=[],
        ),
        UserOption(
            display="TRUNCATE",
            description="Truncate paper text to reduce context",
            action="truncate",
            aliases=[],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # BACKTRACK TRIGGERS
    # ───────────────────────────────────────────────────────────────────
    "backtrack_approval": [
        UserOption(
            display="APPROVE",
            description="Proceed with backtrack",
            action="approve",
            aliases=["YES", "OK", "ACCEPT"],
        ),
        UserOption(
            display="REJECT",
            description="Cancel backtrack",
            action="reject",
            aliases=["NO", "CANCEL"],
        ),
    ],
    
    "backtrack_limit": [
        UserOption(
            display="FORCE_CONTINUE",
            description="Ignore limit and continue",
            action="force_continue",
            aliases=["FORCE", "CONTINUE"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    "invalid_backtrack_decision": [
        UserOption(
            display="CONTINUE",
            description="Continue normally (ignoring invalid backtrack)",
            action="continue",
            aliases=[],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # DEADLOCK
    # ───────────────────────────────────────────────────────────────────
    "deadlock_detected": [
        UserOption(
            display="GENERATE_REPORT",
            description="Generate report with current progress",
            action="generate_report",
            aliases=["REPORT"],
        ),
        UserOption(
            display="REPLAN",
            description="Request replanning",
            action="replan",
            aliases=[],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # ERROR RECOVERY TRIGGERS
    # ───────────────────────────────────────────────────────────────────
    "llm_error": [
        UserOption(
            display="RETRY",
            description="Retry the LLM call",
            action="retry",
            aliases=[],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    "supervisor_error": [
        UserOption(
            display="RETRY",
            description="Attempt to continue the workflow",
            action="retry",
            aliases=[],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    "missing_design": [
        UserOption(
            display="RETRY",
            description="Go back to design phase",
            action="retry",
            aliases=[],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip this stage and continue",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    "unknown_escalation": [
        UserOption(
            display="RETRY",
            description="Attempt to continue the workflow",
            action="retry",
            aliases=[],
        ),
        UserOption(
            display="SKIP_STAGE",
            description="Skip the current stage",
            action="skip",
            aliases=["SKIP"],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # CRITICAL ERRORS (shared handler: missing_paper_text, missing_stage_id, progress_init_failed)
    # ───────────────────────────────────────────────────────────────────
    "critical_error": [
        UserOption(
            display="RETRY",
            description="Retry after error",
            action="retry",
            aliases=[],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
    
    # ───────────────────────────────────────────────────────────────────
    # PLANNING ERRORS (shared handler: no_stages_available, invalid_backtrack_target, backtrack_target_not_found)
    # ───────────────────────────────────────────────────────────────────
    "planning_error": [
        UserOption(
            display="REPLAN",
            description="Request replanning",
            action="replan",
            aliases=[],
        ),
        UserOption(
            display="STOP",
            description="End the workflow",
            action="stop",
            aliases=["QUIT", "EXIT", "ABORT", "END"],
        ),
    ],
}

# Map specific triggers to their shared option sets
_TRIGGER_ALIASES = {
    # Critical errors use "critical_error" options
    "missing_paper_text": "critical_error",
    "missing_stage_id": "critical_error",
    "progress_init_failed": "critical_error",
    # Planning errors use "planning_error" options  
    "no_stages_available": "planning_error",
    "invalid_backtrack_target": "planning_error",
    "backtrack_target_not_found": "planning_error",
}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_options_for_trigger(trigger: str) -> List[UserOption]:
    """
    Get all options for a trigger type.
    
    Args:
        trigger: The trigger name (e.g., "replan_limit")
        
    Returns:
        List of UserOption objects, or empty list if trigger unknown
    """
    # Check for alias first
    resolved_trigger = _TRIGGER_ALIASES.get(trigger, trigger)
    return USER_OPTIONS.get(resolved_trigger, [])


def get_options_prompt(trigger: str) -> str:
    """
    Generate the options text to show to user.
    
    Args:
        trigger: The trigger name
        
    Returns:
        Formatted string like:
        "Options:
        - APPROVE_PLAN: Force-accept the current plan despite issues
        - GUIDANCE: Provide specific guidance for replanning
        - STOP: End the workflow"
    """
    options = get_options_for_trigger(trigger)
    if not options:
        return "Options: (none defined for this trigger)"
    
    lines = ["Options:"]
    for opt in options:
        lines.append(f"- {opt.display}: {opt.description}")
    return "\n".join(lines)


def get_clarification_message(trigger: str) -> str:
    """
    Get message when no option matched.
    
    Args:
        trigger: The trigger name
        
    Returns:
        Message like "Please clarify: APPROVE_PLAN, GUIDANCE, or STOP?"
    """
    options = get_options_for_trigger(trigger)
    if not options:
        return "No valid options defined for this prompt. Please contact support."
    
    displays = [opt.display for opt in options]
    if len(displays) == 1:
        return f"Please clarify: {displays[0]}?"
    elif len(displays) == 2:
        return f"Please clarify: {displays[0]} or {displays[1]}?"
    else:
        return f"Please clarify: {', '.join(displays[:-1])}, or {displays[-1]}?"


def validate_no_collisions() -> None:
    """
    Validate no keyword collisions within each trigger.
    
    Should be called at startup to catch configuration errors early.
    
    Raises:
        ValueError: If any trigger has keyword collisions
    """
    for trigger, options in USER_OPTIONS.items():
        seen: Dict[str, str] = {}
        for opt in options:
            for kw in opt.keywords:
                if kw in seen:
                    raise ValueError(
                        f"Keyword collision in trigger '{trigger}': "
                        f"'{kw}' used by both '{seen[kw]}' and '{opt.display}'"
                    )
                seen[kw] = opt.display
    
    logger.debug(f"Validated {len(USER_OPTIONS)} triggers with no keyword collisions")


# ═══════════════════════════════════════════════════════════════════════
# MATCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def match_option_by_keywords(trigger: str, response: str) -> Optional[UserOption]:
    """
    Match user response to an option using keyword matching.
    
    Args:
        trigger: The trigger name
        response: User's response text
        
    Returns:
        Matched UserOption, or None if no match
    """
    from src.agents.base import check_keywords
    
    options = get_options_for_trigger(trigger)
    
    for opt in options:
        if check_keywords(response, opt.keywords):
            return opt
    
    return None


def classify_with_local_llm(trigger: str, response: str) -> Optional[UserOption]:
    """
    Use local LLM to classify user intent.
    
    Only called if REPROLAB_USE_LOCAL_LLM=1 environment variable is set.
    
    Args:
        trigger: The trigger name
        response: User's response text
        
    Returns:
        Matched UserOption, or None if unavailable/no match
    """
    # Check if local LLM is enabled
    if os.environ.get("REPROLAB_USE_LOCAL_LLM", "0") != "1":
        return None
    
    options = get_options_for_trigger(trigger)
    if not options:
        return None
    
    try:
        import ollama
    except ImportError:
        logger.debug("ollama not installed, skipping LLM classification")
        return None
    
    # Build prompt
    options_text = "\n".join(
        f"- {opt.display}: {opt.description}" for opt in options
    )
    
    prompt = f"""Classify the user's response into one of these options.

Available options:
{options_text}

User's response: "{response}"

Reply with ONLY the option name (e.g., "{options[0].display}") or "UNKNOWN" if the response doesn't clearly match any option.
Do not explain, just output the option name."""

    try:
        # Use a small, fast model
        model = os.environ.get("REPROLAB_LOCAL_LLM_MODEL", "phi3:mini")
        result = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}  # Deterministic
        )
        
        answer = result["message"]["content"].strip().upper()
        logger.debug(f"Local LLM classified '{response}' as '{answer}'")
        
        # Find matching option
        for opt in options:
            if opt.display.upper() == answer:
                return opt
        
        return None
        
    except Exception as e:
        logger.warning(f"Local LLM classification failed: {e}")
        return None


def match_user_response(trigger: str, response: str) -> Optional[UserOption]:
    """
    Match user response to an option using hybrid approach.
    
    Strategy:
    1. Try keyword matching first (fast, deterministic)
    2. If no match and local LLM enabled, try LLM classification
    3. Return None if no match found
    
    Args:
        trigger: The trigger name
        response: User's response text
        
    Returns:
        Matched UserOption, or None if no match
    """
    # 1. Keyword matching (fast path)
    option = match_option_by_keywords(trigger, response)
    if option:
        logger.debug(f"Matched '{response[:50]}...' to '{option.display}' via keywords")
        return option
    
    # 2. LLM fallback (slow path, optional)
    option = classify_with_local_llm(trigger, response)
    if option:
        logger.info(f"Matched '{response[:50]}...' to '{option.display}' via local LLM")
        return option
    
    # 3. No match
    logger.debug(f"No match for '{response[:50]}...' in trigger '{trigger}'")
    return None


# ═══════════════════════════════════════════════════════════════════════
# UTILITY FOR EXTRACTING GUIDANCE TEXT
# ═══════════════════════════════════════════════════════════════════════

def extract_guidance_text(response: str, keywords: Optional[List[str]] = None) -> str:
    """
    Extract guidance text from user response, stripping keyword prefixes.
    
    Handles responses like:
    - "GUIDANCE: please focus on the resonance peak"
    - "HINT please focus on the resonance peak"
    - "RETRY_WITH_GUIDANCE: check the boundary conditions"
    
    Args:
        response: User's response text (can also be a list, which gets converted)
        keywords: Keywords to strip (defaults to common guidance keywords)
        
    Returns:
        Cleaned guidance text
    """
    if keywords is None:
        keywords = ["GUIDANCE", "HINT", "PROVIDE_HINT", "RETRY_WITH_GUIDANCE"]
    
    # Handle non-string inputs gracefully
    if isinstance(response, list):
        response = " ".join(str(item) for item in response if item)
    elif not isinstance(response, str):
        response = str(response) if response else ""
    
    text = response.strip()
    text_upper = text.upper()
    
    for kw in keywords:
        kw_upper = kw.upper()
        if text_upper.startswith(kw_upper):
            # Remove keyword
            remainder = text[len(kw):]
            # Remove optional colon and whitespace
            if remainder.startswith(":"):
                remainder = remainder[1:]
            return remainder.strip()
    
    return text

