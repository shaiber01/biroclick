"""
ASK_USER Trigger Definitions and Documentation

This module documents all valid ask_user_trigger values and their expected
handling by the SupervisorAgent. It centralizes the documentation for the
ask_user contract in the ReproLab workflow.

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════

When a node needs to ask the user for input, it should:
1. Set state["ask_user_trigger"] to one of the documented trigger values
2. Set state["last_node_before_ask_user"] to identify the triggering node
3. Set state["pending_user_questions"] with the questions to ask

The Supervisor will use these fields to:
1. Understand what the user was responding to
2. Route appropriately based on the user's answer

See: supervisor_agent.md and agents.py:supervisor_node
"""

from typing import Dict, List, Optional, Any


# ═══════════════════════════════════════════════════════════════════════
# Trigger Definitions
# ═══════════════════════════════════════════════════════════════════════

ASK_USER_TRIGGERS: Dict[str, Dict[str, Any]] = {
    "material_checkpoint": {
        "description": "Mandatory Stage 0 material validation requires user confirmation",
        "source_node": "material_checkpoint",
        "expected_response_keys": ["verdict", "notes"],
        "valid_verdicts": ["APPROVE", "CHANGE_DATABASE", "CHANGE_MATERIAL", "NEED_HELP", "STOP"],
        "supervisor_action": {
            "APPROVE": "Set supervisor_verdict='ok_continue', proceed to select_stage",
            "CHANGE_DATABASE": "Invalidate Stage 0, update assumptions, rerun Stage 0",
            "CHANGE_MATERIAL": "Route to plan with supervisor_feedback about material change",
            "NEED_HELP": "Route back to ask_user with additional context",
            "STOP": "Route to generate_report, abort workflow",
        }
    },
    "code_review_limit": {
        "description": "Code review revision limit (MAX_CODE_REVISIONS) exceeded",
        "source_node": "code_review",
        "expected_response_keys": ["action", "hint"],
        "valid_verdicts": ["PROVIDE_HINT", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "PROVIDE_HINT": "Reset code_revision_count=0, add hint to reviewer_feedback, route to generate_code",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "design_review_limit": {
        "description": "Design review revision limit (MAX_DESIGN_REVISIONS) exceeded",
        "source_node": "design_review",
        "expected_response_keys": ["action", "hint"],
        "valid_verdicts": ["PROVIDE_HINT", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "PROVIDE_HINT": "Reset design_revision_count=0, add hint to reviewer_feedback, route to design",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "design_flaw_limit": {
        "description": "Design flaw detected by physics_check, design revision limit (MAX_DESIGN_REVISIONS) exceeded",
        "source_node": "physics_check",
        "expected_response_keys": ["action", "hint"],
        "valid_verdicts": ["PROVIDE_HINT", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "PROVIDE_HINT": "Reset design_revision_count=0, add hint to design_feedback, route to design",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "execution_failure_limit": {
        "description": "Execution failure limit (MAX_EXECUTION_FAILURES) exceeded",
        "source_node": "execution_check",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["RETRY_WITH_GUIDANCE", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY_WITH_GUIDANCE": "Reset execution_failure_count=0, add guidance to execution_feedback, route to generate_code",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "physics_failure_limit": {
        "description": "Physics sanity check failure limit (MAX_PHYSICS_FAILURES) exceeded",
        "source_node": "physics_check",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["RETRY_WITH_GUIDANCE", "ACCEPT_PARTIAL", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY_WITH_GUIDANCE": "Reset physics_failure_count=0, add guidance, route to generate_code or design",
            "ACCEPT_PARTIAL": "Mark stage completed_partial, proceed to analyze",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "analysis_limit": {
        "description": "Analysis revision limit (MAX_ANALYSIS_REVISIONS) exceeded",
        "source_node": "comparison_check",
        "expected_response_keys": ["action", "hint"],
        "valid_verdicts": ["ACCEPT_PARTIAL", "PROVIDE_HINT", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "ACCEPT_PARTIAL": "Mark stage completed_partial, proceed to select_stage",
            "PROVIDE_HINT": "Reset analysis_revision_count=0, add hint to analysis_feedback, route to analyze",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "context_overflow": {
        "description": "LLM context window would overflow, recovery options needed",
        "source_node": "any (detected by check_context_before_node)",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["SUMMARIZE_FEEDBACK", "TRUNCATE_PAPER", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "SUMMARIZE_FEEDBACK": "Apply summarize_feedback recovery action",
            "TRUNCATE_PAPER": "Apply truncate_paper_to_methods recovery action",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "replan_limit": {
        "description": "Replan limit (MAX_REPLANS) exceeded",
        "source_node": "plan_review",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["FORCE_ACCEPT", "PROVIDE_GUIDANCE", "STOP"],
        "supervisor_action": {
            "FORCE_ACCEPT": "Accept plan as-is, route to select_stage",
            "PROVIDE_GUIDANCE": "Reset replan_count=0, add guidance, route to plan",
            "STOP": "Route to generate_report",
        }
    },
    "backtrack_approval": {
        "description": "Backtrack suggestion requires user confirmation",
        "source_node": "supervisor",
        "expected_response_keys": ["approve", "alternative"],
        "valid_verdicts": ["APPROVE_BACKTRACK", "REJECT_BACKTRACK", "ALTERNATIVE"],
        "supervisor_action": {
            "APPROVE_BACKTRACK": "Route to handle_backtrack",
            "REJECT_BACKTRACK": "Clear backtrack_suggestion, continue normally",
            "ALTERNATIVE": "Apply user's alternative suggestion",
        }
    },
    "deadlock_detected": {
        "description": "Workflow deadlock detected (no progress)",
        "source_node": "stage_selection",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["GENERATE_REPORT", "REPLAN", "STOP"],
        "supervisor_action": {
            "GENERATE_REPORT": "Route to generate_report",
            "REPLAN": "Route to plan",
            "STOP": "Stop workflow",
        }
    },
    "llm_error": {
        "description": "LLM API error persisted after retries",
        "source_node": "any",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY": "Retry the operation",
            "SKIP_STAGE": "Mark stage blocked, route to select_stage",
            "STOP": "Stop workflow",
        }
    },
    "clarification": {
        "description": "Ambiguous paper information requires user clarification",
        "source_node": "any planning/design agent",
        "expected_response_keys": ["clarification"],
        "valid_verdicts": None,  # Free-form response
        "supervisor_action": "Add clarification to assumptions, continue from last_node_before_ask_user",
    },
    "missing_paper_text": {
        "description": "Critical error: Paper text is missing from state",
        "source_node": "planning/design",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "STOP"],
        "supervisor_action": {
            "RETRY": "Route to select_stage (attempt to restart)",
            "STOP": "Stop workflow",
        }
    },
    "missing_stage_id": {
        "description": "Critical error: Stage ID is missing in a context where it is required",
        "source_node": "various",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "STOP"],
        "supervisor_action": {
            "RETRY": "Route to select_stage (attempt to restart)",
            "STOP": "Stop workflow",
        }
    },
    "no_stages_available": {
        "description": "Critical error: No stages available in plan",
        "source_node": "stage_selection",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["REPLAN", "STOP"],
        "supervisor_action": {
            "REPLAN": "Route to plan to create new stages",
            "STOP": "Stop workflow",
        }
    },
    "progress_init_failed": {
        "description": "Critical error: Failed to initialize progress tracking",
        "source_node": "stage_selection",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "STOP"],
        "supervisor_action": {
            "RETRY": "Route to select_stage (attempt to restart)",
            "STOP": "Stop workflow",
        }
    },
    "backtrack_limit": {
        "description": "Backtrack limit (MAX_BACKTRACKS) exceeded",
        "source_node": "reporting/supervisor",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["STOP", "FORCE_CONTINUE"],
        "supervisor_action": {
            "STOP": "Stop workflow",
            "FORCE_CONTINUE": "Route to select_stage (ignore limit)",
        }
    },
    "invalid_backtrack_target": {
        "description": "Critical error: Backtrack target stage not found in plan",
        "source_node": "reporting",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["STOP", "REPLAN"],
        "supervisor_action": {
            "STOP": "Stop workflow",
            "REPLAN": "Route to plan to fix dependencies",
        }
    },
    "backtrack_target_not_found": {
        "description": "Critical error: Backtrack target stage ID does not exist",
        "source_node": "reporting",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["STOP", "REPLAN"],
        "supervisor_action": {
            "STOP": "Stop workflow",
            "REPLAN": "Route to plan",
        }
    },
    "invalid_backtrack_decision": {
        "description": "Critical error: Backtrack decision structure is invalid",
        "source_node": "reporting",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["STOP", "CONTINUE"],
        "supervisor_action": {
            "STOP": "Stop workflow",
            "CONTINUE": "Route to select_stage (ignore invalid decision)",
        }
    },
    "supervisor_error": {
        "description": "Supervisor node failed to produce a valid verdict",
        "source_node": "supervisor",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "STOP"],
        "supervisor_action": {
            "RETRY": "Retry the supervisor operation",
            "STOP": "Stop workflow",
        }
    },
    "missing_design": {
        "description": "Code generation attempted without required design description",
        "source_node": "code_generator",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY": "Route back to design phase",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Stop workflow",
        }
    },
    "reviewer_escalation": {
        "description": "Reviewer LLM explicitly requested user input via escalate_to_user field",
        "source_node": "code_review/design_review/plan_review",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["PROVIDE_GUIDANCE", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "PROVIDE_GUIDANCE": "Add guidance to reviewer_feedback, continue review process",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Stop workflow",
        }
    },
    "unknown_escalation": {
        "description": "Generic fallback for unexpected workflow errors",
        "source_node": "any (safety net)",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["RETRY", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY": "Attempt to continue the workflow",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Stop workflow",
        }
    },
    "unknown": {
        "description": "Unknown trigger - fallback handling",
        "source_node": "unknown",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["CONTINUE", "STOP"],
        "supervisor_action": {
            "CONTINUE": "Route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def get_ask_user_trigger_info(trigger: str) -> Dict[str, Any]:
    """
    Get documentation for an ask_user trigger.
    
    Args:
        trigger: The ask_user_trigger value
        
    Returns:
        Dict with trigger documentation. Returns "unknown" trigger info
        if the trigger is not recognized.
        
    Example:
        >>> info = get_ask_user_trigger_info("material_checkpoint")
        >>> print(info["description"])
        "Mandatory Stage 0 material validation requires user confirmation"
    """
    return ASK_USER_TRIGGERS.get(trigger, ASK_USER_TRIGGERS["unknown"])


def get_valid_triggers() -> List[str]:
    """
    Get list of all valid trigger names.
    
    Returns:
        List of trigger name strings
    """
    return list(ASK_USER_TRIGGERS.keys())


def get_valid_verdicts_for_trigger(trigger: str) -> Optional[List[str]]:
    """
    Get the valid verdicts for a specific trigger.
    
    Args:
        trigger: The ask_user_trigger value
        
    Returns:
        List of valid verdict strings, or None if free-form response expected
    """
    info = get_ask_user_trigger_info(trigger)
    return info.get("valid_verdicts")
