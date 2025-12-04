"""
Supervision package: supervisor_node and trigger handlers.

This package contains the supervisor agent logic, split into:
- supervisor_node: Main supervisor decision-making
- trigger_handlers: Individual handlers for each ask_user_trigger type
"""

from .supervisor import supervisor_node, _get_dependent_stages
from .trigger_handlers import (
    handle_trigger,
    handle_material_checkpoint,
    handle_code_review_limit,
    handle_design_review_limit,
    handle_execution_failure_limit,
    handle_physics_failure_limit,
    handle_analysis_limit,
    handle_context_overflow,
    handle_replan_limit,
    handle_backtrack_approval,
    handle_deadlock_detected,
    handle_llm_error,
    handle_supervisor_error,
    handle_missing_design,
    handle_unknown_escalation,
)

__all__ = [
    "supervisor_node",
    "_get_dependent_stages",
    "handle_trigger",
    "handle_material_checkpoint",
    "handle_code_review_limit",
    "handle_design_review_limit",
    "handle_execution_failure_limit",
    "handle_physics_failure_limit",
    "handle_analysis_limit",
    "handle_context_overflow",
    "handle_replan_limit",
    "handle_backtrack_approval",
    "handle_deadlock_detected",
    "handle_llm_error",
    "handle_supervisor_error",
    "handle_missing_design",
    "handle_unknown_escalation",
]

