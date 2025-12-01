"""
ReproLab Schemas Package

This package contains:
- JSON schemas defining data structures for agents and state
- Python TypedDict definitions in state.py
- Auto-generated types in generated_types.py

JSON schemas are the source of truth for:
- Agent output formats (used with function calling)
- Plan, progress, and report structures
- Assumptions and metrics tracking

Key exports:
- ReproState: Main workflow state TypedDict
- create_initial_state: Factory function for new state
- Validation and helper functions
"""

from .state import (
    ReproState,
    create_initial_state,
    get_validation_hierarchy,
    update_progress_stage_status,
    archive_stage_outputs_to_progress,
    validate_state_for_node,
    check_context_before_node,
    estimate_context_for_node,
    # Constants
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
    CONTEXT_WINDOW_LIMITS,
)

__all__ = [
    # Core types
    "ReproState",
    "create_initial_state",
    # Validation helpers
    "get_validation_hierarchy",
    "update_progress_stage_status",
    "archive_stage_outputs_to_progress",
    "validate_state_for_node",
    # Context management
    "check_context_before_node",
    "estimate_context_for_node",
    # Constants
    "MAX_DESIGN_REVISIONS",
    "MAX_CODE_REVISIONS",
    "MAX_EXECUTION_FAILURES",
    "MAX_PHYSICS_FAILURES",
    "MAX_ANALYSIS_REVISIONS",
    "MAX_REPLANS",
    "CONTEXT_WINDOW_LIMITS",
]

