"""
Constants and enums for agent nodes.

Using string enums allows both type safety and JSON serialization compatibility.
"""

from enum import Enum


class ReviewVerdict(str, Enum):
    """Verdict values returned by reviewer nodes."""
    APPROVE = "approve"
    NEEDS_REVISION = "needs_revision"


class SupervisorVerdict(str, Enum):
    """Verdict values returned by supervisor node."""
    OK_CONTINUE = "ok_continue"
    BACKTRACK = "backtrack"
    BACKTRACK_TO_STAGE = "backtrack_to_stage"
    FINISH = "finish"
    MATERIAL_CHECKPOINT = "material_checkpoint"
    REPLAN = "replan"


class ExecutionVerdict(str, Enum):
    """Verdict values for execution validation."""
    PASS = "pass"
    FAIL = "fail"


class PhysicsVerdict(str, Enum):
    """Verdict values for physics sanity checks."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    DESIGN_FLAW = "design_flaw"


class AnalysisClassification(str, Enum):
    """Classification values for analysis results."""
    EXCELLENT_MATCH = "EXCELLENT_MATCH"
    ACCEPTABLE_MATCH = "ACCEPTABLE_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    POOR_MATCH = "POOR_MATCH"
    FAILED = "FAILED"
    NO_TARGETS = "NO_TARGETS"
    PENDING_VALIDATION = "pending_validation"
    MATCH = "match"
    MISMATCH = "mismatch"


class WorkflowPhase(str, Enum):
    """Workflow phase values set by nodes."""
    ADAPTING_PROMPTS = "adapting_prompts"
    PLANNING = "planning"
    PLAN_REVIEW = "plan_review"
    STAGE_SELECTION = "stage_selection"
    DESIGN = "design"
    DESIGN_REVIEW = "design_review"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    EXECUTION = "execution"
    EXECUTION_VALIDATION = "execution_validation"
    PHYSICS_VALIDATION = "physics_validation"
    ANALYSIS = "analysis"
    COMPARISON_VALIDATION = "comparison_validation"
    SUPERVISION = "supervision"
    MATERIAL_CHECKPOINT = "material_checkpoint"
    AWAITING_USER = "awaiting_user"
    BACKTRACKING = "backtracking"
    BACKTRACKING_LIMIT = "backtracking_limit"
    REPORTING = "reporting"


class AskUserTrigger(str, Enum):
    """Trigger types for user interaction requests."""
    MATERIAL_CHECKPOINT = "material_checkpoint"
    CODE_REVIEW_LIMIT = "code_review_limit"
    DESIGN_REVIEW_LIMIT = "design_review_limit"
    EXECUTION_FAILURE_LIMIT = "execution_failure_limit"
    PHYSICS_FAILURE_LIMIT = "physics_failure_limit"
    BACKTRACK_APPROVAL = "backtrack_approval"
    BACKTRACK_LIMIT = "backtrack_limit"
    REPLAN_LIMIT = "replan_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    LLM_ERROR = "llm_error"
    MISSING_PAPER_TEXT = "missing_paper_text"
    MISSING_STAGE_ID = "missing_stage_id"
    MISSING_DESIGN = "missing_design"
    NO_STAGES_AVAILABLE = "no_stages_available"
    DEADLOCK_DETECTED = "deadlock_detected"
    INVALID_BACKTRACK_DECISION = "invalid_backtrack_decision"
    INVALID_BACKTRACK_TARGET = "invalid_backtrack_target"
    BACKTRACK_TARGET_NOT_FOUND = "backtrack_target_not_found"
    PROGRESS_INIT_FAILED = "progress_init_failed"


class StageStatus(str, Enum):
    """Status values for progress stages."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    NEEDS_RERUN = "needs_rerun"
    BLOCKED = "blocked"
    INVALIDATED = "invalidated"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_PARTIAL = "completed_partial"
    COMPLETED_FAILED = "completed_failed"


class StageType(str, Enum):
    """Stage type values from plan."""
    MATERIAL_VALIDATION = "MATERIAL_VALIDATION"
    SINGLE_STRUCTURE = "SINGLE_STRUCTURE"
    ARRAY_SYSTEM = "ARRAY_SYSTEM"
    PARAMETER_SWEEP = "PARAMETER_SWEEP"
    COMPLEX_PHYSICS = "COMPLEX_PHYSICS"


