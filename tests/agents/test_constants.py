"""Unit tests for src/agents/constants.py"""

import pytest
import json
from src.agents.constants import (
    ReviewVerdict,
    SupervisorVerdict,
    ExecutionVerdict,
    PhysicsVerdict,
    AnalysisClassification,
    WorkflowPhase,
    AskUserTrigger,
    StageStatus,
    StageType,
)


class TestReviewVerdict:
    """Tests for ReviewVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert ReviewVerdict.APPROVE == "approve"
        assert ReviewVerdict.NEEDS_REVISION == "needs_revision"


class TestSupervisorVerdict:
    """Tests for SupervisorVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert SupervisorVerdict.OK_CONTINUE == "ok_continue"
        assert SupervisorVerdict.BACKTRACK == "backtrack"
        assert SupervisorVerdict.BACKTRACK_TO_STAGE == "backtrack_to_stage"
        assert SupervisorVerdict.FINISH == "finish"
        assert SupervisorVerdict.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert SupervisorVerdict.REPLAN == "replan"


class TestExecutionVerdict:
    """Tests for ExecutionVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert ExecutionVerdict.PASS == "pass"
        assert ExecutionVerdict.FAIL == "fail"


class TestPhysicsVerdict:
    """Tests for PhysicsVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert PhysicsVerdict.PASS == "pass"
        assert PhysicsVerdict.WARNING == "warning"
        assert PhysicsVerdict.FAIL == "fail"
        assert PhysicsVerdict.DESIGN_FLAW == "design_flaw"


class TestAnalysisClassification:
    """Tests for AnalysisClassification enum."""

    def test_values(self):
        """Should have correct values.
        
        Note: Enforcing uppercase consistency for all values.
        Current implementation has mixed case which is considered a bug/inconsistency.
        """
        assert AnalysisClassification.EXCELLENT_MATCH == "EXCELLENT_MATCH"
        assert AnalysisClassification.ACCEPTABLE_MATCH == "ACCEPTABLE_MATCH"
        assert AnalysisClassification.PARTIAL_MATCH == "PARTIAL_MATCH"
        assert AnalysisClassification.POOR_MATCH == "POOR_MATCH"
        assert AnalysisClassification.FAILED == "FAILED"
        assert AnalysisClassification.NO_TARGETS == "NO_TARGETS"
        
        # Failing assertions expected until component is fixed to use UPPERCASE
        assert AnalysisClassification.PENDING_VALIDATION == "PENDING_VALIDATION"
        assert AnalysisClassification.MATCH == "MATCH"
        assert AnalysisClassification.MISMATCH == "MISMATCH"


class TestWorkflowPhase:
    """Tests for WorkflowPhase enum."""

    def test_values(self):
        """Should have correct values."""
        assert WorkflowPhase.ADAPTING_PROMPTS == "adapting_prompts"
        assert WorkflowPhase.PLANNING == "planning"
        assert WorkflowPhase.PLAN_REVIEW == "plan_review"
        assert WorkflowPhase.STAGE_SELECTION == "stage_selection"
        assert WorkflowPhase.DESIGN == "design"
        assert WorkflowPhase.DESIGN_REVIEW == "design_review"
        assert WorkflowPhase.CODE_GENERATION == "code_generation"
        assert WorkflowPhase.CODE_REVIEW == "code_review"
        assert WorkflowPhase.EXECUTION == "execution"
        assert WorkflowPhase.EXECUTION_VALIDATION == "execution_validation"
        assert WorkflowPhase.PHYSICS_VALIDATION == "physics_validation"
        assert WorkflowPhase.ANALYSIS == "analysis"
        assert WorkflowPhase.COMPARISON_VALIDATION == "comparison_validation"
        assert WorkflowPhase.SUPERVISION == "supervision"
        assert WorkflowPhase.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert WorkflowPhase.AWAITING_USER == "awaiting_user"
        assert WorkflowPhase.BACKTRACKING == "backtracking"
        assert WorkflowPhase.BACKTRACKING_LIMIT == "backtracking_limit"
        assert WorkflowPhase.REPORTING == "reporting"


class TestAskUserTrigger:
    """Tests for AskUserTrigger enum."""

    def test_values(self):
        """Should have correct values."""
        assert AskUserTrigger.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert AskUserTrigger.CODE_REVIEW_LIMIT == "code_review_limit"
        assert AskUserTrigger.DESIGN_REVIEW_LIMIT == "design_review_limit"
        assert AskUserTrigger.EXECUTION_FAILURE_LIMIT == "execution_failure_limit"
        assert AskUserTrigger.PHYSICS_FAILURE_LIMIT == "physics_failure_limit"
        assert AskUserTrigger.BACKTRACK_APPROVAL == "backtrack_approval"
        assert AskUserTrigger.BACKTRACK_LIMIT == "backtrack_limit"
        assert AskUserTrigger.REPLAN_LIMIT == "replan_limit"
        assert AskUserTrigger.CONTEXT_OVERFLOW == "context_overflow"
        assert AskUserTrigger.LLM_ERROR == "llm_error"
        assert AskUserTrigger.MISSING_PAPER_TEXT == "missing_paper_text"
        assert AskUserTrigger.MISSING_STAGE_ID == "missing_stage_id"
        assert AskUserTrigger.MISSING_DESIGN == "missing_design"
        assert AskUserTrigger.NO_STAGES_AVAILABLE == "no_stages_available"
        assert AskUserTrigger.DEADLOCK_DETECTED == "deadlock_detected"
        assert AskUserTrigger.INVALID_BACKTRACK_DECISION == "invalid_backtrack_decision"
        assert AskUserTrigger.INVALID_BACKTRACK_TARGET == "invalid_backtrack_target"
        assert AskUserTrigger.BACKTRACK_TARGET_NOT_FOUND == "backtrack_target_not_found"
        assert AskUserTrigger.PROGRESS_INIT_FAILED == "progress_init_failed"


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_values(self):
        """Should have correct values."""
        assert StageStatus.NOT_STARTED == "not_started"
        assert StageStatus.IN_PROGRESS == "in_progress"
        assert StageStatus.NEEDS_RERUN == "needs_rerun"
        assert StageStatus.BLOCKED == "blocked"
        assert StageStatus.INVALIDATED == "invalidated"
        assert StageStatus.COMPLETED_SUCCESS == "completed_success"
        assert StageStatus.COMPLETED_PARTIAL == "completed_partial"
        assert StageStatus.COMPLETED_FAILED == "completed_failed"


class TestStageType:
    """Tests for StageType enum."""

    def test_values(self):
        """Should have correct values."""
        assert StageType.MATERIAL_VALIDATION == "MATERIAL_VALIDATION"
        assert StageType.SINGLE_STRUCTURE == "SINGLE_STRUCTURE"
        assert StageType.ARRAY_SYSTEM == "ARRAY_SYSTEM"
        assert StageType.PARAMETER_SWEEP == "PARAMETER_SWEEP"
        assert StageType.COMPLEX_PHYSICS == "COMPLEX_PHYSICS"


class TestGeneralEnumBehavior:
    """Tests for general behavior common to all enums."""

    ENUM_CLASSES = [
        ReviewVerdict,
        SupervisorVerdict,
        ExecutionVerdict,
        PhysicsVerdict,
        AnalysisClassification,
        WorkflowPhase,
        AskUserTrigger,
        StageStatus,
        StageType,
    ]

    def test_string_comparison(self):
        """Should be comparable to strings."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert member == member.value
                assert member.value == member

    def test_json_serializable(self):
        """Should be JSON serializable as string."""
        for enum_cls in self.ENUM_CLASSES:
            member = list(enum_cls)[0]
            data = {"value": member}
            serialized = json.dumps(data)
            assert f'"{member.value}"' in serialized

    def test_instantiation_from_value(self):
        """Should be instantiable from valid string value."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert enum_cls(member.value) == member

    def test_invalid_value_raises_error(self):
        """Should raise ValueError for invalid values."""
        for enum_cls in self.ENUM_CLASSES:
            with pytest.raises(ValueError):
                enum_cls("this_value_definitely_does_not_exist_12345")

    def test_no_duplicate_values(self):
        """Ensure no enum class has duplicate values."""
        for enum_cls in self.ENUM_CLASSES:
            values = [e.value for e in enum_cls]
            assert len(values) == len(set(values)), f"{enum_cls.__name__} has duplicate values"

    def test_inheritance(self):
        """Ensure all enums inherit from str and Enum."""
        from enum import Enum
        for enum_cls in self.ENUM_CLASSES:
            assert issubclass(enum_cls, str)
            assert issubclass(enum_cls, Enum)
