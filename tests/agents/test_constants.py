"""Unit tests for src/agents/constants.py"""

import pytest

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

    def test_approve_value(self):
        """Should have correct approve value."""
        assert ReviewVerdict.APPROVE == "approve"
        assert ReviewVerdict.APPROVE.value == "approve"

    def test_needs_revision_value(self):
        """Should have correct needs_revision value."""
        assert ReviewVerdict.NEEDS_REVISION == "needs_revision"

    def test_string_comparison(self):
        """Should be comparable to strings."""
        assert ReviewVerdict.APPROVE == "approve"
        assert "approve" == ReviewVerdict.APPROVE

    def test_json_serializable(self):
        """Should be JSON serializable as string."""
        import json
        data = {"verdict": ReviewVerdict.APPROVE}
        serialized = json.dumps(data)
        assert '"approve"' in serialized


class TestSupervisorVerdict:
    """Tests for SupervisorVerdict enum."""

    def test_all_verdicts_defined(self):
        """Should have all expected verdicts."""
        assert SupervisorVerdict.OK_CONTINUE == "ok_continue"
        assert SupervisorVerdict.BACKTRACK == "backtrack"
        assert SupervisorVerdict.FINISH == "finish"
        assert SupervisorVerdict.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert SupervisorVerdict.REPLAN == "replan"


class TestExecutionVerdict:
    """Tests for ExecutionVerdict enum."""

    def test_pass_fail_values(self):
        """Should have pass and fail values."""
        assert ExecutionVerdict.PASS == "pass"
        assert ExecutionVerdict.FAIL == "fail"


class TestPhysicsVerdict:
    """Tests for PhysicsVerdict enum."""

    def test_all_verdicts_defined(self):
        """Should have all expected verdicts."""
        assert PhysicsVerdict.PASS == "pass"
        assert PhysicsVerdict.WARNING == "warning"
        assert PhysicsVerdict.FAIL == "fail"
        assert PhysicsVerdict.DESIGN_FLAW == "design_flaw"


class TestAnalysisClassification:
    """Tests for AnalysisClassification enum."""

    def test_match_classifications(self):
        """Should have match classification values."""
        assert AnalysisClassification.EXCELLENT_MATCH == "EXCELLENT_MATCH"
        assert AnalysisClassification.ACCEPTABLE_MATCH == "ACCEPTABLE_MATCH"
        assert AnalysisClassification.PARTIAL_MATCH == "PARTIAL_MATCH"
        assert AnalysisClassification.MATCH == "match"
        assert AnalysisClassification.MISMATCH == "mismatch"


class TestWorkflowPhase:
    """Tests for WorkflowPhase enum."""

    def test_planning_phases(self):
        """Should have planning-related phases."""
        assert WorkflowPhase.PLANNING == "planning"
        assert WorkflowPhase.PLAN_REVIEW == "plan_review"

    def test_execution_phases(self):
        """Should have execution-related phases."""
        assert WorkflowPhase.EXECUTION == "execution"
        assert WorkflowPhase.EXECUTION_VALIDATION == "execution_validation"
        assert WorkflowPhase.PHYSICS_VALIDATION == "physics_validation"

    def test_all_phases_are_strings(self):
        """All phases should be string values."""
        for phase in WorkflowPhase:
            assert isinstance(phase.value, str)


class TestAskUserTrigger:
    """Tests for AskUserTrigger enum."""

    def test_checkpoint_triggers(self):
        """Should have checkpoint triggers."""
        assert AskUserTrigger.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert AskUserTrigger.BACKTRACK_APPROVAL == "backtrack_approval"

    def test_limit_triggers(self):
        """Should have limit-related triggers."""
        assert AskUserTrigger.CODE_REVIEW_LIMIT == "code_review_limit"
        assert AskUserTrigger.DESIGN_REVIEW_LIMIT == "design_review_limit"
        assert AskUserTrigger.EXECUTION_FAILURE_LIMIT == "execution_failure_limit"

    def test_error_triggers(self):
        """Should have error-related triggers."""
        assert AskUserTrigger.LLM_ERROR == "llm_error"
        assert AskUserTrigger.MISSING_PAPER_TEXT == "missing_paper_text"


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_initial_statuses(self):
        """Should have initial status values."""
        assert StageStatus.NOT_STARTED == "not_started"
        assert StageStatus.IN_PROGRESS == "in_progress"

    def test_completion_statuses(self):
        """Should have completion status values."""
        assert StageStatus.COMPLETED_SUCCESS == "completed_success"
        assert StageStatus.COMPLETED_PARTIAL == "completed_partial"
        assert StageStatus.COMPLETED_FAILED == "completed_failed"

    def test_special_statuses(self):
        """Should have special status values."""
        assert StageStatus.NEEDS_RERUN == "needs_rerun"
        assert StageStatus.BLOCKED == "blocked"
        assert StageStatus.INVALIDATED == "invalidated"


class TestStageType:
    """Tests for StageType enum."""

    def test_all_stage_types_defined(self):
        """Should have all expected stage types."""
        assert StageType.MATERIAL_VALIDATION == "MATERIAL_VALIDATION"
        assert StageType.SINGLE_STRUCTURE == "SINGLE_STRUCTURE"
        assert StageType.ARRAY_SYSTEM == "ARRAY_SYSTEM"
        assert StageType.PARAMETER_SWEEP == "PARAMETER_SWEEP"
        assert StageType.COMPLEX_PHYSICS == "COMPLEX_PHYSICS"


class TestEnumImports:
    """Tests for enum imports from package."""

    def test_imports_from_agents_package(self):
        """Should be importable from src.agents."""
        from src.agents import (
            ReviewVerdict,
            SupervisorVerdict,
            WorkflowPhase,
        )
        assert ReviewVerdict.APPROVE == "approve"
        assert SupervisorVerdict.OK_CONTINUE == "ok_continue"
        assert WorkflowPhase.PLANNING == "planning"

