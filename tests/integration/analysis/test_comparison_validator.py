"""Integration tests for comparison_validator_node.

These tests are designed to FIND BUGS, not just pass.
Every test verifies specific behavior and would fail if bugs exist.
"""

import pytest
from src.agents.constants import AnalysisClassification


class TestComparisonValidatorEarlyReturns:
    """Test early return conditions."""

    def test_early_return_when_awaiting_user_input(self, base_state):
        """Should return empty dict when awaiting_user_input is True (no state updates)."""
        from src.agents.analysis import comparison_validator_node

        base_state["awaiting_user_input"] = True
        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "targets": ["material_gold"]}]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        # Should return empty dict (no state updates) when awaiting user input
        assert result == {}
        # State should remain unchanged (awaiting_user_input still True)
        assert base_state["awaiting_user_input"] is True

    def test_none_stage_id_handled(self, base_state):
        """Should handle None stage_id gracefully."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = None
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "targets": ["material_gold"]}]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        # Should handle None stage_id without crashing
        assert "comparison_verdict" in result
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "comparison_validation"

    def test_empty_stage_id_handled(self, base_state):
        """Should handle empty string stage_id gracefully."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = ""
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "targets": ["material_gold"]}]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert "comparison_verdict" in result
        assert result["workflow_phase"] == "comparison_validation"


class TestComparisonValidatorNoTargets:
    """Test behavior when stage has no targets."""

    def test_approves_no_targets_empty_list(self, base_state):
        """Should approve when targets is empty list."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "targets": [], "target_details": []}]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 2

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert "analysis_revision_count" not in result
        assert result.get("analysis_feedback") is None
        assert result["workflow_phase"] == "comparison_validation"
        assert "Stage has no reproducible targets" in result["comparison_feedback"]

    def test_approves_no_targets_missing_field(self, base_state):
        """Should approve when targets field is missing."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "target_details": []}]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result.get("analysis_feedback") is None
        assert "Stage has no reproducible targets" in result["comparison_feedback"]

    def test_approves_no_targets_empty_target_details(self, base_state):
        """Should approve when target_details is empty."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "targets": [],
                    "target_details": []
                }
            ]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result.get("analysis_feedback") is None


class TestComparisonValidatorMissingComparisons:
    """Test behavior when comparisons are missing."""

    def test_rejects_missing_comparisons_with_targets(self, base_state, valid_plan):
        """Should reject when targets exist but no comparisons."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 1

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 2
        assert result.get("analysis_feedback") is not None
        feedback = result.get("comparison_feedback", "")
        assert "comparison" in feedback.lower() or "report" in feedback.lower()
        assert result["workflow_phase"] == "comparison_validation"

    def test_rejects_missing_comparisons_wrong_stage(self, base_state, valid_plan):
        """Should reject when comparisons exist for different stage."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_1",  # Wrong stage
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result.get("comparison_feedback", "")

    def test_rejects_partial_missing_comparisons(self, base_state, valid_plan):
        """Should reject when only some comparisons are present."""
        from src.agents.analysis import comparison_validator_node

        # Plan has 2 targets but only 1 comparison
        plan = valid_plan.copy()
        plan["stages"][0]["targets"] = ["material_gold", "material_silver"]

        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_silver" in result.get("comparison_feedback", "")


class TestComparisonValidatorBreakdownClassifications:
    """Test breakdown of comparison classifications."""

    def test_rejects_missing_classification(self, base_state, valid_plan):
        """Should reject when classification indicates missing output."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "missing_output",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "missing_output",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result.get("comparison_feedback", "")

    def test_rejects_mismatch_classification(self, base_state, valid_plan):
        """Should reject when classification is mismatch."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "mismatch",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "mismatch",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result.get("comparison_feedback", "")

    def test_rejects_pending_classification(self, base_state, valid_plan):
        """Should reject when classification is pending."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "pending_validation",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "pending_validation",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result.get("comparison_feedback", "")
        assert "pending" in result.get("comparison_feedback", "").lower()

    def test_rejects_partial_match_classification(self, base_state, valid_plan):
        """Should reject when classification is partial_match."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "partial_match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "partial_match",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1

    def test_approves_match_classification(self, base_state, valid_plan):
        """Should approve when classification is match."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert "analysis_revision_count" not in result
        assert result.get("analysis_feedback") is None
        assert result["workflow_phase"] == "comparison_validation"

    def test_handles_enum_classifications(self, base_state, valid_plan):
        """Should handle AnalysisClassification enum values."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": AnalysisClassification.MISMATCH,
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": AnalysisClassification.MISMATCH,
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1


class TestComparisonValidatorMissingReports:
    """Test behavior when analysis reports are missing."""

    def test_rejects_missing_reports(self, base_state, valid_plan):
        """Should reject when reports are missing."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "Missing quantitative reports" in result["comparison_feedback"]
        assert "material_gold" in result["comparison_feedback"]

    def test_rejects_partial_missing_reports(self, base_state, valid_plan):
        """Should reject when only some reports are present."""
        from src.agents.analysis import comparison_validator_node

        plan = valid_plan.copy()
        plan["stages"][0]["targets"] = ["material_gold", "material_silver"]

        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            },
            {
                "stage_id": "stage_0",
                "figure_id": "material_silver",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
            }
            # Missing report for material_silver
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_silver" in result["comparison_feedback"]

    def test_rejects_reports_for_wrong_stage(self, base_state, valid_plan):
        """Should reject when reports exist for different stage."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",  # Wrong stage
                "target_figure": "material_gold",
                "status": "match",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]


class TestComparisonValidatorReportValidation:
    """Test validation of analysis reports."""

    def test_rejects_excellent_precision_without_metrics(self, base_state, valid_plan):
        """Should reject when excellent precision required but no metrics."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "precision_requirement": "excellent",
                "quantitative_metrics": {},  # Empty metrics
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]
        assert "excellent precision" in result["comparison_feedback"].lower()

    def test_rejects_match_with_high_error(self, base_state, valid_plan):
        """Should reject when classified as match but error exceeds threshold."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "quantitative_metrics": {
                    "peak_position_error_percent": 15.0,  # High error
                },
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]
        assert "match" in result["comparison_feedback"].lower() or "error" in result["comparison_feedback"].lower()

    def test_rejects_pending_with_excessive_error(self, base_state, valid_plan):
        """Should reject when pending but error exceeds investigate threshold."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "pending_validation",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "pending_validation",
                "quantitative_metrics": {
                    "peak_position_error_percent": 25.0,  # Very high error
                },
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]

    def test_rejects_reports_with_criteria_failures(self, base_state, valid_plan):
        """Should reject when reports have criteria failures."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": ["Peak wavelength mismatch"],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]
        assert "Peak wavelength mismatch" in result["comparison_feedback"]

    def test_handles_multiple_criteria_failures(self, base_state, valid_plan):
        """Should handle multiple criteria failures."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [
                    "Peak wavelength mismatch",
                    "Bandwidth too wide",
                    "Q factor too low",
                ],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        feedback = result["comparison_feedback"]
        assert "material_gold" in feedback
        # Should include at least one failure message
        assert any(
            failure in feedback
            for failure in ["Peak wavelength mismatch", "Bandwidth too wide", "Q factor too low"]
        )


class TestComparisonValidatorMultipleIssues:
    """Test behavior when multiple issues exist."""

    def test_combines_multiple_issues(self, base_state, valid_plan):
        """Should combine multiple issues in feedback."""
        from src.agents.analysis import comparison_validator_node

        plan = valid_plan.copy()
        plan["stages"][0]["targets"] = ["material_gold", "material_silver"]

        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
            # Missing comparison for material_silver
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": ["Test failure"],
            }
            # Missing report for material_silver
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        feedback = result["comparison_feedback"]
        # Should mention multiple issues
        assert "material_silver" in feedback or "Test failure" in feedback

    def test_truncates_feedback_when_many_issues(self, base_state, valid_plan):
        """Should truncate feedback when more than 3 issues."""
        from src.agents.analysis import comparison_validator_node

        plan = valid_plan.copy()
        plan["stages"][0]["targets"] = [
            "target1", "target2", "target3", "target4", "target5"
        ]

        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []  # All missing
        base_state["analysis_result_reports"] = []  # All missing
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        feedback = result["comparison_feedback"]
        # Should truncate and show count
        assert "+" in feedback or len(feedback.split(";")) <= 4


class TestComparisonValidatorRevisionCount:
    """Test analysis_revision_count increment logic."""

    def test_increments_revision_count_on_needs_revision(self, base_state, valid_plan):
        """Should increment revision count when verdict is needs_revision."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1

    def test_does_not_increment_on_approve(self, base_state, valid_plan):
        """Should not increment revision count when verdict is approve."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 5

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert "analysis_revision_count" not in result

    def test_sets_analysis_feedback_on_needs_revision(self, base_state, valid_plan):
        """Should set analysis_feedback when verdict is needs_revision."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_feedback") is not None
        assert result["analysis_feedback"] == result["comparison_feedback"]

    def test_clears_analysis_feedback_on_approve(self, base_state, valid_plan):
        """Should clear analysis_feedback when verdict is approve."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            }
        ]
        base_state["analysis_feedback"] = "Previous feedback"

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result.get("analysis_feedback") is None


class TestComparisonValidatorWorkflowPhase:
    """Test workflow_phase setting."""

    def test_sets_workflow_phase(self, base_state, valid_plan):
        """Should always set workflow_phase to comparison_validation."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert result["workflow_phase"] == "comparison_validation"

    def test_sets_workflow_phase_on_approve(self, base_state, valid_plan):
        """Should set workflow_phase even when approving."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["workflow_phase"] == "comparison_validation"


class TestComparisonValidatorMultiStage:
    """Test behavior with multiple stages."""

    def test_filters_comparisons_by_stage(self, base_state):
        """Should only consider comparisons for current stage."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["material_gold"]},
                {"stage_id": "stage_1", "targets": ["material_silver"]},
            ]
        }
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            },
            {
                "stage_id": "stage_1",
                "figure_id": "material_silver",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            },
            {
                "stage_id": "stage_1",
                "target_figure": "material_silver",
                "status": "match",
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        # Should approve because stage_0 has valid comparisons
        assert result["comparison_verdict"] == "approve"

    def test_filters_reports_by_stage(self, base_state):
        """Should only consider reports for current stage."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["material_gold"]},
            ]
        }
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            },
            {
                "stage_id": "stage_1",  # Different stage
                "target_figure": "material_silver",
                "status": "match",
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        # Should approve because stage_0 has valid report
        assert result["comparison_verdict"] == "approve"


class TestComparisonValidatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_handles_missing_plan(self, base_state):
        """Should handle missing plan gracefully."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = None
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert "comparison_verdict" in result
        assert result["workflow_phase"] == "comparison_validation"

    def test_handles_missing_stage_in_plan(self, base_state):
        """Should handle missing stage in plan."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {"stages": []}
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert "comparison_verdict" in result
        assert result["workflow_phase"] == "comparison_validation"

    def test_handles_none_comparisons(self, base_state, valid_plan):
        """Should handle None figure_comparisons."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = None
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"

    def test_handles_none_reports(self, base_state, valid_plan):
        """Should handle None analysis_result_reports."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = None

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"

    def test_handles_comparison_without_figure_id(self, base_state, valid_plan):
        """Should handle comparison missing figure_id."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                # Missing figure_id
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)

        # Should still process and reject due to missing reports
        assert result["comparison_verdict"] == "needs_revision"

    def test_handles_report_without_target_figure(self, base_state, valid_plan):
        """Should handle report missing target_figure."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                # Missing target_figure
                "status": "match",
            }
        ]

        result = comparison_validator_node(base_state)

        # Should still process
        assert "comparison_verdict" in result

    def test_handles_target_details_format(self, base_state):
        """Should handle targets from target_details."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "target_details": [
                        {"figure_id": "material_gold"},
                        {"figure_id": "material_silver"},
                    ]
                }
            ]
        }
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        # Should reject because material_silver is missing
        assert result["comparison_verdict"] == "needs_revision"
        assert "material_silver" in result["comparison_feedback"]
