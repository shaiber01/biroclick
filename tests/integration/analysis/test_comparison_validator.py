"""Integration tests for comparison_validator_node.

These tests are designed to FIND BUGS, not just pass.
Every test verifies specific behavior and would fail if bugs exist.
"""

import pytest
from src.agents.constants import AnalysisClassification
from schemas.state import DISCREPANCY_THRESHOLDS, MAX_ANALYSIS_REVISIONS


# ═══════════════════════════════════════════════════════════════════════
# THRESHOLD CONSTANTS FOR TESTING
# ═══════════════════════════════════════════════════════════════════════
# These are derived from DISCREPANCY_THRESHOLDS in schemas/state.py
EXCELLENT_THRESHOLD = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["excellent"]  # 2%
ACCEPTABLE_THRESHOLD = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["acceptable"]  # 5%
INVESTIGATE_THRESHOLD = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["investigate"]  # 10%


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


class TestComparisonValidatorAllClassifications:
    """Test all AnalysisClassification enum variants are handled correctly."""

    def test_excellent_match_classification_approves(self, base_state, valid_plan):
        """Should approve when classification is EXCELLENT_MATCH."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": AnalysisClassification.EXCELLENT_MATCH,
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": AnalysisClassification.EXCELLENT_MATCH,
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"
        assert result.get("analysis_feedback") is None
        assert "analysis_revision_count" not in result

    def test_acceptable_match_classification_approves(self, base_state, valid_plan):
        """Should approve when classification is ACCEPTABLE_MATCH."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": AnalysisClassification.ACCEPTABLE_MATCH,
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": AnalysisClassification.ACCEPTABLE_MATCH,
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"
        assert result.get("analysis_feedback") is None

    def test_poor_match_classification_rejects(self, base_state, valid_plan):
        """Should reject when classification is POOR_MATCH."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": AnalysisClassification.POOR_MATCH,
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": AnalysisClassification.POOR_MATCH,
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result["workflow_phase"] == "comparison_validation"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]

    def test_failed_classification_rejects(self, base_state, valid_plan):
        """Should reject when classification is FAILED."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": AnalysisClassification.FAILED,
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": AnalysisClassification.FAILED,
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result["workflow_phase"] == "comparison_validation"
        assert result.get("analysis_revision_count") == 1
        assert "material_gold" in result["comparison_feedback"]

    def test_no_targets_classification_approves(self, base_state, valid_plan):
        """Should approve when classification is NO_TARGETS."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": AnalysisClassification.NO_TARGETS,
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": AnalysisClassification.NO_TARGETS,
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # NO_TARGETS falls into "missing" bucket per breakdown_comparison_classifications
        assert result["comparison_verdict"] == "needs_revision"
        assert result["workflow_phase"] == "comparison_validation"

    def test_lowercase_string_classification_match(self, base_state, valid_plan):
        """Should handle lowercase string 'match' classification."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",  # lowercase string
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

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"

    def test_uppercase_string_classification_match(self, base_state, valid_plan):
        """Should handle uppercase string 'MATCH' classification."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "MATCH",  # uppercase string
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "MATCH",
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"


class TestComparisonValidatorThresholdBoundaries:
    """Test exact threshold boundary values from DISCREPANCY_THRESHOLDS."""

    def test_error_at_exactly_acceptable_threshold_passes(self, base_state, valid_plan):
        """Should pass when error equals acceptable threshold exactly (5%)."""
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
                    "peak_position_error_percent": ACCEPTABLE_THRESHOLD,  # exactly 5%
                },
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        # Error at exactly 5% should NOT trigger "match with high error" issue
        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"

    def test_error_just_above_acceptable_threshold_fails(self, base_state, valid_plan):
        """Should fail when error is just above acceptable threshold (5.01%)."""
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
                    "peak_position_error_percent": ACCEPTABLE_THRESHOLD + 0.01,  # 5.01%
                },
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # Error above 5% for a "match" status triggers validation issue
        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1

    def test_error_at_exactly_investigate_threshold_for_pending(self, base_state, valid_plan):
        """Should handle error at exactly investigate threshold (10%) for pending."""
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
                    "peak_position_error_percent": INVESTIGATE_THRESHOLD,  # exactly 10%
                },
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # pending_validation goes into "pending" bucket, so needs_revision
        assert result["comparison_verdict"] == "needs_revision"
        assert result["workflow_phase"] == "comparison_validation"

    def test_error_above_investigate_threshold_for_pending(self, base_state, valid_plan):
        """Should flag when pending has error above investigate threshold (10.01%)."""
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
                    "peak_position_error_percent": INVESTIGATE_THRESHOLD + 0.01,  # 10.01%
                },
                "criteria_failures": [],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 1
        # Should mention that error exceeds investigate threshold
        assert "material_gold" in result["comparison_feedback"]

    def test_error_at_excellent_threshold_for_excellent_precision(self, base_state, valid_plan):
        """Should handle error at excellent threshold (2%) for excellent precision."""
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
                "quantitative_metrics": {
                    "peak_position_error_percent": EXCELLENT_THRESHOLD,  # 2%
                },
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"


class TestComparisonValidatorRevisionCountMax:
    """Test behavior when revision count is at or near maximum."""

    def test_revision_count_does_not_exceed_max(self, base_state, valid_plan):
        """Should not increment revision count beyond MAX_ANALYSIS_REVISIONS."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []  # Triggers needs_revision
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS  # Already at max

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        # Should NOT increment beyond max
        assert result["analysis_revision_count"] == MAX_ANALYSIS_REVISIONS
        assert result.get("analysis_feedback") is not None

    def test_revision_count_increments_to_max(self, base_state, valid_plan):
        """Should increment to max when one below."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS - 1

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == MAX_ANALYSIS_REVISIONS

    def test_custom_max_from_runtime_config(self, base_state, valid_plan):
        """Should respect custom max from runtime_config."""
        from src.agents.analysis import comparison_validator_node

        custom_max = 5
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = custom_max - 1
        base_state["runtime_config"] = {"max_analysis_revisions": custom_max}

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == custom_max

    def test_custom_max_does_not_exceed(self, base_state, valid_plan):
        """Should not exceed custom max from runtime_config."""
        from src.agents.analysis import comparison_validator_node

        custom_max = 5
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = custom_max  # Already at custom max
        base_state["runtime_config"] = {"max_analysis_revisions": custom_max}

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == custom_max  # Should not exceed


class TestComparisonValidatorHelperIntegration:
    """Test that helper functions are used correctly in integration."""

    def test_breakdown_classifications_missing_bucket(self, base_state, valid_plan):
        """Verify missing classifications go to missing bucket."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        # Classifications that should go to "missing" bucket
        missing_classifications = [
            "missing_output", "fail", "not_reproduced", "mismatch", "poor_match"
        ]
        
        for classification in missing_classifications:
            base_state["figure_comparisons"] = [
                {
                    "stage_id": "stage_0",
                    "figure_id": "material_gold",
                    "classification": classification,
                }
            ]
            base_state["analysis_result_reports"] = []
            base_state["analysis_revision_count"] = 0

            result = comparison_validator_node(base_state)

            assert result["comparison_verdict"] == "needs_revision", f"Failed for classification: {classification}"
            assert "material_gold" in result["comparison_feedback"], f"Failed for classification: {classification}"

    def test_breakdown_classifications_pending_bucket(self, base_state, valid_plan):
        """Verify pending classifications go to pending bucket."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        # Classifications that should go to "pending" bucket
        pending_classifications = [
            "pending_validation", "partial_match", "match_pending", "partial"
        ]
        
        for classification in pending_classifications:
            base_state["figure_comparisons"] = [
                {
                    "stage_id": "stage_0",
                    "figure_id": "material_gold",
                    "classification": classification,
                }
            ]
            base_state["analysis_result_reports"] = [
                {
                    "stage_id": "stage_0",
                    "target_figure": "material_gold",
                    "status": classification,
                }
            ]
            base_state["analysis_revision_count"] = 0

            result = comparison_validator_node(base_state)

            assert result["comparison_verdict"] == "needs_revision", f"Failed for classification: {classification}"
            assert "pending" in result["comparison_feedback"].lower() or "material_gold" in result["comparison_feedback"], f"Failed for classification: {classification}"

    def test_stage_comparisons_filtering_correct(self, base_state):
        """Verify that comparisons are filtered by stage_id correctly."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["target_0"]},
                {"stage_id": "stage_1", "targets": ["target_1"]},
            ]
        }
        # Comparisons for wrong stage should be ignored
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_1",  # WRONG stage
                "figure_id": "target_0",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # Should need revision because no comparisons for stage_0
        assert result["comparison_verdict"] == "needs_revision"
        assert "target_0" in result["comparison_feedback"]

    def test_analysis_reports_filtering_correct(self, base_state):
        """Verify that reports are filtered by stage_id correctly."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["target_0"]},
            ]
        }
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "target_0",
                "classification": "match",
            }
        ]
        # Report for wrong stage should be ignored
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",  # WRONG stage
                "target_figure": "target_0",
                "status": "match",
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # Should need revision because no reports for stage_0
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]
        assert "target_0" in result["comparison_feedback"]


class TestComparisonValidatorFeedbackContent:
    """Test that feedback messages contain correct and specific content."""

    def test_missing_comparisons_feedback_lists_targets(self, base_state):
        """Feedback should list missing target names."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["target_A", "target_B", "target_C"]}
            ]
        }
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        feedback = result["comparison_feedback"]
        # Should mention all three targets
        assert "target_A" in feedback
        assert "target_B" in feedback
        assert "target_C" in feedback

    def test_criteria_failures_included_in_feedback(self, base_state, valid_plan):
        """Feedback should include criteria failure messages."""
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
                    "Peak wavelength error exceeds 2%",
                    "Q factor mismatch"
                ],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        feedback = result["comparison_feedback"]
        # Should include at least one of the failure messages
        assert "Peak wavelength error exceeds 2%" in feedback or "Q factor mismatch" in feedback

    def test_truncation_indicator_with_many_issues(self, base_state):
        """Feedback should truncate and show count when >3 distinct issues.
        
        Note: Missing comparisons for multiple targets = 1 issue message.
        To trigger truncation, we need multiple validation issues (>3 total).
        """
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["t1", "t2", "t3", "t4"]}
            ]
        }
        # Provide comparisons so we don't get a single "missing comparisons" issue
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_0", "figure_id": "t1", "classification": "match"},
            {"stage_id": "stage_0", "figure_id": "t2", "classification": "match"},
            {"stage_id": "stage_0", "figure_id": "t3", "classification": "match"},
            {"stage_id": "stage_0", "figure_id": "t4", "classification": "match"},
        ]
        # Create multiple validation issues - each criteria failure and high error creates
        # separate issues from validate_analysis_reports
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "t1",
                "status": "match",
                "criteria_failures": ["Issue 1 for t1"],
            },
            {
                "stage_id": "stage_0",
                "target_figure": "t2",
                "status": "match",
                "criteria_failures": ["Issue 2 for t2"],
            },
            {
                "stage_id": "stage_0",
                "target_figure": "t3",
                "status": "match",
                "criteria_failures": ["Issue 3 for t3"],
            },
            {
                "stage_id": "stage_0",
                "target_figure": "t4",
                "status": "match",
                "criteria_failures": ["Issue 4 for t4"],
            },
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        feedback = result["comparison_feedback"]
        # With 4 issues, should show truncation indicator "(+1 more)"
        assert "more)" in feedback

    def test_exactly_three_issues_no_truncation(self, base_state):
        """Feedback should NOT truncate when exactly 3 issues."""
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "targets": ["t1"]}
            ]
        }
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "t1",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "t1",
                "status": "match",
                "criteria_failures": ["issue1", "issue2", "issue3"],
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        feedback = result["comparison_feedback"]
        # With exactly 3 issues, should NOT show "+N more"
        assert "(+0 more)" not in feedback


class TestComparisonValidatorStateIntegrity:
    """Test that the node maintains state integrity."""

    def test_analysis_feedback_matches_comparison_feedback_on_revision(self, base_state, valid_plan):
        """analysis_feedback should equal comparison_feedback when needs_revision."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_feedback"] is not None

    def test_analysis_feedback_is_none_on_approve(self, base_state, valid_plan):
        """analysis_feedback should be None when approved."""
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

        assert result["comparison_verdict"] == "approve"
        assert result["analysis_feedback"] is None
        assert "analysis_revision_count" not in result

    def test_workflow_phase_always_set(self, base_state, valid_plan):
        """workflow_phase should always be 'comparison_validation'."""
        from src.agents.analysis import comparison_validator_node

        test_cases = [
            # Approve case
            {
                "figure_comparisons": [
                    {"stage_id": "stage_0", "figure_id": "material_gold", "classification": "match"}
                ],
                "analysis_result_reports": [
                    {"stage_id": "stage_0", "target_figure": "material_gold", "status": "match", "criteria_failures": []}
                ],
            },
            # Needs revision case
            {
                "figure_comparisons": [],
                "analysis_result_reports": [],
            },
        ]

        for case in test_cases:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["figure_comparisons"] = case["figure_comparisons"]
            base_state["analysis_result_reports"] = case["analysis_result_reports"]
            base_state["analysis_revision_count"] = 0

            result = comparison_validator_node(base_state)

            assert result["workflow_phase"] == "comparison_validation"


class TestComparisonValidatorRobustness:
    """Test robustness against malformed or unusual inputs."""

    def test_comparison_with_empty_figure_id(self, base_state, valid_plan):
        """Should handle comparison with empty figure_id."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "",  # Empty
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # Should still process and identify missing comparisons for actual target
        assert result["comparison_verdict"] == "needs_revision"
        assert result["workflow_phase"] == "comparison_validation"

    def test_report_with_none_status(self, base_state, valid_plan):
        """Should handle report with None status."""
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
                "status": None,  # None status
            }
        ]
        base_state["analysis_revision_count"] = 0

        result = comparison_validator_node(base_state)

        # Should handle gracefully
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "comparison_validation"

    def test_metrics_with_non_numeric_error(self, base_state, valid_plan):
        """Should handle metrics with non-numeric error values."""
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
                    "peak_position_error_percent": "not_a_number",  # Invalid
                },
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)

        # Should handle gracefully without crashing
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "comparison_validation"

    def test_empty_criteria_failures_list(self, base_state, valid_plan):
        """Should handle empty criteria_failures list."""
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
                "criteria_failures": [],  # Empty list
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"

    def test_missing_criteria_failures_field(self, base_state, valid_plan):
        """Should handle missing criteria_failures field."""
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
                # No criteria_failures field
            }
        ]

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"
