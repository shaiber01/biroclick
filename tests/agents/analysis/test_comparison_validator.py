"""
Tests for ComparisonValidatorAgent (comparison_validator_node).
"""

from unittest.mock import patch

import pytest

from src.agents.analysis import comparison_validator_node
from src.agents.constants import AnalysisClassification


@pytest.fixture(name="base_state")
def analysis_base_state_alias(analysis_state):
    return analysis_state


class TestComparisonValidatorNode:
    """Tests for comparison_validator_node."""

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_approve(self, mock_check, base_state):
        """Test approval when all comparisons exist and match."""
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "approve"
        assert result["analysis_feedback"] is None
        assert "All required comparisons present" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_analysis_reports_none(self, mock_check, base_state):
        """Test validation when analysis_result_reports is None."""
        base_state["analysis_result_reports"] = None
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        
        # Should handle None gracefully or fail safely
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_missing_comparison(self, mock_check, base_state):
        """Test rejection when comparison is missing for a target."""
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig1" in result["comparison_feedback"] or "missing" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_missing_quantitative_data(self, mock_check, base_state):
        """Test rejection when quantitative reports are missing."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [] # Missing reports
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_revision_limit(self, mock_check, base_state):
        """Test that revision count increments."""
        base_state["figure_comparisons"] = [] # Fail
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_increments_revision_count_on_needs_revision(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons, base_state
    ):
        """Should increment analysis_revision_count when needs revision."""
        mock_context.return_value = None
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = []
        mock_validate.return_value = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_respects_max_revisions(self, mock_check, base_state):
        """Should not increment beyond max revisions."""
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 3
        base_state["runtime_config"] = {"max_analysis_revisions": 3}
        
        result = comparison_validator_node(base_state)
        
        assert result["analysis_revision_count"] == 3

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_approves_when_no_targets(self, mock_check, base_state):
        """Should approve when stage has no reproducible targets."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "approve"
        assert "no reproducible targets" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_pending_checks(self, mock_check, base_state):
        """Should need revision when comparisons are pending."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.PENDING_VALIDATION}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.PENDING_VALIDATION}
        ]
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "pending" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_on_report_issues(self, mock_check, base_state):
        """Should need revision when analysis reports have validation issues."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1_sim", 
                "target_figure": "Fig1", 
                "status": AnalysisClassification.MATCH,
                "quantitative_metrics": {"peak_position_error_percent": 10.0}, # High error for match
                "precision_requirement": "acceptable"
            }
        ]
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        # High error should flag an issue
        assert "error" in result["comparison_feedback"] or "peak error" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_clears_feedback_on_approve(self, mock_check, base_state):
        """Should clear analysis_feedback on approval."""
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_feedback"] = "Old feedback"
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "approve"
        assert result["analysis_feedback"] is None

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_sets_feedback_on_needs_revision(self, mock_check, base_state):
        """Should set analysis_feedback on needs_revision."""
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_feedback"] is not None
        assert result["analysis_feedback"] == result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_truncates_many_issues(self, mock_check, base_state):
        """Should truncate feedback when many issues exist."""
        base_state["plan"]["stages"][0]["targets"] = ["F1", "F2", "F3", "F4", "F5"]
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        feedback = result["comparison_feedback"]
        with patch("src.agents.analysis.validate_analysis_reports", return_value=["I1", "I2", "I3", "I4"]):
             result = comparison_validator_node(base_state)
             assert "(+2 more)" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_validator_escalation_on_context_overflow(self, mock_context, base_state):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "workflow_phase": "comparison_validation",  # Added missing phase
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        # Ensure we are testing the decorated function, or the logic inside handles it.
        # If mock_context is called, it returns the dict.
        # The decorator in base.py sees this and should return it.
        # If this test fails, it means the decorator logic isn't picking up the return value correctly
        # OR the patch isn't targeting the right function.
        # Since we patched 'src.agents.analysis.check_context_or_escalate', and analysis.py imports it,
        # but the decorator is in base.py...
        
        # Let's try patching where it is defined, to cover all usages.
        with patch("src.agents.helpers.context.check_context_before_node") as mock_check_before:
             mock_check_before.return_value = {
                 "ok": False,
                 "escalate": True,
                 "user_question": "Context overflow",
                 "state_updates": None
             }
             
             result = comparison_validator_node(base_state)
             
             assert result["awaiting_user_input"] is True

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_no_comparisons_but_has_targets(self, mock_check, base_state):
        """Should need revision when targets exist but no comparisons."""
        base_state["plan"]["stages"][0]["targets"] = ["Fig1"]
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_missing_outputs(self, mock_check, base_state):
        """Should need revision when outputs are missing."""
        # Missing output files should result in empty comparisons or comparisons marked as missing_output
        # If results_analyzer_node runs first, it produces comparisons.
        # But if they are missing, breakdown will show "missing"
        base_state["figure_comparisons"] = [
             {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": "missing_output"}
        ]
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "missing" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_approves_when_all_comparisons_present(self, mock_check, base_state):
        """Should approve when all required comparisons are present."""
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH},
            {"stage_id": "stage_1_sim", "figure_id": "Fig2", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH},
            {"stage_id": "stage_1_sim", "target_figure": "Fig2", "status": AnalysisClassification.MATCH}
        ]
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "approve"

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_missing_comparisons(self, mock_check, base_state):
        """Should need revision when comparisons are missing for some targets."""
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        # Fig2 missing
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_missing_report_targets(self, mock_check, base_state):
        """Should detect missing quantitative reports for targets."""
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH},
            {"stage_id": "stage_1_sim", "figure_id": "Fig2", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        # Fig2 report missing
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_target_details_format(self, mock_check, base_state):
        """Should handle target_details format instead of targets list."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = [{"figure_id": "Fig1"}]
        
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
