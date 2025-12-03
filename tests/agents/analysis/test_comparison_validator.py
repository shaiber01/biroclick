"""
Tests for ComparisonValidatorAgent (comparison_validator_node).
"""

from unittest.mock import patch, MagicMock

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
        
        # Verify all return fields are present and correct
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
        assert result["comparison_feedback"] == "All required comparisons present."
        assert result["analysis_feedback"] is None
        # Verify revision count is NOT incremented on approve
        assert "analysis_revision_count" not in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_analysis_reports_none(self, mock_check, base_state):
        """Test validation when analysis_result_reports is None."""
        base_state["analysis_result_reports"] = None
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_revision_count"] = 0
        
        # Should handle None gracefully or fail safely
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_missing_comparison(self, mock_check, base_state):
        """Test rejection when comparison is missing for a target."""
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig1" in result["comparison_feedback"]
        assert "Results analyzer did not produce comparisons for" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_missing_quantitative_data(self, mock_check, base_state):
        """Test rejection when quantitative reports are missing."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [] # Missing reports
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_revision_limit(self, mock_check, base_state):
        """Test that revision count increments."""
        base_state["figure_comparisons"] = [] # Fail
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 1
        assert result["analysis_feedback"] is not None

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
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 3  # Should not increment beyond max
        assert result["analysis_feedback"] is not None
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_respects_max_revisions_at_exact_max(self, mock_check, base_state):
        """Should not increment when already at max revisions."""
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 2
        base_state["runtime_config"] = {"max_analysis_revisions": 3}
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 3  # Increments to max
        assert result["analysis_feedback"] is not None
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_respects_max_revisions_default(self, mock_check, base_state):
        """Should use default max when runtime_config missing."""
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        base_state["runtime_config"] = {}  # Missing max_analysis_revisions
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 1
        assert result["analysis_feedback"] is not None
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_validator_respects_max_revisions_none_config(self, mock_check, base_state):
        """Should use default max when runtime_config is None."""
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        base_state["runtime_config"] = None
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 1
        assert result["analysis_feedback"] is not None

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_approves_when_no_targets(self, mock_check, base_state):
        """Should approve when stage has no reproducible targets."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
        assert result["comparison_feedback"] == "Stage has no reproducible targets; nothing to compare."
        assert result["analysis_feedback"] is None
        assert "analysis_revision_count" not in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_pending_checks(self, mock_check, base_state):
        """Should need revision when comparisons are pending."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.PENDING_VALIDATION}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.PENDING_VALIDATION}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "pending" in result["comparison_feedback"].lower()
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

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
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        # High error should flag an issue
        assert "Fig1" in result["comparison_feedback"]
        assert "error" in result["comparison_feedback"].lower() or "peak" in result["comparison_feedback"].lower()
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

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
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
        assert result["comparison_feedback"] == "All required comparisons present."
        assert result["analysis_feedback"] is None
        assert "analysis_revision_count" not in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_sets_feedback_on_needs_revision(self, mock_check, base_state):
        """Should set analysis_feedback on needs_revision."""
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_feedback"] is not None
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_truncates_many_issues(self, mock_check, base_state):
        """Should truncate feedback when many issues exist."""
        base_state["plan"]["stages"][0]["targets"] = ["F1", "F2", "F3", "F4", "F5"]
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        with patch("src.agents.analysis.validate_analysis_reports", return_value=["I1", "I2", "I3", "I4"]):
            result = comparison_validator_node(base_state)
            # Should show first 3 issues and indicate remaining issues
            assert result["comparison_verdict"] == "needs_revision"
            # Total issues: 1 missing comparisons + 4 report issues + 1 missing reports = 6
            # Shows first 3, so should show "(+3 more)"
            assert "(+3 more)" in result["comparison_feedback"] or "(+2 more)" in result["comparison_feedback"]
            # Verify feedback contains truncation indicator
            assert "more)" in result["comparison_feedback"]
            assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_truncates_exactly_three_issues(self, mock_check, base_state):
        """Should not truncate when exactly 3 issues exist."""
        base_state["plan"]["stages"][0]["targets"] = ["F1", "F2", "F3"]
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        with patch("src.agents.analysis.validate_analysis_reports", return_value=["I1", "I2", "I3"]):
            result = comparison_validator_node(base_state)
            assert result["comparison_verdict"] == "needs_revision"
            assert "(+0 more)" not in result["comparison_feedback"]
            assert "(+1 more)" not in result["comparison_feedback"]
            assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_truncates_five_issues(self, mock_check, base_state):
        """Should truncate when 5 issues exist."""
        base_state["plan"]["stages"][0]["targets"] = ["F1"]
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        with patch("src.agents.analysis.validate_analysis_reports", return_value=["I1", "I2", "I3", "I4", "I5"]):
            result = comparison_validator_node(base_state)
            assert result["comparison_verdict"] == "needs_revision"
            # Total issues: 1 missing comparisons + 5 report issues + 1 missing reports = 7
            # Shows first 3, so should show "(+4 more)"
            assert "(+4 more)" in result["comparison_feedback"]
            assert result["analysis_revision_count"] == 1

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
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        # Should include both missing comparisons and missing reports messages
        assert "Results analyzer did not produce comparisons for" in result["comparison_feedback"]
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
        
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_missing_outputs(self, mock_check, base_state):
        """Should need revision when outputs are missing."""
        # Missing output files should result in empty comparisons or comparisons marked as missing_output
        # If results_analyzer_node runs first, it produces comparisons.
        # But if they are missing, breakdown will show "missing"
        base_state["figure_comparisons"] = [
             {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": "missing_output"}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "missing" in result["comparison_feedback"].lower()
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_mismatch_classification(self, mock_check, base_state):
        """Should need revision when comparisons have mismatch classification."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MISMATCH}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "missing" in result["comparison_feedback"].lower()
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_partial_match_classification(self, mock_check, base_state):
        """Should need revision when comparisons have partial_match classification."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.PARTIAL_MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.PARTIAL_MATCH}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        # PARTIAL_MATCH goes into "pending" bucket, so should show pending message
        assert "pending" in result["comparison_feedback"].lower()
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

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
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
        assert result["comparison_feedback"] == "All required comparisons present."
        assert result["analysis_feedback"] is None
        assert "analysis_revision_count" not in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_needs_revision_when_missing_comparisons(self, mock_check, base_state):
        """Should need revision when comparisons are missing for some targets."""
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_revision_count"] = 0
        # Fig2 missing
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]
        assert "Results analyzer did not produce comparisons for" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

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
        base_state["analysis_revision_count"] = 0
        # Fig2 report missing
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]
        assert "Missing quantitative reports" in result["comparison_feedback"]
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1

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
        assert result["comparison_feedback"] == "All required comparisons present."
        assert result["analysis_feedback"] is None
        assert "analysis_revision_count" not in result
    
    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases and Error Conditions
    # ═══════════════════════════════════════════════════════════════════════
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_current_stage_id_none(self, mock_check, base_state):
        """Should handle None current_stage_id gracefully."""
        base_state["current_stage_id"] = None
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        # Should return empty comparisons and reports for None stage_id
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"  # No targets expected
        assert "no reproducible targets" in result["comparison_feedback"].lower()
        assert result["analysis_feedback"] is None
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_figure_comparisons_none(self, mock_check, base_state):
        """Should handle None figure_comparisons gracefully."""
        base_state["figure_comparisons"] = None
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_feedback"] == result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_plan_none(self, mock_check, base_state):
        """Should handle None plan gracefully."""
        base_state["plan"] = None
        base_state["figure_comparisons"] = []
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"  # No targets expected
        assert "no reproducible targets" in result["comparison_feedback"].lower()
        assert result["analysis_feedback"] is None
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_plan_stages_empty(self, mock_check, base_state):
        """Should handle empty plan stages."""
        base_state["plan"]["stages"] = []
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"  # No targets expected
        assert "no reproducible targets" in result["comparison_feedback"].lower()
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_plan_stages_none(self, mock_check, base_state):
        """Should handle None plan stages."""
        base_state["plan"]["stages"] = None
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        # Should handle gracefully - get_plan_stage returns None
        assert result["comparison_verdict"] == "approve"  # No targets expected
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_awaiting_user_input_true(self, mock_check, base_state):
        """Should return early when awaiting_user_input is already True."""
        base_state["awaiting_user_input"] = True
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        # Should return state unchanged (early return)
        assert result["awaiting_user_input"] is True
        # Should not have processed - no workflow_phase set
        assert "workflow_phase" not in result or result.get("workflow_phase") != "comparison_validation"
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_comparisons_for_wrong_stage(self, mock_check, base_state):
        """Should ignore comparisons for different stage_id."""
        base_state["current_stage_id"] = "stage_1_sim"
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_2_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        # Should treat as missing comparisons for current stage
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_reports_for_wrong_stage(self, mock_check, base_state):
        """Should ignore reports for different stage_id."""
        base_state["current_stage_id"] = "stage_1_sim"
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_2_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        # Should detect missing report for current stage
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]
        assert "Fig1" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_multiple_stages_with_correct_filtering(self, mock_check, base_state):
        """Should only consider comparisons/reports for current stage."""
        base_state["current_stage_id"] = "stage_1_sim"
        base_state["plan"]["stages"] = [
            {"stage_id": "stage_1_sim", "targets": ["Fig1"]},
            {"stage_id": "stage_2_sim", "targets": ["Fig2"]}
        ]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH},
            {"stage_id": "stage_2_sim", "figure_id": "Fig2", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH},
            {"stage_id": "stage_2_sim", "target_figure": "Fig2", "status": AnalysisClassification.MATCH}
        ]
        
        result = comparison_validator_node(base_state)
        
        # Should only validate stage_1_sim
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
        assert result["comparison_feedback"] == "All required comparisons present."
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_target_details_with_missing_figure_id(self, mock_check, base_state):
        """Should handle target_details entries without figure_id."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = [
            {"figure_id": "Fig1"},
            {"description": "Some target"}  # Missing figure_id
        ]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        
        result = comparison_validator_node(base_state)
        
        # Should only consider Fig1 (the one with figure_id)
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "approve"
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_report_with_criteria_failures(self, mock_check, base_state):
        """Should detect criteria_failures in reports."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1_sim",
                "target_figure": "Fig1",
                "status": AnalysisClassification.MATCH,
                "criteria_failures": ["Peak position mismatch", "Amplitude too low"]
            }
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig1" in result["comparison_feedback"]
        assert "Peak position mismatch" in result["comparison_feedback"] or "Amplitude too low" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_report_with_excellent_precision_no_metrics(self, mock_check, base_state):
        """Should detect missing metrics for excellent precision requirement."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1_sim",
                "target_figure": "Fig1",
                "status": AnalysisClassification.MATCH,
                "precision_requirement": "excellent",
                "quantitative_metrics": {}  # Missing metrics
            }
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig1" in result["comparison_feedback"]
        assert "excellent precision requires quantitative metrics" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_handles_report_with_pending_status_high_error(self, mock_check, base_state):
        """Should detect high error for pending/partial_match status."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.PENDING_VALIDATION}
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1_sim",
                "target_figure": "Fig1",
                "status": AnalysisClassification.PENDING_VALIDATION,
                "quantitative_metrics": {"peak_position_error_percent": 50.0},  # Very high error
                "precision_requirement": "acceptable"
            }
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig1" in result["comparison_feedback"]
        # Should flag that error exceeds investigate threshold
        assert "error" in result["comparison_feedback"].lower() or "exceeds" in result["comparison_feedback"].lower()
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_verdict_priority_missing_overrides_approve(self, mock_check, base_state):
        """Test that missing comparisons override initial approve verdict."""
        # Setup: comparisons exist but missing for one target
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        # Should be needs_revision because Fig2 is missing
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_verdict_priority_report_issues_override_approve(self, mock_check, base_state):
        """Test that report issues override initial approve verdict."""
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1_sim",
                "target_figure": "Fig1",
                "status": AnalysisClassification.MATCH,
                "criteria_failures": ["Test failure"]
            }
        ]
        base_state["analysis_revision_count"] = 0
        
        result = comparison_validator_node(base_state)
        
        # Should be needs_revision because of report issues
        assert result["comparison_verdict"] == "needs_revision"
        assert "Test failure" in result["comparison_feedback"]
        assert result["analysis_revision_count"] == 1
    
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    def test_state_not_mutated(self, mock_check, base_state):
        """Test that original state is not mutated."""
        original_state = base_state.copy()
        original_figure_comparisons = base_state.get("figure_comparisons", []).copy()
        original_analysis_reports = base_state.get("analysis_result_reports", []).copy()
        
        base_state["figure_comparisons"] = [
            {"stage_id": "stage_1_sim", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        base_state["analysis_result_reports"] = [
            {"stage_id": "stage_1_sim", "target_figure": "Fig1", "status": AnalysisClassification.MATCH}
        ]
        
        result = comparison_validator_node(base_state)
        
        # Verify result is a dict (state updates), not the state itself
        assert isinstance(result, dict)
        # Verify original state keys are unchanged (function should only return updates)
        assert base_state["current_stage_id"] == original_state["current_stage_id"]
        assert base_state["paper_id"] == original_state["paper_id"]
