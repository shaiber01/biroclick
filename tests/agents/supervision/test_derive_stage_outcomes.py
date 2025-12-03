"""Tests for _derive_stage_completion_outcome helper."""

from unittest.mock import patch, call

import pytest

from src.agents.supervision.supervisor import _derive_stage_completion_outcome

class TestDeriveStageCompletionOutcome:
    """Tests for _derive_stage_completion_outcome logic."""

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_failed_on_missing_outputs(self, mock_breakdown, mock_comparisons):
        """Should return completed_failed if outputs are missing."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": ["absorption.csv"], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": "Good match",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"
        assert "Missing outputs" in summary
        assert "absorption.csv" in summary
        mock_comparisons.assert_called_once_with(state, "stage1")
        mock_breakdown.assert_called_once_with([])

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_failed_on_missing_outputs_overrides_everything(self, mock_breakdown, mock_comparisons):
        """Missing outputs should override all other conditions, even success classifications."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": ["output1.csv", "output2.csv"], "pending": [], "match": []}
        
        # Even with excellent match, missing outputs should fail
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"
        assert "Missing outputs" in summary
        assert "output1.csv" in summary
        assert "output2.csv" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_partial_on_pending_comparisons(self, mock_breakdown, mock_comparisons):
        """Should return completed_partial if comparisons are pending."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": ["fig1"], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_partial"
        assert "Comparisons pending" in summary
        assert "fig1" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_pending_only_affects_success_status(self, mock_breakdown, mock_comparisons):
        """Pending comparisons should only downgrade success to partial, not override failures."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": ["fig1"], "match": []}
        
        # With a failed classification, pending should not override
        state = {
            "analysis_overall_classification": "FAILED",
            "comparison_verdict": "mismatch",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # Should remain failed, not become partial
        assert status == "completed_failed"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_failed_on_bad_classification(self, mock_breakdown, mock_comparisons):
        """Should return completed_failed on FAILED or POOR_MATCH."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        for classification in ["FAILED", "POOR_MATCH"]:
            state = {
                "analysis_overall_classification": classification,
                "comparison_verdict": "mismatch",
                "physics_verdict": "pass",
                "analysis_summary": "Bad results",
            }
            
            status, summary = _derive_stage_completion_outcome(state, "stage1")
            
            assert status == "completed_failed"
            assert isinstance(summary, str)

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_all_classification_values(self, mock_breakdown, mock_comparisons):
        """Test all classification values map correctly."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        test_cases = [
            ("FAILED", "completed_failed"),
            ("POOR_MATCH", "completed_failed"),
            ("PARTIAL_MATCH", "completed_partial"),
            ("ACCEPTABLE_MATCH", "completed_success"),
            ("EXCELLENT_MATCH", "completed_success"),
            ("NO_TARGETS", "completed_success"),
        ]
        
        for classification, expected_status in test_cases:
            state = {
                "analysis_overall_classification": classification,
                "comparison_verdict": "match",
                "physics_verdict": "pass",
            }
            
            status, summary = _derive_stage_completion_outcome(state, "stage1")
            
            assert status == expected_status, f"Classification {classification} should map to {expected_status}, got {status}"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_unknown_classification_defaults_to_success(self, mock_breakdown, mock_comparisons):
        """Unknown classification should default to completed_success."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "UNKNOWN_CLASSIFICATION",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_success"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_partial_on_needs_revision(self, mock_breakdown, mock_comparisons):
        """Should return completed_partial if comparison_verdict is needs_revision."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "needs_revision",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_partial"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_needs_revision_overrides_success(self, mock_breakdown, mock_comparisons):
        """needs_revision should override success status to partial."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": "needs_revision",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_partial"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_needs_revision_should_not_override_failure(self, mock_breakdown, mock_comparisons):
        """needs_revision should not override failure status - failures are more severe."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "FAILED",
            "comparison_verdict": "needs_revision",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # FAILED classification should result in completed_failed, not be overridden by needs_revision
        # If this test fails, it reveals a bug: needs_revision unconditionally sets status to partial
        # which incorrectly overrides failure status
        assert status == "completed_failed", (
            f"FAILED classification should result in completed_failed, but got {status}. "
            "This indicates needs_revision is incorrectly overriding failure status."
        )

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_partial_on_physics_warning(self, mock_breakdown, mock_comparisons):
        """Should return completed_partial if physics_verdict is warning."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "warning",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_partial"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_physics_warning_only_affects_success(self, mock_breakdown, mock_comparisons):
        """Physics warning should only downgrade success to partial, not override failures."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "FAILED",
            "comparison_verdict": "mismatch",
            "physics_verdict": "warning",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # Should remain failed, not become partial
        assert status == "completed_failed"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_returns_failed_on_physics_fail(self, mock_breakdown, mock_comparisons):
        """Should return completed_failed if physics_verdict is fail."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "fail",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_physics_fail_overrides_success(self, mock_breakdown, mock_comparisons):
        """Physics fail should override success status to failed."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "fail",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_empty_classification_with_physics_fail(self, mock_breakdown, mock_comparisons):
        """Empty classification with physics fail should result in failed."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "",
            "comparison_verdict": "match",
            "physics_verdict": "fail",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_empty_classification_without_physics_fail(self, mock_breakdown, mock_comparisons):
        """Empty classification without physics fail - potential bug: condition checks empty but only handles fail."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # The code checks `classification == ""` but only sets status if physics_verdict == "fail"
        # So empty classification with pass should default to success
        # This test documents the actual behavior - might be a bug
        assert status == "completed_success"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_priority_order_missing_overrides_all(self, mock_breakdown, mock_comparisons):
        """Missing outputs should override everything, including physics fail."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": ["output.csv"], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "fail",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"
        assert "Missing outputs" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_priority_order_pending_overrides_success(self, mock_breakdown, mock_comparisons):
        """Pending should override success but not failure."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": ["fig1"], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_partial"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_combination_needs_revision_and_physics_warning(self, mock_breakdown, mock_comparisons):
        """Test combination of needs_revision and physics warning."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "needs_revision",
            "physics_verdict": "warning",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # Both conditions set to partial, so should be partial
        assert status == "completed_partial"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_combination_physics_fail_and_needs_revision(self, mock_breakdown, mock_comparisons):
        """Test combination of physics fail and needs_revision - fail should win."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "needs_revision",
            "physics_verdict": "fail",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # Physics fail should override needs_revision
        assert status == "completed_failed"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_extracts_summary_from_analysis_summary(self, mock_breakdown, mock_comparisons):
        """Should extract correct summary text from analysis_summary dict."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"notes": "Specific notes", "totals": {"matches": 1}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert summary == "Specific notes"
        assert status == "completed_success"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_extracts_summary_from_totals(self, mock_breakdown, mock_comparisons):
        """Should extract summary from totals if no notes."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"totals": {"matches": 2, "targets": 3}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert summary == "2/3 targets matched"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_summary_from_totals_with_zero_matches(self, mock_breakdown, mock_comparisons):
        """Should handle zero matches in totals."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"totals": {"matches": 0, "targets": 5}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert summary == "0/5 targets matched"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_summary_from_totals_missing_keys(self, mock_breakdown, mock_comparisons):
        """Should handle missing keys in totals."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"totals": {}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert summary == "0/0 targets matched"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_summary_from_string_analysis_summary(self, mock_breakdown, mock_comparisons):
        """Should use string analysis_summary when not a dict."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": "Simple string summary",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert summary == "Simple string summary"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_summary_fallback_when_no_analysis_summary(self, mock_breakdown, mock_comparisons):
        """Should generate fallback summary when analysis_summary is missing."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert isinstance(summary, str)
        assert "ACCEPTABLE_MATCH" in summary or "match" in summary or "OK_CONTINUE" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_summary_fallback_with_no_classification_or_verdict(self, mock_breakdown, mock_comparisons):
        """Should handle fallback when both classification and verdict are missing."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert isinstance(summary, str)
        assert "OK_CONTINUE" in summary or "Stage classified" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_none_stage_id(self, mock_breakdown, mock_comparisons):
        """Should handle None stage_id."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, None)
        
        assert status == "completed_success"
        mock_comparisons.assert_called_once_with(state, None)

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_missing_state_keys(self, mock_breakdown, mock_comparisons):
        """Should handle missing state keys gracefully."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {}
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # Should default to success when everything is missing
        assert status == "completed_success"
        assert isinstance(summary, str)

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_none_values_in_state(self, mock_breakdown, mock_comparisons):
        """Should handle None values in state keys."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": None,
            "comparison_verdict": None,
            "physics_verdict": None,
            "analysis_summary": None,
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_success"
        assert isinstance(summary, str)

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_classification_case_insensitive(self, mock_breakdown, mock_comparisons):
        """Classification should be case-insensitive (converted to uppercase)."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "acceptable_match",  # lowercase
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_success"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_multiple_missing_outputs(self, mock_breakdown, mock_comparisons):
        """Should handle multiple missing outputs in summary."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": ["output1.csv", "output2.csv", "output3.csv"], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_failed"
        assert "output1.csv" in summary
        assert "output2.csv" in summary
        assert "output3.csv" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_multiple_pending_comparisons(self, mock_breakdown, mock_comparisons):
        """Should handle multiple pending comparisons in summary."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": ["fig1", "fig2", "table1"], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_partial"
        assert "fig1" in summary
        assert "fig2" in summary
        assert "table1" in summary

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_analysis_summary_notes_empty_string(self, mock_breakdown, mock_comparisons):
        """Should fall back to totals when notes is empty string."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"notes": "", "totals": {"matches": 3, "targets": 4}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        # Empty string is falsy, so should use totals
        assert summary == "3/4 targets matched"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_analysis_summary_notes_none(self, mock_breakdown, mock_comparisons):
        """Should fall back to totals when notes is None."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"notes": None, "totals": {"matches": 5, "targets": 6}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert summary == "5/6 targets matched"

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_verifies_function_calls(self, mock_breakdown, mock_comparisons):
        """Should verify that helper functions are called correctly."""
        mock_comparisons.return_value = [{"target": "fig1", "status": "match"}]
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": ["fig1"]}
        
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
        }
        
        _derive_stage_completion_outcome(state, "stage1")
        
        mock_comparisons.assert_called_once_with(state, "stage1")
        mock_breakdown.assert_called_once_with([{"target": "fig1", "status": "match"}])

    @patch("src.agents.supervision.supervisor.stage_comparisons_for_stage")
    @patch("src.agents.supervision.supervisor.breakdown_comparison_classifications")
    def test_complete_success_case(self, mock_breakdown, mock_comparisons):
        """Test complete success case with all conditions met."""
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": [], "match": []}
        
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": "match",
            "physics_verdict": "pass",
            "analysis_summary": {"notes": "Perfect match!", "totals": {"matches": 5, "targets": 5}},
        }
        
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        
        assert status == "completed_success"
        assert summary == "Perfect match!"
