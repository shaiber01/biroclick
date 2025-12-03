"""Tests for _derive_stage_completion_outcome helper."""

from unittest.mock import patch

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
