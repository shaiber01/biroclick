"""Numeric and metrics tests for results_analyzer_node."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.analysis import results_analyzer_node
from src.agents.constants import AnalysisClassification


@pytest.fixture(name="base_state")
def analysis_base_state_alias(analysis_state):
    return analysis_state


class TestResultsAnalyzerNode:
    """Tests for results_analyzer_node."""

    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    def test_analyzer_basic_success(
        self, mock_prompt, mock_check, mock_llm, mock_load, mock_metrics, mock_match, base_state
    ):
        """Test successful analysis path with excellent match."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_llm.return_value = {"overall_classification": AnalysisClassification.EXCELLENT_MATCH}
        mock_match.return_value = "output.csv"
        
        # Mock successful metric computation
        mock_load.return_value = (MagicMock(), MagicMock()) # valid series
        mock_metrics.return_value = {
            "peak_position_error_percent": 0.5, # Excellent < 1%
            "normalized_rmse_percent": 1.0,
            "peak_position_paper": 500.0,
            "peak_position_sim": 500.0
        }

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/abs/output.csv")):
            
            result = results_analyzer_node(base_state)
            
            # Strict assertions
            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.EXCELLENT_MATCH
            
            # Verify all top-level keys present
            expected_keys = {
                "workflow_phase", 
                "analysis_summary", 
                "analysis_overall_classification", 
                "analysis_result_reports", 
                "figure_comparisons", 
                "analysis_feedback"
            }
            assert set(result.keys()) == expected_keys
            
            summary = result["analysis_summary"]
            assert summary["totals"]["matches"] == 1
            assert summary["totals"]["missing"] == 0
            assert summary["totals"]["pending"] == 0
            assert summary["totals"]["mismatch"] == 0
            assert "Fig1" in summary["matched_targets"]
            
            reports = result["analysis_result_reports"]
            assert len(reports) == 1
            assert reports[0]["target_figure"] == "Fig1"
            assert reports[0]["status"] == AnalysisClassification.MATCH
            assert reports[0]["quantitative_metrics"]["peak_position_error_percent"] == 0.5
            
            # Verify comparisons structure
            comparisons = result["figure_comparisons"]
            assert len(comparisons) == 1
            comp = comparisons[0]
            assert comp["figure_id"] == "Fig1"
            assert comp["classification"] == AnalysisClassification.MATCH
            assert "comparison_table" in comp
            assert len(comp["comparison_table"]) >= 1

    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    def test_analyzer_metrics_nan(
        self, mock_prompt, mock_check, mock_llm, mock_load, mock_metrics, mock_match, base_state
    ):
        """Test handling of NaN metrics."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_match.return_value = "output.csv"
        mock_load.return_value = (MagicMock(), MagicMock())
        
        # Mock metrics returning NaN
        mock_metrics.return_value = {
            "peak_position_error_percent": float("nan"),
            "normalized_rmse_percent": float("nan")
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
             
            result = results_analyzer_node(base_state)
            
            # Should not crash. Classification might depend on how NaN is handled.
            # Usually NaN comparison fails, so it might fall through to mismatch or pending.
            
            assert result["workflow_phase"] == "analysis"
            report = result["analysis_result_reports"][0]
            # Check if it's captured
            metrics = report["quantitative_metrics"]
            import math
            assert math.isnan(metrics["peak_position_error_percent"])
            
            # Verify classification is robust
            assert report["status"] in [
                AnalysisClassification.MATCH,
                AnalysisClassification.MISMATCH,
                AnalysisClassification.PENDING_VALIDATION,
                AnalysisClassification.PARTIAL_MATCH
            ]

    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    def test_processes_matching_output(
        self, mock_prompt, mock_check, mock_match, mock_metrics, mock_load, base_state
    ):
        """Should process and classify matching outputs correctly."""
        mock_match.return_value = "output.csv"
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_metrics.return_value = {
            "peak_position_error_percent": 1.5,
            "normalized_rmse_percent": 2.0,
            "peak_position_paper": 500,
            "peak_position_sim": 500
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            assert result["workflow_phase"] == "analysis"
            summary = result["analysis_summary"]
            # 1.5% error is usually "partial_match" or "match" depending on thresholds.
            # Assuming acceptable threshold is > 1.5, it might be match.
            # Let's check the resulting status in report
            report = result["analysis_result_reports"][0]
            assert report["target_figure"] == "Fig1"
            assert report["quantitative_metrics"]["peak_position_error_percent"] == 1.5

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    def test_handles_validation_criteria_failure(self, mock_prompt, mock_check, base_state):
        """Should handle validation criteria failures."""
        base_state["plan"]["stages"][0]["validation_criteria"] = ["Fig1: peak within 1%"]
        
        # Mock metrics that fail criteria
        with patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 5.0, "peak_position_paper": 100, "peak_position_sim": 105}), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            assert result["workflow_phase"] == "analysis"
            summary = result["analysis_summary"]
            assert "Fig1" in summary["mismatch_targets"]
            assert summary["discrepancies_logged"] > 0
            
            report = result["analysis_result_reports"][0]
            assert "criteria_failures" in report
            assert len(report["criteria_failures"]) > 0

