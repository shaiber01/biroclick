"""Tests for generate_report_node."""

import copy
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.reporting import generate_report_node


class TestGenerateReportNode:
    """Tests for generate_report_node function."""

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_generates_report_on_success(self, mock_prompt, mock_call):
        """Should generate final report with all expected fields."""
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "Pass"}]},
            "key_findings": ["Finding 1"],
            "recommendations": ["Recommendation 1"],
            "paper_citation": {"title": "Generated Title"},
            "assumptions": {"assumed": "true"},
            "figure_comparisons": [{"fig": "1"}],
            "systematic_discrepancies": ["disc1"],
            "conclusions": ["Conclusion 1"]
        }
        
        state = {
            "paper_id": "test_paper",
            "progress": {"stages": [{"stage_id": "stage1", "status": "completed_success"}]},
            "metrics": {"agent_calls": []},
            # Pre-existing citation should be overwritten if agent returns one? 
            # The code says: 
            # if agent_output.get("paper_citation"): result["paper_citation"] = agent_output["paper_citation"]
            "paper_citation": {"title": "Original Title"} 
        }
        
        result = generate_report_node(state)
        
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Test"
        assert result["paper_citation"]["title"] == "Generated Title" # Verifies agent output overwrites/updates
        assert result["assumptions"] == {"assumed": "true"}
        assert result["figure_comparisons"] == [{"fig": "1"}]
        assert result["systematic_discrepancies_identified"] == ["disc1"] # Check mapping
        assert result["report_conclusions"] == ["Conclusion 1"] # Check mapping

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_constructs_user_content_correctly(self, mock_prompt, mock_call):
        """Should verify user content includes all state components."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "paper_title": "Test Paper",
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "summary": "Done well"}
                ]
            },
            "figure_comparisons": [{"fig_id": "fig1", "status": "match"}],
            "assumptions": {"temp": "300K"},
            "discrepancies": [
                {"parameter": "gap", "classification": "minor", "likely_cause": "noise"}
            ]
        }
        
        generate_report_node(state)
        
        # Verify call arguments
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        assert "Paper ID: test_paper" in user_content
        assert "stage1: completed_success - Done well" in user_content
        assert '"fig_id": "fig1"' in user_content
        assert '"temp": "300K"' in user_content
        assert "- gap: minor - noise" in user_content

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_llm_error(self, mock_prompt, mock_call):
        """Should handle LLM call failure gracefully and return partial results."""
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "paper_id": "test_paper",
            "completed_stages": ["stage1"],
            "paper_title": "Fallback Title"
        }
        
        result = generate_report_node(state)
        
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True
        # Should have defaults
        assert result["paper_citation"]["title"] == "Fallback Title"
        assert "executive_summary" in result
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Material Properties"

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_calculates_metrics_cost(self, mock_prompt, mock_call):
        """Should correctly calculate token costs in metrics summary."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        # Cost formula in code: (input * 3.0 + output * 15.0) / 1_000_000
        state = {
            "paper_id": "test_paper",
            "metrics": {
                "agent_calls": [
                    {"input_tokens": 1000, "output_tokens": 100}, # 3000 + 1500 = 4500
                    {"input_tokens": 2000, "output_tokens": 200}, # 6000 + 3000 = 9000
                ]
            }
        }
        # Total input: 3000, Total output: 300
        # Total cost = (3000 * 3 + 300 * 15) / 1M = (9000 + 4500) / 1M = 0.0135
        
        result = generate_report_node(state)
        
        metrics = result["metrics"]
        summary = metrics["token_summary"]
        
        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 300
        assert summary["estimated_cost"] == pytest.approx(0.0135)

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_builds_quantitative_summary(self, mock_prompt, mock_call):
        """Should build quantitative summary table from reports."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "analysis_result_reports": [
                {
                    "stage_id": "stage1",
                    "target_figure": "Fig1",
                    "status": "match",
                    "precision_requirement": "acceptable",
                    "quantitative_metrics": {
                        "peak_position_error_percent": 1.5,
                        "normalized_rmse_percent": 2.0,
                        "correlation": 0.95,
                        "n_points_compared": 100
                    }
                },
                {
                    # Partial data
                    "stage_id": "stage2",
                    "status": "fail",
                    "quantitative_metrics": None
                }
            ]
        }
        
        result = generate_report_node(state)
        
        summary = result.get("quantitative_summary", [])
        assert len(summary) == 2
        
        row1 = summary[0]
        assert row1["figure_id"] == "Fig1"
        assert row1["peak_position_error_percent"] == 1.5
        assert row1["correlation"] == 0.95
        
        row2 = summary[1]
        assert row2["stage_id"] == "stage2"
        assert row2["peak_position_error_percent"] is None # Should handle missing metrics

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_missing_metrics(self, mock_prompt, mock_call):
        """Should handle missing metrics in state gracefully."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {"paper_id": "test_paper"} # No metrics key
        
        result = generate_report_node(state)
        
        assert "metrics" in result
        assert result["metrics"]["token_summary"]["total_input_tokens"] == 0
