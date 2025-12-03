"""Prompt/LLM tests for results_analyzer_node."""

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

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_calls_llm_for_visual_analysis(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should call LLM for visual comparison when images available."""
        mock_images.return_value = ["fig1.png"]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Visual analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            mock_call.assert_called_once()
            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.ACCEPTABLE_MATCH

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_llm_error_gracefully(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM error and use quantitative results only."""
        mock_images.return_value = ["fig1.png"]
        mock_call.side_effect = Exception("API error")
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            assert result["workflow_phase"] == "analysis"
            # Should fallback to quantitative classification (which defaults to ACCEPTABLE if matched)
            # If metrics are empty, classification might be "missing_output" if match failed?
            # Here match succeeded, metrics empty -> classification might be 'pending_validation' (no ref) or 'match' if qualitative
            # Assuming qualitative because precision is 'acceptable' and no ref data in mock
            assert result["analysis_overall_classification"] in [
                AnalysisClassification.ACCEPTABLE_MATCH,
                AnalysisClassification.PARTIAL_MATCH,
                AnalysisClassification.EXCELLENT_MATCH
            ]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=None)
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_missing_reference_image(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check, base_state
    ):
        """Should warn but proceed when reference image is missing."""
        base_state["paper_figures"][0]["image_path"] = "missing_fig.png"
        
        # Mock path behavior: output exists, image missing
        def path_exists_side_effect(self):
            path_str = str(self)
            if "missing_fig.png" in path_str:
                return False
            return True

        with patch("pathlib.Path.exists", side_effect=path_exists_side_effect, autospec=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            assert result["workflow_phase"] == "analysis"
            comp = result["figure_comparisons"][0]
            assert comp["paper_image_path"] is None

