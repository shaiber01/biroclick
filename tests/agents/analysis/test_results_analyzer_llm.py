"""Prompt/LLM tests for results_analyzer_node."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

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
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Visual analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify LLM was called exactly once
            mock_call.assert_called_once()
            
            # Verify call arguments are correct
            call_args = mock_call.call_args
            assert call_args.kwargs["agent_name"] == "results_analyzer"
            assert call_args.kwargs["system_prompt"] == "prompt"
            assert call_args.kwargs["user_content"] is not None
            assert "QUANTITATIVE ANALYSIS RESULTS" in call_args.kwargs["user_content"]
            assert call_args.kwargs["images"] == [Path("fig1.png")]
            assert call_args.kwargs["state"] == base_state
            
            # Verify result structure
            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.ACCEPTABLE_MATCH
            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict)
            assert result["analysis_summary"].get("llm_qualitative_analysis") == "Visual analysis complete"
            
            # Verify all required fields are present
            assert "analysis_result_reports" in result
            assert "figure_comparisons" in result
            assert "analysis_feedback" in result

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={
        "peak_position_error_percent": 0.5,
        "normalized_rmse_percent": 1.0
    })
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_overrides_quantitative_classification(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should allow LLM to override quantitative classification."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        # LLM returns different classification than quantitative would suggest
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.POOR_MATCH,
            "summary": "Visual inspection reveals significant discrepancies",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify LLM classification overrides quantitative
            assert result["analysis_overall_classification"] == AnalysisClassification.POOR_MATCH
            assert result["analysis_summary"]["llm_qualitative_analysis"] == "Visual inspection reveals significant discrepancies"
            
            # Verify quantitative metrics are still present
            assert len(result["analysis_result_reports"]) > 0
            assert "quantitative_metrics" in result["analysis_result_reports"][0]

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_merges_figure_comparisons(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should merge LLM figure comparisons into existing comparisons."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "shape_comparison": ["Peak matches well", "Minor shape differences"],
                    "reason_for_difference": "Slight numerical differences"
                }
            ]
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify figure comparison was merged
            comparisons = result["figure_comparisons"]
            assert len(comparisons) > 0
            
            fig1_comp = next((c for c in comparisons if c.get("figure_id") == "Fig1"), None)
            assert fig1_comp is not None
            assert fig1_comp["shape_comparison"] == ["Peak matches well", "Minor shape differences"]
            assert fig1_comp["reason_for_difference"] == "Slight numerical differences"
            
            # Verify other fields are preserved
            assert "classification" in fig1_comp
            assert "comparison_table" in fig1_comp

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_handles_partial_figure_comparisons(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM returning comparisons for only some figures."""
        # Add multiple targets
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        mock_images.return_value = [Path("fig1.png"), Path("fig2.png")]
        mock_user_content.return_value = "Analysis content"
        # LLM only returns comparison for Fig1
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "shape_comparison": ["Good match"],
                    "reason_for_difference": ""
                }
            ]
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify both figures have comparisons
            comparisons = result["figure_comparisons"]
            assert len(comparisons) >= 2
            
            # Fig1 should have LLM comparison merged
            fig1_comp = next((c for c in comparisons if c.get("figure_id") == "Fig1"), None)
            assert fig1_comp is not None
            assert fig1_comp["shape_comparison"] == ["Good match"]
            
            # Fig2 should still exist but without LLM comparison
            fig2_comp = next((c for c in comparisons if c.get("figure_id") == "Fig2"), None)
            assert fig2_comp is not None
            assert fig2_comp.get("shape_comparison") == [] or "shape_comparison" not in fig2_comp

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
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.side_effect = Exception("API error")
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify LLM was attempted
            mock_call.assert_called_once()
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            
            # Verify quantitative results are still present
            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict)
            assert "analysis_result_reports" in result
            assert len(result["analysis_result_reports"]) > 0
            
            # Verify LLM summary is NOT present (error occurred)
            assert "llm_qualitative_analysis" not in result["analysis_summary"]
            
            # Verify figure comparisons exist but without LLM data
            assert "figure_comparisons" in result
            assert len(result["figure_comparisons"]) > 0
            # Shape comparisons should be empty (no LLM data)
            for comp in result["figure_comparisons"]:
                assert comp.get("shape_comparison") == [] or "shape_comparison" not in comp

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_llm_returning_none(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM returning None gracefully."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = None
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            
            # Verify quantitative results are still present
            assert "analysis_summary" in result
            assert "llm_qualitative_analysis" not in result["analysis_summary"]

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_llm_malformed_response(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle malformed LLM response without crashing."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        # Missing required fields
        mock_call.return_value = {
            "summary": "Some text"
            # Missing overall_classification and figure_comparisons
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            
            # Verify summary is added if present
            if "summary" in mock_call.return_value:
                assert result["analysis_summary"].get("llm_qualitative_analysis") == "Some text"

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_skips_llm_when_no_images(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should skip LLM call when no images are available."""
        mock_images.return_value = []
        mock_user_content.return_value = "Analysis content"
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify LLM was NOT called
            mock_call.assert_not_called()
            
            # Verify workflow continues with quantitative only
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            assert "llm_qualitative_analysis" not in result["analysis_summary"]

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_skips_llm_when_no_figure_comparisons(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should skip LLM call when no figure comparisons exist."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        
        # Remove targets so no comparisons are created
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        
        with patch("src.agents.analysis.ensure_stub_figures", return_value=[]), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify LLM was NOT called (no figure_comparisons to analyze)
            mock_call.assert_not_called()
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_limits_images_to_10(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should limit images passed to LLM to maximum of 10."""
        # Create 15 images
        mock_images.return_value = [Path(f"fig{i}.png") for i in range(15)]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify only 10 images were passed
            call_args = mock_call.call_args
            assert len(call_args.kwargs["images"]) == 10
            assert call_args.kwargs["images"] == [Path(f"fig{i}.png") for i in range(10)]

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={
        "peak_position_error_percent": 2.0,
        "normalized_rmse_percent": 3.0
    })
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_user_content_includes_quantitative_results(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should include quantitative results in user content passed to LLM."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Base content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify user content includes quantitative results
            call_args = mock_call.call_args
            user_content = call_args.kwargs["user_content"]
            
            assert "QUANTITATIVE ANALYSIS RESULTS" in user_content
            assert "Overall:" in user_content
            assert "Matched:" in user_content
            assert "Pending:" in user_content
            assert "Missing:" in user_content
            assert "Mismatch:" in user_content
            
            # Verify base content is included
            assert "Base content" in user_content

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_missing_reference_image(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should warn but proceed when reference image is missing."""
        base_state["paper_figures"][0]["image_path"] = "missing_fig.png"
        
        # Mock LLM functions
        mock_images.return_value = []  # No images since reference is missing
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": []
        }
        
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
            
            # Verify workflow continues normally
            assert "analysis_overall_classification" in result
            assert "analysis_summary" in result

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_figure_id_mismatch_handled(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM returning comparison for non-existent figure_id."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        # LLM returns comparison for figure that doesn't exist
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": [
                {
                    "figure_id": "NonExistentFig",
                    "shape_comparison": ["Some analysis"],
                    "reason_for_difference": "Unknown"
                }
            ]
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues without crashing
            assert result["workflow_phase"] == "analysis"
            
            # Verify the non-existent figure comparison is not merged
            comparisons = result["figure_comparisons"]
            non_existent = next((c for c in comparisons if c.get("figure_id") == "NonExistentFig"), None)
            assert non_existent is None
            
            # Verify existing comparisons are still present
            assert len(comparisons) > 0

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_empty_figure_comparisons_list(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM returning empty figure_comparisons list."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": []  # Empty list
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.ACCEPTABLE_MATCH
            
            # Verify figure comparisons still exist (from quantitative analysis)
            assert len(result["figure_comparisons"]) > 0
            
            # Verify no LLM data was merged (empty list)
            for comp in result["figure_comparisons"]:
                assert comp.get("shape_comparison") == [] or "shape_comparison" not in comp

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_missing_figure_comparisons_key(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM response missing figure_comparisons key."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete"
            # Missing figure_comparisons key
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues without crashing
            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.ACCEPTABLE_MATCH
            
            # Verify figure comparisons still exist
            assert len(result["figure_comparisons"]) > 0

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_preserves_existing_comparisons_from_other_stages(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should preserve figure comparisons from other stages."""
        # Add existing comparisons from different stage
        base_state["figure_comparisons"] = [
            {
                "figure_id": "FigOther",
                "stage_id": "other_stage",
                "classification": AnalysisClassification.MATCH
            }
        ]
        
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify existing comparisons are preserved
            comparisons = result["figure_comparisons"]
            other_stage_comp = next((c for c in comparisons if c.get("stage_id") == "other_stage"), None)
            assert other_stage_comp is not None
            assert other_stage_comp["figure_id"] == "FigOther"
            
            # Verify current stage comparisons are also present
            current_stage_id = base_state["current_stage_id"]
            current_stage_comps = [c for c in comparisons if c.get("stage_id") == current_stage_id]
            assert len(current_stage_comps) > 0

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_get_images_returning_none(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle get_images_for_analyzer returning None gracefully."""
        mock_images.return_value = None
        mock_user_content.return_value = "Analysis content"
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify LLM was NOT called (None is falsy)
            mock_call.assert_not_called()
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_llm_figure_comparisons_not_list(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM returning figure_comparisons that is not a list."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": "not a list"  # Wrong type
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            # Should not crash - should handle gracefully
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            
            # Verify figure comparisons are NOT corrupted by bad LLM response
            # The bad response should be ignored, not merged
            comparisons = result["figure_comparisons"]
            assert len(comparisons) > 0
            # All comparisons should have proper structure (not corrupted by iterating over string)
            for comp in comparisons:
                assert isinstance(comp, dict)
                assert "figure_id" in comp
                # shape_comparison should be a list or not present, not a string character
                if "shape_comparison" in comp:
                    assert isinstance(comp["shape_comparison"], list) or comp["shape_comparison"] == []

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_llm_figure_comparison_missing_figure_id(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should handle LLM figure comparison missing figure_id field."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": [
                {
                    # Missing figure_id
                    "shape_comparison": ["Some analysis"],
                    "reason_for_difference": "Unknown"
                }
            ]
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify workflow continues without crashing
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            
            # Verify the comparison without figure_id is not merged
            comparisons = result["figure_comparisons"]
            # All comparisons should have figure_id
            for comp in comparisons:
                assert "figure_id" in comp

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_classification_only_overrides_when_present(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should only override classification if LLM provides overall_classification."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        # LLM response without overall_classification
        mock_call.return_value = {
            "summary": "Analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify quantitative classification is used (not overridden)
            assert "analysis_overall_classification" in result
            # Should be based on quantitative metrics, not LLM
            assert result["analysis_overall_classification"] in [
                AnalysisClassification.ACCEPTABLE_MATCH,
                AnalysisClassification.PARTIAL_MATCH,
                AnalysisClassification.EXCELLENT_MATCH,
                AnalysisClassification.MATCH,
                AnalysisClassification.PENDING_VALIDATION
            ]
            
            # Verify summary is still added
            assert result["analysis_summary"].get("llm_qualitative_analysis") == "Analysis complete"

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_summary_only_added_when_present(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should only add summary to analysis_summary if LLM provides it."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        # LLM response without summary
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify classification is set
            assert result["analysis_overall_classification"] == AnalysisClassification.ACCEPTABLE_MATCH
            
            # Verify summary is NOT added if not present
            assert "llm_qualitative_analysis" not in result["analysis_summary"]

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_merges_multiple_comparisons_correctly(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should correctly merge multiple LLM figure comparisons."""
        # Add multiple targets
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2", "Fig3"]
        mock_images.return_value = [Path("fig1.png"), Path("fig2.png"), Path("fig3.png")]
        mock_user_content.return_value = "Analysis content"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "shape_comparison": ["Match 1"],
                    "reason_for_difference": "Reason 1"
                },
                {
                    "figure_id": "Fig2",
                    "shape_comparison": ["Match 2"],
                    "reason_for_difference": "Reason 2"
                },
                {
                    "figure_id": "Fig3",
                    "shape_comparison": ["Match 3"],
                    "reason_for_difference": "Reason 3"
                }
            ]
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify all three comparisons were merged
            comparisons = result["figure_comparisons"]
            assert len(comparisons) >= 3
            
            fig1_comp = next((c for c in comparisons if c.get("figure_id") == "Fig1"), None)
            assert fig1_comp is not None
            assert fig1_comp["shape_comparison"] == ["Match 1"]
            assert fig1_comp["reason_for_difference"] == "Reason 1"
            
            fig2_comp = next((c for c in comparisons if c.get("figure_id") == "Fig2"), None)
            assert fig2_comp is not None
            assert fig2_comp["shape_comparison"] == ["Match 2"]
            assert fig2_comp["reason_for_difference"] == "Reason 2"
            
            fig3_comp = next((c for c in comparisons if c.get("figure_id") == "Fig3"), None)
            assert fig3_comp is not None
            assert fig3_comp["shape_comparison"] == ["Match 3"]
            assert fig3_comp["reason_for_difference"] == "Reason 3"

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.quantitative_curve_metrics", return_value={})
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_llm_call_uses_correct_system_prompt(
        self, mock_match, mock_metrics, mock_load, mock_prompt, mock_check,
        mock_user_content, mock_images, mock_call, base_state
    ):
        """Should use the system prompt built for results_analyzer."""
        mock_images.return_value = [Path("fig1.png")]
        mock_user_content.return_value = "Analysis content"
        mock_prompt.return_value = "Custom system prompt for analyzer"
        mock_call.return_value = {
            "overall_classification": AnalysisClassification.ACCEPTABLE_MATCH,
            "summary": "Analysis complete",
            "figure_comparisons": []
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            
            # Verify system prompt was built and used
            mock_prompt.assert_called_once()
            call_args = mock_prompt.call_args
            assert call_args[0][0] == "results_analyzer"  # agent_name
            assert call_args[0][1] == base_state  # state
            
            # Verify LLM was called with the correct prompt
            llm_call_args = mock_call.call_args
            assert llm_call_args.kwargs["system_prompt"] == "Custom system prompt for analyzer"

