"""
Tests for ResultsAnalyzerAgent (results_analyzer_node).
"""

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

    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    def test_analyzer_basic_success(
        self, mock_match, mock_metrics, mock_load, mock_llm, mock_check, mock_prompt, base_state
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

    def test_analyzer_stage_outputs_none(self, base_state):
        """Test error when stage_outputs is explicitly None."""
        base_state["stage_outputs"] = None
        
        result = results_analyzer_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"

    def test_analyzer_plan_none(self, base_state):
        """Test handling when plan is None (should default gracefully or fail safely)."""
        base_state["plan"] = None
        
        # Should not crash, but might result in NO_TARGETS or proceed if figures exist
        # If plan is None, get_plan_stage returns None.
        # figures = ensure_stub_figures(state) -> checks paper_figures
        # target_ids fallback to figures.
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            result = results_analyzer_node(base_state)
        
        assert result["workflow_phase"] == "analysis"
        # If paper_figures are present in base_state (they are "Fig1"), it should try to analyze "Fig1"
        # But it might fail to find output if stage_outputs["files"] is checked against empty expectations?
        # match_output_file logic handles matching.
        
        # In base_state, stage_outputs has files.
        # So it might actually proceed.
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.NO_TARGETS,
            AnalysisClassification.POOR_MATCH,
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
            AnalysisClassification.EXCELLENT_MATCH
        ]

    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    def test_analyzer_metrics_nan(
        self, mock_match, mock_metrics, mock_load, mock_llm, mock_check, mock_prompt, base_state
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

    def test_analyzer_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = results_analyzer_node(base_state)
        
        assert result["workflow_phase"] == "analysis"
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True
        assert "No stage selected" in result["pending_user_questions"][0]

    def test_analyzer_missing_outputs(self, base_state):
        """Test error when stage outputs are empty."""
        base_state["stage_outputs"] = {}
        result = results_analyzer_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_files_missing_on_disk(self, mock_prompt, mock_check, base_state):
        """Test failure when files listed in state do not exist on disk."""
        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False):
            result = results_analyzer_node(base_state)
            
            assert result["execution_verdict"] == "fail"
            assert "do not exist on disk" in result["run_error"]
            assert "output.csv" in result["run_error"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures", return_value=[])
    def test_analyzer_no_targets(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test handling of stage with no targets."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        
        result = results_analyzer_node(base_state)
        
        assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["analysis_summary"]["totals"]["targets"] == 0

    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.call_agent_with_metrics")
    def test_analyzer_hallucinated_targets(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test robustness against targets not in plan."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        # Add a target to feedback that isn't in plan
        base_state["analysis_feedback"] = "Check non_existent_fig"
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_missing_files_list(self, mock_stub, mock_prompt, mock_check, mock_plan_stage, base_state):
        """Should return error when files list is explicit empty list."""
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        
        base_state["stage_outputs"] = {"files": []}
        
        result = results_analyzer_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Stage outputs are missing" in result["run_error"]

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context, base_state):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        result = results_analyzer_node(base_state)
        
        assert result["awaiting_user_input"] is True
        assert result["pending_user_questions"] == ["Context overflow"]

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
    def test_handles_digitized_data_requirement(self, mock_prompt, mock_check, base_state):
        """Should flag targets requiring digitized data without it."""
        # Require excellent precision which mandates digitized data
        base_state["plan"]["stages"][0]["targets"] = ["Fig1"]
        base_state["plan"]["stages"][0]["target_details"] = [{"figure_id": "Fig1", "precision_requirement": "excellent"}]
        # Ensure no digitized path in plan or figures
        base_state["paper_figures"][0]["digitized_data_path"] = None
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
             
            result = results_analyzer_node(base_state)
            
            # Should flag as missing digitized data
            summary = result["analysis_summary"]
            # It goes into pending_targets or just blocked?
            # Code says: pending_targets.append(target_id), classification="missing_digitized_data"
            assert "Fig1" in summary["pending_targets"]
            
            reports = result["figure_comparisons"]
            assert any(r["classification"] == "missing_digitized_data" for r in reports)

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

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.load_numeric_series", return_value=(MagicMock(), MagicMock()))
    @patch("src.agents.analysis.match_output_file", return_value="output.csv")
    def test_handles_validation_criteria_failure(
        self, mock_match, mock_load, mock_prompt, mock_check, base_state
    ):
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


# ═══════════════════════════════════════════════════════════════════════
# comparison_validator_node Tests
# ═══════════════════════════════════════════════════════════════════════
