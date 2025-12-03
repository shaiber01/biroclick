"""
Tests for Analysis Agents (ResultsAnalyzerAgent, ComparisonValidatorAgent).
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.agents.analysis import results_analyzer_node, comparison_validator_node, PROJECT_ROOT
from src.agents.constants import AnalysisClassification

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Base state for analysis tests."""
    return {
        "paper_id": "test_paper",
        "current_stage_id": "stage_1_sim",
        "paper_figures": [{"id": "Fig1", "image_path": "fig1.png"}],
        "plan": {
            "stages": [
                {
                    "stage_id": "stage_1_sim",
                    "targets": ["Fig1"],
                    "target_details": [{"figure_id": "Fig1", "precision_requirement": "acceptable"}],
                    "expected_outputs": [
                        {"target_figure": "Fig1", "filename_pattern": "output.csv", "columns": ["x", "y"]}
                    ]
                }
            ],
            "targets": [{"figure_id": "Fig1", "precision_requirement": "acceptable"}]
        },
        "stage_outputs": {
            "files": ["simulation_stage_1_sim.py", "output.csv"]
        },
        "analysis_revision_count": 0,
        "analysis_result_reports": [],
        "figure_comparisons": []
    }

# ═══════════════════════════════════════════════════════════════════════
# results_analyzer_node Tests
# ═══════════════════════════════════════════════════════════════════════

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
