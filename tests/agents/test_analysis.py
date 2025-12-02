"""Unit tests for src/agents/analysis.py"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents.analysis import (
    results_analyzer_node,
    comparison_validator_node,
)


class TestResultsAnalyzerNode:
    """Tests for results_analyzer_node function."""

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_errors_on_missing_stage_id(self, mock_context):
        """Should return error when current_stage_id is None."""
        mock_context.return_value = None
        
        state = {
            "current_stage_id": None,
            "stage_outputs": {"files": ["output.csv"]},
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True

    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_no_targets(self, mock_stub, mock_prompt, mock_context, mock_plan_stage):
        """Should skip analysis when stage has no targets."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = []
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": []}
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": ["output.csv"]},
            "plan": {"stages": []},
        }
        
        result = results_analyzer_node(state)
        
        assert result["analysis_overall_classification"] == "NO_TARGETS"
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_empty_stage_outputs(self, mock_stub, mock_prompt, mock_context, mock_plan_stage):
        """Should return error when stage_outputs is empty."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},
            "plan": {"stages": []},
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "missing" in result.get("run_error", "").lower()

    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_missing_files_list(self, mock_stub, mock_prompt, mock_context, mock_plan_stage):
        """Should return error when files list is empty."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": []},
            "plan": {"stages": []},
        }
        
        result = results_analyzer_node(state)
        
        assert result["execution_verdict"] == "fail"

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "stage_outputs": {}}
        
        result = results_analyzer_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.analysis.Path")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    def test_processes_matching_output(
        self, mock_metrics, mock_load, mock_stub, mock_prompt, mock_context, mock_plan_stage, mock_path
    ):
        """Should process and classify matching outputs."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1", "description": "Test figure"}]
        mock_plan_stage.return_value = {
            "stage_id": "stage1",
            "targets": ["Fig1"],
            "target_details": [{"figure_id": "Fig1", "precision_requirement": "acceptable"}],
        }
        mock_load.return_value = {"x": [1, 2, 3], "y": [1, 2, 3]}
        mock_metrics.return_value = {
            "peak_position_error_percent": 1.5,
            "normalized_rmse_percent": 2.0,
        }
        
        # Mock path to simulate existing file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.is_file.return_value = True
        mock_file.resolve.return_value = "/outputs/stage1/fig1_output.csv"
        mock_path.return_value = mock_file
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["fig1_output.csv"]},
            "plan": {"stages": [], "targets": []},
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result

    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_digitized_data_requirement(self, mock_stub, mock_prompt, mock_context, mock_plan_stage):
        """Should flag targets requiring digitized data without it."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {
            "stage_id": "stage1",
            "targets": ["Fig1"],
            "target_details": [{"figure_id": "Fig1", "precision_requirement": "excellent"}],  # Requires digitized
        }
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["output.csv"]},
            "plan": {"stages": [], "targets": [{"figure_id": "Fig1", "precision_requirement": "excellent"}]},
        }
        
        result = results_analyzer_node(state)
        
        # Should flag the missing digitized data
        assert result["workflow_phase"] == "analysis"
        # Missing digitized data should result in pending or error
        summary = result.get("analysis_summary", {})
        if isinstance(summary, dict):
            assert summary.get("totals", {}).get("pending", 0) > 0 or "pending" in str(summary).lower()

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.Path")
    def test_calls_llm_for_visual_analysis(
        self, mock_path_cls, mock_match, mock_metrics, mock_load, mock_stub, mock_prompt,
        mock_context, mock_plan_stage, mock_user_content, mock_images, mock_call, validated_results_analyzer_response
    ):
        """Should call LLM for visual comparison when images available (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1", "image_path": "/fig1.png"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        mock_load.return_value = None
        mock_metrics.return_value = {}
        mock_match.return_value = "/outputs/output.csv"
        mock_images.return_value = ["/fig1.png", "/output.png"]
        mock_user_content.return_value = "Analysis content"
        
        mock_response = validated_results_analyzer_response.copy()
        mock_response["overall_classification"] = "ACCEPTABLE_MATCH"
        mock_call.return_value = mock_response
        
        # Mock Path to simulate existing files
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.is_file.return_value = True
        mock_file.is_absolute.return_value = True
        mock_file.resolve.return_value = "/outputs/output.csv"
        mock_file.__truediv__ = MagicMock(return_value=mock_file)
        mock_path_cls.return_value = mock_file
        mock_path_cls.__truediv__ = MagicMock(return_value=mock_file)
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["/outputs/output.csv"]},  # Use absolute path
            "plan": {"stages": [], "targets": []},
        }
        
        result = results_analyzer_node(state)
        
        mock_call.assert_called_once()
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.get_images_for_analyzer")
    @patch("src.agents.analysis.build_user_content_for_analyzer")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    def test_handles_llm_error_gracefully(
        self, mock_match, mock_metrics, mock_load, mock_stub, mock_prompt,
        mock_context, mock_plan_stage, mock_user_content, mock_images, mock_call
    ):
        """Should handle LLM error and use quantitative results only."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        mock_load.return_value = None
        mock_metrics.return_value = {}
        mock_match.return_value = "output.csv"
        mock_images.return_value = ["/fig1.png"]
        mock_user_content.return_value = "Content"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["output.csv"]},
            "plan": {"stages": [], "targets": []},
        }
        
        result = results_analyzer_node(state)
        
        # Should still return analysis results despite LLM failure
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.Path")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_missing_output_files_on_disk(
        self, mock_stub, mock_prompt, mock_context, mock_plan_stage, mock_path
    ):
        """Should fail when output files in state are missing from disk."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        
        # Mock path to simulate MISSING file
        mock_file = MagicMock()
        mock_file.exists.return_value = False # File missing
        mock_file.is_absolute.return_value = False
        mock_file.__str__.return_value = "missing.csv"
        mock_path.return_value = mock_file
        mock_path.return_value.__truediv__.return_value = mock_file # Handle path joins
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["missing.csv"]},
            "plan": {"stages": []},
        }
        
        result = results_analyzer_node(state)
        
        assert result["execution_verdict"] == "fail"
        assert "do not exist on disk" in result["run_error"]

    @patch("src.agents.analysis.Path")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    def test_handles_missing_reference_image(
        self, mock_match, mock_metrics, mock_load, mock_stub, mock_prompt,
        mock_context, mock_plan_stage, mock_path
    ):
        """Should warn but proceed when reference image is missing."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        # Figure has image path
        mock_stub.return_value = [{"id": "Fig1", "image_path": "missing_fig.png"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        mock_load.return_value = None
        mock_metrics.return_value = {}
        mock_match.return_value = "output.csv"
        
        # Mock path behavior:
        # 1. Output file exists
        # 2. Image file missing
        def path_side_effect(path_str):
            m = MagicMock()
            if "output.csv" in str(path_str):
                m.exists.return_value = True
                m.is_file.return_value = True
                m.resolve.return_value = str(path_str)
            elif "missing_fig.png" in str(path_str):
                m.exists.return_value = False # Missing image
            return m
            
        mock_path.side_effect = path_side_effect
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["output.csv"]},
            "plan": {"stages": []},
        }
        
        # Should not crash
        result = results_analyzer_node(state)
        assert result["workflow_phase"] == "analysis"
        # Image path should be None in comparison because it was missing
        comp = result["figure_comparisons"][0]
        assert comp["paper_image_path"] is None

    @patch("src.agents.analysis.evaluate_validation_criteria")
    @patch("src.agents.analysis.record_discrepancy")
    @patch("src.agents.analysis.Path")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    def test_handles_validation_criteria_failure(
        self, mock_match, mock_metrics, mock_load, mock_stub, mock_prompt,
        mock_context, mock_plan_stage, mock_path, mock_record, mock_eval
    ):
        """Should handle validation criteria failures."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {
            "stage_id": "stage1", 
            "targets": ["Fig1"],
            "validation_criteria": ["peak error < 1%"]
        }
        mock_load.return_value = None
        mock_metrics.return_value = {"peak_error": 5.0}
        mock_match.return_value = "output.csv"
        
        # Criteria failure
        mock_eval.return_value = (False, ["peak error too high"])
        
        # Mock file exists
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.is_file.return_value = True
        mock_file.resolve.return_value = "output.csv"
        mock_path.return_value = mock_file
        
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            "stage_outputs": {"files": ["output.csv"]},
            "plan": {"stages": []},
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"
        # Should be classified as mismatch due to criteria failure
        summary = result["analysis_summary"]
        assert "Fig1" in summary["mismatch_targets"]
        # Should record discrepancy
        mock_record.assert_called()


class TestComparisonValidatorNode:
    """Tests for comparison_validator_node function."""

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_approves_when_no_targets(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should approve when stage has no reproducible targets."""
        mock_context.return_value = None
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": []}
        mock_reports.return_value = []
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "approve"
        assert "no reproducible targets" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_needs_revision_when_no_comparisons_but_has_targets(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should need revision when targets exist but no comparisons."""
        mock_context.return_value = None
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1", "Fig2"]}
        mock_reports.return_value = []
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_needs_revision_when_missing_outputs(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should need revision when outputs are missing."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]
        mock_breakdown.return_value = {"missing": ["Fig1"], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "missing" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_needs_revision_when_pending_checks(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should need revision when comparisons are pending."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]
        mock_breakdown.return_value = {"missing": [], "pending": ["Fig1"]}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "pending" in result["comparison_feedback"].lower()

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_approves_when_all_comparisons_present(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should approve when all required comparisons are present."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}]
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1", "Fig2"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}, {"target_figure": "Fig2"}]
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "approve"

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_needs_revision_when_missing_comparisons(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should need revision when comparisons are missing for some targets."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]  # Only one comparison
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1", "Fig2"]}  # Two targets
        mock_reports.return_value = [{"target_figure": "Fig1"}]
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_needs_revision_on_report_issues(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should need revision when analysis reports have issues."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]
        mock_validate.return_value = ["Report has missing fields", "Invalid metrics"]
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_increments_revision_count_on_needs_revision(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should increment analysis_revision_count when needs revision."""
        mock_context.return_value = None
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = []
        mock_validate.return_value = []
        
        state = {
            "current_stage_id": "stage1",
            "analysis_revision_count": 0,  # Start at 0, expect increment to 1
        }
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_revision_count"] == 1

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_respects_max_revisions(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should not exceed max revisions."""
        mock_context.return_value = None
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = []
        mock_validate.return_value = []
        
        state = {
            "current_stage_id": "stage1",
            "analysis_revision_count": 10,  # Already at max
            "runtime_config": {"max_analysis_revisions": 3},
        }
        
        result = comparison_validator_node(state)
        
        # Should not increment beyond max
        assert result["analysis_revision_count"] == 10

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_clears_feedback_on_approve(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should clear analysis_feedback on approval."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]
        mock_validate.return_value = []
        
        state = {
            "current_stage_id": "stage1",
            "analysis_feedback": "Previous feedback",
        }
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "approve"
        assert result["analysis_feedback"] is None

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_sets_feedback_on_needs_revision(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should set analysis_feedback on needs_revision."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]
        mock_breakdown.return_value = {"missing": ["output.csv"], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert result["analysis_feedback"] is not None

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_handles_missing_report_targets(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should detect missing quantitative reports for targets."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}]
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1", "Fig2"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]  # Missing Fig2 report
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        assert result["comparison_verdict"] == "needs_revision"
        assert "Fig2" in result["comparison_feedback"]

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_truncates_many_issues(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should truncate feedback when many issues exist."""
        mock_context.return_value = None
        mock_comparisons.return_value = []
        mock_breakdown.return_value = {"missing": [], "pending": []}
        mock_plan_stage.return_value = {"targets": ["Fig1"]}
        mock_reports.return_value = [{"target_figure": "Fig1"}]  # Report exists for Fig1
        mock_validate.return_value = [
            "Issue 1", "Issue 2", "Issue 3", "Issue 4", "Issue 5"
        ]
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        # Should mention there are more issues (5 issues - 3 shown = +2 more)
        assert "+2 more" in result["comparison_feedback"]

    @patch("src.agents.analysis.stage_comparisons_for_stage")
    @patch("src.agents.analysis.breakdown_comparison_classifications")
    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.analysis_reports_for_stage")
    @patch("src.agents.analysis.validate_analysis_reports")
    @patch("src.agents.analysis.check_context_or_escalate")
    def test_handles_target_details_format(
        self, mock_context, mock_validate, mock_reports, mock_plan_stage,
        mock_breakdown, mock_comparisons
    ):
        """Should handle target_details format instead of targets list."""
        mock_context.return_value = None
        mock_comparisons.return_value = [{"figure_id": "Fig1"}]
        mock_breakdown.return_value = {"missing": [], "pending": []}
        # Use target_details format instead of targets list
        mock_plan_stage.return_value = {
            "target_details": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}]
        }
        mock_reports.return_value = [{"target_figure": "Fig1"}, {"target_figure": "Fig2"}]
        mock_validate.return_value = []
        
        state = {"current_stage_id": "stage1"}
        
        result = comparison_validator_node(state)
        
        # Should still work with target_details format
        assert result["workflow_phase"] == "comparison_validation"
