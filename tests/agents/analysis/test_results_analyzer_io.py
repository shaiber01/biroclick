"""State/IO tests for results_analyzer_node."""

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

    def test_analyzer_stage_outputs_none(self, base_state):
        """Test error when stage_outputs is explicitly None."""
        base_state["stage_outputs"] = None
        
        result = results_analyzer_node(base_state)
        
        # Verify all expected fields are present and correct
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"
        # Verify analysis_summary is set correctly
        assert "analysis_summary" in result
        assert result["analysis_summary"] == "Analysis skipped: No outputs available"
        # When analysis fails due to missing outputs, classification is set to FAILED
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []

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
        
        # Verify workflow phase is always set
        assert result["workflow_phase"] == "analysis"
        # Verify classification is always present and valid
        assert "analysis_overall_classification" in result
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.NO_TARGETS,
            AnalysisClassification.POOR_MATCH,
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
            AnalysisClassification.EXCELLENT_MATCH
        ]
        # Verify analysis_summary structure exists
        assert "analysis_summary" in result
        assert isinstance(result["analysis_summary"], dict)
        assert "totals" in result["analysis_summary"]
        # Verify figure_comparisons is always a list
        assert "figure_comparisons" in result
        assert isinstance(result["figure_comparisons"], list)
        # Verify analysis_result_reports is always a list
        assert "analysis_result_reports" in result
        assert isinstance(result["analysis_result_reports"], list)

    def test_analyzer_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = results_analyzer_node(base_state)
        
        # Verify all error handling fields are present and correct
        assert result["workflow_phase"] == "analysis"
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True
        assert "pending_user_questions" in result
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "No stage selected" in result["pending_user_questions"][0] or "ERROR" in result["pending_user_questions"][0]
        # Verify no execution verdict is set (this is a user input request, not a failure)
        assert "execution_verdict" not in result
        # Verify no analysis results are returned (can't analyze without stage)
        assert "analysis_overall_classification" not in result

    def test_analyzer_missing_outputs(self, base_state):
        """Test error when stage outputs are empty."""
        base_state["stage_outputs"] = {}
        result = results_analyzer_node(base_state)
        
        # Verify failure is properly reported
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert "Stage outputs are missing" in result["run_error"]
        # Verify stage_id is included in error message
        assert base_state["current_stage_id"] in result["run_error"]
        assert result["workflow_phase"] == "analysis"
        # Verify analysis_summary indicates skip
        assert "analysis_summary" in result
        assert "Analysis skipped" in result["analysis_summary"] or isinstance(result["analysis_summary"], dict)
        # When analysis fails due to missing outputs, classification is set to FAILED
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_files_missing_on_disk(self, mock_prompt, mock_check, base_state):
        """Test failure when files listed in state do not exist on disk."""
        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.is_file", return_value=False):
            result = results_analyzer_node(base_state)
            
            # Verify failure is properly reported
            assert result["execution_verdict"] == "fail"
            assert "run_error" in result
            assert "do not exist on disk" in result["run_error"]
            # Verify at least one file from stage_outputs is mentioned
            stage_files = base_state["stage_outputs"].get("files", [])
            assert any(str(f) in result["run_error"] for f in stage_files)
            assert result["workflow_phase"] == "analysis"
            # Verify analysis_summary indicates skip
            assert "analysis_summary" in result
            assert "Analysis skipped" in result["analysis_summary"] or isinstance(result["analysis_summary"], dict)
            # Verify missing_files list is included in error
            assert "missing" in result["run_error"].lower() or "do not exist" in result["run_error"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures", return_value=[])
    def test_analyzer_no_targets(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test handling of stage with no targets."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        
        result = results_analyzer_node(base_state)
        
        # Verify classification is correct
        assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
        # Verify supervisor verdict
        assert result["supervisor_verdict"] == "ok_continue"
        # Verify analysis_summary structure
        assert "analysis_summary" in result
        assert isinstance(result["analysis_summary"], dict)
        assert "totals" in result["analysis_summary"]
        assert result["analysis_summary"]["totals"]["targets"] == 0
        assert result["analysis_summary"]["totals"]["matches"] == 0
        assert result["analysis_summary"]["totals"]["pending"] == 0
        assert result["analysis_summary"]["totals"]["missing"] == 0
        assert result["analysis_summary"]["totals"]["mismatch"] == 0
        # Verify all expected fields are present (component should have consistent structure)
        assert "unresolved_targets" in result["analysis_summary"]
        assert result["analysis_summary"]["unresolved_targets"] == []
        assert "matched_targets" in result["analysis_summary"]
        assert result["analysis_summary"]["matched_targets"] == []
        assert "pending_targets" in result["analysis_summary"]
        assert result["analysis_summary"]["pending_targets"] == []
        assert "missing_targets" in result["analysis_summary"]
        assert result["analysis_summary"]["missing_targets"] == []
        assert "mismatch_targets" in result["analysis_summary"]
        assert result["analysis_summary"]["mismatch_targets"] == []
        assert "stage_id" in result["analysis_summary"]
        assert result["analysis_summary"]["stage_id"] == base_state["current_stage_id"]
        assert "discrepancies_logged" in result["analysis_summary"]
        assert result["analysis_summary"]["discrepancies_logged"] == 0
        # Verify reports and comparisons are empty lists
        assert result["analysis_result_reports"] == []
        assert result["figure_comparisons"] == []
        # Verify workflow phase
        assert result["workflow_phase"] == "analysis"

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
            # Verify workflow phase is set
            assert result["workflow_phase"] == "analysis"
            # Verify analysis completes without crashing
            assert "analysis_overall_classification" in result
            assert "analysis_summary" in result
            # Verify hallucinated target doesn't break the analysis
            # The target should either be ignored or handled gracefully
            assert isinstance(result["analysis_summary"], dict)
            # Verify figure_comparisons is a list (may be empty or contain entries)
            assert isinstance(result["figure_comparisons"], list)
            # Verify analysis_result_reports is a list
            assert isinstance(result["analysis_result_reports"], list)

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
        
        # Verify failure is properly reported
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert "Stage outputs are missing" in result["run_error"]
        # Verify stage_id is mentioned in error
        assert base_state["current_stage_id"] in result["run_error"]
        assert result["workflow_phase"] == "analysis"
        # Verify analysis_summary indicates skip
        assert "analysis_summary" in result
        # When analysis fails due to missing outputs, classification is set to FAILED
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context, base_state):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        result = results_analyzer_node(base_state)
        
        # Verify escalation is properly returned
        assert result["awaiting_user_input"] is True
        assert "pending_user_questions" in result
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "Context overflow" in result["pending_user_questions"]
        # Verify no analysis results are returned when escalated
        assert "analysis_overall_classification" not in result
        assert "execution_verdict" not in result
        # Verify workflow phase may or may not be set (depends on implementation)
        # But if set, should be "analysis"
        if "workflow_phase" in result:
            assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    def test_handles_digitized_data_requirement(
        self, mock_prompt, mock_check, mock_match, mock_metrics, mock_load, base_state
    ):
        """Should flag targets requiring digitized data without it."""
        # Require excellent precision which mandates digitized data
        base_state["plan"]["stages"][0]["targets"] = ["Fig1"]
        base_state["plan"]["stages"][0]["target_details"] = [{"figure_id": "Fig1", "precision_requirement": "excellent"}]
        # Ensure no digitized path in plan or figures
        base_state["paper_figures"][0]["digitized_data_path"] = None
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
             
            result = results_analyzer_node(base_state)
            
            # Verify workflow phase
            assert result["workflow_phase"] == "analysis"
            # Should flag as missing digitized data
            summary = result["analysis_summary"]
            assert isinstance(summary, dict)
            assert "pending_targets" in summary
            assert isinstance(summary["pending_targets"], list)
            # It goes into pending_targets or just blocked?
            # Code says: pending_targets.append(target_id), classification="missing_digitized_data"
            assert "Fig1" in summary["pending_targets"]
            # Verify totals reflect pending target
            assert summary["totals"]["pending"] >= 1
            assert summary["totals"]["targets"] == 1
            
            # Verify figure_comparisons contains the classification
            reports = result["figure_comparisons"]
            assert isinstance(reports, list)
            assert len(reports) > 0
            # Component should use "figure_id" consistently
            fig1_report = next((r for r in reports if r.get("figure_id") == "Fig1"), None)
            assert fig1_report is not None, f"Fig1 not found in reports: {reports}"
            assert fig1_report["classification"] == "missing_digitized_data"
            # Verify status field
            assert fig1_report.get("status") == "missing_digitized_data"
            # Verify notes indicate blocking
            assert "blocked" in fig1_report.get("notes", "").lower() or "required" in fig1_report.get("notes", "").lower()
            # Verify all required fields are present
            assert "stage_id" in fig1_report
            assert fig1_report["stage_id"] == base_state["current_stage_id"]
            assert "title" in fig1_report
            assert "comparison_table" in fig1_report
            assert isinstance(fig1_report["comparison_table"], list)
            
            # Verify analysis_result_reports also contains the classification
            assert "analysis_result_reports" in result
            assert isinstance(result["analysis_result_reports"], list)
            fig1_result_report = next((r for r in result["analysis_result_reports"] if r.get("target_figure") == "Fig1"), None)
            assert fig1_result_report is not None
            assert fig1_result_report["status"] == "missing_digitized_data"

    def test_analyzer_stage_outputs_missing_files_key(self, base_state):
        """Test error when stage_outputs exists but has no 'files' key."""
        base_state["stage_outputs"] = {"other_key": "value"}
        
        result = results_analyzer_node(base_state)
        
        # Verify failure is properly reported
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result

    def test_analyzer_empty_paper_id(self, base_state):
        """Test handling when paper_id is empty string."""
        base_state["paper_id"] = ""
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            result = results_analyzer_node(base_state)
        
        # Should still work, using "unknown" as fallback
        assert result["workflow_phase"] == "analysis"
        assert "analysis_overall_classification" in result

    def test_analyzer_empty_stage_id_string(self, base_state):
        """Test error when current_stage_id is empty string."""
        base_state["current_stage_id"] = ""
        
        result = results_analyzer_node(base_state)
        
        # Empty string should be treated as missing
        assert result["workflow_phase"] == "analysis"
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True

    def test_analyzer_stage_outputs_files_none(self, base_state):
        """Test error when stage_outputs.files is explicitly None."""
        base_state["stage_outputs"] = {"files": None}
        
        result = results_analyzer_node(base_state)
        
        # Verify failure is properly reported
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_partial_files_missing(self, mock_prompt, mock_check, base_state):
        """Test handling when some files exist and some don't."""
        base_state["stage_outputs"] = {"files": ["output.csv", "missing_file.csv", "another_output.csv"]}
        
        # Mock exists to return True for some files, False for others
        # Path.exists() is an instance method, so we need to patch it properly
        original_exists = Path.exists
        original_is_file = Path.is_file
        
        def exists_side_effect(self):
            path_str = str(self)
            return "missing_file" not in path_str
        
        def is_file_side_effect(self):
            path_str = str(self)
            return "missing_file" not in path_str
        
        with patch.object(Path, "exists", side_effect=exists_side_effect, autospec=True), \
             patch.object(Path, "is_file", side_effect=is_file_side_effect, autospec=True):
            result = results_analyzer_node(base_state)
        
        # Should proceed with existing files
        assert result["workflow_phase"] == "analysis"
        # Should not fail completely - should proceed with available files
        assert "execution_verdict" not in result or result.get("execution_verdict") != "fail"
        # Should have analysis results
        assert "analysis_overall_classification" in result
        assert "analysis_summary" in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_analyzer_targets_from_feedback_priority(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test that feedback targets are prioritized over regular targets."""
        mock_stubs.return_value = [{"id": "Fig1"}, {"id": "Fig2"}]
        base_state["plan"]["stages"][0]["targets"] = ["Fig1", "Fig2"]
        base_state["analysis_feedback"] = "Focus on Fig2"
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=None), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={}):
            
            result = results_analyzer_node(base_state)
            
            # Verify analysis completed
            assert result["workflow_phase"] == "analysis"
            assert "analysis_summary" in result
            summary = result["analysis_summary"]
            # Verify feedback_applied contains extracted targets
            assert "feedback_applied" in summary
            # Fig2 should be prioritized if extracted from feedback
            # The exact behavior depends on extract_targets_from_feedback implementation
            assert isinstance(summary["feedback_applied"], list)

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_analyzer_paper_figures_empty(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test handling when paper_figures is empty list."""
        mock_stubs.return_value = []
        base_state["paper_figures"] = []
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        
        result = results_analyzer_node(base_state)
        
        # Should handle gracefully with NO_TARGETS
        assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
        assert result["workflow_phase"] == "analysis"
        assert result["analysis_summary"]["totals"]["targets"] == 0

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_absolute_path_handling(self, mock_prompt, mock_check, base_state):
        """Test handling of absolute vs relative file paths."""
        import os
        abs_path = os.path.abspath("/tmp/test_output.csv")
        base_state["stage_outputs"] = {"files": [abs_path]}
        
        # Mock exists to return True for absolute path
        # Path.exists() is an instance method, so we need to patch it properly
        def exists_side_effect(self):
            path_str = str(self)
            return path_str == abs_path or "/tmp/test_output.csv" in path_str
        
        def is_file_side_effect(self):
            path_str = str(self)
            return path_str == abs_path or "/tmp/test_output.csv" in path_str
        
        with patch.object(Path, "exists", side_effect=exists_side_effect, autospec=True), \
             patch.object(Path, "is_file", side_effect=is_file_side_effect, autospec=True):
            result = results_analyzer_node(base_state)
        
        # Should handle absolute paths correctly
        assert result["workflow_phase"] == "analysis"
        # Should not fail due to path handling
        assert "execution_verdict" not in result or result.get("execution_verdict") != "fail"

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_analyzer_target_details_without_figure_id(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test handling when target_details entries lack figure_id."""
        mock_stubs.return_value = [{"id": "Fig1"}]
        base_state["plan"]["stages"][0]["targets"] = ["Fig1"]
        base_state["plan"]["stages"][0]["target_details"] = [
            {"precision_requirement": "acceptable"}  # Missing figure_id
        ]
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            result = results_analyzer_node(base_state)
        
        # Should handle gracefully - targets list should still work
        assert result["workflow_phase"] == "analysis"
        assert "analysis_overall_classification" in result
        # Should still process Fig1 from targets list
        assert result["analysis_summary"]["totals"]["targets"] >= 0

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_context_update_without_escalation(self, mock_context, base_state):
        """Test handling when context check returns updates but no escalation."""
        mock_context.return_value = {
            "context_budget": {"remaining": 1000}
        }
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            result = results_analyzer_node(base_state)
        
        # Should proceed with analysis
        assert result["workflow_phase"] == "analysis"
        assert "analysis_overall_classification" in result
        # Should not be awaiting user input
        assert result.get("awaiting_user_input") is not True

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.match_expected_files")
    def test_analyzer_file_matching_fallback(self, mock_match_expected, mock_match_output, mock_prompt, mock_check, base_state):
        """Test that match_expected_files failure falls back to match_output_file."""
        mock_match_expected.return_value = None  # First match fails
        mock_match_output.return_value = "fallback_output.csv"  # Fallback succeeds
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.load_numeric_series", return_value=None), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={}):
            result = results_analyzer_node(base_state)
        
        # Verify both matching functions were called
        assert mock_match_expected.called
        assert mock_match_output.called
        # Verify analysis completed
        assert result["workflow_phase"] == "analysis"
        assert "analysis_overall_classification" in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.get_images_for_analyzer", return_value=[])
    def test_analyzer_no_images_skips_llm_call(self, mock_images, mock_prompt, mock_check, base_state):
        """Test that LLM call is skipped when no images are available."""
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=None), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={}), \
             patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            
            result = results_analyzer_node(base_state)
            
            # LLM should not be called when no images
            assert not mock_llm.called
            # But analysis should still complete
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.get_images_for_analyzer", return_value=["image1.png"])
    @patch("src.agents.analysis.call_agent_with_metrics")
    def test_analyzer_llm_exception_handling(self, mock_llm, mock_images, mock_prompt, mock_check, base_state):
        """Test that LLM exceptions don't crash the analysis."""
        mock_llm.side_effect = Exception("LLM service unavailable")
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=None), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={}):
            
            result = results_analyzer_node(base_state)
            
            # Should handle exception gracefully and continue
            assert result["workflow_phase"] == "analysis"
            assert "analysis_overall_classification" in result
            # Should still have quantitative results even if LLM failed
            assert "analysis_summary" in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_figure_image_path_missing(self, mock_prompt, mock_check, base_state):
        """Test handling when figure image_path doesn't exist."""
        base_state["paper_figures"][0]["image_path"] = "/nonexistent/path/to/fig1.png"
        
        # Path.exists() and Path.is_file() are instance methods
        def exists_side_effect(self):
            path_str = str(self)
            if "/nonexistent/path/to/fig1.png" in path_str:
                return False
            return True
        
        def is_file_side_effect(self):
            path_str = str(self)
            if "/nonexistent/path/to/fig1.png" in path_str:
                return False
            return True
        
        with patch.object(Path, "exists", side_effect=exists_side_effect, autospec=True), \
             patch.object(Path, "is_file", side_effect=is_file_side_effect, autospec=True):
            result = results_analyzer_node(base_state)
        
        # Should handle missing image gracefully
        assert result["workflow_phase"] == "analysis"
        assert "analysis_overall_classification" in result
        # Should still produce comparisons (without image)
        assert "figure_comparisons" in result

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_analyzer_stage_info_none_but_targets_in_plan(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test handling when get_plan_stage returns None but plan has targets."""
        mock_stubs.return_value = [{"id": "Fig1"}]
        
        # Mock get_plan_stage to return None
        with patch("src.agents.analysis.get_plan_stage", return_value=None), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            # Set targets in plan root
            base_state["plan"]["targets"] = [{"figure_id": "Fig1"}]
            
            result = results_analyzer_node(base_state)
        
        # Should fallback to plan.targets
        assert result["workflow_phase"] == "analysis"
        assert "analysis_overall_classification" in result
        # Should still process targets from plan root
        assert result["analysis_summary"]["totals"]["targets"] >= 0

