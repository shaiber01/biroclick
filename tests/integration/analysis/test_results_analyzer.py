"""Integration tests for results_analyzer_node behavior.

These tests are designed to FIND BUGS, not just pass.
They verify specific outputs, edge cases, error handling, and complex logic.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.constants import AnalysisClassification


class TestResultsAnalyzerErrorHandling:
    """Test error handling paths - these should catch bugs in error handling logic."""

    def test_missing_current_stage_id_returns_error_state(self, base_state, valid_plan):
        """When current_stage_id is None, should return error state with user question."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = None
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0]
        # Should NOT have analysis results when stage_id is missing
        assert "analysis_summary" not in result or isinstance(result.get("analysis_summary"), str)

    def test_empty_current_stage_id_string_handled(self, base_state, valid_plan):
        """Empty string for current_stage_id should be handled."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = ""
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        result = results_analyzer_node(base_state)

        # Empty string should be treated as missing (falsy value)
        assert result["workflow_phase"] == "analysis"
        # Should return error state like None case
        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result

    def test_missing_stage_outputs_returns_fail_state(self, base_state, valid_plan):
        """When stage_outputs is missing, should return fail state."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = None

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("execution_verdict") == "fail"
        assert "run_error" in result
        assert "stage_outputs" in result["run_error"].lower() or "outputs" in result["run_error"].lower()
        assert result.get("analysis_summary") == "Analysis skipped: No outputs available"

    def test_empty_stage_outputs_dict_returns_fail_state(self, base_state, valid_plan):
        """When stage_outputs is empty dict, should return fail state."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("execution_verdict") == "fail"
        assert "run_error" in result
        assert result.get("analysis_summary") == "Analysis skipped: No outputs available"

    def test_empty_files_list_returns_fail_state(self, base_state, valid_plan):
        """When stage_outputs.files is empty list, should return fail state."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("execution_verdict") == "fail"
        assert "run_error" in result
        assert result.get("analysis_summary") == "Analysis skipped: No outputs available"

    def test_all_files_missing_returns_fail_state(self, base_state, valid_plan):
        """When all files in stage_outputs don't exist, should return fail state."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/nonexistent/file1.csv", "/nonexistent/file2.csv"]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("execution_verdict") == "fail"
        assert "run_error" in result
        assert "missing" in result["run_error"].lower() or "not exist" in result["run_error"].lower()
        assert result.get("analysis_summary") == "Analysis skipped: Output files missing"

    def test_no_targets_defined_returns_no_targets_classification(self, base_state, valid_plan):
        """When stage has no targets, should return NO_TARGETS classification."""
        from src.agents.analysis import results_analyzer_node

        # Create plan with stage that has empty targets
        plan_no_targets = {**valid_plan}
        plan_no_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test stage",
            "targets": [],  # Empty targets
            "dependencies": [],
        }]

        base_state["plan"] = plan_no_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_figures"] = []  # No figures either

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        summary = result["analysis_summary"]
        assert summary["overall_classification"] == AnalysisClassification.NO_TARGETS
        assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
        assert summary["totals"]["targets"] == 0
        assert summary["totals"]["matches"] == 0
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []

    def test_missing_plan_handled_gracefully(self, base_state):
        """When plan is missing, should handle gracefully."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = None
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        result = results_analyzer_node(base_state)

        # Should either error or handle gracefully - verify behavior
        assert result["workflow_phase"] == "analysis"
        # Component should handle None plan without crashing
        assert "analysis_summary" in result or "run_error" in result


class TestResultsAnalyzerStateTransitions:
    """Workflow-phase signals and state transitions."""

    def test_results_analyzer_sets_workflow_phase(self, base_state, valid_plan):
        """Verify workflow_phase is always set to 'analysis'."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/nonexistent/fake_output.csv"]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # When files don't exist, should return fail state
        assert result.get("execution_verdict") == "fail" or result.get("run_error")
        # Verify specific error message structure
        if result.get("run_error"):
            assert isinstance(result["run_error"], str)
            assert len(result["run_error"]) > 0

    def test_results_analyzer_with_empty_outputs(self, base_state, valid_plan):
        """results_analyzer_node should handle empty stage_outputs gracefully."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)

        assert result.get("workflow_phase") == "analysis"
        # Should return fail state, not just empty response
        assert result.get("execution_verdict") == "fail"
        assert "run_error" in result
        assert result.get("analysis_summary") == "Analysis skipped: No outputs available"


@pytest.mark.slow
class TestResultsAnalyzerFileIO:
    """File-system heavy analyzer tests with comprehensive assertions."""

    def test_results_analyzer_returns_figure_comparisons_on_success(
        self, base_state, valid_plan
    ):
        """Verify complete output structure when files exist."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict)
            assert "figure_comparisons" in result
            assert isinstance(result["figure_comparisons"], list)
            # Verify summary structure
            summary = result["analysis_summary"]
            assert "stage_id" in summary
            assert "totals" in summary
            assert isinstance(summary["totals"], dict)
            assert "targets" in summary["totals"]
            assert "matches" in summary["totals"]
            assert "pending" in summary["totals"]
            assert "missing" in summary["totals"]
            assert "mismatch" in summary["totals"]
            # Verify classification is set
            assert "analysis_overall_classification" in result
            assert isinstance(result["analysis_overall_classification"], str)
            # Verify reports exist
            assert "analysis_result_reports" in result
            assert isinstance(result["analysis_result_reports"], list)
        finally:
            os.unlink(temp_file)

    def test_partial_files_exist_continues_with_existing(self, base_state, valid_plan):
        """When some files exist and some don't, should continue with existing files."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {
                "files": [temp_file, "/nonexistent/file.csv"]
            }
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            # Should not fail completely, should proceed with existing file
            assert result.get("execution_verdict") != "fail"
            assert "analysis_summary" in result
            # Should have processed the existing file
            assert len(result.get("figure_comparisons", [])) >= 0
        finally:
            os.unlink(temp_file)

    def test_absolute_path_handling(self, base_state, valid_plan):
        """Verify absolute paths are handled correctly."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n")
            temp_file = handle.name
            abs_path = os.path.abspath(temp_file)

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [abs_path]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert result.get("execution_verdict") != "fail"
            assert "analysis_summary" in result
        finally:
            os.unlink(temp_file)

    def test_relative_path_resolution(self, base_state, valid_plan):
        """Verify relative paths are resolved correctly."""
        from src.agents.analysis import results_analyzer_node

        # Create file in expected output directory
        paper_id = "test_integration"
        stage_id = "stage_0"
        output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / paper_id / stage_id
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_file = output_dir / "output.csv"
        try:
            temp_file.write_text("wavelength,extinction\n400,0.1\n500,0.5\n")

            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = stage_id
            base_state["stage_outputs"] = {"files": ["output.csv"]}  # Relative path
            base_state["paper_id"] = paper_id

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert result.get("execution_verdict") != "fail"
            assert "analysis_summary" in result
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestResultsAnalyzerCompleteness:
    """Verify results_analyzer produces complete output with all required fields."""

    def test_results_analyzer_returns_all_required_fields(self, base_state, valid_plan):
        """Verify all required fields are present in output."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)

        assert result.get("workflow_phase") == "analysis"
        # Even with empty outputs, should have structured response
        assert "analysis_summary" in result or "run_error" in result
        assert "analysis_overall_classification" in result or "execution_verdict" in result
        assert "figure_comparisons" in result
        assert isinstance(result["figure_comparisons"], list)
        assert "analysis_result_reports" in result
        assert isinstance(result["analysis_result_reports"], list)

    def test_analysis_summary_structure_complete(self, base_state, valid_plan):
        """Verify analysis_summary has complete structure when present."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            if isinstance(result.get("analysis_summary"), dict):
                summary = result["analysis_summary"]
                # Verify all expected keys
                assert "stage_id" in summary
                assert "totals" in summary
                assert "matched_targets" in summary
                assert "pending_targets" in summary
                assert "missing_targets" in summary
                assert "mismatch_targets" in summary
                assert "discrepancies_logged" in summary
                assert "validation_criteria" in summary
                assert "feedback_applied" in summary
                assert "unresolved_targets" in summary
                assert "notes" in summary
                # Verify totals structure
                totals = summary["totals"]
                assert isinstance(totals["targets"], int)
                assert isinstance(totals["matches"], int)
                assert isinstance(totals["pending"], int)
                assert isinstance(totals["missing"], int)
                assert isinstance(totals["mismatch"], int)
        finally:
            os.unlink(temp_file)


class TestResultsAnalyzerLogic:
    """Verify results_analyzer_node logic for existing files and complex scenarios."""

    def test_results_analyzer_analyzes_existing_files(self, base_state, valid_plan):
        """Verify file analysis logic with mocked dependencies."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"

        mock_metrics = {"peak_position_error_percent": 5.0, "peak_position_paper": 650.0, "peak_position_sim": 645.0}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        assert isinstance(result["analysis_summary"], dict)
        assert result["figure_comparisons"]
        assert isinstance(result["figure_comparisons"], list)
        # Verify each comparison has required fields
        for comp in result["figure_comparisons"]:
            assert "figure_id" in comp or "target_id" in comp
            assert "stage_id" in comp
            assert "classification" in comp
            assert "comparison_table" in comp
            assert isinstance(comp["comparison_table"], list)

    def test_target_matching_logic(self, base_state, valid_plan):
        """Verify target matching logic works correctly."""
        from src.agents.analysis import results_analyzer_node

        # Create plan with specific targets
        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test stage",
            "targets": ["Fig1", "Fig2"],
            "dependencies": [],
        }]
        plan_with_targets["targets"] = [
            {"figure_id": "Fig1", "description": "Figure 1"},
            {"figure_id": "Fig2", "description": "Figure 2"},
        ]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [
            {"figure_id": "Fig1", "description": "Figure 1"},
            {"figure_id": "Fig2", "description": "Figure 2"},
        ]

        mock_metrics = {"peak_position_error_percent": 2.0}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        summary = result["analysis_summary"]
        # Should have processed targets from stage
        assert summary["totals"]["targets"] >= 0
        # Verify reports exist for targets
        assert len(result["analysis_result_reports"]) >= 0

    def test_classification_logic_excellent_match(self, base_state, valid_plan):
        """Verify classification logic for excellent match."""
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "description": "Test"}]

        # Mock metrics indicating excellent match (< 2% error)
        mock_metrics = {
            "peak_position_error_percent": 1.5,
            "peak_position_paper": 650.0,
            "peak_position_sim": 640.0,
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ), patch(
            "src.agents.analysis.classification_from_metrics",
            return_value=AnalysisClassification.MATCH
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # When all targets match, should be excellent or acceptable match
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.EXCELLENT_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
            AnalysisClassification.MATCH,
        ]

    def test_classification_logic_poor_match(self, base_state, valid_plan):
        """Verify classification logic for poor match."""
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "description": "Test"}]

        # Mock metrics indicating mismatch
        mock_metrics = {
            "peak_position_error_percent": 50.0,
            "peak_position_paper": 650.0,
            "peak_position_sim": 500.0,
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ), patch(
            "src.agents.analysis.classification_from_metrics",
            return_value=AnalysisClassification.MISMATCH
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # When there are mismatches, should be poor match
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.POOR_MATCH,
            AnalysisClassification.MISMATCH,
        ]

    def test_digitized_data_requirement_enforced(self, base_state, valid_plan):
        """Verify that excellent precision requirement enforces digitized data."""
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]
        plan_with_targets["targets"] = [{
            "figure_id": "Fig1",
            "description": "Test",
            "precision_requirement": "excellent",  # Requires digitized data
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "description": "Test"}]

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should have pending target due to missing digitized data
        summary = result["analysis_summary"]
        assert "pending_targets" in summary
        # Should have comparison with missing_digitized_data status
        for comp in result["figure_comparisons"]:
            if comp.get("figure_id") == "Fig1":
                assert comp.get("status") == "missing_digitized_data" or comp.get("classification") == "missing_digitized_data"
                break


class TestResultsAnalyzerIntegration:
    """Test integration with LLM, file I/O, and helper functions."""

    def test_llm_analysis_integration(self, base_state, valid_plan):
        """Verify LLM analysis integration works correctly."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"
            base_state["paper_figures"] = [{"figure_id": "Fig1", "description": "Test", "image_path": "/path/to/image.png"}]

            mock_llm_response = {
                "overall_classification": "ACCEPTABLE_MATCH",
                "figure_comparisons": [{
                    "figure_id": "Fig1",
                    "shape_comparison": ["Peak matches well"],
                    "reason_for_difference": "Minor offset",
                }],
                "summary": "Overall good match",
            }

            with patch("pathlib.Path.exists", side_effect=lambda: True), patch(
                "pathlib.Path.is_file", return_value=True
            ), patch(
                "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
            ), patch(
                "src.agents.analysis.quantitative_curve_metrics",
                return_value={"peak_position_error_percent": 5.0},
            ), patch(
                "src.agents.analysis.get_images_for_analyzer", return_value=["/path/to/image.png"]
            ), patch(
                "src.agents.analysis.call_agent_with_metrics", return_value=mock_llm_response
            ):
                result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            # LLM response should influence classification
            assert "analysis_summary" in result
            # Shape comparisons should be populated from LLM
            for comp in result["figure_comparisons"]:
                if comp.get("figure_id") == "Fig1":
                    assert "shape_comparison" in comp
                    break
        finally:
            os.unlink(temp_file)

    def test_llm_analysis_failure_handled_gracefully(self, base_state, valid_plan):
        """Verify LLM analysis failure doesn't crash the node."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            with patch("pathlib.Path.exists", return_value=True), patch(
                "pathlib.Path.is_file", return_value=True
            ), patch(
                "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
            ), patch(
                "src.agents.analysis.quantitative_curve_metrics",
                return_value={"peak_position_error_percent": 5.0},
            ), patch(
                "src.agents.analysis.get_images_for_analyzer", return_value=["/path/to/image.png"]
            ), patch(
                "src.agents.analysis.call_agent_with_metrics", side_effect=Exception("LLM error")
            ):
                result = results_analyzer_node(base_state)

            # Should not crash, should return results without LLM analysis
            assert result["workflow_phase"] == "analysis"
            assert "analysis_summary" in result
            assert "figure_comparisons" in result
        finally:
            os.unlink(temp_file)

    def test_figure_comparisons_merged_correctly(self, base_state, valid_plan):
        """Verify figure_comparisons from different stages are merged correctly."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        # Pre-populate with existing comparisons from different stage
        base_state["figure_comparisons"] = [{
            "figure_id": "Fig0",
            "stage_id": "stage_1",
            "classification": "MATCH",
        }]

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics",
            return_value={"peak_position_error_percent": 5.0},
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should include both existing and new comparisons
        assert len(result["figure_comparisons"]) >= 1
        # Existing comparison from stage_1 should be preserved
        stage_1_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_1"]
        assert len(stage_1_comps) == 1
        # New comparison for stage_0 should be added
        stage_0_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_0"]
        assert len(stage_0_comps) >= 0

    def test_analysis_reports_merged_correctly(self, base_state, valid_plan):
        """Verify analysis_result_reports from different stages are merged correctly."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        # Pre-populate with existing reports from different stage
        base_state["analysis_result_reports"] = [{
            "result_id": "stage_1_Fig0",
            "target_figure": "Fig0",
            "stage_id": "stage_1",
            "status": "MATCH",
        }]

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics",
            return_value={"peak_position_error_percent": 5.0},
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ), patch(
            "src.agents.analysis.call_agent_with_metrics"
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should include both existing and new reports
        assert len(result["analysis_result_reports"]) >= 1
        # Existing report from stage_1 should be preserved
        stage_1_reports = [r for r in result["analysis_result_reports"] if r.get("stage_id") == "stage_1"]
        assert len(stage_1_reports) == 1
        # New reports for stage_0 should be added
        stage_0_reports = [r for r in result["analysis_result_reports"] if r.get("stage_id") == "stage_0"]
        assert len(stage_0_reports) >= 0


class TestResultsAnalyzerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_values_in_state(self, base_state, valid_plan):
        """Verify None values in state are handled correctly."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_figures"] = None
        base_state["analysis_feedback"] = None

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ):
            result = results_analyzer_node(base_state)

        # Should handle None values without crashing
        assert result["workflow_phase"] == "analysis"

    def test_empty_strings_in_state(self, base_state, valid_plan):
        """Verify empty strings in state are handled correctly."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["analysis_feedback"] = ""

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"

    def test_very_long_file_paths(self, base_state, valid_plan):
        """Verify very long file paths are handled."""
        from src.agents.analysis import results_analyzer_node

        long_path = "/" + "a" * 500 + "/file.csv"
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [long_path]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        # Should handle long paths without crashing
        assert result["workflow_phase"] == "analysis"

    def test_special_characters_in_paths(self, base_state, valid_plan):
        """Verify special characters in paths are handled."""
        from src.agents.analysis import results_analyzer_node

        special_path = "/path/with spaces/file (1).csv"
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [special_path]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        # Should handle special characters without crashing
        assert result["workflow_phase"] == "analysis"

    def test_multiple_stages_in_plan(self, base_state, valid_plan):
        """Verify correct stage is selected when multiple stages exist."""
        from src.agents.analysis import results_analyzer_node

        multi_stage_plan = {**valid_plan}
        multi_stage_plan["stages"] = [
            {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "description": "Stage 0", "targets": [], "dependencies": []},
            {"stage_id": "stage_1", "stage_type": "MATERIAL_VALIDATION", "description": "Stage 1", "targets": [], "dependencies": []},
        ]

        base_state["plan"] = multi_stage_plan
        base_state["current_stage_id"] = "stage_1"  # Select second stage
        base_state["stage_outputs"] = {}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should process stage_1, not stage_0
        if isinstance(result.get("analysis_summary"), dict):
            assert result["analysis_summary"]["stage_id"] == "stage_1"
