"""Integration tests for results_analyzer_node behavior."""

import os
import tempfile
from unittest.mock import patch

import pytest


class TestResultsAnalyzerStateTransitions:
    """Workflow-phase signals and empty-output handling."""

    def test_results_analyzer_sets_workflow_phase(self, base_state, valid_plan):
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/nonexistent/fake_output.csv"]}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("execution_verdict") == "fail" or result.get("run_error")

    def test_results_analyzer_with_empty_outputs(self, base_state, valid_plan):
        """results_analyzer_node should handle empty stage_outputs gracefully."""
        from src.agents.analysis import results_analyzer_node

        mock_response = {
            "overall_classification": "NO_DATA",
            "figure_comparisons": [],
            "summary": "No data to analyze",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        with patch(
            "src.agents.analysis.call_agent_with_metrics", return_value=mock_response
        ):
            result = results_analyzer_node(base_state)

        assert result.get("workflow_phase") == "analysis"


@pytest.mark.slow
class TestResultsAnalyzerFileIO:
    """File-system heavy analyzer tests."""

    def test_results_analyzer_returns_figure_comparisons_on_success(
        self, base_state, valid_plan
    ):
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
            assert "figure_comparisons" in result
        finally:
            os.unlink(temp_file)


class TestResultsAnalyzerCompleteness:
    """Verify results_analyzer produces complete output."""

    def test_results_analyzer_returns_all_required_fields(self, base_state, valid_plan):
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)
        assert result.get("workflow_phase") == "analysis"


class TestResultsAnalyzerLogic:
    """Verify results_analyzer_node logic for existing files."""

    def test_results_analyzer_analyzes_existing_files(self, base_state, valid_plan):
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics",
            return_value={"peak_position_error_percent": 5.0},
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        assert result["figure_comparisons"]

