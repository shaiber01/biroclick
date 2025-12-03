from unittest.mock import patch

from src.agents import results_analyzer_node
from src.agents.constants import AnalysisClassification

from tests.workflow.fixtures import MockResponseFactory


class TestResultsAnalyzerLogic:
    """Test results analyzer logic."""

    def test_analysis_classification_update(self, base_state, tmp_path):
        """Test that analysis results update the state correctly."""
        base_state["current_stage_id"] = "stage_1_extinction"

        # Create dummy output file
        d = tmp_path / "data.csv"
        d.write_text("header\n1,2")

        base_state["stage_outputs"] = {"files": [str(d)]}
        # Add target to plan for reference
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        # Patch get_images so the LLM is called
        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm, patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=["img.png"]
        ):
            mock_llm.return_value = MockResponseFactory.analyzer_response(
                AnalysisClassification.EXCELLENT_MATCH
            )

            result = results_analyzer_node(base_state)

            assert result["analysis_overall_classification"] == AnalysisClassification.EXCELLENT_MATCH
            assert "figure_comparisons" in result
            assert len(result["figure_comparisons"]) > 0
            assert result["workflow_phase"] == "analysis"


