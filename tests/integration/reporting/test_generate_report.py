"""Integration tests for generate_report_node covering multiple scenarios."""

from unittest.mock import patch

import pytest

from tests.integration.helpers.agent_responses import reporting_summary_response


class TestGenerateReportNode:
    """Verify generate_report_node produces complete reports."""

    def test_generate_report_node_creates_report(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        mock_response = reporting_summary_response(
            executive_summary={"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
            conclusions={"main_physics_reproduced": True, "key_findings": ["Test finding"]},
            paper_citation={"title": "Test Paper", "authors": "Test Author"},
        )

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
            ]
        }
        base_state["paper_id"] = "test_paper_id"

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result.get("workflow_complete") is True
        assert result["workflow_phase"] == "reporting"
        assert "metrics" in result and "token_summary" in result["metrics"]
        assert result["executive_summary"] == mock_response["executive_summary"]
        assert result["paper_citation"] == mock_response["paper_citation"]
        assert result["report_conclusions"] == mock_response["conclusions"]

    def test_report_includes_token_summary(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
                {"agent_name": "designer", "input_tokens": 2000, "output_tokens": 800},
            ]
        }
        mock_response = reporting_summary_response(
            executive_summary={"overall_assessment": []},
            conclusions={"main_physics_reproduced": True},
        )

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        metrics = result.get("metrics", {})
        token_summary = metrics.get("token_summary", {})
        assert token_summary.get("total_input_tokens") == 3000
        assert token_summary.get("total_output_tokens") == 1300

        expected_cost = (3000 * 3.0 + 1300 * 15.0) / 1_000_000
        assert token_summary.get("estimated_cost") == pytest.approx(expected_cost)

    def test_report_generation_handles_llm_failure(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state.pop("executive_summary", None)

        with patch(
            "src.agents.reporting.call_agent_with_metrics",
            side_effect=Exception("LLM Error"),
        ):
            result = generate_report_node(base_state)

        assert result["workflow_complete"] is True
        assert "executive_summary" in result
        assert result["executive_summary"]["overall_assessment"] is not None
        assert "paper_citation" in result

    def test_report_generation_quantitative_summary(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",
                "target_figure": "Fig 1",
                "status": "pass",
                "precision_requirement": "high",
                "quantitative_metrics": {
                    "peak_position_error_percent": 0.5,
                    "normalized_rmse_percent": 1.2,
                    "correlation": 0.99,
                    "n_points_compared": 100,
                },
            }
        ]

        mock_response = {}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None
        assert len(summary) == 1
        row = summary[0]
        assert row["stage_id"] == "stage_1"
        assert row["peak_position_error_percent"] == 0.5
        assert row["normalized_rmse_percent"] == 1.2

    def test_report_generation_populates_missing_structures(
        self, base_state, valid_plan
    ):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("paper_citation", None)
        base_state.pop("executive_summary", None)
        base_state["paper_title"] = "My Paper"

        mock_response = {}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["paper_citation"]["title"] == "My Paper"
        assert result["paper_citation"]["authors"] == "Unknown"
        assert "executive_summary" in result

    def test_report_includes_rich_state_in_context(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = [{"fig": "1", "diff": "small"}]
        base_state["assumptions"] = {"param": "value"}
        base_state["discrepancies"] = [{"parameter": "p1", "classification": "minor"}]

        mock_response = {}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "Figure Comparisons" in user_content
        assert "Assumptions" in user_content
        assert "Discrepancies" in user_content
        assert "p1" in user_content

    def test_report_includes_token_summary_misc(self, base_state, valid_plan):
        """Regression test migrated from mixed execution/analysis suite."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
                {"agent_name": "designer", "input_tokens": 2000, "output_tokens": 800},
            ]
        }

        mock_response = reporting_summary_response(
            executive_summary={"overall_assessment": []},
            conclusions={"main_physics_reproduced": True},
        )

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        metrics = result.get("metrics", {})
        token_summary = metrics.get("token_summary", {})
        assert token_summary.get("total_input_tokens") == 3000
        assert "total_output_tokens" in token_summary

    def test_report_marks_workflow_complete_misc(self, base_state, valid_plan):
        """Regression test migrated from mixed execution/analysis suite."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}

        mock_response = reporting_summary_response(
            executive_summary={},
            conclusions={},
        )

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result.get("workflow_complete") is True

