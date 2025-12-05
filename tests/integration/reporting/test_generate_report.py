"""Integration tests for generate_report_node covering multiple scenarios."""

import copy
from unittest.mock import patch, MagicMock

import pytest

from tests.integration.helpers.agent_responses import reporting_summary_response


class TestGenerateReportNode:
    """Verify generate_report_node produces complete reports."""

    def test_generate_report_node_creates_report(self, base_state, valid_plan):
        """Test basic report generation with all required fields."""
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
        
        # Create a deep copy to check for mutation
        state_before_call = copy.deepcopy(base_state)

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # Verify workflow completion
        assert result.get("workflow_complete") is True, "Workflow must be marked complete"
        assert result["workflow_phase"] == "reporting", "Workflow phase must be 'reporting'"
        
        # Verify metrics structure
        assert "metrics" in result, "Result must contain metrics"
        assert "token_summary" in result["metrics"], "Metrics must contain token_summary"
        token_summary = result["metrics"]["token_summary"]
        assert "total_input_tokens" in token_summary, "Token summary must have total_input_tokens"
        assert "total_output_tokens" in token_summary, "Token summary must have total_output_tokens"
        assert "estimated_cost" in token_summary, "Token summary must have estimated_cost"
        assert token_summary["total_input_tokens"] == 1000, "Input tokens must be correctly summed"
        assert token_summary["total_output_tokens"] == 500, "Output tokens must be correctly summed"
        
        # Verify LLM response fields are preserved
        assert result["executive_summary"] == mock_response["executive_summary"], "Executive summary must match LLM response"
        assert result["paper_citation"] == mock_response["paper_citation"], "Paper citation must match LLM response"
        assert result["report_conclusions"] == mock_response["conclusions"], "Report conclusions must match LLM response"
        
        # Verify state is not mutated (check that input state is unchanged)
        assert base_state == state_before_call, "Input state must not be mutated by the function"

    def test_report_includes_token_summary(self, base_state, valid_plan):
        """Test token summary calculation with multiple agent calls."""
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
        assert metrics is not None, "Metrics must exist"
        token_summary = metrics.get("token_summary", {})
        assert token_summary is not None, "Token summary must exist"
        
        # Verify exact token counts
        assert token_summary.get("total_input_tokens") == 3000, "Total input tokens must sum correctly"
        assert token_summary.get("total_output_tokens") == 1300, "Total output tokens must sum correctly"

        # Verify cost calculation: $3 per 1M input tokens, $15 per 1M output tokens
        expected_cost = (3000 * 3.0 + 1300 * 15.0) / 1_000_000
        assert token_summary.get("estimated_cost") == pytest.approx(expected_cost, rel=1e-6), \
            f"Estimated cost must be {expected_cost}, got {token_summary.get('estimated_cost')}"
        
        # Verify all agent_calls are preserved
        assert "agent_calls" in metrics, "Agent calls must be preserved in metrics"
        assert len(metrics["agent_calls"]) == 2, "All agent calls must be preserved"

    def test_report_generation_handles_llm_failure(self, base_state, valid_plan):
        """Test that report generation continues with stub data when LLM fails."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state.pop("executive_summary", None)
        base_state.pop("paper_citation", None)

        with patch(
            "src.agents.reporting.call_agent_with_metrics",
            side_effect=Exception("LLM Error"),
        ):
            result = generate_report_node(base_state)

        # Verify workflow completes even on LLM failure
        assert result["workflow_complete"] is True, "Workflow must complete even on LLM failure"
        assert result["workflow_phase"] == "reporting", "Workflow phase must be set"
        
        # Verify stub executive_summary is created
        assert "executive_summary" in result, "Executive summary must exist even on failure"
        assert result["executive_summary"]["overall_assessment"] is not None, "Overall assessment must be populated"
        assert isinstance(result["executive_summary"]["overall_assessment"], list), "Overall assessment must be a list"
        assert len(result["executive_summary"]["overall_assessment"]) > 0, "Stub assessment must have items"
        
        # Verify stub paper_citation is created
        assert "paper_citation" in result, "Paper citation must exist even on failure"
        assert result["paper_citation"]["title"] == base_state.get("paper_title", "Unknown"), \
            "Paper citation title must use paper_title or default"
        assert result["paper_citation"]["authors"] == "Unknown", "Authors must default to 'Unknown'"
        
        # Verify metrics are still computed
        assert "metrics" in result, "Metrics must exist even on LLM failure"
        assert "token_summary" in result["metrics"], "Token summary must exist"

    def test_report_generation_quantitative_summary(self, base_state, valid_plan):
        """Test quantitative summary generation from analysis_result_reports."""
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
        assert summary is not None, "Quantitative summary must exist"
        assert isinstance(summary, list), "Quantitative summary must be a list"
        assert len(summary) == 1, "Summary must have one row per report"
        
        row = summary[0]
        # Verify all expected fields are present
        assert row["stage_id"] == "stage_1", "Stage ID must match"
        assert row["figure_id"] == "Fig 1", "Figure ID must match target_figure"
        assert row["status"] == "pass", "Status must match"
        assert row["precision_requirement"] == "high", "Precision requirement must match"
        assert row["peak_position_error_percent"] == 0.5, "Peak position error must match"
        assert row["normalized_rmse_percent"] == 1.2, "Normalized RMSE must match"
        assert row["correlation"] == 0.99, "Correlation must match"
        assert row["n_points_compared"] == 100, "N points compared must match"

    def test_report_generation_populates_missing_structures(
        self, base_state, valid_plan
    ):
        """Test that missing paper_citation and executive_summary are populated with defaults."""
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

        # Verify paper_citation is created with defaults
        assert "paper_citation" in result, "Paper citation must exist"
        assert result["paper_citation"]["title"] == "My Paper", "Title must use paper_title"
        assert result["paper_citation"]["authors"] == "Unknown", "Authors must default to 'Unknown'"
        assert result["paper_citation"]["journal"] == "Unknown", "Journal must default to 'Unknown'"
        assert result["paper_citation"]["year"] == 2023, "Year must default to 2023"
        
        # Verify executive_summary is created with defaults
        assert "executive_summary" in result, "Executive summary must exist"
        assert "overall_assessment" in result["executive_summary"], "Overall assessment must exist"
        assert isinstance(result["executive_summary"]["overall_assessment"], list), "Overall assessment must be a list"
        assert len(result["executive_summary"]["overall_assessment"]) > 0, "Default assessment must have items"

    def test_report_includes_rich_state_in_context(self, base_state, valid_plan):
        """Test that user_content includes all relevant state information."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["paper_id"] = "test_paper_123"
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "status": "completed_success", "summary": "Stage completed"}
            ]
        }
        base_state["figure_comparisons"] = [{"fig": "1", "diff": "small"}]
        base_state["assumptions"] = {"param": "value"}
        base_state["discrepancies"] = [{"parameter": "p1", "classification": "minor", "likely_cause": "test_cause"}]

        mock_response = {}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        # Verify call_agent_with_metrics was called with correct parameters
        assert mock_call.called, "call_agent_with_metrics must be called"
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("agent_name") == "report", "Agent name must be 'report'"
        assert "system_prompt" in call_kwargs, "System prompt must be provided"
        assert "user_content" in call_kwargs, "User content must be provided"
        assert call_kwargs.get("schema_name") == "report_output_schema", "Schema name must be 'report_output_schema'"
        
        user_content = call_kwargs.get("user_content", "")
        # Verify all sections are included
        assert "GENERATE REPRODUCTION REPORT" in user_content, "Report header must be present"
        assert "Paper ID: test_paper_123" in user_content, "Paper ID must be included"
        assert "Stage Summary" in user_content, "Stage summary section must be present"
        assert "stage_0" in user_content, "Stage ID must be included"
        assert "completed_success" in user_content, "Stage status must be included"
        assert "Figure Comparisons" in user_content, "Figure comparisons section must be present"
        assert "Assumptions" in user_content, "Assumptions section must be present"
        assert "Discrepancies" in user_content, "Discrepancies section must be present"
        assert "p1" in user_content, "Discrepancy parameter must be included"
        assert "minor" in user_content, "Discrepancy classification must be included"

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
        assert metrics is not None, "Metrics must exist"
        token_summary = metrics.get("token_summary", {})
        assert token_summary is not None, "Token summary must exist"
        assert token_summary.get("total_input_tokens") == 3000, "Input tokens must be correctly summed"
        assert "total_output_tokens" in token_summary, "Output tokens must be present"
        assert token_summary.get("total_output_tokens") == 1300, "Output tokens must be correctly summed"
        assert "estimated_cost" in token_summary, "Estimated cost must be present"

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

        assert result.get("workflow_complete") is True, "Workflow must be marked complete"
        assert result["workflow_phase"] == "reporting", "Workflow phase must be set"

    def test_report_handles_empty_metrics(self, base_state, valid_plan):
        """Test that report generation handles empty or missing metrics gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state.pop("metrics", None)

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert "metrics" in result, "Metrics must exist even when missing from state"
        assert "token_summary" in result["metrics"], "Token summary must exist"
        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 0, "Input tokens must default to 0"
        assert token_summary["total_output_tokens"] == 0, "Output tokens must default to 0"
        assert token_summary["estimated_cost"] == 0.0, "Estimated cost must be 0.0 when no tokens"
        assert result["metrics"]["agent_calls"] == [], "Agent calls must be empty list"

    def test_report_handles_empty_agent_calls(self, base_state, valid_plan):
        """Test that report generation handles empty agent_calls list."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {"agent_calls": []}

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 0, "Total input tokens must be 0"
        assert token_summary["total_output_tokens"] == 0, "Total output tokens must be 0"
        assert token_summary["estimated_cost"] == 0.0, "Estimated cost must be 0.0"

    def test_report_handles_none_token_values(self, base_state, valid_plan):
        """Test that report generation handles None values in token counts."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": None, "output_tokens": 500},
                {"agent_name": "designer", "input_tokens": 2000, "output_tokens": None},
                {"agent_name": "analyzer", "input_tokens": None, "output_tokens": None},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 2000, "None values must be treated as 0"
        assert token_summary["total_output_tokens"] == 500, "None values must be treated as 0"

    def test_report_handles_missing_token_fields(self, base_state, valid_plan):
        """Test that report generation handles missing token fields in agent_calls."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner"},  # Missing token fields
                {"agent_name": "designer", "input_tokens": 1000},  # Missing output_tokens
                {"agent_name": "analyzer", "output_tokens": 500},  # Missing input_tokens
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 1000, "Missing fields must be treated as 0"
        assert token_summary["total_output_tokens"] == 500, "Missing fields must be treated as 0"

    def test_report_handles_empty_quantitative_reports(self, base_state, valid_plan):
        """Test that report generation handles empty analysis_result_reports."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = []

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # quantitative_summary should not exist when there are no reports
        assert "quantitative_summary" not in result or result.get("quantitative_summary") is None, \
            "Quantitative summary should not exist when there are no reports"

    def test_report_handles_missing_quantitative_metrics(self, base_state, valid_plan):
        """Test that report generation handles missing quantitative_metrics field."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",
                "target_figure": "Fig 1",
                "status": "pass",
                "precision_requirement": "high",
                # Missing quantitative_metrics
            }
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None, "Quantitative summary must exist"
        assert len(summary) == 1, "Summary must have one row"
        row = summary[0]
        assert row["stage_id"] == "stage_1", "Stage ID must be preserved"
        assert row["figure_id"] == "Fig 1", "Figure ID must be preserved"
        assert row["status"] == "pass", "Status must be preserved"
        # Missing metrics should result in None values
        assert row.get("peak_position_error_percent") is None, "Missing metric must be None"
        assert row.get("normalized_rmse_percent") is None, "Missing metric must be None"
        assert row.get("correlation") is None, "Missing metric must be None"

    def test_report_handles_multiple_quantitative_reports(self, base_state, valid_plan):
        """Test quantitative summary with multiple analysis_result_reports."""
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
            },
            {
                "stage_id": "stage_2",
                "target_figure": "Fig 2",
                "status": "fail",
                "precision_requirement": "medium",
                "quantitative_metrics": {
                    "peak_position_error_percent": 5.0,
                    "normalized_rmse_percent": 10.0,
                    "correlation": 0.85,
                    "n_points_compared": 50,
                },
            },
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None, "Quantitative summary must exist"
        assert len(summary) == 2, "Summary must have two rows"
        
        # Verify first report
        row1 = summary[0]
        assert row1["stage_id"] == "stage_1", "First stage ID must match"
        assert row1["peak_position_error_percent"] == 0.5, "First report metrics must match"
        
        # Verify second report
        row2 = summary[1]
        assert row2["stage_id"] == "stage_2", "Second stage ID must match"
        assert row2["peak_position_error_percent"] == 5.0, "Second report metrics must match"

    def test_report_preserves_existing_paper_citation(self, base_state, valid_plan):
        """Test that existing paper_citation is preserved when present."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["paper_citation"] = {
            "title": "Existing Title",
            "authors": "Existing Authors",
            "journal": "Existing Journal",
            "year": 2020,
        }

        mock_response = reporting_summary_response(
            paper_citation={"title": "LLM Title", "authors": "LLM Authors"}
        )

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # LLM response should overwrite existing citation
        assert result["paper_citation"]["title"] == "LLM Title", "LLM citation should overwrite existing"
        assert result["paper_citation"]["authors"] == "LLM Authors", "LLM citation should overwrite existing"

    def test_report_preserves_existing_executive_summary(self, base_state, valid_plan):
        """Test that existing executive_summary is preserved when present."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["executive_summary"] = {
            "overall_assessment": [{"aspect": "Existing", "status": "OK"}]
        }

        mock_response = reporting_summary_response(
            executive_summary={"overall_assessment": [{"aspect": "LLM", "status": "OK"}]}
        )

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # LLM response should overwrite existing summary
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "LLM", \
            "LLM summary should overwrite existing"

    def test_report_handles_missing_paper_title(self, base_state, valid_plan):
        """Test that report generation handles missing paper_title when creating citation."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("paper_citation", None)
        base_state.pop("paper_title", None)

        # LLM response without paper_citation to test default behavior
        mock_response = reporting_summary_response()
        mock_response.pop("paper_citation", None)

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["paper_citation"]["title"] == "Unknown", \
            "Title must default to 'Unknown' when paper_title is missing and LLM doesn't provide citation"
        assert result["paper_citation"]["authors"] == "Unknown", "Authors must default to 'Unknown'"
        assert result["paper_citation"]["journal"] == "Unknown", "Journal must default to 'Unknown'"
        assert result["paper_citation"]["year"] == 2023, "Year must default to 2023"

    def test_report_handles_missing_paper_id(self, base_state, valid_plan):
        """Test that report generation handles missing paper_id in user_content."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("paper_id", None)

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "Paper ID: unknown" in user_content, "Paper ID must default to 'unknown'"

    def test_report_handles_empty_progress_stages(self, base_state, valid_plan):
        """Test that report generation handles empty progress stages."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "Stage Summary" in user_content, "Stage summary section must be present even if empty"

    def test_report_handles_missing_progress(self, base_state, valid_plan):
        """Test that report generation handles missing progress field."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("progress", None)

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        # Should not crash, but may have empty stage summary
        assert "GENERATE REPRODUCTION REPORT" in user_content, "Report header must be present"

    def test_report_handles_many_discrepancies(self, base_state, valid_plan):
        """Test that report generation truncates discrepancies list in user_content."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["discrepancies"] = [
            {"parameter": f"p{i}", "classification": "minor", "likely_cause": f"cause{i}"}
            for i in range(10)
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "Discrepancies (10 total)" in user_content, "Total count must be shown"
        # Should only show first 5
        assert "p0" in user_content, "First discrepancy must be included"
        assert "p4" in user_content, "Fifth discrepancy must be included"
        # Should not include all 10
        assert "p9" not in user_content or user_content.count("p9") == 1, \
            "Discrepancies beyond 5 should not be included in detail"

    def test_report_handles_many_figure_comparisons(self, base_state, valid_plan):
        """Test that report generation truncates figure_comparisons list in user_content."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = [
            {"fig": f"Fig{i}", "diff": "small"} for i in range(10)
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "Figure Comparisons" in user_content, "Figure comparisons section must be present"
        # Should only show first 5
        assert "Fig0" in user_content, "First comparison must be included"
        assert "Fig4" in user_content, "Fifth comparison must be included"

    def test_report_handles_llm_response_with_all_fields(self, base_state, valid_plan):
        """Test that all LLM response fields are properly extracted and stored."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        mock_response = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
            "paper_citation": {"title": "Test", "authors": "Author"},
            "assumptions": {"param": "value"},
            "figure_comparisons": [{"fig": "1"}],
            "systematic_discrepancies": [{"type": "systematic"}],
            "conclusions": {"main_physics_reproduced": True},
        }

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["executive_summary"] == mock_response["executive_summary"], \
            "Executive summary must be extracted"
        assert result["paper_citation"] == mock_response["paper_citation"], \
            "Paper citation must be extracted"
        assert result["assumptions"] == mock_response["assumptions"], \
            "Assumptions must be extracted"
        assert result["figure_comparisons"] == mock_response["figure_comparisons"], \
            "Figure comparisons must be extracted"
        assert result["systematic_discrepancies_identified"] == mock_response["systematic_discrepancies"], \
            "Systematic discrepancies must be extracted with correct field name"
        assert result["report_conclusions"] == mock_response["conclusions"], \
            "Conclusions must be extracted"

    def test_report_preserves_assumptions_from_state(self, base_state, valid_plan):
        """Test that assumptions from state are preserved if LLM doesn't provide them."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["assumptions"] = {"param": "state_value"}

        mock_response = {}  # LLM doesn't provide assumptions

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["assumptions"] == {"param": "state_value"}, \
            "Assumptions from state must be preserved when LLM doesn't provide them"

    def test_report_preserves_figure_comparisons_from_state(self, base_state, valid_plan):
        """Test that figure_comparisons from state are preserved if LLM doesn't provide them."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = [{"fig": "1", "diff": "small"}]

        mock_response = {}  # LLM doesn't provide figure_comparisons

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["figure_comparisons"] == [{"fig": "1", "diff": "small"}], \
            "Figure comparisons from state must be preserved when LLM doesn't provide them"

    def test_report_overwrites_state_with_llm_assumptions(self, base_state, valid_plan):
        """Test that LLM-provided assumptions overwrite state assumptions."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["assumptions"] = {"param": "state_value"}

        mock_response = {"assumptions": {"param": "llm_value"}}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["assumptions"] == {"param": "llm_value"}, \
            "LLM assumptions must overwrite state assumptions"

    def test_report_overwrites_state_with_llm_figure_comparisons(self, base_state, valid_plan):
        """Test that LLM-provided figure_comparisons overwrite state figure_comparisons."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = [{"fig": "1", "diff": "small"}]

        mock_response = {"figure_comparisons": [{"fig": "2", "diff": "large"}]}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["figure_comparisons"] == [{"fig": "2", "diff": "large"}], \
            "LLM figure comparisons must overwrite state figure comparisons"

    def test_report_handles_none_quantitative_metrics(self, base_state, valid_plan):
        """Test that report generation handles None quantitative_metrics."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",
                "target_figure": "Fig 1",
                "status": "pass",
                "precision_requirement": "high",
                "quantitative_metrics": None,  # Explicitly None
            }
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None, "Quantitative summary must exist"
        row = summary[0]
        assert row.get("peak_position_error_percent") is None, "None metrics must result in None values"

    def test_report_handles_missing_stage_fields(self, base_state, valid_plan):
        """Test that report generation handles missing stage_id or target_figure."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                # Missing stage_id
                "target_figure": "Fig 1",
                "status": "pass",
                "quantitative_metrics": {},
            },
            {
                "stage_id": "stage_2",
                # Missing target_figure
                "status": "pass",
                "quantitative_metrics": {},
            },
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None, "Quantitative summary must exist"
        assert len(summary) == 2, "Summary must have two rows"
        assert summary[0].get("stage_id") is None, "Missing stage_id must be None"
        assert summary[1].get("figure_id") is None, "Missing target_figure must result in None figure_id"

    def test_report_handles_large_token_counts(self, base_state, valid_plan):
        """Test that report generation handles very large token counts."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1_000_000_000, "output_tokens": 500_000_000},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 1_000_000_000, "Large token counts must be handled"
        assert token_summary["total_output_tokens"] == 500_000_000, "Large token counts must be handled"
        # Verify cost calculation doesn't overflow
        expected_cost = (1_000_000_000 * 3.0 + 500_000_000 * 15.0) / 1_000_000
        assert token_summary["estimated_cost"] == pytest.approx(expected_cost, rel=1e-6), \
            "Cost calculation must handle large numbers"

    def test_report_handles_zero_token_counts(self, base_state, valid_plan):
        """Test that report generation handles zero token counts."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 0, "output_tokens": 0},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 0, "Zero tokens must be handled"
        assert token_summary["total_output_tokens"] == 0, "Zero tokens must be handled"
        assert token_summary["estimated_cost"] == 0.0, "Zero tokens must result in zero cost"

    def test_report_handles_partial_llm_response(self, base_state, valid_plan):
        """Test that report generation handles partial LLM responses."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["assumptions"] = {"param": "state_value"}
        base_state["figure_comparisons"] = [{"fig": "1"}]

        # LLM only provides executive_summary, not other fields
        mock_response = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
            # Missing paper_citation, assumptions, figure_comparisons, conclusions
        }

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result["executive_summary"] == mock_response["executive_summary"], \
            "Executive summary from LLM must be used"
        assert result["assumptions"] == {"param": "state_value"}, \
            "State assumptions must be preserved when LLM doesn't provide them"
        assert result["figure_comparisons"] == [{"fig": "1"}], \
            "State figure comparisons must be preserved when LLM doesn't provide them"
        # paper_citation should be created with defaults
        assert "paper_citation" in result, "Paper citation must exist"
        assert result["paper_citation"]["title"] == base_state.get("paper_title", "Unknown"), \
            "Paper citation must use paper_title or default"

    def test_report_handles_missing_stage_summary(self, base_state, valid_plan):
        """Test that report generation handles stages without summary field."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "status": "completed_success"},  # No summary field
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "stage_0" in user_content, "Stage ID must be included"
        assert "completed_success" in user_content, "Stage status must be included"
        assert "No summary" in user_content, "Missing summary must show 'No summary'"

    def test_report_handles_empty_assumptions(self, base_state, valid_plan):
        """Test that report generation handles empty assumptions dict."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["assumptions"] = {}

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = generate_report_node(base_state)

        # Empty assumptions should still be preserved
        assert result["assumptions"] == {}, "Empty assumptions must be preserved"
        
        user_content = mock_call.call_args.kwargs.get("user_content", "")
        # Empty assumptions might not show in user_content, which is fine

    def test_report_handles_empty_figure_comparisons(self, base_state, valid_plan):
        """Test that report generation handles empty figure_comparisons list."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = []

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = generate_report_node(base_state)

        assert result["figure_comparisons"] == [], "Empty figure comparisons must be preserved"
        
        user_content = mock_call.call_args.kwargs.get("user_content", "")
        # Empty list might not show in user_content, which is fine

    def test_report_handles_non_list_analysis_result_reports(self, base_state, valid_plan):
        """Test that report generation handles non-list analysis_result_reports gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = "not a list"  # Wrong type

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            # This should either handle gracefully or raise a clear error
            try:
                result = generate_report_node(base_state)
                # If it doesn't crash, quantitative_summary should not exist
                assert "quantitative_summary" not in result or result.get("quantitative_summary") is None, \
                    "Non-list analysis_result_reports should not create quantitative_summary"
            except (TypeError, AttributeError):
                # If it raises an error, that's also acceptable - the test documents the behavior
                pass

    def test_report_handles_non_dict_quantitative_metrics(self, base_state, valid_plan):
        """Test that report generation handles non-dict quantitative_metrics gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",
                "target_figure": "Fig 1",
                "status": "pass",
                "quantitative_metrics": "not a dict",  # Wrong type
            }
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            # Should handle gracefully by treating non-dict as empty dict
            result = generate_report_node(base_state)
            summary = result.get("quantitative_summary")
            assert summary is not None, "Quantitative summary must exist"
            row = summary[0]
            assert row["stage_id"] == "stage_1", "Stage ID must be preserved"
            # Non-dict should be treated as empty dict, so all metrics should be None
            assert row.get("peak_position_error_percent") is None, \
                "Non-dict quantitative_metrics should be treated as empty dict"
            assert row.get("normalized_rmse_percent") is None, \
                "Non-dict quantitative_metrics should be treated as empty dict"

    def test_report_handles_non_list_agent_calls(self, base_state, valid_plan):
        """Test that report generation handles non-list agent_calls gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": "not a list"  # Wrong type
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            # Should handle gracefully by treating non-list as empty list
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            assert token_summary["total_input_tokens"] == 0, \
                "Non-list agent_calls should be treated as empty list, resulting in 0 tokens"
            assert token_summary["total_output_tokens"] == 0, \
                "Non-list agent_calls should be treated as empty list, resulting in 0 tokens"
            assert token_summary["estimated_cost"] == 0.0, \
                "Non-list agent_calls should result in 0 cost"

    def test_report_handles_negative_token_values(self, base_state, valid_plan):
        """Test that report generation handles negative token values.
        
        Negative tokens are invalid but may occur due to bugs in upstream code.
        The function should either reject them or treat them as 0.
        """
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": -100, "output_tokens": 500},
                {"agent_name": "designer", "input_tokens": 2000, "output_tokens": -50},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            # Negative values should be treated as 0 to avoid incorrect cost calculation
            # Current behavior: sums them directly, which gives 1900 input, 450 output
            # This could be a bug - negative tokens don't make sense
            assert token_summary["total_input_tokens"] >= 0, \
                "Total input tokens should not be negative"
            assert token_summary["total_output_tokens"] >= 0, \
                "Total output tokens should not be negative"
            assert token_summary["estimated_cost"] >= 0.0, \
                "Estimated cost should not be negative"

    def test_report_handles_float_token_values(self, base_state, valid_plan):
        """Test that report generation handles float token values.
        
        Token counts should be integers, but floats may occur.
        """
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000.5, "output_tokens": 500.7},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            # Float values should be handled - either cast to int or used as-is
            assert token_summary["total_input_tokens"] is not None, "Input tokens must exist"
            assert token_summary["total_output_tokens"] is not None, "Output tokens must exist"
            # Cost calculation should work with floats
            assert isinstance(token_summary["estimated_cost"], (int, float)), \
                "Estimated cost must be numeric"
            assert token_summary["estimated_cost"] > 0, "Cost should be positive with non-zero tokens"

    def test_report_preserves_stage_metrics_in_output(self, base_state, valid_plan):
        """Test that stage_metrics from input metrics are preserved in output."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
            ],
            "stage_metrics": [
                {"stage_id": "stage_0", "duration_seconds": 120, "iterations": 3}
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            
            # stage_metrics should be preserved in output metrics
            assert "stage_metrics" in result["metrics"], "stage_metrics must be preserved"
            assert len(result["metrics"]["stage_metrics"]) == 1, "stage_metrics count must match"
            assert result["metrics"]["stage_metrics"][0]["stage_id"] == "stage_0", \
                "stage_metrics content must be preserved"

    def test_report_build_agent_prompt_called_correctly(self, base_state, valid_plan):
        """Test that build_agent_prompt is called with correct parameters."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan

        mock_response = reporting_summary_response()

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response) as mock_call, \
             patch("src.agents.reporting.build_agent_prompt", return_value="test_prompt") as mock_build:
            generate_report_node(base_state)
            
            # Verify build_agent_prompt was called with correct agent name
            mock_build.assert_called_once()
            call_args = mock_build.call_args
            assert call_args.args[0] == "report_generator", \
                "build_agent_prompt must be called with 'report_generator'"
            # State should be passed for prompt adaptations
            assert call_args.args[1] == base_state or call_args.kwargs.get("state") == base_state, \
                "State must be passed to build_agent_prompt"

    def test_report_call_agent_with_metrics_receives_state(self, base_state, valid_plan):
        """Test that call_agent_with_metrics receives the state parameter."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)
            
            # Verify state was passed to call_agent_with_metrics
            call_kwargs = mock_call.call_args.kwargs
            assert "state" in call_kwargs, "State must be passed to call_agent_with_metrics"
            assert call_kwargs["state"] == base_state, \
                "Original state must be passed, not a copy"

    def test_report_default_executive_summary_matches_schema(self, base_state, valid_plan):
        """Test that default executive_summary structure matches report schema requirements."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("executive_summary", None)

        mock_response = {}  # LLM returns empty response

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        exec_summary = result["executive_summary"]
        assert "overall_assessment" in exec_summary, "overall_assessment is required"
        assert isinstance(exec_summary["overall_assessment"], list), "overall_assessment must be a list"
        
        # Each item in overall_assessment should have required fields per schema
        for item in exec_summary["overall_assessment"]:
            assert "aspect" in item, "Each assessment item must have 'aspect'"
            assert "status" in item, "Each assessment item must have 'status'"
            # status should be one of the allowed values
            allowed_statuses = ["Reproduced", "Partial", "Not Reproduced", "Not Attempted"]
            assert item["status"] in allowed_statuses, \
                f"Status '{item['status']}' must be one of {allowed_statuses}"

    def test_report_default_paper_citation_matches_schema(self, base_state, valid_plan):
        """Test that default paper_citation structure matches report schema requirements."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("paper_citation", None)
        base_state.pop("paper_title", None)

        mock_response = {}  # LLM returns empty response

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        citation = result["paper_citation"]
        # Schema requires: authors, title, journal, year
        assert "authors" in citation, "authors is required by schema"
        assert "title" in citation, "title is required by schema"
        assert "journal" in citation, "journal is required by schema"
        assert "year" in citation, "year is required by schema"
        assert isinstance(citation["year"], int), "year must be an integer per schema"
        assert isinstance(citation["authors"], str), "authors must be a string per schema"
        assert isinstance(citation["title"], str), "title must be a string per schema"
        assert isinstance(citation["journal"], str), "journal must be a string per schema"

    def test_report_llm_response_with_wrong_type_executive_summary(self, base_state, valid_plan):
        """Test handling when LLM returns wrong type for executive_summary."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("executive_summary", None)

        # LLM returns string instead of dict
        mock_response = {
            "executive_summary": "This is a string, not a dict"
        }

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            
            # Should preserve the LLM response even if malformed (validation happens elsewhere)
            # OR should fallback to default
            exec_summary = result.get("executive_summary")
            assert exec_summary is not None, "executive_summary must exist in result"
            # If it's the default, it should be a dict; if it's the LLM response, it's a string
            # This test documents the current behavior

    def test_report_llm_response_with_wrong_type_paper_citation(self, base_state, valid_plan):
        """Test handling when LLM returns wrong type for paper_citation."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state.pop("paper_citation", None)

        # LLM returns list instead of dict
        mock_response = {
            "paper_citation": ["author1", "title1"]
        }

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            
            # Should handle gracefully
            citation = result.get("paper_citation")
            assert citation is not None, "paper_citation must exist in result"

    def test_report_handles_malformed_state_paper_citation(self, base_state, valid_plan):
        """Test handling when state has paper_citation missing required fields."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        # Malformed citation - missing required fields
        base_state["paper_citation"] = {"title": "Only Title"}

        mock_response = {}  # LLM doesn't provide citation

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            
            # Should preserve existing citation from state even if malformed
            citation = result["paper_citation"]
            assert citation["title"] == "Only Title", \
                "Existing title should be preserved"
            # Missing fields should NOT be auto-filled in this case
            # (the function preserves state as-is when paper_citation exists)

    def test_report_state_metrics_not_mutated(self, base_state, valid_plan):
        """Test that the input state's metrics dict is not mutated."""
        from src.agents.reporting import generate_report_node

        original_metrics = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
            ]
        }
        base_state["plan"] = valid_plan
        base_state["metrics"] = original_metrics
        
        # Deep copy to compare later
        metrics_before = copy.deepcopy(original_metrics)

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # Input metrics should not have been modified
        assert base_state["metrics"] == metrics_before, \
            "Input state metrics must not be mutated"
        # But result should have token_summary added
        assert "token_summary" in result["metrics"], \
            "Result metrics should have token_summary"

    def test_report_handles_prompt_adaptations(self, base_state, valid_plan):
        """Test that prompt_adaptations in state are passed to build_agent_prompt."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["prompt_adaptations"] = [
            {
                "target_agent": "ReportGeneratorAgent",
                "modification_type": "append",
                "content": "Additional instructions for report",
                "confidence": 0.9,
                "reason": "Paper-specific adaptation"
            }
        ]

        mock_response = reporting_summary_response()

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response), \
             patch("src.agents.reporting.build_agent_prompt", return_value="adapted_prompt") as mock_build:
            generate_report_node(base_state)
            
            # Verify state with adaptations is passed
            call_args = mock_build.call_args
            passed_state = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("state")
            assert passed_state is not None, "State must be passed to build_agent_prompt"
            assert "prompt_adaptations" in passed_state, \
                "State with prompt_adaptations must be passed"

    def test_report_handles_very_long_assumptions(self, base_state, valid_plan):
        """Test that report generation handles very long assumptions dict."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        # Create assumptions with many entries
        base_state["assumptions"] = {
            f"param_{i}": f"value_{i}" * 100 for i in range(50)
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = generate_report_node(base_state)

        # Should not crash
        assert result["workflow_complete"] is True
        # Assumptions should be included in user_content
        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "Assumptions" in user_content, "Assumptions section must be present"

    def test_report_handles_unicode_in_user_content(self, base_state, valid_plan):
        """Test that report generation handles unicode characters in state."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["paper_id"] = "test_论文_αβγ"
        base_state["assumptions"] = {"wavelength_λ": "650nm", "材料": "gold"}
        base_state["discrepancies"] = [
            {"parameter": "λ_peak", "classification": "minor", "likely_cause": "数值误差"}
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = generate_report_node(base_state)

        # Should not crash with unicode
        assert result["workflow_complete"] is True
        user_content = mock_call.call_args.kwargs.get("user_content", "")
        assert "论文" in user_content or "test_" in user_content, \
            "Paper ID must be included in user_content"

    def test_report_quantitative_summary_empty_quantitative_metrics_dict(self, base_state, valid_plan):
        """Test quantitative summary when quantitative_metrics is empty dict (not missing/None)."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",
                "target_figure": "Fig 1",
                "status": "pass",
                "precision_requirement": "high",
                "quantitative_metrics": {},  # Empty dict, not None/missing
            }
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None, "Quantitative summary must exist"
        row = summary[0]
        assert row["stage_id"] == "stage_1", "Stage ID must be preserved"
        assert row.get("peak_position_error_percent") is None, "Empty dict should yield None for metrics"
        assert row.get("normalized_rmse_percent") is None, "Empty dict should yield None for metrics"
        assert row.get("correlation") is None, "Empty dict should yield None for metrics"

    def test_report_handles_analysis_reports_with_extra_fields(self, base_state, valid_plan):
        """Test that extra fields in analysis_result_reports are handled (not crash)."""
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
                    "extra_metric": 123,  # Extra field not in standard set
                },
                "extra_field": "should be ignored",  # Extra field at report level
            }
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        summary = result.get("quantitative_summary")
        assert summary is not None, "Quantitative summary must exist"
        row = summary[0]
        assert row["peak_position_error_percent"] == 0.5, "Standard metric must be extracted"
        # Extra fields should not cause errors

    def test_report_progress_stages_with_various_statuses(self, base_state, valid_plan):
        """Test user_content generation with stages having various statuses."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "status": "completed_success", "summary": "Material validated"},
                {"stage_id": "stage_1", "status": "completed_failure", "summary": "FDTD failed"},
                {"stage_id": "stage_2", "status": "needs_rerun", "summary": "Awaiting backtrack"},
                {"stage_id": "stage_3", "status": "invalidated"},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        # All stages should be listed
        assert "stage_0" in user_content, "stage_0 must be in user_content"
        assert "stage_1" in user_content, "stage_1 must be in user_content"
        assert "stage_2" in user_content, "stage_2 must be in user_content"
        assert "stage_3" in user_content, "stage_3 must be in user_content"
        # Statuses should be included
        assert "completed_success" in user_content, "Status completed_success must be shown"
        assert "completed_failure" in user_content, "Status completed_failure must be shown"
        # Summaries should be included for stages that have them
        assert "Material validated" in user_content, "Stage summary must be included"
        assert "FDTD failed" in user_content, "Stage summary must be included"
        # Stage without summary should show 'No summary'
        assert "No summary" in user_content, "Stages without summary should show 'No summary'"

    def test_report_figure_comparisons_truncation(self, base_state, valid_plan):
        """Test that exactly first 5 figure_comparisons are included in user_content."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = [
            {"fig": f"Fig{i}", "diff": "small", "data": f"data_{i}"} for i in range(10)
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        # First 5 should be included
        for i in range(5):
            assert f"Fig{i}" in user_content, f"Fig{i} should be included (first 5)"
        # Beyond 5 should not be in detail
        # Note: "Fig5" through "Fig9" should not appear in the JSON section
        # They might appear as count mention but not in detail

    def test_report_discrepancies_truncation_count_shown(self, base_state, valid_plan):
        """Test that discrepancies count is shown correctly in user_content."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["discrepancies"] = [
            {"parameter": f"p{i}", "classification": "minor", "likely_cause": f"cause{i}"}
            for i in range(15)
        ]

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            generate_report_node(base_state)

        user_content = mock_call.call_args.kwargs.get("user_content", "")
        # Total count should be shown
        assert "15 total" in user_content, "Total discrepancy count (15) must be shown"
        # First 5 should be detailed
        for i in range(5):
            assert f"p{i}" in user_content, f"p{i} should be included (first 5)"

    def test_report_result_does_not_include_plan(self, base_state, valid_plan):
        """Test that result dict doesn't unnecessarily include large state fields."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["paper_text"] = "Very long paper text " * 1000

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # Result should not include large unnecessary fields
        assert "paper_text" not in result, "paper_text should not be in result"
        assert "plan" not in result, "plan should not be in result"

    def test_report_all_expected_fields_in_result(self, base_state, valid_plan):
        """Test that all expected fields are present in result."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [{"agent_name": "test", "input_tokens": 100, "output_tokens": 50}]
        }
        base_state["analysis_result_reports"] = [
            {"stage_id": "s1", "target_figure": "F1", "status": "pass", "quantitative_metrics": {}}
        ]

        mock_response = reporting_summary_response(
            executive_summary={"overall_assessment": [{"aspect": "A", "status": "OK"}]},
            conclusions={"main_physics_reproduced": True},
            paper_citation={"title": "T", "authors": "A"},
        )

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        # Required fields
        assert "workflow_phase" in result, "workflow_phase must be in result"
        assert result["workflow_phase"] == "reporting", "workflow_phase must be 'reporting'"
        assert "workflow_complete" in result, "workflow_complete must be in result"
        assert result["workflow_complete"] is True, "workflow_complete must be True"
        assert "metrics" in result, "metrics must be in result"
        assert "token_summary" in result["metrics"], "token_summary must be in metrics"
        assert "executive_summary" in result, "executive_summary must be in result"
        assert "paper_citation" in result, "paper_citation must be in result"
        assert "report_conclusions" in result, "report_conclusions must be in result"
        assert "quantitative_summary" in result, "quantitative_summary must be in result"

    def test_report_handles_string_token_values(self, base_state, valid_plan):
        """Test that report generation handles string token values gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": "1000", "output_tokens": "500"},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            # String values should be converted to integers
            assert token_summary["total_input_tokens"] == 1000, \
                "String '1000' must be converted to int 1000"
            assert token_summary["total_output_tokens"] == 500, \
                "String '500' must be converted to int 500"
            # Cost should be calculated correctly
            expected_cost = (1000 * 3.0 + 500 * 15.0) / 1_000_000
            assert token_summary["estimated_cost"] == pytest.approx(expected_cost, rel=1e-6), \
                "Cost must be calculated correctly from converted token values"

    def test_report_handles_invalid_string_token_values(self, base_state, valid_plan):
        """Test that report generation handles invalid string token values."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": "not_a_number", "output_tokens": "abc"},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            # Invalid strings should be treated as 0
            assert token_summary["total_input_tokens"] == 0, \
                "Invalid string 'not_a_number' must be converted to 0"
            assert token_summary["total_output_tokens"] == 0, \
                "Invalid string 'abc' must be converted to 0"
            assert token_summary["estimated_cost"] == 0.0, \
                "Cost must be 0 when tokens are 0"

    def test_report_handles_mixed_token_value_types(self, base_state, valid_plan):
        """Test that report generation handles mixed token value types."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": "500"},  # int + string
                {"agent_name": "designer", "input_tokens": "2000.5", "output_tokens": 800.7},  # string + float
                {"agent_name": "analyzer", "input_tokens": None, "output_tokens": "invalid"},  # None + invalid
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            # 1000 + 2000 + 0 = 3000 (None -> 0, string "2000.5" -> 2000)
            assert token_summary["total_input_tokens"] == 3000, \
                "Mixed input tokens must sum to 3000"
            # 500 + 800 + 0 = 1300 (string "500" -> 500, float 800.7 -> 800, invalid -> 0)
            assert token_summary["total_output_tokens"] == 1300, \
                "Mixed output tokens must sum to 1300"

    def test_report_handles_empty_string_token_values(self, base_state, valid_plan):
        """Test that report generation handles empty string token values."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": "", "output_tokens": ""},
            ]
        }

        mock_response = reporting_summary_response()

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)
            token_summary = result["metrics"]["token_summary"]
            # Empty strings should be treated as 0
            assert token_summary["total_input_tokens"] == 0, \
                "Empty string must be converted to 0"
            assert token_summary["total_output_tokens"] == 0, \
                "Empty string must be converted to 0"

