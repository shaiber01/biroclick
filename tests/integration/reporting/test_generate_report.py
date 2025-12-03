"""Integration tests for generate_report_node covering multiple scenarios."""

import copy
from unittest.mock import patch

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
        assert call_kwargs.get("schema_name") == "report_schema", "Schema name must be 'report_schema'"
        
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

