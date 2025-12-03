"""Unit tests for src/agents/reporting.py"""

import pytest
from unittest.mock import patch, MagicMock, ANY
import copy

from src.agents.reporting import (
    generate_report_node,
    handle_backtrack_node,
)


class TestGenerateReportNode:
    """Tests for generate_report_node function."""

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_generates_report_on_success(self, mock_prompt, mock_call):
        """Should generate final report with all expected fields."""
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "Pass"}]},
            "key_findings": ["Finding 1"],
            "recommendations": ["Recommendation 1"],
            "paper_citation": {"title": "Generated Title"},
            "assumptions": {"assumed": "true"},
            "figure_comparisons": [{"fig": "1"}],
            "systematic_discrepancies": ["disc1"],
            "conclusions": ["Conclusion 1"]
        }
        
        state = {
            "paper_id": "test_paper",
            "progress": {"stages": [{"stage_id": "stage1", "status": "completed_success"}]},
            "metrics": {"agent_calls": []},
            # Pre-existing citation should be overwritten if agent returns one? 
            # The code says: 
            # if agent_output.get("paper_citation"): result["paper_citation"] = agent_output["paper_citation"]
            "paper_citation": {"title": "Original Title"} 
        }
        
        result = generate_report_node(state)
        
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Test"
        assert result["paper_citation"]["title"] == "Generated Title" # Verifies agent output overwrites/updates
        assert result["assumptions"] == {"assumed": "true"}
        assert result["figure_comparisons"] == [{"fig": "1"}]
        assert result["systematic_discrepancies_identified"] == ["disc1"] # Check mapping
        assert result["report_conclusions"] == ["Conclusion 1"] # Check mapping

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_constructs_user_content_correctly(self, mock_prompt, mock_call):
        """Should verify user content includes all state components."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "paper_title": "Test Paper",
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "summary": "Done well"}
                ]
            },
            "figure_comparisons": [{"fig_id": "fig1", "status": "match"}],
            "assumptions": {"temp": "300K"},
            "discrepancies": [
                {"parameter": "gap", "classification": "minor", "likely_cause": "noise"}
            ]
        }
        
        generate_report_node(state)
        
        # Verify call arguments
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        assert "Paper ID: test_paper" in user_content
        assert "stage1: completed_success - Done well" in user_content
        assert '"fig_id": "fig1"' in user_content
        assert '"temp": "300K"' in user_content
        assert "- gap: minor - noise" in user_content

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_llm_error(self, mock_prompt, mock_call):
        """Should handle LLM call failure gracefully and return partial results."""
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "paper_id": "test_paper",
            "completed_stages": ["stage1"],
            "paper_title": "Fallback Title"
        }
        
        result = generate_report_node(state)
        
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True
        # Should have defaults
        assert result["paper_citation"]["title"] == "Fallback Title"
        assert "executive_summary" in result
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Material Properties"

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_calculates_metrics_cost(self, mock_prompt, mock_call):
        """Should correctly calculate token costs in metrics summary."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        # Cost formula in code: (input * 3.0 + output * 15.0) / 1_000_000
        state = {
            "paper_id": "test_paper",
            "metrics": {
                "agent_calls": [
                    {"input_tokens": 1000, "output_tokens": 100}, # 3000 + 1500 = 4500
                    {"input_tokens": 2000, "output_tokens": 200}, # 6000 + 3000 = 9000
                ]
            }
        }
        # Total input: 3000, Total output: 300
        # Total cost = (3000 * 3 + 300 * 15) / 1M = (9000 + 4500) / 1M = 0.0135
        
        result = generate_report_node(state)
        
        metrics = result["metrics"]
        summary = metrics["token_summary"]
        
        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 300
        assert summary["estimated_cost"] == pytest.approx(0.0135)

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_builds_quantitative_summary(self, mock_prompt, mock_call):
        """Should build quantitative summary table from reports."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "analysis_result_reports": [
                {
                    "stage_id": "stage1",
                    "target_figure": "Fig1",
                    "status": "match",
                    "precision_requirement": "acceptable",
                    "quantitative_metrics": {
                        "peak_position_error_percent": 1.5,
                        "normalized_rmse_percent": 2.0,
                        "correlation": 0.95,
                        "n_points_compared": 100
                    }
                },
                {
                    # Partial data
                    "stage_id": "stage2",
                    "status": "fail",
                    "quantitative_metrics": None
                }
            ]
        }
        
        result = generate_report_node(state)
        
        summary = result.get("quantitative_summary", [])
        assert len(summary) == 2
        
        row1 = summary[0]
        assert row1["figure_id"] == "Fig1"
        assert row1["peak_position_error_percent"] == 1.5
        assert row1["correlation"] == 0.95
        
        row2 = summary[1]
        assert row2["stage_id"] == "stage2"
        assert row2["peak_position_error_percent"] is None # Should handle missing metrics

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_missing_metrics(self, mock_prompt, mock_call):
        """Should handle missing metrics in state gracefully."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {"paper_id": "test_paper"} # No metrics key
        
        result = generate_report_node(state)
        
        assert "metrics" in result
        assert result["metrics"]["token_summary"]["total_input_tokens"] == 0


class TestHandleBacktrackNode:
    """Tests for handle_backtrack_node function."""

    def test_backtracks_to_stage(self):
        """Should handle backtrack to a specific stage and update status."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "outputs": ["data"]},
                    {"stage_id": "stage2", "status": "completed_success"},
                    {"stage_id": "stage3", "status": "pending"}
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["current_stage_id"] == "stage1"
        assert result["backtrack_count"] == 1
        
        # Verify stage updates
        new_stages = result["progress"]["stages"]
        stage1 = next(s for s in new_stages if s["stage_id"] == "stage1")
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        stage3 = next(s for s in new_stages if s["stage_id"] == "stage3")
        
        assert stage1["status"] == "needs_rerun"
        assert stage1["outputs"] == [] # Should be cleared
        assert "discrepancies" in stage1 and stage1["discrepancies"] == [] # Should be cleared
        
        assert stage2["status"] == "invalidated"
        
        assert stage3["status"] == "pending" # Unaffected

    def test_errors_on_missing_decision(self):
        """Should error when backtrack_decision is missing."""
        state = {"backtrack_decision": None}
        
        result = handle_backtrack_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"

    def test_errors_on_unaccepted_decision(self):
        """Should error when backtrack_decision is not accepted."""
        state = {
            "backtrack_decision": {
                "accepted": False,
                "target_stage_id": "stage1"
            }
        }
        
        result = handle_backtrack_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"

    def test_errors_on_missing_target(self):
        """Should error when target_stage_id is empty."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "",
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "invalid_backtrack_target"

    def test_increments_backtrack_count(self):
        """Should increment backtrack count."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 1,
        }
        
        result = handle_backtrack_node(state)
        
        assert result["backtrack_count"] == 2

    def test_clears_all_downstream_state(self):
        """Should clear all relevant downstream state fields on backtrack."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "code": "old code",
            "design_description": {"old": "design"},
            "stage_outputs": {"stage1": "output"},
            "run_error": "Some error",
            "analysis_summary": "Summary",
            "invalidated_stages": [],
            "last_design_review_verdict": "approved",
            "last_code_review_verdict": "approved",
            "supervisor_verdict": "backtrack",
            # Counters that should be reset
            "design_revision_count": 3,
            "code_revision_count": 2,
            "execution_failure_count": 1,
            "analysis_revision_count": 1,
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["code"] is None
        assert result["design_description"] is None
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        assert result["analysis_summary"] is None
        assert result["last_design_review_verdict"] is None
        assert result["last_code_review_verdict"] is None
        assert result["supervisor_verdict"] is None
        assert result["backtrack_decision"] is None
        
        # Verify counters are reset
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_handles_backtrack_limit(self):
        """Should handle backtrack limit reached."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 3,
            "runtime_config": {"max_backtracks": 2},
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking_limit"
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "backtrack_limit"

    def test_backtrack_to_stage_0_clears_materials(self):
        """Should clear materials when backtracking to Stage 0 (MATERIAL_VALIDATION)."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage0",
                "stages_to_invalidate": ["stage1"],
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0", 
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_success"
                    },
                    {"stage_id": "stage1", "status": "completed_success"},
                ]
            },
            "validated_materials": ["gold.py"],
            "pending_validated_materials": ["silver.py"]
        }
        
        result = handle_backtrack_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []

    def test_errors_on_missing_target_in_progress(self):
        """Should error when target stage is not in progress history."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stageX", # Not in progress
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["ask_user_trigger"] == "backtrack_target_not_found"
        assert result["awaiting_user_input"] is True

    def test_deep_copies_progress(self):
        """Should deep copy progress to avoid side effects on input state."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            }
        }
        
        result = handle_backtrack_node(state)
        
        # Modify result progress
        result["progress"]["stages"][0]["status"] = "modified"
        
        # Input state should remain unchanged
        assert state["progress"]["stages"][0]["status"] == "completed_success"
