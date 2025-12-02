"""Unit tests for src/agents/reporting.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.reporting import (
    generate_report_node,
    handle_backtrack_node,
)


class TestGenerateReportNode:
    """Tests for generate_report_node function."""

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_generates_report_on_success(self, mock_prompt, mock_call):
        """Should generate final report."""
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": []},
            "key_findings": ["Finding 1"],
            "recommendations": ["Recommendation 1"],
        }
        
        state = {
            "paper_id": "test_paper",
            "progress": {"stages": [{"stage_id": "stage1", "status": "completed_success"}]},
        }
        
        result = generate_report_node(state)
        
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_llm_error(self, mock_prompt, mock_call):
        """Should handle LLM call failure gracefully."""
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "paper_id": "test_paper",
            "completed_stages": ["stage1"],
        }
        
        result = generate_report_node(state)
        
        # Should still produce a basic report
        assert "workflow_phase" in result

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_includes_metrics_summary(self, mock_prompt, mock_call):
        """Should include metrics summary in report."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "progress": {"stages": []},
            "metrics": {"agent_calls": [{"input_tokens": 100, "output_tokens": 50}]},
        }
        
        result = generate_report_node(state)
        
        assert "metrics" in result
        assert "token_summary" in result["metrics"]


class TestHandleBacktrackNode:
    """Tests for handle_backtrack_node function."""

    def test_backtracks_to_stage(self):
        """Should handle backtrack to a specific stage."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success"},
                    {"stage_id": "stage2", "status": "completed_success"},
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["current_stage_id"] == "stage1"
        assert result["backtrack_count"] == 1

    def test_errors_on_missing_decision(self):
        """Should error when backtrack_decision is missing."""
        state = {"backtrack_decision": None}
        
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

    def test_clears_downstream_state(self):
        """Should clear downstream state on backtrack."""
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
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["code"] is None
        assert result["design_description"] is None

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

