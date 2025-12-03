"""replan_limit trigger tests."""

from unittest.mock import patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestReplanLimitTrigger:
    """Tests for replan_limit trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_force_accepts_plan(self, mock_context):
        """Should force accept plan on FORCE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "FORCE accept the current plan"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "force" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_accepts_plan_on_accept(self, mock_context):
        """Should accept plan on ACCEPT."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "ACCEPT"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_retries_replan_on_guidance(self, mock_context):
        """Should retry replan with guidance on GUIDANCE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "GUIDANCE: Focus on single wavelength first"},
            "replan_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert result["replan_count"] == 0

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"

class TestHandleReplanLimit:
    """Direct handler tests for replan limit."""

    def test_handle_replan_limit_force(self, mock_state, mock_result):
        user_input = {"q1": "FORCE accept"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_replan_limit_guidance(self, mock_state, mock_result):
        user_input = {"q1": "GUIDANCE: Try this."}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "replan_count", 0)
        assert mock_result["supervisor_verdict"] == "replan_needed"

    def test_handle_replan_limit_unknown(self, mock_state, mock_result):
        user_input = {"q1": "Unknown"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
