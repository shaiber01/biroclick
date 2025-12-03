"""code_review_limit trigger tests."""

from unittest.mock import ANY, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestCodeReviewLimitTrigger:
    """Tests for code_review_limit trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_hint(self, mock_context):
        """Should reset code revision count when user provides hint."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "PROVIDE_HINT: Use numpy instead"},
            "code_revision_count": 5,
        }
        
        result = supervisor_node(state)
        
        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "I dont know"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"

class TestHandleCodeReviewLimit:
    """Direct handler tests for code review limit."""

    def test_handle_code_review_limit_hint(self, mock_state, mock_result):
        user_input = {"q1": "PROVIDE_HINT: Use a loop instead."}
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "code_revision_count", 0)
        assert "Use a loop" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_code_review_limit_skip(self, mock_update, mock_state, mock_result):
        user_input = {"q1": "SKIP"}
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_with(mock_state, "stage1", "blocked", summary=ANY)

    def test_handle_code_review_limit_stop(self, mock_state, mock_result):
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_code_review_limit_unknown(self, mock_state, mock_result):
        user_input = {"q1": "Just keep going"}
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
