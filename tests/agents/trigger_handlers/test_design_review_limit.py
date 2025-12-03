"""design_review_limit trigger tests."""

from unittest.mock import patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestDesignReviewLimitTrigger:
    """Tests for design_review_limit trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_hint(self, mock_context):
        """Should reset design revision count when user provides hint."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "HINT: Try using a larger simulation domain"},
            "design_revision_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["design_revision_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
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
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

class TestHandleDesignReviewLimit:
    """Direct handler tests for design review limit."""

    def test_handle_design_review_limit_hint(self, mock_state, mock_result):
        user_input = {"q1": "PROVIDE_HINT: Fix the dimensions."}
        trigger_handlers.handle_design_review_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "design_revision_count", 0)
        assert "Fix the dimensions" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"
