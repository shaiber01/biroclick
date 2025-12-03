"""Context overflow trigger tests."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestContextOverflowTrigger:
    """Tests for context_overflow trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_continues_on_summarize(self, mock_context):
        """Should continue with summarization on SUMMARIZE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SUMMARIZE"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "summariz" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_truncates_paper_on_truncate(self, mock_context):
        """Should truncate paper on TRUNCATE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": "A" * 30000,  # Long paper
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in result["paper_text"]
        assert len(result["paper_text"]) < 30000

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_short_paper_on_truncate(self, mock_context):
        """Should handle short paper gracefully on TRUNCATE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": "Short paper",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "short enough" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage on SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"


class TestHandleContextOverflow:
    """Direct handler tests for context overflow mitigation."""

    def test_handle_context_overflow_summarize(self, mock_state, mock_result):
        """Should apply summarization."""
        user_input = {"q1": "SUMMARIZE"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "summarization" in mock_result["supervisor_feedback"]

    def test_handle_context_overflow_truncate_long(self, mock_state, mock_result):
        """Should truncate long paper text."""
        long_text = "a" * 25000
        mock_state["paper_text"] = long_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "TRUNCATED" in mock_result["paper_text"]
        assert len(mock_result["paper_text"]) < 25000
        assert mock_result["paper_text"].startswith(long_text[:15000])
        assert mock_result["paper_text"].endswith(long_text[-5000:])

    def test_handle_context_overflow_truncate_short(self, mock_state, mock_result):
        """Should not truncate if text is short."""
        mock_state["paper_text"] = "a" * 100
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "paper_text" not in mock_result
        assert "short enough" in mock_result["supervisor_feedback"]

    def test_handle_context_overflow_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown input."""
        user_input = {"q1": "What is this?"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
