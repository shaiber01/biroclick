"""Supervisor tests for context overflow and backtrack approval triggers."""

from unittest.mock import patch

from src.agents.supervision import supervisor_node


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
            "paper_text": "A" * 30000,
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


class TestBacktrackApprovalTrigger:
    """Tests for backtrack_approval trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_on_approve(self, mock_context):
        """Should approve backtrack on APPROVE."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE the backtrack"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "stages_to_invalidate" in result["backtrack_decision"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejects_backtrack_on_reject(self, mock_context):
        """Should reject backtrack on REJECT."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "REJECT"},
            "backtrack_suggestion": {"target": "stage1"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["backtrack_suggestion"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_defaults_to_continue_on_unclear(self, mock_context):
        """Should default to continue on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "maybe"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"

