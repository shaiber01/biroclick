"""Supervisor tests for deadlock and unknown trigger handling."""

from unittest.mock import patch

from src.agents.supervision import supervisor_node


class TestDeadlockTrigger:
    """Tests for deadlock_detected trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_generates_report_on_generate_report(self, mock_context):
        """Should generate report on GENERATE_REPORT."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_replans_on_replan(self, mock_context):
        """Should trigger replan on REPLAN."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPLAN"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "STOP"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "I'm not sure"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"


class TestUnknownTrigger:
    """Tests for unknown trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unknown_trigger(self, mock_context):
        """Should handle unknown trigger gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "some_unknown_trigger",
            "user_responses": {"Question": "Response"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "unknown" in result["supervisor_feedback"].lower()

