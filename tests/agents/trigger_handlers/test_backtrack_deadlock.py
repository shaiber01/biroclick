"""Backtrack approval and deadlock trigger tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


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


class TestHandleBacktrackAndDeadlockHandlers:
    """Direct handler tests for backtrack approval and deadlock triggers."""

    def test_handle_backtrack_approval_approve(self, mock_state, mock_result):
        """Should calculate dependents and set verdict to backtrack."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=["stage1", "stage2"])

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert mock_result["backtrack_decision"]["stages_to_invalidate"] == ["stage1", "stage2"]
        mock_get_dependents.assert_called_once()

    def test_handle_backtrack_approval_reject(self, mock_state, mock_result):
        """Should clear suggestion and continue."""
        user_input = {"q1": "REJECT"}
        trigger_handlers.handle_backtrack_approval(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["backtrack_suggestion"] is None

    def test_handle_deadlock_report(self, mock_state, mock_result):
        """Should stop and generate report."""
        user_input = {"q1": "GENERATE_REPORT"}
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_replan(self, mock_state, mock_result):
        """Should request replan."""
        user_input = {"q1": "REPLAN"}
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "replan_needed"

    def test_handle_deadlock_unknown(self, mock_state, mock_result):
        """Should ask for clarification."""
        user_input = {"q1": "Help"}
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
