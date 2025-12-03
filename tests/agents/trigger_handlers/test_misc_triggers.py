"""Tests for context overflow, deadlock, and miscellaneous triggers."""

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


class TestArchiveErrorRecovery:
    """Tests for archive error recovery logic."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry failed archive operations."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None  # Successful retry
        
        state = {
            "archive_errors": [{"stage_id": "stage1", "error": "Failed"}],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["archive_errors"] == []
        mock_archive.assert_called_once_with(state, "stage1")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_keeps_failed_retries(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should keep archive errors that still fail on retry."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.side_effect = Exception("Still failing")
        
        state = {
            "archive_errors": [{"stage_id": "stage1", "error": "Failed"}],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        mock_archive.assert_called_once_with(state, "stage1")


class TestUserInteractionLogging:
    """Tests for user interaction logging."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction(self, mock_context):
        """Should log user interaction to progress."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Material question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "material_checkpoint"
        assert interaction["id"] == "U1"
        assert "timestamp" in interaction
        assert interaction["context"]["stage_id"] == "stage0"
        assert interaction["context"]["agent"] == "SupervisorAgent"
        assert interaction["user_response"] == "APPROVE"



class TestInvalidUserResponses:
    """Tests for handling invalid user_responses."""

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle non-dict user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": "invalid string",  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()



class TestHandleContextAndMiscTriggers:
    """Tests for context overflow, backtrack, deadlock, and dispatcher handlers."""

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

    def test_handle_llm_error_retry(self, mock_state, mock_result):
        """Should retry."""
        user_input = {"q1": "RETRY"}
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_llm_error_skip(self, mock_update, mock_state, mock_result):
        """Should skip stage."""
        user_input = {"q1": "SKIP"}

        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")

        mock_update.assert_called_with(mock_state, "stage1", "blocked", summary=ANY)
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_llm_error_unknown(self, mock_state, mock_result):
        """Should ask for clarification."""
        user_input = {"q1": "What?"}
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_trigger_dispatcher(self, mock_state, mock_result):
        """Should dispatch to correct handler."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_trigger("code_review_limit", mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"

    def test_handle_trigger_unknown(self, mock_state, mock_result):
        """Should handle unknown trigger gracefully."""
        user_input = {"q1": "foo"}
        trigger_handlers.handle_trigger("unknown_trigger_xyz", mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "unknown trigger" in mock_result["supervisor_feedback"]
