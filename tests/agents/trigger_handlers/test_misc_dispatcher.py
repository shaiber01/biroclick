"""Miscellaneous trigger handler tests."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


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


class TestHandleLlmAndDispatcherHandlers:
    """Direct handler tests for LLM errors and dispatcher fallbacks."""

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
