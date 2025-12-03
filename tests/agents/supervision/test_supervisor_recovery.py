"""Supervisor recovery and logging tests."""

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

from src.agents.supervision import supervisor_node


class TestArchiveErrorRecovery:
    """Tests for archive error recovery logic."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry failed archive operations and clear errors on success."""
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
        
        # Verify archive errors are cleared
        assert result["archive_errors"] == []
        # Verify archive was called with correct arguments
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify normal supervision still runs
        assert result["supervisor_verdict"] == "ok_continue"
        assert "workflow_phase" in result

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
        
        # Verify error is preserved
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert result["archive_errors"][0]["error"] == "Failed"
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify normal supervision still runs
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_multiple_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry all failed archive operations."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None  # All succeed
        
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Failed"},
                {"stage_id": "stage2", "error": "Failed"},
                {"stage_id": "stage3", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify all errors are cleared
        assert result["archive_errors"] == []
        # Verify all archives were called
        assert mock_archive.call_count == 3
        assert mock_archive.call_args_list[0][0] == (state, "stage1")
        assert mock_archive.call_args_list[1][0] == (state, "stage2")
        assert mock_archive.call_args_list[2][0] == (state, "stage3")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_partial_retry_success(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should keep only failed retries when some succeed."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        def archive_side_effect(state, stage_id):
            if stage_id == "stage2":
                raise Exception("Still failing")
            return None
        
        mock_archive.side_effect = archive_side_effect
        
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Failed"},
                {"stage_id": "stage2", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify only failed retry is kept
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage2"
        assert mock_archive.call_count == 2

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_empty_archive_errors(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle empty archive_errors list."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive_errors is cleared
        assert result["archive_errors"] == []
        # Verify archive was not called
        mock_archive.assert_not_called()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_missing_archive_errors(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle missing archive_errors key."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive_errors is set to empty list
        assert result["archive_errors"] == []
        # Verify archive was not called
        mock_archive.assert_not_called()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_archive_error_without_stage_id(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should skip archive errors without stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [
                {"error": "Failed"},  # Missing stage_id
                {"stage_id": "stage1", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive was only called for stage1
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify error without stage_id is kept (not retried)
        assert len(result["archive_errors"]) == 1
        assert "stage_id" not in result["archive_errors"][0] or result["archive_errors"][0].get("stage_id") != "stage1"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_archive_error_with_none_stage_id(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should skip archive errors with None stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [
                {"stage_id": None, "error": "Failed"},
                {"stage_id": "stage1", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive was only called for stage1
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify error with None stage_id is kept (not retried)
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] is None

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_non_list_archive_errors(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle archive_errors that is not a list."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": "not a list",  # Invalid type
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        # This should not crash - the code should handle it gracefully
        # If it crashes, that's a bug in the component
        result = supervisor_node(state)
        
        # Verify function completes
        assert "supervisor_verdict" in result
        # Archive should not be called with invalid archive_errors
        mock_archive.assert_not_called()
class TestUserInteractionLogging:
    """Tests for user interaction logging."""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction(self, mock_context, mock_handle_trigger):
        """Should log user interaction to progress."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Material question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify progress is updated
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        
        # Verify interaction details
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "material_checkpoint"
        assert interaction["id"] == "U1"
        assert "timestamp" in interaction
        assert isinstance(interaction["timestamp"], str)
        assert interaction["context"]["stage_id"] == "stage0"
        assert interaction["context"]["agent"] == "SupervisorAgent"
        assert interaction["context"]["reason"] == "material_checkpoint"
        assert interaction["user_response"] == "APPROVE"
        assert interaction["question"] == "Material question"
        assert "impact" in interaction
        
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_multiple_user_interactions(self, mock_context, mock_handle_trigger):
        """Should append to existing user interactions."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Material question"],
            "current_stage_id": "stage0",
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1", "interaction_type": "previous"}
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # Verify new interaction is appended
        assert len(result["progress"]["user_interactions"]) == 2
        assert result["progress"]["user_interactions"][0]["id"] == "U1"
        assert result["progress"]["user_interactions"][1]["id"] == "U2"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_empty_questions(self, mock_context, mock_handle_trigger):
        """Should handle empty pending_user_questions."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": [],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_missing_questions(self, mock_context, mock_handle_trigger):
        """Should handle missing pending_user_questions."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_none_stage_id(self, mock_context, mock_handle_trigger):
        """Should handle None current_stage_id."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Question"],
            "current_stage_id": None,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["context"]["stage_id"] is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_multiple_responses(self, mock_context, mock_handle_trigger):
        """Should use last user response when multiple exist."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {
                "Question1": "REJECT",
                "Question2": "APPROVE",
            },
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        # Should use last value from dict
        assert interaction["user_response"] == "APPROVE"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_empty_responses(self, mock_context, mock_handle_trigger):
        """Should handle empty user_responses."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["user_response"] == ""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_does_not_log_when_no_trigger(self, mock_context, mock_handle_trigger):
        """Should not log interaction when ask_user_trigger is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": None,
            "user_responses": {"Question": "APPROVE"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should not have user_interactions in result
        assert "user_interactions" not in result.get("progress", {})

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_interaction_even_with_empty_responses(self, mock_context, mock_handle_trigger):
        """Should log interaction even when user_responses is empty (to track that user was asked)."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should log interaction even with empty responses (to track that user was asked)
        assert "user_interactions" in result.get("progress", {})
        assert len(result["progress"]["user_interactions"]) == 1
        assert result["progress"]["user_interactions"][0]["user_response"] == ""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_missing_progress(self, mock_context, mock_handle_trigger):
        """Should handle missing progress key."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
        }
        
        result = supervisor_node(state)
        
        # Should create progress with user_interactions
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_missing_user_interactions(self, mock_context, mock_handle_trigger):
        """Should handle progress without user_interactions key."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should create user_interactions list
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
class TestInvalidUserResponses:
    """Tests for handling invalid user_responses."""

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_string(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle string user_responses gracefully and log warning."""
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
        
        # Verify function completes successfully
        assert result["supervisor_verdict"] == "ok_continue"
        # Verify warning was logged
        assert mock_logger.warning.called
        # Verify warning message contains expected info
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "user_responses" in warning_call_args.lower()
        assert "dict" in warning_call_args.lower()
        # Verify user_responses is converted to empty dict internally
        # (We can't directly check this, but function should work)

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_list(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle list user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": ["response1", "response2"],  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_int(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle int user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": 42,  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_none(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle None user_responses gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": None,  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # None should be handled (isinstance(None, dict) is False)
        mock_logger.warning.assert_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_bool(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle bool user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": True,  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_set(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle set user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": {"response1", "response2"},  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_valid_dict_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle valid dict user_responses without warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": {"Question": "APPROVE"},  # Valid dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Should not log warning for valid dict
        mock_logger.warning.assert_not_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_missing_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle missing user_responses key."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Should not crash when user_responses is missing
        # get() returns None, which is handled
        # Actually, get() returns None, and isinstance(None, dict) is False, so it logs warning
        # This might be a bug - missing key should probably default to {} not None
        # But we're testing the component as-is


class TestNormalSupervision:
    """Tests for normal supervision path (when not handling user response)."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_calls_llm(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should call LLM for normal supervision."""
        mock_context.return_value = None
        mock_prompt.return_value = "system_prompt"
        mock_call.return_value = {"verdict": "ok_continue", "reasoning": "All good"}
        mock_archive.return_value = None
        
        state = {
            "ask_user_trigger": None,  # No trigger = normal supervision
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
            "workflow_phase": "design",
        }
        
        result = supervisor_node(state)
        
        # Verify LLM was called
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args.kwargs["agent_name"] == "supervisor"
        assert call_args.kwargs["system_prompt"] == "system_prompt"
        assert "user_content" in call_args.kwargs
        assert call_args.kwargs["state"] == state
        
        # Verify verdict is set
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["supervisor_feedback"] == "All good"
        assert result["workflow_phase"] == "supervision"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_archives_current_stage(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should archive outputs for current stage."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive was called for current stage
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify status update was called
        mock_update_status.assert_called_once()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_handles_archive_error(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle archive errors during normal supervision."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.side_effect = Exception("Archive failed")
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive error is recorded
        assert "archive_errors" in result
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert "error" in result["archive_errors"][0]
        assert "timestamp" in result["archive_errors"][0]
        # Verify supervision still completes
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_no_current_stage(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle normal supervision without current_stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive is not called when no current stage
        mock_archive.assert_not_called()
        mock_update_status.assert_not_called()
        # Verify supervision still completes
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_handles_llm_exception(self, mock_prompt, mock_context, mock_call):
        """Should handle LLM call exceptions gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("LLM unavailable")
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify default verdict is set
        assert result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in result
        assert "LLM unavailable" in result["supervisor_feedback"]

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_backtrack_verdict(self, mock_prompt, mock_context, mock_call):
        """Should handle backtrack_to_stage verdict."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "backtrack_to_stage",
            "backtrack_target": "stage0",
            "reasoning": "Need to restart"
        }
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify backtrack decision is set
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "stage0"
        assert result["backtrack_decision"]["reason"] == "Need to restart"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_missing_verdict(self, mock_prompt, mock_context, mock_call):
        """Should default verdict when LLM doesn't return one."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}  # No verdict
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify default verdict
        assert result["supervisor_verdict"] == "ok_continue"


class TestContextCheck:
    """Tests for context check and escalation handling."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_context_check_returns_awaiting_user_input(self, mock_prompt, mock_context, mock_call):
        """Should return early when context check requires user input."""
        mock_context.return_value = {"awaiting_user_input": True}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should return context_update directly
        assert result["awaiting_user_input"] is True
        # Should not call LLM
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_context_check_updates_state(self, mock_prompt, mock_context, mock_call):
        """Should merge context_update into state."""
        mock_context.return_value = {"some_update": "value"}
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify function completes (state is merged internally)
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_context_check_returns_none(self, mock_prompt, mock_context, mock_call):
        """Should continue normally when context check returns None."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify function continues normally
        assert result["supervisor_verdict"] == "ok_continue"
