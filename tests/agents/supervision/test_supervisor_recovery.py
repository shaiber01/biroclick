"""Supervisor recovery and logging tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node


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
