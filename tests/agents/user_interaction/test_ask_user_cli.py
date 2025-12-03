"""CLI-oriented tests for ask_user_node."""

import os
import signal
from unittest.mock import call, patch

import pytest

from src.agents.user_interaction import ask_user_node

pytestmark = pytest.mark.slow

class TestAskUserNode:
    """Tests for ask_user_node function."""

    def test_returns_not_awaiting_when_no_questions(self):
        """Should return awaiting_user_input=False when no questions pending."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test_trigger",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert result["workflow_phase"] == "awaiting_user"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_formats_question_from_pending(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should format question from pending questions and collect responses."""
        # Simulate user input: "Answer 1", empty line (submit), "Answer 2", empty line (submit)
        mock_input.side_effect = ["Answer 1", "", "Answer 2", ""]
        
        state = {
            "pending_user_questions": ["Question 1", "Question 2"],
            "ask_user_trigger": "multi_question",
        }
        
        result = ask_user_node(state)
        
        # After collecting all responses, awaiting_user_input should be False
        assert result["awaiting_user_input"] is False
        assert "user_responses" in result
        assert len(result["user_responses"]) == 2
        assert result["user_responses"]["Question 1"] == "Answer 1"
        assert result["user_responses"]["Question 2"] == "Answer 2"
        # Verify pending_user_questions is cleared
        assert result["pending_user_questions"] == []

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_collects_user_response(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should collect user response via CLI."""
        # Simulate user input: "User response", empty line (submit)
        mock_input.side_effect = ["User response", ""]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert "user_responses" in result
        assert result["user_responses"]["Question?"] == "User response"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0", "REPROLAB_USER_TIMEOUT_SECONDS": "10"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_sets_custom_timeout(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should set timeout from environment variable."""
        mock_input.side_effect = ["Response", ""]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        ask_user_node(state)
        
        # Check if alarm was called with 10 seconds (and then 0 to cancel)
        mock_alarm.assert_has_calls([call(10), call(0)])

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_clears_pending_questions_on_success(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should clear pending_user_questions upon successful response collection."""
        mock_input.side_effect = ["Response", ""]
        state = {
            "pending_user_questions": ["Q1"],
            "ask_user_trigger": "test",
        }
        result = ask_user_node(state)
        assert result["pending_user_questions"] == []

class TestNonInteractiveMode:
    """Tests for non-interactive mode."""

    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "1"})
    def test_exits_in_non_interactive_mode(self, mock_save):
        """Should save checkpoint and exit in non-interactive mode."""
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test_trigger",
            "paper_id": "test_paper",
        }
        
        with pytest.raises(SystemExit) as exc_info:
            ask_user_node(state)
        
        assert exc_info.value.code == 0
        mock_save.assert_called_once()

class TestMergeExistingResponses:
    """Tests for merging with existing user responses."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_merges_with_existing_responses(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should merge new responses with existing ones."""
        mock_input.side_effect = ["new answer", ""]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["New question?"],
            "ask_user_trigger": "test",
            "user_responses": {"Previous question": "previous answer"},
        }
        
        result = ask_user_node(state)
        
        assert result["user_responses"]["Previous question"] == "previous answer"
        assert result["user_responses"]["New question?"] == "new answer"

