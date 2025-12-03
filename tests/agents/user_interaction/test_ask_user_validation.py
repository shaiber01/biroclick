"""Validation and error handling tests for ask_user_node."""

import logging
import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from src.agents.user_interaction import ask_user_node

pytestmark = pytest.mark.slow

class TestValidationErrorHandling:
    """Tests for validation error handling."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_returns_error_on_validation_failure(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should return validation error when response is invalid."""
        mock_input.side_effect = ["invalid response", ""]
        mock_validate.return_value = ["Response must contain APPROVE or REJECT"]
        
        state = {
            "pending_user_questions": ["Material checkpoint: APPROVE or REJECT?"],
            "ask_user_trigger": "material_checkpoint",
        }
        
        result = ask_user_node(state)
        
        # Should return with awaiting_user_input=True and error message
        assert result["awaiting_user_input"] is True
        assert "validation errors" in result["pending_user_questions"][0].lower()
        assert result["ask_user_trigger"] == "material_checkpoint"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_preserves_all_questions_on_validation_failure(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should preserve ALL questions when validation fails, not just the first one."""
        # User answers all three, but validation fails
        mock_input.side_effect = ["Ans1", "", "Ans2", "", "Ans3", ""]
        mock_validate.return_value = ["Something is wrong"]
        
        questions = ["Q1", "Q2", "Q3"]
        state = {
            "pending_user_questions": questions,
            "ask_user_trigger": "multi_test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is True
        # The returned pending questions should include the error message AND re-ask needed questions.
        # If the implementation replaces the list with a single string (error + Q1), 
        # then Q2 and Q3 are lost.
        
        # We expect the returned list to either be:
        # 1. [Error + Q1, Q2, Q3]
        # 2. [Error Message, Q1, Q2, Q3]
        # 3. Or at least contain the text of all questions if combined.
        
        # For now, let's assert that Q2 and Q3 are still present in some form in the pending questions
        # so the user is prompted for them again.
        pending_text = " ".join(result["pending_user_questions"])
        assert "Q2" in pending_text, "Question 2 was lost after validation failure"
        assert "Q3" in pending_text, "Question 3 was lost after validation failure"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_increments_validation_attempt_counter(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should increment validation attempt counter on each failure."""
        mock_input.side_effect = ["invalid", ""]
        mock_validate.return_value = ["Invalid response"]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
            "user_validation_attempts_material_checkpoint": 1,  # Already tried once
        }
        
        result = ask_user_node(state)
        
        # Should increment to 2
        assert result["user_validation_attempts_material_checkpoint"] == 2
        assert result["awaiting_user_input"] is True

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_accepts_after_max_validation_attempts(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should accept response after max validation attempts exceeded."""
        mock_input.side_effect = ["still invalid", ""]
        mock_validate.return_value = ["Invalid response"]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
            "user_validation_attempts_material_checkpoint": 2,  # Already tried twice (max=3)
        }
        
        with patch("src.agents.user_interaction.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = ask_user_node(state)
            
            # Verify warning logged
            mock_logger.warning.assert_called()
        
        # Should accept despite validation errors
        assert result["awaiting_user_input"] is False
        assert "user_responses" in result
        assert "supervisor_feedback" in result  # Should note the validation override
        # Counter should be reset
        assert result["user_validation_attempts_material_checkpoint"] == 0

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_resets_validation_counter_on_valid_response(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should reset validation counter when response is valid."""
        mock_input.side_effect = ["APPROVE", ""]
        mock_validate.return_value = []  # No validation errors
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
            "user_validation_attempts_material_checkpoint": 2,
        }
        
        result = ask_user_node(state)
        
        # Counter should be reset to 0
        assert result["user_validation_attempts_material_checkpoint"] == 0
        assert result["awaiting_user_input"] is False

class TestSingleLineResponseFallback:
    """Tests for single-line response fallback."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_prompts_for_single_line_on_eof_multiline(
        self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input
    ):
        """Should use single-line input when multiline encounters EOF."""
        # First input raises EOFError (empty multiline), then single-line response
        mock_input.side_effect = [EOFError(), "APPROVE"]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert result["user_responses"]["Question?"] == "APPROVE"

class TestExceptionHandling:
    """Tests for exception handling (KeyboardInterrupt, TimeoutError, EOFError)."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_handles_keyboard_interrupt(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should save checkpoint on KeyboardInterrupt."""
        mock_input.side_effect = KeyboardInterrupt()
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        with pytest.raises(SystemExit) as exc_info:
            ask_user_node(state)
        
        assert exc_info.value.code == 0
        mock_save.assert_called_once()
        # Verify checkpoint name contains trigger and 'interrupted'
        args, _ = mock_save.call_args
        assert "interrupted_test" in args[1]

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_handles_timeout_error(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should save checkpoint on TimeoutError."""
        # Simulate TimeoutError raised during input
        mock_input.side_effect = TimeoutError()
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        with pytest.raises(SystemExit) as exc_info:
            ask_user_node(state)
            
        assert exc_info.value.code == 0
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert "timeout_test" in args[1]

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_handles_eof_error_at_top_level(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should save checkpoint on EOFError when not inside multiline loop."""
        # This EOFError simulates immediate EOF on input, which might bubble up
        # However, the code catches EOFError inside the inner loop.
        # To test the outer EOFError, we need the inner loop to propagate it or the single line input to propagate it.
        
        # The code:
        # try:
        #   line = input() ... except EOFError: break
        #
        # if not response: response = input("Your response (single line): ")
        
        # So if the inner loop breaks due to EOF, response is empty.
        # Then it calls input() again. If THAT raises EOFError, it's caught by the outer try/except.
        
        mock_input.side_effect = [EOFError(), EOFError()]
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        with pytest.raises(SystemExit) as exc_info:
            ask_user_node(state)
            
        assert exc_info.value.code == 0
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert "eof_test" in args[1]

