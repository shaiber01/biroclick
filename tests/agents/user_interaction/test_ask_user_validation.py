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
        # Verify validation counter is initialized to 1 (first attempt)
        assert result["user_validation_attempts_material_checkpoint"] == 1
        # Verify error message contains the actual validation error
        assert "Response must contain APPROVE or REJECT" in result["pending_user_questions"][0]
        # Verify attempt count is shown in error message
        assert "attempt 1/3" in result["pending_user_questions"][0].lower()
        # Verify last_node_before_ask_user is preserved if present
        assert "last_node_before_ask_user" not in result or result.get("last_node_before_ask_user") == state.get("last_node_before_ask_user")

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
        # Verify exact structure: first item should have error + Q1, then Q2, then Q3
        assert len(result["pending_user_questions"]) == 3, "Should have exactly 3 questions (error+Q1, Q2, Q3)"
        assert "Q1" in result["pending_user_questions"][0], "First question should contain Q1"
        assert result["pending_user_questions"][1] == "Q2", "Second question should be exactly Q2"
        assert result["pending_user_questions"][2] == "Q3", "Third question should be exactly Q3"
        # Verify error message is in first question
        assert "Something is wrong" in result["pending_user_questions"][0]

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
        # Verify attempt count is shown correctly in error message
        assert "attempt 2/3" in result["pending_user_questions"][0].lower()
        # Verify counter is initialized from 0 if not present
        state_no_counter = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
        }
        mock_input.side_effect = ["invalid", ""]
        result2 = ask_user_node(state_no_counter)
        assert result2["user_validation_attempts_material_checkpoint"] == 1, "Counter should initialize to 1 if missing"

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
            
            # Verify warning logged with correct message
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "3 times" in warning_call or "validation failed" in warning_call.lower()
            assert "material_checkpoint" in warning_call
        
        # Should accept despite validation errors
        assert result["awaiting_user_input"] is False
        assert "user_responses" in result
        assert result["user_responses"]["Question?"] == "still invalid", "Response should be stored"
        assert "supervisor_feedback" in result  # Should note the validation override
        assert "validation errors" in result["supervisor_feedback"].lower() or "3 attempts" in result["supervisor_feedback"]
        # Counter should be reset
        assert result["user_validation_attempts_material_checkpoint"] == 0
        # Pending questions should be cleared
        assert result["pending_user_questions"] == []
        # Workflow phase should be set
        assert result["workflow_phase"] == "awaiting_user"

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
        # Response should be stored
        assert result["user_responses"]["Question?"] == "APPROVE"
        # Pending questions should be cleared
        assert result["pending_user_questions"] == []
        # Workflow phase should be set
        assert result["workflow_phase"] == "awaiting_user"

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
        # Verify single-line prompt was called
        assert mock_input.call_count == 2
        # Verify second call was for single-line input
        assert "single line" in mock_input.call_args_list[1][0][0].lower()

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
        # Verify state was passed correctly
        assert args[0] == state


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_empty_questions_list(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should handle empty questions list gracefully."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert result["workflow_phase"] == "awaiting_user"
        # Should not call input when no questions
        mock_input.assert_not_called()
        mock_validate.assert_not_called()

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_missing_ask_user_trigger(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should handle missing ask_user_trigger with safety net."""
        mock_input.side_effect = ["APPROVE", ""]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            # Missing ask_user_trigger - safety net will set "unknown_escalation"
        }
        
        result = ask_user_node(state)
        
        # Safety net should set "unknown_escalation" as trigger
        assert result["awaiting_user_input"] is False
        assert result["user_validation_attempts_unknown_escalation"] == 0
        # Verify validate_user_responses was called with "unknown_escalation" trigger
        mock_validate.assert_called_once()
        call_args = mock_validate.call_args
        assert call_args[0][0] == "unknown_escalation"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_empty_response_string(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should handle empty response string."""
        # Empty multiline: first "" gets appended, second "" breaks the loop
        # Then single-line prompt returns "APPROVE"
        mock_input.side_effect = ["", "", "APPROVE"]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert result["user_responses"]["Question?"] == "APPROVE"
        # Verify single-line prompt was called (3rd call)
        assert mock_input.call_count == 3
        # Verify third call was for single-line input
        assert "single line" in mock_input.call_args_list[2][0][0].lower()

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_whitespace_only_response(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should handle whitespace-only response."""
        # Whitespace line, then empty line to break multiline loop
        # After strip(), response is empty, so single-line prompt returns "APPROVE"
        mock_input.side_effect = ["   \t  ", "", "APPROVE"]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        # After stripping, response is empty, so should use single-line
        assert result["awaiting_user_input"] is False
        assert result["user_responses"]["Question?"] == "APPROVE"
        # Verify single-line prompt was called
        assert mock_input.call_count == 3

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_multiple_questions_all_responses_stored(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should store all responses for multiple questions."""
        mock_input.side_effect = ["Response1", "", "Response2", "", "Response3", ""]
        mock_validate.return_value = []
        
        questions = ["Q1?", "Q2?", "Q3?"]
        state = {
            "pending_user_questions": questions,
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert len(result["user_responses"]) == 3
        assert result["user_responses"]["Q1?"] == "Response1"
        assert result["user_responses"]["Q2?"] == "Response2"
        assert result["user_responses"]["Q3?"] == "Response3"
        # Verify validate_user_responses was called with all responses
        mock_validate.assert_called_once()
        call_args = mock_validate.call_args
        assert len(call_args[0][1]) == 3
        assert call_args[0][1]["Q1?"] == "Response1"
        assert call_args[0][1]["Q2?"] == "Response2"
        assert call_args[0][1]["Q3?"] == "Response3"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_multiline_response_preserved(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should preserve multiline responses correctly."""
        multiline_response = "Line 1\nLine 2\nLine 3"
        mock_input.side_effect = ["Line 1", "Line 2", "Line 3", ""]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert result["user_responses"]["Question?"] == multiline_response

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_validation_counter_initializes_from_zero(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should initialize validation counter to 1 if missing from state."""
        mock_input.side_effect = ["invalid", ""]
        mock_validate.return_value = ["Error"]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "new_trigger",
            # No validation counter key
        }
        
        result = ask_user_node(state)
        
        assert result["user_validation_attempts_new_trigger"] == 1
        assert "attempt 1/3" in result["pending_user_questions"][0].lower()

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_max_validation_attempts_boundary(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should accept response exactly at max attempts (3)."""
        mock_input.side_effect = ["invalid", ""]
        mock_validate.return_value = ["Error"]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "user_validation_attempts_test": 2,  # This will become 3, which is max
        }
        
        with patch("src.agents.user_interaction.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = ask_user_node(state)
            
            # Should accept after 3 attempts
            assert result["awaiting_user_input"] is False
            assert result["user_validation_attempts_test"] == 0
            mock_logger.warning.assert_called_once()


class TestStatePreservation:
    """Tests for state preservation and merging."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_preserves_existing_user_responses(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should merge new responses with existing user_responses."""
        mock_input.side_effect = ["NewResponse", ""]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["NewQuestion?"],
            "ask_user_trigger": "test",
            "user_responses": {
                "OldQuestion?": "OldResponse",
            },
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False
        assert len(result["user_responses"]) == 2
        assert result["user_responses"]["OldQuestion?"] == "OldResponse"
        assert result["user_responses"]["NewQuestion?"] == "NewResponse"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_preserves_last_node_before_ask_user(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should preserve last_node_before_ask_user on validation failure."""
        mock_input.side_effect = ["invalid", ""]
        mock_validate.return_value = ["Error"]
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "last_node_before_ask_user": "some_node",
        }
        
        result = ask_user_node(state)
        
        assert result["last_node_before_ask_user"] == "some_node"

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_preserves_paper_id(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should preserve paper_id in state (read-only, but verify it's used)."""
        mock_input.side_effect = ["APPROVE", ""]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper_123",
        }
        
        # Verify paper_id is used in output (printed, but we can't easily test that)
        # But we can verify the function completes successfully
        result = ask_user_node(state)
        assert result["awaiting_user_input"] is False

    @patch("builtins.input")
    @patch("src.agents.user_interaction.validate_user_responses")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_does_not_preserve_last_node_on_success(self, mock_alarm, mock_signal, mock_save, mock_validate, mock_input):
        """Should NOT preserve last_node_before_ask_user on successful validation (cleared after input)."""
        mock_input.side_effect = ["APPROVE", ""]
        mock_validate.return_value = []
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "last_node_before_ask_user": "some_node",
        }
        
        result = ask_user_node(state)
        
        # last_node_before_ask_user should NOT be in result (cleared after successful input)
        assert "last_node_before_ask_user" not in result
        assert result["awaiting_user_input"] is False


class TestNonInteractiveMode:
    """Tests for non-interactive mode."""

    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "1"})
    def test_non_interactive_mode_saves_checkpoint(self, mock_save):
        """Should save checkpoint and exit in non-interactive mode."""
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper",
        }
        
        with pytest.raises(SystemExit) as exc_info:
            ask_user_node(state)
        
        assert exc_info.value.code == 0
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert args[0] == state
        assert "awaiting_user_test" in args[1]

    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "1"})
    def test_non_interactive_mode_with_multiple_questions(self, mock_save):
        """Should handle multiple questions in non-interactive mode."""
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Q1?", "Q2?", "Q3?"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper",
        }
        
        with pytest.raises(SystemExit) as exc_info:
            ask_user_node(state)
        
        assert exc_info.value.code == 0
        mock_save.assert_called_once()

    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "1"})
    def test_non_interactive_mode_empty_questions(self, mock_save):
        """Should handle empty questions in non-interactive mode."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        # Should return normally without saving checkpoint
        assert result["awaiting_user_input"] is False
        mock_save.assert_not_called()


class TestSignalHandling:
    """Tests for signal handling and timeout."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0", "REPROLAB_USER_TIMEOUT_SECONDS": "3600"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_timeout_environment_variable(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should use REPROLAB_USER_TIMEOUT_SECONDS environment variable."""
        mock_input.side_effect = KeyboardInterrupt()
        mock_save.return_value = "/path/to/checkpoint"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        with pytest.raises(SystemExit):
            ask_user_node(state)
        
        # Verify signal.alarm was called with timeout from env var
        mock_alarm.assert_called()
        # Should be called with 3600 seconds
        assert 3600 in [call[0][0] for call in mock_alarm.call_args_list if call[0]]

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_signal_restored_after_success(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should restore signal handler after successful input."""
        mock_input.side_effect = ["APPROVE", ""]
        
        with patch("src.agents.user_interaction.validate_user_responses") as mock_validate:
            mock_validate.return_value = []
            
            state = {
                "pending_user_questions": ["Question?"],
                "ask_user_trigger": "test",
            }
            
            result = ask_user_node(state)
            
            assert result["awaiting_user_input"] is False
            # Verify alarm was cancelled (set to 0)
            assert any(call[0][0] == 0 for call in mock_alarm.call_args_list)
            # Verify signal handler was restored
            assert mock_signal.call_count >= 2  # Set and restore


class TestValidationIntegration:
    """Tests for integration with validation function."""

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_validate_user_responses_called_with_correct_args(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should call validate_user_responses with correct arguments."""
        mock_input.side_effect = ["Response1", "", "Response2", ""]
        
        with patch("src.agents.user_interaction.validate_user_responses") as mock_validate:
            mock_validate.return_value = []
            
            questions = ["Q1?", "Q2?"]
            state = {
                "pending_user_questions": questions,
                "ask_user_trigger": "test_trigger",
            }
            
            result = ask_user_node(state)
            
            assert result["awaiting_user_input"] is False
            # Verify validate_user_responses was called correctly
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args[0]
            assert call_args[0] == "test_trigger"
            assert isinstance(call_args[1], dict)
            assert call_args[1]["Q1?"] == "Response1"
            assert call_args[1]["Q2?"] == "Response2"
            assert call_args[2] == questions

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_multiple_validation_errors_preserved(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should preserve all validation errors in error message."""
        mock_input.side_effect = ["invalid", ""]
        
        with patch("src.agents.user_interaction.validate_user_responses") as mock_validate:
            mock_validate.return_value = [
                "Error 1: Missing keyword",
                "Error 2: Invalid format",
                "Error 3: Wrong trigger",
            ]
            
            state = {
                "pending_user_questions": ["Question?"],
                "ask_user_trigger": "test",
            }
            
            result = ask_user_node(state)
            
            assert result["awaiting_user_input"] is True
            error_text = result["pending_user_questions"][0]
            assert "Error 1: Missing keyword" in error_text
            assert "Error 2: Invalid format" in error_text
            assert "Error 3: Wrong trigger" in error_text

    @patch("builtins.input")
    @patch("src.agents.user_interaction.save_checkpoint")
    @patch.dict("os.environ", {"REPROLAB_NON_INTERACTIVE": "0"})
    @patch("signal.signal")
    @patch("signal.alarm")
    def test_validation_exception_handling(self, mock_alarm, mock_signal, mock_save, mock_input):
        """Should handle exceptions from validate_user_responses gracefully."""
        mock_input.side_effect = ["Response", ""]
        
        with patch("src.agents.user_interaction.validate_user_responses") as mock_validate:
            mock_validate.side_effect = Exception("Validation error")
            
            state = {
                "pending_user_questions": ["Question?"],
                "ask_user_trigger": "test",
            }
            
            # Should propagate the exception (not catch it)
            with pytest.raises(Exception) as exc_info:
                ask_user_node(state)
            
            assert "Validation error" in str(exc_info.value)

