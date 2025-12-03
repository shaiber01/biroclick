"""Unit tests for src/agents/user_interaction.py"""

import os
import pytest
import signal
import logging
from unittest.mock import patch, MagicMock, call

from src.agents.user_interaction import (
    ask_user_node,
    material_checkpoint_node,
)


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


class TestMaterialCheckpointNode:
    """Tests for material_checkpoint_node function."""

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_triggers_material_validation(self, mock_format, mock_extract):
        """Should trigger material validation checkpoint."""
        mock_extract.return_value = [
            {"material_id": "gold", "name": "Gold", "path": "gold.csv"}
        ]
        mock_format.return_value = "Material checkpoint question"
        
        state = {
            "current_stage_id": "stage0_material_validation",
            "stage_outputs": {"files": ["gold.csv"]},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "material_checkpoint"
        assert "pending_validated_materials" in result
        # Check that extracted materials are passed to the result
        assert result["pending_validated_materials"] == mock_extract.return_value

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_includes_pending_validated_materials(self, mock_format, mock_extract):
        """Should include pending validated materials in result."""
        mock_extract.return_value = [
            {"material_id": "gold", "name": "Gold"},
            {"material_id": "silver", "name": "Silver"},
        ]
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert len(result["pending_validated_materials"]) == 2
        assert result["pending_validated_materials"][0]["material_id"] == "gold"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_no_materials(self, mock_format, mock_extract):
        """Should handle case with no materials detected."""
        mock_extract.return_value = []
        mock_format.return_value = "No materials detected"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["pending_validated_materials"] == []
        # Should include warning about no materials in the question text
        assert "WARNING" in result["pending_user_questions"][0]

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_extracts_plot_files(self, mock_format, mock_extract):
        """Should extract plot files from stage outputs."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {
                "files": ["data.csv", "plot.png", "comparison.pdf", "results.jpg"],
            },
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        # Format function should be called with plot files
        call_args = mock_format.call_args
        plot_files = call_args[0][2]  # Third argument is plot_files
        assert "plot.png" in plot_files
        assert "comparison.pdf" in plot_files
        assert "results.jpg" in plot_files
        assert "data.csv" not in plot_files

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_finds_material_validation_stage(self, mock_format, mock_extract):
        """Should find MATERIAL_VALIDATION stage in progress."""
        mock_extract.return_value = [{"material_id": "gold"}]
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "summary": "Validated gold"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"},
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should pass stage0_info to format function
        call_args = mock_format.call_args
        stage_info = call_args[0][1]  # Second argument is stage_info
        assert stage_info["stage_type"] == "MATERIAL_VALIDATION"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_empty_progress(self, mock_format, mock_extract):
        """Should handle empty progress gracefully."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {},  # Empty progress
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_sets_workflow_phase(self, mock_format, mock_extract):
        """Should set workflow_phase to material_checkpoint."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["workflow_phase"] == "material_checkpoint"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_sets_last_node_before_ask_user(self, mock_format, mock_extract):
        """Should set last_node_before_ask_user."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["last_node_before_ask_user"] == "material_checkpoint"


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
