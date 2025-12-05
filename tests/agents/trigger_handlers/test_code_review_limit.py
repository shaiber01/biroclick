"""code_review_limit trigger tests."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestCodeReviewLimitTrigger:
    """Tests for code_review_limit trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_hint(self, mock_context):
        """Should reset code revision count when user provides hint."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "PROVIDE_HINT: Use numpy instead"},
            "code_revision_count": 5,
        }
        
        result = supervisor_node(state)
        
        # Verify counter reset
        assert result["code_revision_count"] == 0
        # Verify verdict - routes directly to generate_code
        assert result["supervisor_verdict"] == "retry_generate_code"
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None
        # Verify reviewer feedback contains the hint
        assert "reviewer_feedback" in result
        assert "Use numpy instead" in result["reviewer_feedback"]
        # Verify supervisor feedback is set
        assert result.get("supervisor_feedback") == "Retrying code generation with user hint."
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_hint_lowercase(self, mock_context):
        """Should reset code revision count when user provides hint in lowercase."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "provide_hint: try a different approach"},
            "code_revision_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"
        assert "try a different approach" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_hint_keyword_only(self, mock_context):
        """Should reset code revision count when user provides just HINT keyword."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "HINT: Use pandas"},
            "code_revision_count": 7,
        }
        
        result = supervisor_node(state)
        
        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"
        assert "Use pandas" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_hint_preserves_existing_code_revision_count_if_not_set(self, mock_context):
        """Should set code_revision_count to 0 even if it wasn't in state."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "PROVIDE_HINT: Fix the bug"},
        }
        
        result = supervisor_node(state)
        
        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("ask_user_trigger") is None
        # Verify update_progress_stage_status was called with correct args
        mock_update.assert_called_once_with(
            state, "stage1", "blocked", summary="Skipped by user due to code review issues"
        )
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True
        # Verify code_revision_count is NOT reset
        assert "code_revision_count" not in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip_lowercase(self, mock_update, mock_context):
        """Should skip stage when user says skip in lowercase."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "skip"},
            "current_stage_id": "stage2",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once_with(
            state, "stage2", "blocked", summary="Skipped by user due to code review issues"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skip_does_not_call_update_when_no_stage_id(self, mock_update, mock_context):
        """Should not call update_progress_stage_status when current_stage_id is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Critical: should NOT call update_progress_stage_status when stage_id is None
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skip_does_not_call_update_when_stage_id_missing(self, mock_update, mock_context):
        """Should not call update_progress_stage_status when current_stage_id is missing."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "SKIP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result.get("ask_user_trigger") is None
        # Verify code_revision_count is NOT reset
        assert "code_revision_count" not in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop_lowercase(self, mock_context):
        """Should stop workflow when user says stop in lowercase."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "stop"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "I dont know"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert result.get("ask_user_trigger") is None
        # Verify clarification question is set
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1
        assert "PROVIDE_HINT" in result["pending_user_questions"][0]
        assert "SKIP_STAGE" in result["pending_user_questions"][0] or "SKIP" in result["pending_user_questions"][0]
        assert "STOP" in result["pending_user_questions"][0]
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True
        # Verify code_revision_count is NOT reset
        assert "code_revision_count" not in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_empty_response(self, mock_context):
        """Should ask for clarification on empty response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": ""},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_uses_last_response_when_multiple_responses(self, mock_context):
        """Should use the last response when multiple responses are provided."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {
                "Question1": "SKIP",
                "Question2": "STOP",  # Last one should be used
            },
        }
        
        result = supervisor_node(state)
        
        # Should use STOP (last response), not SKIP
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_state_is_not_mutated(self, mock_context):
        """Should not mutate the input state dict."""
        mock_context.return_value = None
        
        original_state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "PROVIDE_HINT: Test hint"},
            "code_revision_count": 5,
            "other_field": "should_not_change",
        }
        state_copy = original_state.copy()
        
        result = supervisor_node(state_copy)
        
        # Verify state was not mutated
        assert state_copy["ask_user_trigger"] == "code_review_limit"  # Not cleared in state
        assert state_copy["code_revision_count"] == 5  # Not reset in state
        assert state_copy["other_field"] == "should_not_change"
        # But result should have the changes
        assert result["code_revision_count"] == 0


class TestHandleCodeReviewLimit:
    """Direct handler tests for code review limit."""

    def test_handle_code_review_limit_hint(self, mock_state, mock_result):
        """Should handle PROVIDE_HINT response correctly."""
        user_input = {"q1": "PROVIDE_HINT: Use a loop instead."}
        initial_code_revision_count = mock_result.get("code_revision_count")
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Verify counter reset
        assert mock_result["code_revision_count"] == 0
        # Verify reviewer feedback contains the hint
        assert "reviewer_feedback" in mock_result
        assert "User hint:" in mock_result["reviewer_feedback"]
        assert "Use a loop instead." in mock_result["reviewer_feedback"]
        # Verify verdict - routes directly to generate_code
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        # Verify supervisor feedback
        assert mock_result.get("supervisor_feedback") == "Retrying code generation with user hint."
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True
        # Verify state was not mutated
        assert "code_revision_count" not in mock_state

    def test_handle_code_review_limit_hint_with_hint_keyword(self, mock_state, mock_result):
        """Should handle HINT keyword (without PROVIDE_HINT)."""
        user_input = {"q1": "HINT: Try recursion"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["code_revision_count"] == 0
        assert "Try recursion" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_code_review_limit_hint_empty_user_responses(self, mock_state, mock_result):
        """Should handle empty user_responses dict gracefully."""
        user_input = {}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should fall through to else clause (ask_user)
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "code_revision_count" not in mock_result

    def test_handle_code_review_limit_hint_none_stage_id(self, mock_state, mock_result):
        """Should handle hint when current_stage_id is None."""
        user_input = {"q1": "PROVIDE_HINT: Test"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, None)
        
        assert mock_result["code_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_handle_code_review_limit_skip(self, mock_update, mock_state, mock_result):
        """Should handle SKIP response correctly."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Verify _update_progress_with_error_handling was called with correct arguments
        mock_update.assert_called_once_with(
            mock_state, mock_result, "stage1", "blocked", summary="Skipped by user due to code review issues"
        )
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True
        # Verify code_revision_count is NOT reset
        assert "code_revision_count" not in mock_result

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_handle_code_review_limit_skip_none_stage_id(self, mock_update, mock_state, mock_result):
        """Should not call _update_progress_with_error_handling when stage_id is None."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Critical: should NOT call _update_progress_with_error_handling
        mock_update.assert_not_called()

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_handle_code_review_limit_skip_empty_string_stage_id(self, mock_update, mock_state, mock_result):
        """Should not call _update_progress_with_error_handling when stage_id is empty string."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Empty string is falsy, so should NOT call _update_progress_with_error_handling
        mock_update.assert_not_called()

    def test_handle_code_review_limit_stop(self, mock_state, mock_result):
        """Should handle STOP response correctly."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True
        # Verify code_revision_count is NOT reset
        assert "code_revision_count" not in mock_result

    def test_handle_code_review_limit_stop_none_stage_id(self, mock_state, mock_result):
        """Should handle STOP when stage_id is None."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_code_review_limit_unknown(self, mock_state, mock_result):
        """Should handle unknown response correctly."""
        user_input = {"q1": "Just keep going"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        # Verify clarification question is set
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        assert isinstance(mock_result["pending_user_questions"], list)
        question = mock_result["pending_user_questions"][0]
        assert "PROVIDE_HINT" in question
        assert "SKIP" in question or "SKIP_STAGE" in question
        assert "STOP" in question
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True
        # Verify code_revision_count is NOT reset
        assert "code_revision_count" not in mock_result

    def test_handle_code_review_limit_unknown_empty_string(self, mock_state, mock_result):
        """Should handle empty string response as unknown."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_code_review_limit_unknown_whitespace_only(self, mock_state, mock_result):
        """Should handle whitespace-only response as unknown."""
        user_input = {"q1": "   \n\t  "}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_code_review_limit_uses_last_response(self, mock_state, mock_result):
        """Should use the last response when multiple responses provided."""
        user_input = {
            "q1": "SKIP",
            "q2": "STOP",  # Last one should be used
        }
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use STOP (last response), not SKIP
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_code_review_limit_state_not_mutated(self, mock_state, mock_result):
        """Should not mutate the input state dict."""
        original_state = mock_state.copy()
        user_input = {"q1": "PROVIDE_HINT: Test"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Verify state was not mutated
        assert mock_state == original_state
        # But result should have changes
        assert mock_result["code_revision_count"] == 0

    def test_handle_code_review_limit_hint_extracts_raw_response(self, mock_state, mock_result):
        """Should extract the raw response text correctly for hint."""
        user_input = {"q1": "PROVIDE_HINT: This is a detailed hint with multiple words"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should include the full raw response, not just parsed version
        assert "This is a detailed hint with multiple words" in mock_result["reviewer_feedback"]
        assert mock_result["reviewer_feedback"].startswith("User hint:")

    def test_handle_code_review_limit_hint_with_special_characters(self, mock_state, mock_result):
        """Should handle hint with special characters correctly."""
        user_input = {"q1": "PROVIDE_HINT: Use @#$%^&*() and <tags>"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["code_revision_count"] == 0
        assert "@#$%^&*()" in mock_result["reviewer_feedback"]
        assert "<tags>" in mock_result["reviewer_feedback"]

    def test_handle_code_review_limit_partial_match_hint(self, mock_state, mock_result):
        """Should match HINT even when PROVIDE_HINT is not present."""
        user_input = {"q1": "HINT: Use numpy arrays"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["code_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_code_review_limit_partial_match_provide_hint(self, mock_state, mock_result):
        """Should match PROVIDE_HINT correctly."""
        user_input = {"q1": "PROVIDE_HINT: Fix the bug"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["code_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_code_review_limit_case_insensitive_matching(self, mock_state, mock_result):
        """Should match keywords case-insensitively."""
        test_cases = [
            ("provide_hint: test", "retry_generate_code", True, False),
            ("hint: test", "retry_generate_code", True, False),
            ("skip", "ok_continue", False, False),
            ("stop", "all_complete", False, True),
        ]
        
        for user_input_text, expected_verdict, should_reset_count, should_stop in test_cases:
            mock_result.clear()
            user_input = {"q1": user_input_text}
            
            trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == expected_verdict
            if should_reset_count:
                assert mock_result.get("code_revision_count") == 0
            if should_stop:
                assert mock_result.get("should_stop") is True

    def test_handle_code_review_limit_keyword_precedence_hint_before_skip(self, mock_state, mock_result):
        """Should prioritize PROVIDE_HINT over SKIP when both are present."""
        user_input = {"q1": "PROVIDE_HINT: SKIP this stage"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should match PROVIDE_HINT (checked first), not SKIP
        assert mock_result["code_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "SKIP this stage" in mock_result["reviewer_feedback"]

    def test_handle_code_review_limit_keyword_precedence_hint_before_stop(self, mock_state, mock_result):
        """Should prioritize PROVIDE_HINT over STOP when both are present."""
        user_input = {"q1": "PROVIDE_HINT: STOP the workflow"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should match PROVIDE_HINT (checked first), not STOP
        assert mock_result["code_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert mock_result.get("should_stop") is not True

    def test_handle_code_review_limit_keyword_precedence_skip_before_stop(self, mock_state, mock_result):
        """Should prioritize SKIP over STOP when both are present (SKIP checked before STOP)."""
        user_input = {"q1": "SKIP STOP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should match SKIP (checked before STOP)
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("should_stop") is not True

    def test_handle_code_review_limit_empty_string_in_middle_of_responses(self, mock_state, mock_result):
        """Should use the last non-empty response when earlier ones are empty."""
        user_input = {
            "q1": "",
            "q2": "PROVIDE_HINT: Use this hint",
        }
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use the last response (q2), not the empty one
        assert mock_result["code_revision_count"] == 0
        assert "Use this hint" in mock_result["reviewer_feedback"]

    def test_handle_code_review_limit_none_value_in_user_responses(self, mock_state, mock_result):
        """Should handle None values in user_responses gracefully."""
        user_input = {"q1": None}
        
        # parse_user_response converts None to string "None", then uppercases to "NONE"
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should fall through to ask_user since "NONE" won't match keywords
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_code_review_limit_none_user_responses_dict(self, mock_state, mock_result):
        """Should handle None user_responses dict gracefully."""
        # parse_user_response handles None by returning empty string
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, None, "stage1")
        
        # Should fall through to ask_user since empty string won't match keywords
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_code_review_limit_last_response_is_empty_string(self, mock_state, mock_result):
        """Should handle case where last response is empty string."""
        user_input = {
            "q1": "PROVIDE_HINT: First hint",
            "q2": "",  # Last response is empty
        }
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # parse_user_response should return empty string, which won't match keywords
        # So should fall through to ask_user
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_code_review_limit_hint_with_empty_raw_response(self, mock_state, mock_result):
        """Should handle hint keyword with empty raw response text."""
        user_input = {"q1": "HINT:"}  # Keyword present but no hint text
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should still reset counter and set verdict
        assert mock_result["code_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        # reviewer_feedback should still be set, even if empty
        assert "reviewer_feedback" in mock_result
        assert mock_result["reviewer_feedback"].startswith("User hint:")

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_handle_code_review_limit_skip_with_empty_string_stage_id_does_not_call_update(self, mock_update, mock_state, mock_result):
        """Should not call _update_progress_with_error_handling when stage_id is empty string (falsy)."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Empty string is falsy, so should NOT call _update_progress_with_error_handling
        mock_update.assert_not_called()

    def test_handle_code_review_limit_hint_word_boundary(self, mock_state, mock_result):
        """Keywords as substrings should NOT match (word boundary matching)."""
        # Word boundary matching prevents false positives
        user_input = {"q1": "I was HINTING at something"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # "HINTING" does NOT match "HINT" with word boundary matching
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_code_review_limit_skip_word_boundary(self, mock_state, mock_result):
        """SKIP as substring should NOT match (word boundary matching)."""
        user_input = {"q1": "SKIPPING this stage"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # "SKIPPING" does NOT match "SKIP" with word boundary matching
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_code_review_limit_stop_word_boundary(self, mock_state, mock_result):
        """STOP as substring should NOT match (word boundary matching)."""
        user_input = {"q1": "STOPPING the workflow"}
        
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        # "STOPPING" does NOT match "STOP" with word boundary matching
        assert mock_result["supervisor_verdict"] == "ask_user"
