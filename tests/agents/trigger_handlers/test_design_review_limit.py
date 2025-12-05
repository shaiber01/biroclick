"""design_review_limit trigger tests."""

from unittest.mock import ANY, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestDesignReviewLimitTrigger:
    """Tests for design_review_limit trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_hint(self, mock_context):
        """Should reset design revision count when user provides hint."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "HINT: Try using a larger simulation domain"},
            "design_revision_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["design_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_design"
        assert "Try using a larger simulation domain" in result.get("reviewer_feedback", "")
        assert result.get("ask_user_trigger") is None  # Should be cleared

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_provide_hint(self, mock_context):
        """Should reset design revision count when user provides PROVIDE_HINT."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "PROVIDE_HINT: Use a different material"},
            "design_revision_count": 5,
        }
        
        result = supervisor_node(state)
        
        assert result["design_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_design"
        assert "Use a different material" in result.get("reviewer_feedback", "")

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][0] == state  # state passed correctly
        assert call_args[0][1] == "stage1"  # stage_id passed correctly
        assert call_args[0][2] == "blocked"  # status is "blocked"
        assert call_args[1]["summary"] == "Skipped by user due to design review issues"
        assert call_args[1].get("invalidation_reason") is None  # Should be None or not present

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip_without_stage_id(self, mock_update, mock_context):
        """Should skip stage when user says SKIP even without current_stage_id."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "SKIP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()  # Should not call update_progress_stage_status

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "I dont know what to do"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "PROVIDE_HINT" in result["pending_user_questions"][0] or "SKIP_STAGE" in result["pending_user_questions"][0]


class TestHandleDesignReviewLimit:
    """Direct handler tests for design review limit."""

    def test_handle_design_review_limit_hint_provide_hint(self, mock_state, mock_result):
        """Should handle PROVIDE_HINT keyword."""
        user_input = {"q1": "PROVIDE_HINT: Fix the dimensions."}
        return_value = trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert return_value is None  # Handler should return None
        assert mock_result["design_revision_count"] == 0
        assert "Fix the dimensions." in mock_result["reviewer_feedback"]
        assert mock_result["reviewer_feedback"].startswith("User hint:")
        assert mock_result["supervisor_verdict"] == "retry_design"
        assert "supervisor_feedback" not in mock_result  # Should NOT set supervisor_feedback

    def test_handle_design_review_limit_hint_keyword(self, mock_state, mock_result):
        """Should handle HINT keyword."""
        user_input = {"q1": "HINT: Use a larger domain size"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "Use a larger domain size" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "retry_design"

    def test_handle_design_review_limit_hint_case_insensitive(self, mock_state, mock_result):
        """Should handle hint in lowercase (parse_user_response uppercases)."""
        user_input = {"q1": "hint: lowercase hint"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "lowercase hint" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "retry_design"

    def test_handle_design_review_limit_hint_multiple_responses(self, mock_state, mock_result):
        """Should use last response when multiple responses exist."""
        user_input = {
            "q1": "First response",
            "q2": "PROVIDE_HINT: This is the actual hint"
        }
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "This is the actual hint" in mock_result["reviewer_feedback"]
        assert "First response" not in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_hint_empty_hint_text(self, mock_state, mock_result):
        """Should handle empty hint text."""
        user_input = {"q1": "PROVIDE_HINT:"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "User hint:" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "retry_design"

    def test_handle_design_review_limit_hint_empty_user_responses(self, mock_state, mock_result):
        """Should handle empty user_responses dict."""
        user_input = {}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        # Should not match HINT, should fall through to else clause
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert "design_revision_count" not in mock_result  # Should not reset

    def test_handle_design_review_limit_hint_none_user_responses(self, mock_state, mock_result):
        """Should handle None user_responses."""
        user_input = None
        # parse_user_response should handle None, but let's test the handler
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        # parse_user_response returns "" for None, so should fall through to else
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_design_review_limit_skip(self, mock_update, mock_state, mock_result):
        """Should skip stage when user says SKIP."""
        user_input = {"q1": "SKIP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][0] is mock_state  # state passed correctly
        assert call_args[0][1] == "stage1"  # stage_id passed correctly
        assert call_args[0][2] == "blocked"  # status is "blocked"
        assert call_args[1]["summary"] == "Skipped by user due to design review issues"
        assert call_args[1].get("invalidation_reason") is None  # Should be None or not present

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_design_review_limit_skip_no_stage_id(self, mock_update, mock_state, mock_result):
        """Should skip without calling update_progress_stage_status when stage_id is None."""
        user_input = {"q1": "SKIP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, None
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_design_review_limit_skip_empty_stage_id(self, mock_update, mock_state, mock_result):
        """Should skip without calling update_progress_stage_status when stage_id is empty string."""
        user_input = {"q1": "SKIP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, ""
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    def test_handle_design_review_limit_stop(self, mock_state, mock_result):
        """Should stop workflow when user says STOP."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_design_review_limit_stop_case_insensitive(self, mock_state, mock_result):
        """Should handle STOP in lowercase."""
        user_input = {"q1": "stop"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_design_review_limit_unknown_response(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "Just keep going"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        assert "PROVIDE_HINT" in mock_result["pending_user_questions"][0]
        assert "SKIP_STAGE" in mock_result["pending_user_questions"][0]
        assert "STOP" in mock_result["pending_user_questions"][0]

    def test_handle_design_review_limit_empty_string_response(self, mock_state, mock_result):
        """Should ask for clarification on empty string response."""
        user_input = {"q1": ""}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_design_review_limit_whitespace_only_response(self, mock_state, mock_result):
        """Should ask for clarification on whitespace-only response."""
        user_input = {"q1": "   \n\t  "}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_design_review_limit_hint_preserves_existing_result_keys(self, mock_state, mock_result):
        """Should not overwrite unrelated keys in result dict."""
        mock_result["existing_key"] = "existing_value"
        mock_result["another_key"] = 42
        
        user_input = {"q1": "PROVIDE_HINT: Test hint"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["existing_key"] == "existing_value"
        assert mock_result["another_key"] == 42
        assert mock_result["design_revision_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_design"

    def test_handle_design_review_limit_hint_overwrites_existing_count(self, mock_state, mock_result):
        """Should overwrite existing design_revision_count."""
        mock_result["design_revision_count"] = 10
        
        user_input = {"q1": "PROVIDE_HINT: Reset count"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0

    def test_handle_design_review_limit_hint_overwrites_existing_feedback(self, mock_state, mock_result):
        """Should overwrite existing reviewer_feedback."""
        mock_result["reviewer_feedback"] = "Old feedback"
        
        user_input = {"q1": "PROVIDE_HINT: New feedback"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert "New feedback" in mock_result["reviewer_feedback"]
        assert "Old feedback" not in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_hint_does_not_set_supervisor_feedback(self, mock_state, mock_result):
        """Should NOT set supervisor_feedback (unlike code_review_limit)."""
        user_input = {"q1": "PROVIDE_HINT: Test hint"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert "supervisor_feedback" not in mock_result

    def test_handle_design_review_limit_skip_does_not_set_should_stop(self, mock_state, mock_result):
        """Should not set should_stop when skipping."""
        user_input = {"q1": "SKIP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert "should_stop" not in mock_result or mock_result.get("should_stop") is not True

    def test_handle_design_review_limit_stop_sets_should_stop_true(self, mock_state, mock_result):
        """Should set should_stop to True when stopping."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["should_stop"] is True
        assert isinstance(mock_result["should_stop"], bool)

    def test_handle_design_review_limit_hint_with_partial_match(self, mock_state, mock_result):
        """Should handle HINT keyword even when PROVIDE_HINT is not present."""
        user_input = {"q1": "HINT: This is a hint"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "This is a hint" in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_hint_with_both_keywords(self, mock_state, mock_result):
        """Should handle response containing both PROVIDE_HINT and HINT."""
        user_input = {"q1": "PROVIDE_HINT: HINT: Use this"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "HINT: Use this" in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_skip_with_skip_stage_keyword(self, mock_state, mock_result):
        """Should handle SKIP_STAGE keyword."""
        user_input = {"q1": "SKIP_STAGE"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_design_review_limit_state_not_mutated(self, mock_state, mock_result):
        """Should not mutate state dict."""
        original_state = dict(mock_state)
        user_input = {"q1": "PROVIDE_HINT: Test"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_state == original_state

    def test_handle_design_review_limit_result_mutated(self, mock_state, mock_result):
        """Should mutate result dict in place."""
        user_input = {"q1": "PROVIDE_HINT: Test"}
        result_id_before = id(mock_result)
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        result_id_after = id(mock_result)
        
        assert result_id_before == result_id_after  # Same object
        assert "design_revision_count" in mock_result  # Was mutated

    def test_handle_design_review_limit_hint_with_none_value_in_responses(self, mock_state, mock_result):
        """Should handle None value in user_responses dict."""
        user_input = {"q1": None}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        # parse_user_response converts None to "NONE" (str(None).strip().upper())
        # So "HINT" won't be in "NONE", should fall through to else
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_design_review_limit_hint_with_none_in_middle(self, mock_state, mock_result):
        """Should use last response even if earlier ones are None."""
        user_input = {"q1": None, "q2": "PROVIDE_HINT: Actual hint"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "Actual hint" in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_hint_with_non_string_value(self, mock_state, mock_result):
        """Should handle non-string values in user_responses."""
        user_input = {"q1": 12345}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        # parse_user_response converts to string, so "HINT" won't be in "12345"
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_design_review_limit_hint_with_list_value(self, mock_state, mock_result):
        """Should handle list values in user_responses."""
        user_input = {"q1": ["PROVIDE_HINT", "test"]}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        # parse_user_response converts list to string representation
        # str(["PROVIDE_HINT", "test"]) = "['PROVIDE_HINT', 'test']"
        # "HINT" should be in that string, so should match
        assert mock_result["design_revision_count"] == 0

    def test_handle_design_review_limit_invalid_user_responses_type(self, mock_state, mock_result):
        """Should raise TypeError if user_responses is not a dict."""
        # parse_user_response raises TypeError for non-dict
        with pytest.raises(TypeError):
            trigger_handlers.handle_design_review_limit(
                mock_state, mock_result, "not a dict", "stage1"
            )

    def test_handle_design_review_limit_hint_raw_response_extraction_empty_dict(self, mock_state, mock_result):
        """Should handle empty dict when extracting raw_response for hint."""
        user_input = {}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        # Empty dict means parse_user_response returns "", so no HINT match
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_design_review_limit_hint_raw_response_with_empty_string_value(self, mock_state, mock_result):
        """Should handle empty string value when extracting raw_response.
        
        Note: extract_guidance_text now strips the PROVIDE_HINT: prefix,
        so "PROVIDE_HINT:" becomes an empty hint.
        """
        user_input = {"q1": "PROVIDE_HINT:"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        # The prefix is stripped, so the hint is empty
        assert mock_result["reviewer_feedback"] == "User hint: "
        assert mock_result["supervisor_verdict"] == "retry_design"

    def test_handle_design_review_limit_skip_with_falsy_stage_id_values(self, mock_state, mock_result):
        """Should not call update_progress_stage_status for falsy stage_id values."""
        falsy_values = [None, "", 0, False]
        
        for falsy_value in falsy_values:
            mock_result.clear()
            user_input = {"q1": "SKIP"}
            with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
                trigger_handlers.handle_design_review_limit(
                    mock_state, mock_result, user_input, falsy_value
                )
                mock_update.assert_not_called()
                assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_design_review_limit_hint_verifies_exact_count_reset(self, mock_state, mock_result):
        """Should reset count to exactly 0, not just decrement."""
        mock_result["design_revision_count"] = 999
        user_input = {"q1": "PROVIDE_HINT: Reset"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert mock_result["design_revision_count"] is not None

    def test_handle_design_review_limit_stop_verifies_should_stop_type(self, mock_state, mock_result):
        """Should set should_stop to boolean True, not string or other truthy value."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["should_stop"] is True
        assert isinstance(mock_result["should_stop"], bool)
        assert type(mock_result["should_stop"]) is bool  # Check exact type is bool, not int or str
        # Verify it's not a string "True"
        assert not isinstance(mock_result["should_stop"], str)

    def test_handle_design_review_limit_unknown_response_overwrites_existing_questions(self, mock_state, mock_result):
        """Should overwrite existing pending_user_questions."""
        mock_result["pending_user_questions"] = ["Old question"]
        user_input = {"q1": "Unknown response"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["pending_user_questions"] != ["Old question"]
        assert len(mock_result["pending_user_questions"]) == 1
        assert "PROVIDE_HINT" in mock_result["pending_user_questions"][0]

    def test_handle_design_review_limit_hint_with_unicode_characters(self, mock_state, mock_result):
        """Should handle unicode characters in hint text."""
        user_input = {"q1": "PROVIDE_HINT: Use α (alpha) parameter"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "α (alpha)" in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_hint_with_special_characters(self, mock_state, mock_result):
        """Should handle special characters in hint text."""
        user_input = {"q1": "PROVIDE_HINT: Try @#$%^&*()"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "@#$%^&*()" in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_hint_with_newlines(self, mock_state, mock_result):
        """Should handle newlines in hint text."""
        user_input = {"q1": "PROVIDE_HINT: Line 1\nLine 2\nLine 3"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert "Line 1" in mock_result["reviewer_feedback"]
        assert "Line 2" in mock_result["reviewer_feedback"]
        assert "Line 3" in mock_result["reviewer_feedback"]

    def test_handle_design_review_limit_skip_verifies_stage_status_call_args(self, mock_state, mock_result):
        """Should call update_progress_stage_status with correct arguments."""
        user_input = {"q1": "SKIP"}
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_design_review_limit(
                mock_state, mock_result, user_input, "stage42"
            )
            
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args[0][0] is mock_state  # state passed correctly
            assert call_args[0][1] == "stage42"  # stage_id passed correctly
            assert call_args[0][2] == "blocked"  # status is "blocked"
            assert call_args[1]["summary"] == "Skipped by user due to design review issues"

    def test_handle_design_review_limit_hint_does_not_affect_other_counters(self, mock_state, mock_result):
        """Should not affect other revision counters."""
        mock_result["code_revision_count"] = 5
        mock_result["replan_count"] = 3
        user_input = {"q1": "PROVIDE_HINT: Test"}
        trigger_handlers.handle_design_review_limit(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["design_revision_count"] == 0
        assert mock_result["code_revision_count"] == 5
        assert mock_result["replan_count"] == 3
