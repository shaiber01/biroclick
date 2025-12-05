"""reviewer_escalation trigger tests."""

from unittest.mock import patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestReviewerEscalationTrigger:
    """Tests for reviewer_escalation trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_guidance_sets_feedback(self, mock_context):
        """Should set reviewer_feedback when user provides guidance."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "PROVIDE_GUIDANCE: Use the Drude model"},
        }
        
        result = supervisor_node(state)
        
        # Verify verdict
        assert result["supervisor_verdict"] == "ok_continue"
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None
        # Verify reviewer feedback contains the guidance
        assert "reviewer_feedback" in result
        assert "Use the Drude model" in result["reviewer_feedback"]
        # Verify supervisor feedback is set
        assert "Continuing with user guidance" in result.get("supervisor_feedback", "")
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_guidance_lowercase(self, mock_context):
        """Should handle provide_guidance in lowercase."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "provide_guidance: try a different approach"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "try a different approach" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_guidance_alias(self, mock_context):
        """Should handle GUIDANCE alias."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "GUIDANCE: Use tabulated data"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Use tabulated data" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_answer_alias(self, mock_context):
        """Should handle ANSWER alias."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "ANSWER: The spacing is 20nm period"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "The spacing is 20nm period" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skip_marks_stage_blocked(self, mock_update, mock_context):
        """Should mark stage as blocked when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("ask_user_trigger") is None
        # Verify update_progress_stage_status was called with correct args
        mock_update.assert_called_once_with(
            state, "stage1", "blocked", summary="Skipped by user due to reviewer escalation"
        )
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skip_does_not_call_update_when_no_stage_id(self, mock_update, mock_context):
        """Should not call update_progress_stage_status when current_stage_id is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stop_ends_workflow(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_unclear_response_asks_clarification(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "I dont know what to do"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert result.get("ask_user_trigger") is None
        # Verify clarification question is set
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1
        assert "PROVIDE_GUIDANCE" in result["pending_user_questions"][0]
        assert "SKIP" in result["pending_user_questions"][0] or "SKIP_STAGE" in result["pending_user_questions"][0]
        assert "STOP" in result["pending_user_questions"][0]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_empty_response_asks_clarification(self, mock_context):
        """Should ask for clarification on empty response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": ""},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_state_not_mutated(self, mock_context):
        """Should not mutate the input state dict."""
        mock_context.return_value = None
        
        original_state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "PROVIDE_GUIDANCE: Test guidance"},
            "other_field": "should_not_change",
        }
        state_copy = original_state.copy()
        
        result = supervisor_node(state_copy)
        
        # Verify state was not mutated
        assert state_copy["ask_user_trigger"] == "reviewer_escalation"  # Not cleared in state
        assert state_copy["other_field"] == "should_not_change"
        # But result should have the changes
        assert result.get("ask_user_trigger") is None


class TestHandleReviewerEscalation:
    """Direct handler tests for reviewer escalation."""

    def test_provide_guidance_sets_feedback(self, mock_state, mock_result):
        """Should handle PROVIDE_GUIDANCE response correctly."""
        user_input = {"q1": "PROVIDE_GUIDANCE: Use the measured optical constants"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Verify reviewer feedback contains the guidance
        assert "reviewer_feedback" in mock_result
        assert "User guidance:" in mock_result["reviewer_feedback"]
        assert "Use the measured optical constants" in mock_result["reviewer_feedback"]
        # Verify verdict
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Verify supervisor feedback
        assert "Continuing with user guidance" in mock_result.get("supervisor_feedback", "")
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True

    def test_provide_guidance_with_guidance_alias(self, mock_state, mock_result):
        """Should handle GUIDANCE alias."""
        user_input = {"q1": "GUIDANCE: Check the boundary conditions"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "Check the boundary conditions" in mock_result["reviewer_feedback"]

    def test_provide_guidance_empty_user_responses(self, mock_state, mock_result):
        """Should handle empty user_responses dict gracefully."""
        user_input = {}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Should fall through to ask_user
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_provide_guidance_none_stage_id(self, mock_state, mock_result):
        """Should handle guidance when current_stage_id is None."""
        user_input = {"q1": "PROVIDE_GUIDANCE: Test"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_skip_marks_stage_blocked(self, mock_update, mock_state, mock_result):
        """Should handle SKIP response correctly."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Verify _update_progress_with_error_handling was called with correct arguments
        mock_update.assert_called_once_with(
            mock_state, mock_result, "stage1", "blocked", summary="Skipped by user due to reviewer escalation"
        )
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_skip_none_stage_id(self, mock_update, mock_state, mock_result):
        """Should not call _update_progress_with_error_handling when stage_id is None."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    def test_stop_ends_workflow(self, mock_state, mock_result):
        """Should handle STOP response correctly."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_unknown_response(self, mock_state, mock_result):
        """Should handle unknown response correctly."""
        user_input = {"q1": "Just keep going"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        # Verify clarification question is set
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "PROVIDE_GUIDANCE" in question
        assert "SKIP" in question or "SKIP_STAGE" in question
        assert "STOP" in question
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True

    def test_case_insensitive_matching(self, mock_state, mock_result):
        """Should match keywords case-insensitively."""
        test_cases = [
            ("provide_guidance: test", "ok_continue", False),
            ("guidance: test", "ok_continue", False),
            ("skip", "ok_continue", False),
            ("stop", "all_complete", True),
        ]
        
        for user_input_text, expected_verdict, should_stop in test_cases:
            mock_result.clear()
            user_input = {"q1": user_input_text}
            
            trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == expected_verdict, f"Failed for input: {user_input_text}"
            if should_stop:
                assert mock_result.get("should_stop") is True

    def test_state_not_mutated(self, mock_state, mock_result):
        """Should not mutate the input state dict."""
        original_state = mock_state.copy()
        user_input = {"q1": "PROVIDE_GUIDANCE: Test"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Verify state was not mutated
        assert mock_state == original_state

    def test_guidance_extracts_text_after_keyword(self, mock_state, mock_result):
        """Should extract the guidance text correctly."""
        user_input = {"q1": "PROVIDE_GUIDANCE: This is detailed guidance with multiple words"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Should include the full guidance text
        assert "This is detailed guidance with multiple words" in mock_result["reviewer_feedback"]
        assert mock_result["reviewer_feedback"].startswith("User guidance:")

    def test_guidance_with_special_characters(self, mock_state, mock_result):
        """Should handle guidance with special characters correctly."""
        user_input = {"q1": "PROVIDE_GUIDANCE: Use λ = 500nm and ε = -5.0+0.3i"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "λ = 500nm" in mock_result["reviewer_feedback"]
        assert "ε = -5.0+0.3i" in mock_result["reviewer_feedback"]


