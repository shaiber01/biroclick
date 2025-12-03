"""execution/physics limit trigger tests."""

from unittest.mock import ANY, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestExecutionFailureLimitTrigger:
    """Tests for execution_failure_limit trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_retry(self, mock_context):
        """Should reset execution failure count on RETRY."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "RETRY with more memory"},
            "execution_failure_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["execution_failure_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in result
        assert "more memory" in result["supervisor_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_guidance(self, mock_context):
        """Should reset execution failure count on GUIDANCE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "GUIDANCE: reduce resolution"},
            "execution_failure_count": 5,
        }
        
        result = supervisor_node(state)
        
        assert result["execution_failure_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"
        assert "reduce resolution" in result["supervisor_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once_with(
            state, "stage1", "blocked",
            summary="Skipped by user due to execution failures"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "I dont know what to do"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "RETRY_WITH_GUIDANCE" in result["pending_user_questions"][0] or "SKIP_STAGE" in result["pending_user_questions"][0] or "STOP" in result["pending_user_questions"][0]


class TestPhysicsFailureLimitTrigger:
    """Tests for physics_failure_limit trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_retry(self, mock_context):
        """Should reset physics failure count on RETRY."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "RETRY"},
            "physics_failure_count": 2,
        }
        
        result = supervisor_node(state)
        
        assert result["physics_failure_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_accepts_partial_on_accept(self, mock_update, mock_context):
        """Should accept as partial when user says ACCEPT."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "ACCEPT"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once_with(
            state, result, "stage1", "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_accepts_partial_on_partial(self, mock_update, mock_context):
        """Should accept as partial when user says PARTIAL."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "PARTIAL is fine"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once_with(
            state, result, "stage1", "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once_with(
            state, "stage1", "blocked",
            summary="Skipped by user due to physics check failures"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "Maybe continue?"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0


class TestHandleExecutionFailureLimit:
    """Direct handler tests for execution_failure_limit."""

    def test_handle_execution_failure_limit_retry(self, mock_state, mock_result):
        """Test RETRY resets count and sets feedback."""
        user_input = {"q1": "RETRY_WITH_GUIDANCE: Check memory."}
        initial_count = 5
        mock_result["execution_failure_count"] = initial_count
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert "Check memory" in mock_result["supervisor_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["supervisor_feedback"].startswith("User guidance:")

    def test_handle_execution_failure_limit_retry_keyword_only(self, mock_state, mock_result):
        """Test RETRY keyword alone resets count."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 3
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result

    def test_handle_execution_failure_limit_guidance_keyword(self, mock_state, mock_result):
        """Test GUIDANCE keyword resets count."""
        user_input = {"q1": "GUIDANCE: reduce resolution"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "reduce resolution" in mock_result["supervisor_feedback"]

    def test_handle_execution_failure_limit_retry_with_empty_guidance(self, mock_state, mock_result):
        """Test RETRY with empty guidance still resets count."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_execution_failure_limit_skip_with_stage_id(self, mock_state, mock_result):
        """Test SKIP updates stage status when stage_id provided."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, "stage1", "blocked",
                summary="Skipped by user due to execution failures"
            )

    def test_handle_execution_failure_limit_skip_without_stage_id(self, mock_state, mock_result):
        """Test SKIP works without stage_id (no update call)."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, None)
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_not_called()

    def test_handle_execution_failure_limit_stop(self, mock_state, mock_result):
        """Test STOP sets verdict and should_stop flag."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_execution_failure_limit_unknown_response(self, mock_state, mock_result):
        """Test unknown response asks for clarification."""
        user_input = {"q1": "Just keep going"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) > 0
        assert any("RETRY_WITH_GUIDANCE" in q or "SKIP_STAGE" in q or "STOP" in q 
                   for q in mock_result["pending_user_questions"])

    def test_handle_execution_failure_limit_empty_response(self, mock_state, mock_result):
        """Test empty response asks for clarification."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_execution_failure_limit_multiple_responses(self, mock_state, mock_result):
        """Test handler uses last response when multiple provided."""
        user_input = {"q1": "SKIP", "q2": "RETRY with guidance"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use last response (RETRY)
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "guidance" in mock_result["supervisor_feedback"]

    def test_handle_execution_failure_limit_case_insensitive(self, mock_state, mock_result):
        """Test keywords are case-insensitive."""
        user_input = {"q1": "retry with more memory"}
        mock_result["execution_failure_count"] = 3
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_execution_failure_limit_retry_and_guidance_both_present(self, mock_state, mock_result):
        """Test when both RETRY and GUIDANCE keywords present."""
        user_input = {"q1": "RETRY with GUIDANCE: fix memory"}
        mock_result["execution_failure_count"] = 4
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "fix memory" in mock_result["supervisor_feedback"]


class TestHandlePhysicsFailureLimit:
    """Direct handler tests for physics_failure_limit."""

    def test_handle_physics_failure_limit_retry(self, mock_state, mock_result):
        """Test RETRY resets count and sets feedback."""
        user_input = {"q1": "RETRY"}
        initial_count = 3
        mock_result["physics_failure_count"] = initial_count
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result

    def test_handle_physics_failure_limit_retry_with_guidance(self, mock_state, mock_result):
        """Test RETRY with guidance text."""
        user_input = {"q1": "RETRY with better parameters"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "better parameters" in mock_result["supervisor_feedback"]

    def test_handle_physics_failure_limit_accept_with_stage_id(self, mock_state, mock_result):
        """Test ACCEPT marks stage as partial when stage_id provided."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_accept_without_stage_id(self, mock_state, mock_result):
        """Test ACCEPT works without stage_id (no update call)."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, None)
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_not_called()

    def test_handle_physics_failure_limit_partial_with_stage_id(self, mock_state, mock_result):
        """Test PARTIAL marks stage as partial when stage_id provided."""
        user_input = {"q1": "PARTIAL is fine"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_accept_partial_combined(self, mock_state, mock_result):
        """Test ACCEPT_PARTIAL combination."""
        user_input = {"q1": "ACCEPT_PARTIAL results."}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_skip_with_stage_id(self, mock_state, mock_result):
        """Test SKIP updates stage status when stage_id provided."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, "stage1", "blocked",
                summary="Skipped by user due to physics check failures"
            )

    def test_handle_physics_failure_limit_skip_without_stage_id(self, mock_state, mock_result):
        """Test SKIP works without stage_id (no update call)."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, None)
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_not_called()

    def test_handle_physics_failure_limit_stop(self, mock_state, mock_result):
        """Test STOP sets verdict and should_stop flag."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_physics_failure_limit_unknown_response(self, mock_state, mock_result):
        """Test unknown response asks for clarification."""
        user_input = {"q1": "Maybe continue anyway?"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) > 0
        assert any("RETRY" in q or "ACCEPT_PARTIAL" in q or "SKIP_STAGE" in q or "STOP" in q
                   for q in mock_result["pending_user_questions"])

    def test_handle_physics_failure_limit_empty_response(self, mock_state, mock_result):
        """Test empty response asks for clarification."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_physics_failure_limit_multiple_responses(self, mock_state, mock_result):
        """Test handler uses last response when multiple provided."""
        user_input = {"q1": "ACCEPT", "q2": "RETRY"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use last response (RETRY)
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_physics_failure_limit_case_insensitive(self, mock_state, mock_result):
        """Test keywords are case-insensitive."""
        user_input = {"q1": "accept partial"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once()

    def test_handle_physics_failure_limit_retry_takes_precedence_over_accept(self, mock_state, mock_result):
        """Test RETRY takes precedence when both RETRY and ACCEPT present."""
        user_input = {"q1": "RETRY and ACCEPT"}
        mock_result["physics_failure_count"] = 1
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY should be checked first
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_physics_failure_limit_accept_partial_both_keywords(self, mock_state, mock_result):
        """Test when both ACCEPT and PARTIAL keywords present."""
        user_input = {"q1": "ACCEPT PARTIAL results"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_does_not_modify_state(self, mock_state, mock_result):
        """Test handler does not modify state dict."""
        original_state = dict(mock_state)
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_state == original_state

    def test_handle_execution_failure_limit_does_not_modify_state(self, mock_state, mock_result):
        """Test handler does not modify state dict."""
        original_state = dict(mock_state)
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_state == original_state

    def test_handle_physics_failure_limit_none_user_responses(self, mock_state, mock_result):
        """Test handler handles None user_responses."""
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, None, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_physics_failure_limit_empty_dict_user_responses(self, mock_state, mock_result):
        """Test handler handles empty dict user_responses."""
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, {}, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_physics_failure_limit_whitespace_only_response(self, mock_state, mock_result):
        """Test handler handles whitespace-only response."""
        user_input = {"q1": "   \n\t  "}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_physics_failure_limit_non_string_response(self, mock_state, mock_result):
        """Test handler handles non-string response (converted to string)."""
        user_input = {"q1": 99999}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_physics_failure_limit_retry_with_whitespace(self, mock_state, mock_result):
        """Test RETRY keyword with surrounding whitespace."""
        user_input = {"q1": "  RETRY  "}
        mock_result["physics_failure_count"] = 1
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_physics_failure_limit_accept_with_whitespace(self, mock_state, mock_result):
        """Test ACCEPT keyword with surrounding whitespace."""
        user_input = {"q1": "  ACCEPT  "}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once()

    def test_handle_physics_failure_limit_count_not_present_in_result(self, mock_state, mock_result):
        """Test handler sets count even if not present in result initially."""
        # Don't set physics_failure_count in mock_result
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_physics_failure_limit_verdict_overwrites_existing(self, mock_state, mock_result):
        """Test handler overwrites existing verdict."""
        mock_result["supervisor_verdict"] = "some_other_verdict"
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_physics_failure_limit_feedback_overwrites_existing(self, mock_state, mock_result):
        """Test handler overwrites existing feedback."""
        mock_result["supervisor_feedback"] = "Old feedback"
        user_input = {"q1": "RETRY with new guidance"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert "new guidance" in mock_result["supervisor_feedback"]
        assert mock_result["supervisor_feedback"].startswith("User guidance:")

    def test_handle_physics_failure_limit_skip_empty_string(self, mock_state, mock_result):
        """Test SKIP with empty string value."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_physics_failure_limit_stop_sets_should_stop_even_if_not_present(self, mock_state, mock_result):
        """Test STOP sets should_stop even if not present in result."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_execution_failure_limit_stop_sets_should_stop_even_if_not_present(self, mock_state, mock_result):
        """Test STOP sets should_stop even if not present in result."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_execution_failure_limit_none_user_responses(self, mock_state, mock_result):
        """Test handler handles None user_responses."""
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, None, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_execution_failure_limit_empty_dict_user_responses(self, mock_state, mock_result):
        """Test handler handles empty dict user_responses."""
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, {}, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_execution_failure_limit_whitespace_only_response(self, mock_state, mock_result):
        """Test handler handles whitespace-only response."""
        user_input = {"q1": "   \n\t  "}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_execution_failure_limit_non_string_response(self, mock_state, mock_result):
        """Test handler handles non-string response (converted to string)."""
        user_input = {"q1": 12345}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_execution_failure_limit_retry_with_whitespace(self, mock_state, mock_result):
        """Test RETRY keyword with surrounding whitespace."""
        user_input = {"q1": "  RETRY  with guidance  "}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_execution_failure_limit_skip_empty_string(self, mock_state, mock_result):
        """Test SKIP with empty string value."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_execution_failure_limit_verdict_overwrites_existing(self, mock_state, mock_result):
        """Test handler overwrites existing verdict."""
        mock_result["supervisor_verdict"] = "some_other_verdict"
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_execution_failure_limit_feedback_overwrites_existing(self, mock_state, mock_result):
        """Test handler overwrites existing feedback."""
        mock_result["supervisor_feedback"] = "Old feedback"
        user_input = {"q1": "RETRY with new guidance"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert "new guidance" in mock_result["supervisor_feedback"]
        assert mock_result["supervisor_feedback"].startswith("User guidance:")

    def test_handle_execution_failure_limit_count_not_present_in_result(self, mock_state, mock_result):
        """Test handler sets count even if not present in result initially."""
        # Don't set execution_failure_count in mock_result
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "ok_continue"
