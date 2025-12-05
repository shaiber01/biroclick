"""execution/physics limit trigger tests."""

from unittest.mock import ANY, patch, MagicMock

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from src.agents.supervision.trigger_handlers import (
    handle_execution_failure_limit,
    handle_physics_failure_limit,
    handle_trigger,
    TRIGGER_HANDLERS,
)
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
        assert result["supervisor_verdict"] == "retry_generate_code"
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
        assert result["supervisor_verdict"] == "retry_generate_code"
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
        assert result["supervisor_verdict"] == "retry_generate_code"
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
        
        assert result["supervisor_verdict"] == "retry_analyze"
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
        
        assert result["supervisor_verdict"] == "retry_analyze"
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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert mock_result["supervisor_feedback"].startswith("User guidance:")

    def test_handle_execution_failure_limit_retry_keyword_only(self, mock_state, mock_result):
        """Test RETRY keyword alone resets count."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 3
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "supervisor_feedback" in mock_result

    def test_handle_execution_failure_limit_guidance_keyword(self, mock_state, mock_result):
        """Test GUIDANCE keyword resets count."""
        user_input = {"q1": "GUIDANCE: reduce resolution"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "reduce resolution" in mock_result["supervisor_feedback"]

    def test_handle_execution_failure_limit_retry_with_empty_guidance(self, mock_state, mock_result):
        """Test RETRY with empty guidance still resets count."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_execution_failure_limit_skip_with_stage_id(self, mock_state, mock_result):
        """Test SKIP updates stage status when stage_id provided."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "blocked",
                summary="Skipped by user due to execution failures"
            )

    def test_handle_execution_failure_limit_skip_without_stage_id(self, mock_state, mock_result):
        """Test SKIP works without stage_id (no update call)."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "guidance" in mock_result["supervisor_feedback"]

    def test_handle_execution_failure_limit_case_insensitive(self, mock_state, mock_result):
        """Test keywords are case-insensitive."""
        user_input = {"q1": "retry with more memory"}
        mock_result["execution_failure_count"] = 3
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_execution_failure_limit_retry_and_guidance_both_present(self, mock_state, mock_result):
        """Test when both RETRY and GUIDANCE keywords present."""
        user_input = {"q1": "RETRY with GUIDANCE: fix memory"}
        mock_result["execution_failure_count"] = 4
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "supervisor_feedback" in mock_result

    def test_handle_physics_failure_limit_retry_with_guidance(self, mock_state, mock_result):
        """Test RETRY with guidance text."""
        user_input = {"q1": "RETRY with better parameters"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "better parameters" in mock_result["supervisor_feedback"]

    def test_handle_physics_failure_limit_accept_with_stage_id(self, mock_state, mock_result):
        """Test ACCEPT marks stage as partial when stage_id provided."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_accept_without_stage_id(self, mock_state, mock_result):
        """Test ACCEPT works without stage_id (no update call)."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, None)
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
            mock_update.assert_not_called()

    def test_handle_physics_failure_limit_partial_with_stage_id(self, mock_state, mock_result):
        """Test PARTIAL marks stage as partial when stage_id provided."""
        user_input = {"q1": "PARTIAL is fine"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_accept_partial_combined(self, mock_state, mock_result):
        """Test ACCEPT_PARTIAL combination."""
        user_input = {"q1": "ACCEPT_PARTIAL results."}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "completed_partial",
                summary="Accepted as partial by user despite physics issues"
            )

    def test_handle_physics_failure_limit_skip_with_stage_id(self, mock_state, mock_result):
        """Test SKIP updates stage status when stage_id provided."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, mock_result, "stage1", "blocked",
                summary="Skipped by user due to physics check failures"
            )

    def test_handle_physics_failure_limit_skip_without_stage_id(self, mock_state, mock_result):
        """Test SKIP works without stage_id (no update call)."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_physics_failure_limit_case_insensitive(self, mock_state, mock_result):
        """Test keywords are case-insensitive."""
        user_input = {"q1": "accept partial"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
            mock_update.assert_called_once()

    def test_handle_physics_failure_limit_retry_takes_precedence_over_accept(self, mock_state, mock_result):
        """Test RETRY takes precedence when both RETRY and ACCEPT present."""
        user_input = {"q1": "RETRY and ACCEPT"}
        mock_result["physics_failure_count"] = 1
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY should be checked first
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_physics_failure_limit_accept_partial_both_keywords(self, mock_state, mock_result):
        """Test when both ACCEPT and PARTIAL keywords present."""
        user_input = {"q1": "ACCEPT PARTIAL results"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_physics_failure_limit_accept_with_whitespace(self, mock_state, mock_result):
        """Test ACCEPT keyword with surrounding whitespace."""
        user_input = {"q1": "  ACCEPT  "}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "retry_analyze"
            mock_update.assert_called_once()

    def test_handle_physics_failure_limit_count_not_present_in_result(self, mock_state, mock_result):
        """Test handler sets count even if not present in result initially."""
        # Don't set physics_failure_count in mock_result
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_handle_physics_failure_limit_verdict_overwrites_existing(self, mock_state, mock_result):
        """Test handler overwrites existing verdict."""
        mock_result["supervisor_verdict"] = "some_other_verdict"
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

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
        
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

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
        assert mock_result["supervisor_verdict"] == "retry_generate_code"


class TestTriggerHandlersRegistry:
    """Tests for the TRIGGER_HANDLERS registry."""

    def test_execution_failure_limit_in_registry(self):
        """Verify execution_failure_limit handler is registered."""
        assert "execution_failure_limit" in TRIGGER_HANDLERS
        assert TRIGGER_HANDLERS["execution_failure_limit"] == handle_execution_failure_limit

    def test_physics_failure_limit_in_registry(self):
        """Verify physics_failure_limit handler is registered."""
        assert "physics_failure_limit" in TRIGGER_HANDLERS
        assert TRIGGER_HANDLERS["physics_failure_limit"] == handle_physics_failure_limit

    def test_all_registered_handlers_are_callable(self):
        """All handlers in the registry must be callable functions."""
        for trigger_name, handler in TRIGGER_HANDLERS.items():
            assert callable(handler), f"Handler for {trigger_name} is not callable"


class TestHandleTriggerDispatch:
    """Tests for the handle_trigger dispatch function."""

    def test_dispatches_to_execution_failure_limit_handler(self, mock_state, mock_result):
        """handle_trigger should dispatch to execution_failure_limit handler."""
        user_input = {"q1": "RETRY with guidance"}
        mock_result["execution_failure_count"] = 3
        
        handle_trigger(
            trigger="execution_failure_limit",
            state=mock_state,
            result=mock_result,
            user_responses=user_input,
            current_stage_id="stage1",
        )
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "guidance" in mock_result["supervisor_feedback"]

    def test_dispatches_to_physics_failure_limit_handler(self, mock_state, mock_result):
        """handle_trigger should dispatch to physics_failure_limit handler."""
        user_input = {"q1": "RETRY"}
        mock_result["physics_failure_count"] = 2
        
        handle_trigger(
            trigger="physics_failure_limit",
            state=mock_state,
            result=mock_result,
            user_responses=user_input,
            current_stage_id="stage1",
        )
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_unknown_trigger_defaults_to_continue(self, mock_state, mock_result):
        """Unknown triggers should default to ok_continue."""
        handle_trigger(
            trigger="unknown_trigger_xyz",
            state=mock_state,
            result=mock_result,
            user_responses={"q1": "whatever"},
            current_stage_id="stage1",
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "unknown trigger" in mock_result["supervisor_feedback"].lower()

    def test_dispatch_with_none_stage_id(self, mock_state, mock_result):
        """handle_trigger should work with None stage_id."""
        user_input = {"q1": "STOP"}
        
        handle_trigger(
            trigger="execution_failure_limit",
            state=mock_state,
            result=mock_result,
            user_responses=user_input,
            current_stage_id=None,
        )
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True


class TestKeywordPrecedence:
    """Tests for keyword precedence when multiple keywords present."""

    def test_execution_retry_checked_before_skip(self, mock_state, mock_result):
        """RETRY should be checked before SKIP in execution handler."""
        # Response contains both RETRY and SKIP
        user_input = {"q1": "RETRY but maybe SKIP"}
        mock_result["execution_failure_count"] = 3
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY should win (checked first in if-elif chain)
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_execution_retry_checked_before_stop(self, mock_state, mock_result):
        """RETRY should be checked before STOP in execution handler."""
        user_input = {"q1": "RETRY or STOP"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY should win
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert mock_result.get("should_stop") is not True

    def test_execution_skip_checked_before_stop(self, mock_state, mock_result):
        """SKIP should be checked before STOP when RETRY not present."""
        user_input = {"q1": "SKIP or STOP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # SKIP should win (checked before STOP)
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("should_stop") is not True
        mock_update.assert_called_once()

    def test_physics_retry_checked_before_accept(self, mock_state, mock_result):
        """RETRY should be checked before ACCEPT in physics handler."""
        user_input = {"q1": "RETRY and ACCEPT"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY should win (checked first)
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_physics_accept_checked_before_skip(self, mock_state, mock_result):
        """ACCEPT should be checked before SKIP in physics handler."""
        user_input = {"q1": "ACCEPT and SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # ACCEPT should win (checked before SKIP)
        assert mock_result["supervisor_verdict"] == "retry_analyze"
        mock_update.assert_called_once_with(
            mock_state, mock_result, "stage1", "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )

    def test_physics_skip_checked_before_stop(self, mock_state, mock_result):
        """SKIP should be checked before STOP when ACCEPT/RETRY not present."""
        user_input = {"q1": "SKIP or STOP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # SKIP should win
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("should_stop") is not True
        mock_update.assert_called_once()


class TestWordBoundaryKeywordMatching:
    """
    Tests for word boundary keyword matching behavior.
    
    The implementation uses check_keywords() which uses regex word boundaries.
    This means words like "RETRYSTYLE" or "ACCEPTABLE" should NOT match "RETRY" or "ACCEPT".
    These tests verify this correct behavior.
    """

    def test_execution_retry_no_substring_match(self, mock_state, mock_result):
        """RETRY should NOT match as substring - 'RETRYING' does not match 'RETRY' as whole word."""
        user_input = {"q1": "RETRYING soon"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: RETRY in RETRYING = False (not a whole word)
        # Should ask for clarification
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert mock_result["execution_failure_count"] == 2  # Not reset

    def test_execution_guidance_no_substring_match(self, mock_state, mock_result):
        """GUIDANCE should NOT match when part of compound word."""
        user_input = {"q1": "GUIDANCE_NEEDED"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: GUIDANCE_NEEDED contains GUIDANCE but with underscore
        # check_keywords uses \b which treats underscore as word character
        # So GUIDANCE_NEEDED should NOT match GUIDANCE
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert mock_result["execution_failure_count"] == 2  # Not reset

    def test_execution_skip_no_substring_match(self, mock_state, mock_result):
        """SKIP should NOT match as substring - 'SKIPPING' does not match 'SKIP' as whole word."""
        user_input = {"q1": "SKIPPING this stage"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: SKIP in SKIPPING = False
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_physics_accept_no_substring_match(self, mock_state, mock_result):
        """ACCEPT should NOT match as substring - 'ACCEPTABLE' does not match 'ACCEPT' as whole word."""
        user_input = {"q1": "ACCEPTABLE results"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: ACCEPT in ACCEPTABLE = False
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_physics_partial_no_substring_match(self, mock_state, mock_result):
        """PARTIAL should NOT match as substring - 'PARTIALLY' does not match 'PARTIAL' as whole word."""
        user_input = {"q1": "PARTIALLY complete"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: PARTIAL in PARTIALLY = False
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_physics_retry_no_substring_match(self, mock_state, mock_result):
        """RETRY should NOT match as substring."""
        user_input = {"q1": "RETRYSTYLE approach"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: RETRY in RETRYSTYLE = False
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert mock_result["physics_failure_count"] == 2  # Not reset

    def test_execution_retry_matches_whole_word(self, mock_state, mock_result):
        """RETRY should match when it's a whole word."""
        user_input = {"q1": "Please RETRY the operation"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY as whole word = matches
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_physics_accept_matches_whole_word(self, mock_state, mock_result):
        """ACCEPT should match when it's a whole word."""
        user_input = {"q1": "I will ACCEPT the results"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # ACCEPT as whole word = matches
        assert mock_result["supervisor_verdict"] == "retry_analyze"
        mock_update.assert_called_once()


class TestSkipDoesNotResetCounters:
    """Verify SKIP action does not reset failure counters."""

    def test_execution_skip_preserves_failure_count(self, mock_state, mock_result):
        """SKIP should not reset execution_failure_count."""
        user_input = {"q1": "SKIP"}
        mock_result["execution_failure_count"] = 5
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Count should NOT be reset
        assert mock_result["execution_failure_count"] == 5
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_physics_skip_preserves_failure_count(self, mock_state, mock_result):
        """SKIP should not reset physics_failure_count."""
        user_input = {"q1": "SKIP"}
        mock_result["physics_failure_count"] = 4
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Count should NOT be reset
        assert mock_result["physics_failure_count"] == 4
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_physics_accept_preserves_failure_count(self, mock_state, mock_result):
        """ACCEPT should not reset physics_failure_count."""
        user_input = {"q1": "ACCEPT"}
        mock_result["physics_failure_count"] = 3
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Count should NOT be reset - only RETRY resets
        assert mock_result["physics_failure_count"] == 3
        assert mock_result["supervisor_verdict"] == "retry_analyze"


class TestStopActionBehavior:
    """Tests for STOP action specific behavior."""

    def test_execution_stop_preserves_failure_count(self, mock_state, mock_result):
        """STOP should not reset execution_failure_count."""
        user_input = {"q1": "STOP"}
        mock_result["execution_failure_count"] = 5
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Count should be preserved
        assert mock_result["execution_failure_count"] == 5
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_physics_stop_preserves_failure_count(self, mock_state, mock_result):
        """STOP should not reset physics_failure_count."""
        user_input = {"q1": "STOP"}
        mock_result["physics_failure_count"] = 4
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Count should be preserved
        assert mock_result["physics_failure_count"] == 4
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_execution_stop_does_not_update_progress(self, mock_state, mock_result):
        """STOP should not call any progress update functions."""
        user_input = {"q1": "STOP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_wrapper:
                trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        mock_update.assert_not_called()
        mock_wrapper.assert_not_called()

    def test_physics_stop_does_not_update_progress(self, mock_state, mock_result):
        """STOP should not call any progress update functions."""
        user_input = {"q1": "STOP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_wrapper:
                trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        mock_update.assert_not_called()
        mock_wrapper.assert_not_called()


class TestClarificationQuestionContent:
    """Tests verifying the content of clarification questions."""

    def test_execution_clarification_mentions_retry(self, mock_state, mock_result):
        """Unclear response should ask about RETRY_WITH_GUIDANCE."""
        user_input = {"q1": "I'm not sure what to do"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "RETRY_WITH_GUIDANCE" in question
        assert "SKIP_STAGE" in question
        assert "STOP" in question

    def test_physics_clarification_mentions_all_options(self, mock_state, mock_result):
        """Unclear response should mention all physics handler options."""
        user_input = {"q1": "I'm not sure what to do"}
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        question = mock_result["pending_user_questions"][0]
        assert "RETRY_WITH_GUIDANCE" in question
        assert "ACCEPT_PARTIAL" in question
        assert "SKIP_STAGE" in question
        assert "STOP" in question


class TestFeedbackFormat:
    """Tests for the format and content of supervisor feedback."""

    def test_execution_retry_feedback_includes_raw_response(self, mock_state, mock_result):
        """RETRY feedback should include the raw user response."""
        user_input = {"q1": "RETRY with more memory allocation please"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        feedback = mock_result["supervisor_feedback"]
        assert feedback.startswith("User guidance:")
        # Should include the raw response
        assert "RETRY with more memory allocation please" in feedback

    def test_physics_retry_feedback_includes_raw_response(self, mock_state, mock_result):
        """RETRY feedback should include the raw user response."""
        user_input = {"q1": "RETRY with adjusted parameters"}
        mock_result["physics_failure_count"] = 1
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        feedback = mock_result["supervisor_feedback"]
        assert feedback.startswith("User guidance:")
        assert "RETRY with adjusted parameters" in feedback

    def test_execution_retry_empty_guidance_produces_valid_feedback(self, mock_state, mock_result):
        """RETRY without additional text should still produce valid feedback."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        feedback = mock_result["supervisor_feedback"]
        assert feedback.startswith("User guidance:")
        assert "RETRY" in feedback

    def test_execution_guidance_keyword_feedback(self, mock_state, mock_result):
        """GUIDANCE keyword should produce proper feedback."""
        user_input = {"q1": "GUIDANCE: use less memory"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        feedback = mock_result["supervisor_feedback"]
        assert feedback.startswith("User guidance:")
        assert "use less memory" in feedback


class TestMultipleUserResponses:
    """Tests for handling multiple user responses in the dict."""

    def test_execution_uses_last_response_value(self, mock_state, mock_result):
        """Handler should use the last value in user_responses dict."""
        # Dict ordering is preserved in Python 3.7+
        user_input = {"q1": "STOP", "q2": "SKIP", "q3": "RETRY"}
        mock_result["execution_failure_count"] = 3
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # The last response "RETRY" wins for keyword checking (response_text is uppercased concatenation? No, it's just the last value)
        # Actually looking at parse_user_response, it returns the last value
        # But keyword checking is done on response_text which contains all keywords from last response
        # Wait, let me check again... parse_user_response returns str(last_response).strip().upper()
        # So if last is "RETRY", response_text = "RETRY"
        # But the handler checks "RETRY" in response_text which would match
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_physics_uses_last_response_for_raw(self, mock_state, mock_result):
        """Raw response should be from the last response."""
        user_input = {"q1": "First response", "q2": "RETRY with better params"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Raw response should be from last item
        assert "RETRY with better params" in mock_result["supervisor_feedback"]
        assert "First response" not in mock_result["supervisor_feedback"]


class TestProgressUpdateParameters:
    """Tests verifying the exact parameters passed to progress update functions."""

    def test_execution_skip_calls_update_with_correct_params(self, mock_state, mock_result):
        """SKIP should call _update_progress_with_error_handling with exact parameters."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "test_stage_123")
        
        mock_update.assert_called_once_with(
            mock_state,
            mock_result,
            "test_stage_123",
            "blocked",
            summary="Skipped by user due to execution failures"
        )

    def test_physics_skip_calls_update_with_correct_params(self, mock_state, mock_result):
        """SKIP should call _update_progress_with_error_handling with exact parameters."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "physics_stage_456")
        
        mock_update.assert_called_once_with(
            mock_state,
            mock_result,
            "physics_stage_456",
            "blocked",
            summary="Skipped by user due to physics check failures"
        )

    def test_physics_accept_calls_update_with_completed_partial(self, mock_state, mock_result):
        """ACCEPT should call _update_progress_with_error_handling with completed_partial."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "accept_stage")
        
        mock_update.assert_called_once_with(
            mock_state,
            mock_result,
            "accept_stage",
            "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )

    def test_physics_partial_calls_update_with_completed_partial(self, mock_state, mock_result):
        """PARTIAL should call _update_progress_with_error_handling with completed_partial."""
        user_input = {"q1": "PARTIAL"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "partial_stage")
        
        mock_update.assert_called_once_with(
            mock_state,
            mock_result,
            "partial_stage",
            "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )


class TestIntegrationViaSupervisorNode:
    """Integration tests through supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_supervisor_clears_ask_user_trigger_after_handling(self, mock_context):
        """supervisor_node should clear ask_user_trigger after handling."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "RETRY"},
            "execution_failure_count": 3,
        }
        
        result = supervisor_node(state)
        
        # Trigger should be cleared (set to None)
        assert result["ask_user_trigger"] is None
        assert result["execution_failure_count"] == 0

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_supervisor_logs_user_interaction(self, mock_context):
        """supervisor_node should log user interactions to progress."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should have progress with user_interactions
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "physics_failure_limit"
        assert "STOP" in interaction["user_response"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_supervisor_preserves_archive_errors(self, mock_context):
        """supervisor_node should handle archive_errors state."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "RETRY"},
            "execution_failure_count": 2,
            "archive_errors": [],  # Empty list should be preserved
        }
        
        result = supervisor_node(state)
        
        # archive_errors should be in result
        assert "archive_errors" in result
        assert result["archive_errors"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_supervisor_sets_workflow_phase(self, mock_context):
        """supervisor_node should set workflow_phase."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "ACCEPT"},
            "current_stage_id": "stage1",
        }
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            result = supervisor_node(state)
        
        assert result["workflow_phase"] == "supervision"


class TestSpecialCharactersInResponses:
    """Tests for handling special characters in user responses."""

    def test_execution_response_with_newlines(self, mock_state, mock_result):
        """Handler should work with newlines in response."""
        user_input = {"q1": "RETRY\nwith\nmultiline\nguidance"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "multiline" in mock_result["supervisor_feedback"]

    def test_physics_response_with_special_chars(self, mock_state, mock_result):
        """Handler should work with special characters in response."""
        user_input = {"q1": "ACCEPT! @#$%^&*() special chars"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "retry_analyze"

    def test_execution_response_with_unicode(self, mock_state, mock_result):
        """Handler should work with unicode characters."""
        user_input = {"q1": "RETRY avec des caractères français: é, è, ê, ë"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert "français" in mock_result["supervisor_feedback"]


class TestResultDictMutation:
    """Tests verifying that result dict is mutated correctly."""

    def test_execution_handler_only_sets_expected_keys_on_retry(self, mock_state, mock_result):
        """RETRY should only set specific keys."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 3
        mock_result["unrelated_key"] = "should_remain"
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should have set these keys
        assert "execution_failure_count" in mock_result
        assert "supervisor_feedback" in mock_result
        assert "supervisor_verdict" in mock_result
        # Should NOT have touched unrelated keys
        assert mock_result["unrelated_key"] == "should_remain"

    def test_physics_handler_only_sets_expected_keys_on_stop(self, mock_state, mock_result):
        """STOP should only set verdict and should_stop."""
        user_input = {"q1": "STOP"}
        mock_result["unrelated_key"] = "should_remain"
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True
        assert mock_result["unrelated_key"] == "should_remain"

    def test_ask_user_sets_pending_questions_as_list(self, mock_state, mock_result):
        """Unclear response should set pending_user_questions as a list."""
        user_input = {"q1": "unclear response"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert isinstance(mock_result["pending_user_questions"], list)
        assert len(mock_result["pending_user_questions"]) > 0
        # Each question should be a string
        for q in mock_result["pending_user_questions"]:
            assert isinstance(q, str)


class TestErrorHandlingConsistency:
    """
    Tests for error handling consistency between handlers.
    
    All handlers now consistently use _update_progress_with_error_handling
    for progress updates, ensuring exceptions don't crash the workflow.
    """

    def test_execution_skip_uses_error_handling_wrapper(self, mock_state, mock_result):
        """Verify execution SKIP uses error handling wrapper."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_direct:
            with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_wrapper:
                trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Error handling wrapper is used, not direct call
        mock_wrapper.assert_called_once()
        mock_direct.assert_not_called()

    def test_physics_skip_uses_error_handling_wrapper(self, mock_state, mock_result):
        """Verify physics SKIP uses error handling wrapper."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_direct:
            with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_wrapper:
                trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Error handling wrapper is used, not direct call
        mock_wrapper.assert_called_once()
        mock_direct.assert_not_called()

    def test_physics_accept_uses_error_handling_wrapper(self, mock_state, mock_result):
        """Verify physics ACCEPT uses error handling wrapper."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_direct:
            with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_wrapper:
                trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Wrapper is used, not direct call
        mock_wrapper.assert_called_once()
        mock_direct.assert_not_called()


class TestBoundaryConditions:
    """Tests for boundary conditions and extreme inputs."""

    def test_execution_very_long_response(self, mock_state, mock_result):
        """Handler should work with very long responses."""
        long_guidance = "RETRY " + "x" * 10000
        user_input = {"q1": long_guidance}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        # Full response should be in feedback
        assert len(mock_result["supervisor_feedback"]) > 10000

    def test_physics_very_long_response(self, mock_state, mock_result):
        """Handler should work with very long responses."""
        long_response = "RETRY " + "guidance " * 5000
        user_input = {"q1": long_response}
        mock_result["physics_failure_count"] = 1
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_execution_high_failure_count(self, mock_state, mock_result):
        """Handler should work with very high failure counts."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 999999
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0

    def test_physics_negative_failure_count(self, mock_state, mock_result):
        """Handler should work even if count is negative (shouldn't happen but handle gracefully)."""
        user_input = {"q1": "RETRY"}
        mock_result["physics_failure_count"] = -5
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should reset to 0 regardless
        assert mock_result["physics_failure_count"] == 0

    def test_execution_empty_stage_id_string(self, mock_state, mock_result):
        """Handler should treat empty string stage_id as falsy (no progress update)."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "")
        
        # Empty string is falsy in Python, so `if current_stage_id:` returns False
        # This is correct behavior - no update for empty stage_id
        mock_update.assert_not_called()
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_physics_stage_id_with_special_chars(self, mock_state, mock_result):
        """Handler should work with stage IDs containing special characters."""
        user_input = {"q1": "SKIP"}
        special_stage_id = "stage_123_with-dashes.and.dots/and/slashes"
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, special_stage_id)
        
        mock_update.assert_called_once_with(
            mock_state,
            mock_result,
            special_stage_id,
            "blocked",
            summary="Skipped by user due to physics check failures"
        )


class TestDictKeyOrdering:
    """Tests for dict key ordering in user_responses."""

    def test_execution_response_order_matters(self, mock_state, mock_result):
        """Last value in ordered dict should be used for response."""
        # Create dict with specific ordering
        user_input = {}
        user_input["first"] = "STOP"
        user_input["second"] = "SKIP" 
        user_input["third"] = "RETRY with this guidance"
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Last response "RETRY with this guidance" should be used
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        assert "RETRY with this guidance" in mock_result["supervisor_feedback"]

    def test_physics_single_response_key_doesnt_matter(self, mock_state, mock_result):
        """With single response, key name doesn't affect behavior."""
        user_input = {"any_key_name_here": "RETRY"}
        mock_result["physics_failure_count"] = 1
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["physics_failure_count"] == 0


class TestKeywordOnlyResponses:
    """Tests for responses that are exactly the keyword."""

    def test_execution_exact_retry_keyword(self, mock_state, mock_result):
        """Exact 'RETRY' keyword should work."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        # Feedback should still be set
        assert "supervisor_feedback" in mock_result
        assert mock_result["supervisor_feedback"].startswith("User guidance:")

    def test_execution_exact_skip_keyword(self, mock_state, mock_result):
        """Exact 'SKIP' keyword should work."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # No supervisor_feedback should be set for SKIP
        assert "supervisor_feedback" not in mock_result

    def test_execution_exact_stop_keyword(self, mock_state, mock_result):
        """Exact 'STOP' keyword should work."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_physics_exact_accept_keyword(self, mock_state, mock_result):
        """Exact 'ACCEPT' keyword should work."""
        user_input = {"q1": "ACCEPT"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "retry_analyze"

    def test_physics_exact_partial_keyword(self, mock_state, mock_result):
        """Exact 'PARTIAL' keyword should work."""
        user_input = {"q1": "PARTIAL"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling"):
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "retry_analyze"


class TestStateFieldTypes:
    """Tests for handling unusual state field types."""

    def test_execution_count_as_float_in_result(self, mock_state, mock_result):
        """Handler should work even if count is a float."""
        user_input = {"q1": "RETRY"}
        mock_result["execution_failure_count"] = 3.0  # Float instead of int
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should be reset to int 0
        assert mock_result["execution_failure_count"] == 0
        assert isinstance(mock_result["execution_failure_count"], int)

    def test_physics_with_none_values_in_user_responses(self, mock_state, mock_result):
        """Handler should handle None values in user_responses dict."""
        user_input = {"q1": None}  # None value
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should ask for clarification since "None" doesn't contain keywords
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_execution_with_list_value_in_user_responses(self, mock_state, mock_result):
        """Handler should handle list value in user_responses (converted to string)."""
        user_input = {"q1": ["RETRY", "something"]}
        mock_result["execution_failure_count"] = 1
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # List will be converted to string "['RETRY', 'something']" which contains "RETRY"
        assert mock_result["execution_failure_count"] == 0


class TestConcurrentKeywords:
    """Tests for responses containing multiple action keywords."""

    def test_execution_all_keywords_present(self, mock_state, mock_result):
        """When all keywords present, first in precedence order wins."""
        user_input = {"q1": "RETRY GUIDANCE SKIP STOP"}
        mock_result["execution_failure_count"] = 2
        
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY/GUIDANCE checked first
        assert mock_result["execution_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_physics_all_keywords_present(self, mock_state, mock_result):
        """When all keywords present, first in precedence order wins."""
        user_input = {"q1": "RETRY ACCEPT PARTIAL SKIP STOP"}
        mock_result["physics_failure_count"] = 2
        
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # RETRY checked first
        assert mock_result["physics_failure_count"] == 0
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    def test_physics_accept_and_partial_both_trigger_same_path(self, mock_state, mock_result):
        """ACCEPT and PARTIAL both go to same code path."""
        user_input = {"q1": "ACCEPT PARTIAL"}
        
        with patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling") as mock_update:
            trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        
        # Both matched, but only one call
        mock_update.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT TESTS: ask_user_trigger and pending_user_questions Contract
# ═══════════════════════════════════════════════════════════════════════════════
#
# These tests verify that when any path leads to ask_user node, the required
# state fields (ask_user_trigger and pending_user_questions) are properly set.
#
# The contract is:
# 1. ask_user_trigger must be a non-empty string matching a valid trigger
# 2. pending_user_questions must be a non-empty list of strings
#
# ═══════════════════════════════════════════════════════════════════════════════


from src.agents.execution import execution_validator_node, physics_sanity_node
from src.agents.analysis import comparison_validator_node
from src.agents.code import code_reviewer_node
from src.agents.design import design_reviewer_node
from src.agents.planning import plan_reviewer_node
from src.routing import (
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
    route_after_code_review,
    route_after_design_review,
    route_after_plan_review,
)
from schemas.state import (
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_REPLANS,
)
from src.agents.constants import AskUserTrigger


# Get all valid trigger values for strict validation
VALID_TRIGGERS = {t.value for t in AskUserTrigger}


def assert_ask_user_contract(result: dict, expected_trigger: str, context: str) -> None:
    """
    Helper to assert the ask_user contract is satisfied.
    
    This enforces:
    1. ask_user_trigger is a non-empty string matching a valid trigger
    2. pending_user_questions is a non-empty list of non-empty strings
    """
    # Strict assertion for ask_user_trigger
    trigger = result.get("ask_user_trigger")
    assert trigger is not None, (
        f"{context}: ask_user_trigger is None - CONTRACT VIOLATED. "
        f"When routing to ask_user, trigger MUST be set."
    )
    assert isinstance(trigger, str), (
        f"{context}: ask_user_trigger must be str, got {type(trigger).__name__}: {trigger}"
    )
    assert len(trigger) > 0, (
        f"{context}: ask_user_trigger is empty string - CONTRACT VIOLATED"
    )
    assert trigger in VALID_TRIGGERS, (
        f"{context}: ask_user_trigger='{trigger}' is not a valid trigger. "
        f"Valid triggers: {sorted(VALID_TRIGGERS)}"
    )
    assert trigger == expected_trigger, (
        f"{context}: Expected trigger '{expected_trigger}', got '{trigger}'"
    )
    
    # Strict assertion for pending_user_questions
    questions = result.get("pending_user_questions")
    assert questions is not None, (
        f"{context}: pending_user_questions is None - CONTRACT VIOLATED. "
        f"When routing to ask_user, questions MUST be set."
    )
    assert isinstance(questions, list), (
        f"{context}: pending_user_questions must be list, got {type(questions).__name__}"
    )
    assert len(questions) > 0, (
        f"{context}: pending_user_questions is empty - CONTRACT VIOLATED. "
        f"User must be given questions when asked for input."
    )
    for i, q in enumerate(questions):
        assert isinstance(q, str), (
            f"{context}: Question {i} must be str, got {type(q).__name__}: {q}"
        )
        assert len(q) > 0, (
            f"{context}: Question {i} is empty string - CONTRACT VIOLATED"
        )


class TestLimitEscalationContract:
    """
    Contract tests: When limits are reached, ask_user_trigger and pending_user_questions MUST be set.
    
    These tests verify that each validator node properly sets the required state fields
    when a limit is reached, BEFORE the router routes to ask_user.
    """

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    def test_execution_limit_sets_trigger_and_questions(
        self, mock_context, mock_prompt, mock_call
    ):
        """
        execution_validator_node MUST set ask_user_trigger and pending_user_questions
        when execution_failure_count reaches limit.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "summary": "Execution failed",
            "error_analysis": "Memory error"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "execution_failure_count": MAX_EXECUTION_FAILURES - 1,  # Will increment to limit
            "execution_output": "Error: out of memory",
            "execution_result": {"success": False, "error": "OOM"},
            "current_simulation_code": "import meep",
        }
        
        result = execution_validator_node(state)
        
        # Contract assertion
        assert_ask_user_contract(
            result,
            expected_trigger="execution_failure_limit",
            context="execution_validator_node at limit"
        )

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    def test_physics_limit_sets_trigger_and_questions(
        self, mock_context, mock_prompt, mock_call
    ):
        """
        physics_sanity_node MUST set ask_user_trigger and pending_user_questions
        when physics_failure_count reaches limit.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "issues": ["Impossible resonance frequency"],
            "summary": "Physics check failed"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "physics_failure_count": MAX_PHYSICS_FAILURES - 1,  # Will increment to limit
            "analysis_summary": {"peak_wavelength": 500},
            "design_description": "Test design",
        }
        
        result = physics_sanity_node(state)
        
        # Contract assertion
        assert_ask_user_contract(
            result,
            expected_trigger="physics_failure_limit",
            context="physics_sanity_node at limit"
        )

    def test_analysis_limit_sets_trigger_and_questions(self):
        """
        comparison_validator_node MUST set ask_user_trigger and pending_user_questions
        when analysis_revision_count reaches limit.
        
        NOTE: comparison_validator_node does NOT use LLM - it computes verdicts
        programmatically based on state analysis. To trigger needs_revision,
        we set up state where expected targets exist but comparisons are missing.
        """
        # Set up state that triggers needs_revision verdict:
        # - has expected targets from plan
        # - but no comparisons produced yet
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "analysis_revision_count": MAX_ANALYSIS_REVISIONS - 1,  # Will increment to limit
            "plan": {
                "stages": [{
                    "stage_id": "stage_0",
                    "targets": ["fig1"],  # Expected targets
                    "target_details": [{"figure_id": "fig1"}],
                }]
            },
            "progress": {
                "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
            },
            "stage_outputs": {},  # No outputs yet - this causes needs_revision
        }
        
        result = comparison_validator_node(state)
        
        # Verify we got needs_revision verdict
        assert result.get("comparison_verdict") == "needs_revision", (
            f"Expected needs_revision verdict, got: {result.get('comparison_verdict')}"
        )
        
        # Contract assertion
        assert_ask_user_contract(
            result,
            expected_trigger="analysis_limit",
            context="comparison_validator_node at limit"
        )

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    def test_code_review_limit_sets_trigger_and_questions(
        self, mock_context, mock_prompt, mock_call
    ):
        """
        code_reviewer_node MUST set ask_user_trigger and pending_user_questions
        when code_revision_count reaches limit.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": ["Missing imports"],
            "feedback": "Add numpy import"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "code_revision_count": MAX_CODE_REVISIONS - 1,  # Will increment to limit
            "current_simulation_code": "print('hello')",
            "design_description": "Test design",
        }
        
        result = code_reviewer_node(state)
        
        # Contract assertion
        assert_ask_user_contract(
            result,
            expected_trigger="code_review_limit",
            context="code_reviewer_node at limit"
        )

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    def test_design_review_limit_sets_trigger_and_questions(
        self, mock_context, mock_prompt, mock_call
    ):
        """
        design_reviewer_node MUST set ask_user_trigger and pending_user_questions
        when design_revision_count reaches limit.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": ["Missing boundary conditions"],
            "feedback": "Add PML boundaries"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "design_revision_count": MAX_DESIGN_REVISIONS - 1,  # Will increment to limit
            "design_description": "Basic design",
        }
        
        result = design_reviewer_node(state)
        
        # Contract assertion
        assert_ask_user_contract(
            result,
            expected_trigger="design_review_limit",
            context="design_reviewer_node at limit"
        )

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_replan_limit_sets_trigger_and_questions(
        self, mock_context, mock_prompt, mock_call
    ):
        """
        plan_reviewer_node MUST set ask_user_trigger and pending_user_questions
        when replan_count reaches limit.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": ["Missing validation stage"],
            "feedback": "Add material validation"
        }
        
        state = {
            "paper_text": "test paper",
            "replan_count": MAX_REPLANS - 1,  # Will increment to limit
            "plan": {"stages": []},
        }
        
        result = plan_reviewer_node(state)
        
        # Contract assertion
        assert_ask_user_contract(
            result,
            expected_trigger="replan_limit",
            context="plan_reviewer_node at limit"
        )


class TestErrorEscalationContract:
    """
    Contract tests: Error paths to ask_user are handled by the safety net.
    
    When routers return ask_user due to errors (None verdict, invalid type, unknown verdict),
    the router itself does NOT modify state (routers should be pure functions).
    Instead, the ask_user_node's safety net detects the missing trigger and sets
    a fallback "unknown_escalation" trigger.
    
    These tests verify:
    1. Routers correctly route to ask_user for error cases
    2. Routers do NOT modify state (purity)
    3. The ask_user_node safety net handles missing triggers correctly
    """

    @patch("src.routing.save_checkpoint")
    def test_none_execution_verdict_routes_to_ask_user(self, mock_checkpoint):
        """
        When execution_verdict is None, router should route to ask_user.
        Router should NOT modify state (routers are pure functions).
        """
        state = {
            "execution_verdict": None,  # None verdict
            "execution_failure_count": 0,
            "current_stage_id": "stage_0",
        }
        
        route = route_after_execution_check(state)
        
        # Router correctly routes to ask_user
        assert route == "ask_user", "None verdict should route to ask_user"
        
        # Router should NOT modify state - it's a pure function
        # The safety net in ask_user_node will handle missing trigger
        assert state.get("ask_user_trigger") is None, (
            "Router should not modify state - trigger handling is done by ask_user_node safety net"
        )

    @patch("src.routing.save_checkpoint")
    def test_invalid_type_verdict_routes_to_ask_user(self, mock_checkpoint):
        """
        When execution_verdict is invalid type (e.g., int), router should route to ask_user.
        Router should NOT modify state.
        """
        state = {
            "execution_verdict": 123,  # Invalid type - should be string
            "execution_failure_count": 0,
            "current_stage_id": "stage_0",
        }
        
        route = route_after_execution_check(state)
        
        assert route == "ask_user", "Invalid verdict type should route to ask_user"
        
        # Router should NOT modify state
        assert state.get("ask_user_trigger") is None, (
            "Router should not modify state - trigger handling is done by ask_user_node safety net"
        )

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_routes_to_ask_user(self, mock_checkpoint):
        """
        When verdict is unknown string, router should route to ask_user.
        Router should NOT modify state.
        """
        state = {
            "execution_verdict": "garbage_verdict_that_doesnt_exist",
            "execution_failure_count": 0,
            "current_stage_id": "stage_0",
        }
        
        route = route_after_execution_check(state)
        
        assert route == "ask_user", "Unknown verdict should route to ask_user"
        
        # Router should NOT modify state
        assert state.get("ask_user_trigger") is None, (
            "Router should not modify state - trigger handling is done by ask_user_node safety net"
        )

    @patch("src.routing.save_checkpoint")
    def test_none_comparison_verdict_routes_to_ask_user(self, mock_checkpoint):
        """
        When comparison_verdict is None, router should route to ask_user.
        Router should NOT modify state.
        """
        state = {
            "comparison_verdict": None,
            "analysis_revision_count": 0,
            "current_stage_id": "stage_0",
        }
        
        route = route_after_comparison_check(state)
        
        assert route == "ask_user", "None comparison_verdict should route to ask_user"
        
        # Router should NOT modify state
        assert state.get("ask_user_trigger") is None, (
            "Router should not modify state - trigger handling is done by ask_user_node safety net"
        )

    @patch("src.routing.save_checkpoint")
    def test_none_physics_verdict_routes_to_ask_user(self, mock_checkpoint):
        """
        When physics_verdict is None, router should route to ask_user.
        Router should NOT modify state.
        """
        state = {
            "physics_verdict": None,
            "physics_failure_count": 0,
            "current_stage_id": "stage_0",
        }
        
        route = route_after_physics_check(state)
        
        assert route == "ask_user", "None physics_verdict should route to ask_user"
        
        # Router should NOT modify state
        assert state.get("ask_user_trigger") is None, (
            "Router should not modify state - trigger handling is done by ask_user_node safety net"
        )


class TestSupervisorAskUserContract:
    """
    Contract tests: When supervisor verdict is ask_user, questions MUST be set.
    
    When the supervisor decides to ask the user (either via LLM returning ask_user
    verdict, or via a trigger handler asking for clarification), the result MUST
    contain pending_user_questions.
    """

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_supervisor_clarification_has_questions(self, mock_context):
        """
        When a trigger handler asks for clarification (verdict=ask_user),
        pending_user_questions MUST be set with meaningful questions.
        """
        mock_context.return_value = None
        
        # Simulate unclear user response that triggers clarification
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "I'm not sure what to do"},
        }
        
        result = supervisor_node(state)
        
        # If supervisor asks for clarification, questions must be set
        if result.get("supervisor_verdict") == "ask_user":
            questions = result.get("pending_user_questions")
            assert questions is not None, (
                "CONTRACT VIOLATION: supervisor_verdict is 'ask_user' but "
                "pending_user_questions is None"
            )
            assert isinstance(questions, list), (
                f"pending_user_questions must be list, got {type(questions).__name__}"
            )
            assert len(questions) > 0, (
                "CONTRACT VIOLATION: supervisor_verdict is 'ask_user' but "
                "pending_user_questions is empty"
            )
            for i, q in enumerate(questions):
                assert isinstance(q, str) and len(q) > 0, (
                    f"Question {i} must be non-empty string, got: {q!r}"
                )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_all_trigger_handlers_set_questions_on_clarification(self, mock_context):
        """
        For all registered trigger handlers, when they request clarification,
        they MUST set pending_user_questions.
        """
        mock_context.return_value = None
        
        # Test triggers that might ask for clarification
        test_cases = [
            ("execution_failure_limit", {"Q": "maybe"}),
            ("physics_failure_limit", {"Q": "hmm"}),
            ("code_review_limit", {"Q": "not sure"}),
            ("design_review_limit", {"Q": "dunno"}),
            ("replan_limit", {"Q": "confused"}),
            ("analysis_limit", {"Q": "unclear"}),
        ]
        
        for trigger, response in test_cases:
            state = {
                "ask_user_trigger": trigger,
                "user_responses": response,
                "current_stage_id": "stage_0",
            }
            
            result = supervisor_node(state)
            
            # If this trigger handler asks for clarification, verify contract
            if result.get("supervisor_verdict") == "ask_user":
                questions = result.get("pending_user_questions")
                assert questions is not None and len(questions) > 0, (
                    f"CONTRACT VIOLATION: Handler for '{trigger}' returned verdict='ask_user' "
                    f"but pending_user_questions is not set. Result: {result}"
                )


class TestAskUserPathIntegrationContract:
    """
    Integration contract: For any path to ask_user, merged state must have required fields.
    
    These tests simulate the full flow:
    1. Create state with conditions that will trigger ask_user routing
    2. Call the node that produces the verdict
    3. Merge result into state (as LangGraph would)
    4. Call the router
    5. If router returns "ask_user", verify the merged state has trigger and questions
    """

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.routing.save_checkpoint")
    def test_execution_limit_to_ask_user_full_path(
        self, mock_checkpoint, mock_context, mock_prompt, mock_call
    ):
        """
        Full integration: execution_validator_node at limit -> router -> verify contract.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "summary": "Execution failed",
            "error_analysis": "Memory error"
        }
        
        # Initial state at limit
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "execution_failure_count": MAX_EXECUTION_FAILURES - 1,
            "execution_output": "Error: out of memory",
            "execution_result": {"success": False, "error": "OOM"},
            "current_simulation_code": "import meep",
        }
        
        # Step 1: Call the node
        result = execution_validator_node(state)
        
        # Step 2: Merge result into state (as LangGraph would)
        merged_state = {**state, **result}
        
        # Step 3: Call router
        route = route_after_execution_check(merged_state)
        
        # Step 4: Verify contract if routing to ask_user
        assert route == "ask_user", (
            f"Expected ask_user at limit, got {route}. "
            f"Count after node: {merged_state.get('execution_failure_count')}"
        )
        
        # Step 5: Full contract assertion on merged state
        assert_ask_user_contract(
            merged_state,
            expected_trigger="execution_failure_limit",
            context="INTEGRATION: execution_validator -> route_after_execution_check"
        )

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.routing.save_checkpoint")
    def test_physics_limit_to_ask_user_full_path(
        self, mock_checkpoint, mock_context, mock_prompt, mock_call
    ):
        """
        Full integration: physics_sanity_node at limit -> router -> verify contract.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "issues": ["Impossible value"],
            "summary": "Physics check failed"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "physics_failure_count": MAX_PHYSICS_FAILURES - 1,
            "analysis_summary": {"peak_wavelength": 500},
            "design_description": "Test design",
        }
        
        result = physics_sanity_node(state)
        merged_state = {**state, **result}
        route = route_after_physics_check(merged_state)
        
        assert route == "ask_user", f"Expected ask_user at limit, got {route}"
        
        assert_ask_user_contract(
            merged_state,
            expected_trigger="physics_failure_limit",
            context="INTEGRATION: physics_sanity -> route_after_physics_check"
        )

    @patch("src.routing.save_checkpoint")
    def test_analysis_limit_to_ask_user_full_path(self, mock_checkpoint):
        """
        Full integration: comparison_validator_node at limit -> router -> verify contract.
        
        NOTE: comparison_validator_node computes verdicts programmatically (no LLM).
        We set up state that triggers needs_revision by having expected targets
        but no comparisons produced.
        """
        # Set up state that triggers needs_revision verdict
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "analysis_revision_count": MAX_ANALYSIS_REVISIONS - 1,
            "plan": {
                "stages": [{
                    "stage_id": "stage_0",
                    "targets": ["fig1"],
                    "target_details": [{"figure_id": "fig1"}],
                }]
            },
            "progress": {
                "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
            },
            "stage_outputs": {},  # No outputs triggers needs_revision
        }
        
        result = comparison_validator_node(state)
        merged_state = {**state, **result}
        
        # Verify we got needs_revision
        assert merged_state.get("comparison_verdict") == "needs_revision", (
            f"Expected needs_revision, got: {merged_state.get('comparison_verdict')}"
        )
        
        route = route_after_comparison_check(merged_state)
        
        assert route == "ask_user", f"Expected ask_user at limit, got {route}"
        
        assert_ask_user_contract(
            merged_state,
            expected_trigger="analysis_limit",
            context="INTEGRATION: comparison_validator -> route_after_comparison_check"
        )

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.routing.save_checkpoint")
    def test_code_review_limit_to_ask_user_full_path(
        self, mock_checkpoint, mock_context, mock_prompt, mock_call
    ):
        """
        Full integration: code_reviewer_node at limit -> router -> verify contract.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": ["Bug found"],
            "feedback": "Fix bug"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "code_revision_count": MAX_CODE_REVISIONS - 1,
            "current_simulation_code": "print('hello')",
            "design_description": "Test design",
        }
        
        result = code_reviewer_node(state)
        merged_state = {**state, **result}
        route = route_after_code_review(merged_state)
        
        assert route == "ask_user", f"Expected ask_user at limit, got {route}"
        
        assert_ask_user_contract(
            merged_state,
            expected_trigger="code_review_limit",
            context="INTEGRATION: code_reviewer -> route_after_code_review"
        )

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.routing.save_checkpoint")
    def test_design_review_limit_to_ask_user_full_path(
        self, mock_checkpoint, mock_context, mock_prompt, mock_call
    ):
        """
        Full integration: design_reviewer_node at limit -> router -> verify contract.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": ["Missing PML"],
            "feedback": "Add boundaries"
        }
        
        state = {
            "paper_text": "test paper",
            "current_stage_id": "stage_0",
            "design_revision_count": MAX_DESIGN_REVISIONS - 1,
            "design_description": "Basic design",
        }
        
        result = design_reviewer_node(state)
        merged_state = {**state, **result}
        route = route_after_design_review(merged_state)
        
        assert route == "ask_user", f"Expected ask_user at limit, got {route}"
        
        assert_ask_user_contract(
            merged_state,
            expected_trigger="design_review_limit",
            context="INTEGRATION: design_reviewer -> route_after_design_review"
        )

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.routing.save_checkpoint")
    def test_replan_limit_to_ask_user_full_path(
        self, mock_checkpoint, mock_context, mock_prompt, mock_call
    ):
        """
        Full integration: plan_reviewer_node at limit -> router -> verify contract.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "test prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": ["Missing stage"],
            "feedback": "Add validation"
        }
        
        state = {
            "paper_text": "test paper",
            "replan_count": MAX_REPLANS - 1,
            "plan": {"stages": []},
        }
        
        result = plan_reviewer_node(state)
        merged_state = {**state, **result}
        route = route_after_plan_review(merged_state)
        
        assert route == "ask_user", f"Expected ask_user at limit, got {route}"
        
        assert_ask_user_contract(
            merged_state,
            expected_trigger="replan_limit",
            context="INTEGRATION: plan_reviewer -> route_after_plan_review"
        )


class TestFeedbackClearingOnTransitions:
    """
    Integration tests for feedback clearing behavior.
    
    These tests verify that stale feedback is properly cleared:
    1. When code/design review approves (issue 3)
    2. When transitioning between stages (issue 4)
    
    Without these fixes, agents receive confusing feedback from previous
    iterations or stages, leading to incorrect behavior like empty code generation.
    """

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_stale_reviewer_feedback_cleared_on_code_review_approval(
        self, mock_llm, mock_prompt
    ):
        """
        Integration test: code_reviewer_node must clear reviewer_feedback on approval.
        
        Scenario: Code was rejected twice, then approved. The approval must clear
        the stale feedback so subsequent code generation doesn't receive it.
        """
        from src.agents.code import code_reviewer_node
        
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        
        # State after two rejections
        state = {
            "paper_id": "test_paper",
            "code": "import meep; # good code",
            "current_stage_id": "stage_1",
            "code_revision_count": 2,
            "reviewer_feedback": "Previous rejection: missing flux monitors",
        }
        
        result = code_reviewer_node(state)
        
        # Verify approval
        assert result["last_code_review_verdict"] == "approve"
        
        # Critical: feedback must be cleared
        assert "reviewer_feedback" in result, (
            "reviewer_feedback key must be in result to clear it from state"
        )
        assert result["reviewer_feedback"] is None, (
            f"Stale feedback must be None after approval, got: {result['reviewer_feedback']!r}"
        )
        
        # Simulate state merge (as LangGraph would do)
        merged_state = {**state, **result}
        assert merged_state["reviewer_feedback"] is None, (
            "After state merge, reviewer_feedback should be None"
        )

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_stale_reviewer_feedback_cleared_on_design_review_approval(
        self, mock_llm, mock_prompt, mock_check
    ):
        """
        Integration test: design_reviewer_node must clear reviewer_feedback on approval.
        """
        from src.agents.design import design_reviewer_node
        
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        
        state = {
            "paper_id": "test_paper",
            "paper_text": "Test paper",
            "design_description": {"geometry": "disk"},
            "current_stage_id": "stage_1",
            "design_revision_count": 1,
            "reviewer_feedback": "Previous rejection: geometry dimensions unclear",
        }
        
        result = design_reviewer_node(state)
        
        assert result["last_design_review_verdict"] == "approve"
        assert "reviewer_feedback" in result
        assert result["reviewer_feedback"] is None, (
            f"Stale feedback must be None after design approval, got: {result['reviewer_feedback']!r}"
        )

    def test_feedback_cleared_on_stage_transition(self):
        """
        Integration test: select_stage_node must clear all feedback fields on stage transition.
        
        This prevents feedback from Stage N leaking into Stage N+1.
        """
        from src.agents.stage_selection import select_stage_node
        from tests.integration.helpers.state_factories import make_plan, make_progress, make_stage
        
        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        
        state = {
            "paper_id": "test_paper",
            "paper_text": "Test paper",
            "plan": make_plan(plan_stages),
            "progress": make_progress(progress_stages),
            "current_stage_id": "stage_0",
            # Stale feedback from Stage 0
            "reviewer_feedback": "Stage 0 code review feedback",
            "physics_feedback": "Stage 0 physics feedback",
            "execution_feedback": "Stage 0 execution feedback",
            "analysis_feedback": "Stage 0 analysis feedback",
            "design_feedback": "Stage 0 design feedback",
            "comparison_feedback": "Stage 0 comparison feedback",
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage_1", "Should select Stage 1"
        
        # All feedback fields must be cleared
        feedback_fields = [
            "reviewer_feedback",
            "physics_feedback",
            "execution_feedback", 
            "analysis_feedback",
            "design_feedback",
            "comparison_feedback",
        ]
        
        for field in feedback_fields:
            assert result.get(field) is None, (
                f"{field} must be None on stage transition, got: {result.get(field)!r}"
            )

    def test_physics_feedback_not_leaked_to_next_stage(self):
        """
        Integration test: physics_feedback from Stage 0 must not appear in Stage 1.
        
        This is the specific bug that caused the empty code generation issue.
        """
        from src.agents.stage_selection import select_stage_node
        from src.agents.code import code_generator_node
        from tests.integration.helpers.state_factories import make_plan, make_progress, make_stage
        
        # Setup: Stage 0 completed with physics feedback, moving to Stage 1
        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        
        state = {
            "paper_id": "test_paper",
            "paper_text": "Test paper",
            "plan": make_plan(plan_stages),
            "progress": make_progress(progress_stages),
            "current_stage_id": "stage_0",
            "physics_feedback": "Stage 0: T+R+A != 1, energy conservation violated",
        }
        
        # Transition to Stage 1
        select_result = select_stage_node(state)
        merged_state = {**state, **select_result}
        
        # Verify physics_feedback was cleared
        assert merged_state.get("physics_feedback") is None, (
            f"physics_feedback should be None after stage transition, "
            f"got: {merged_state.get('physics_feedback')!r}"
        )
        
        # Now if code_generator_node runs, it shouldn't see any stale feedback
        # (We don't call it directly here as it requires more mocking, but the
        # state is now clean)
