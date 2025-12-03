"""Tests for various limit-based trigger handlers."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value

class TestCodeReviewLimitTrigger:
    """Tests for code_review_limit trigger handling."""

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
        
        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"

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
        mock_update.assert_called_once()

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
class TestDesignReviewLimitTrigger:
    """Tests for design_review_limit trigger handling."""

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
        assert result["supervisor_verdict"] == "ok_continue"

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
class TestExecutionFailureLimitTrigger:
    """Tests for execution_failure_limit trigger handling."""

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

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_resets_count_on_guidance(self, mock_context):
        """Should reset execution failure count on GUIDANCE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "GUIDANCE: reduce resolution"},
        }
        
        result = supervisor_node(state)
        
        assert result["execution_failure_count"] == 0

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
class TestPhysicsFailureLimitTrigger:
    """Tests for physics_failure_limit trigger handling."""

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

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
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
        mock_update.assert_called_with(
            state, "stage1", "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
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
class TestReplanLimitTrigger:
    """Tests for replan_limit trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_force_accepts_plan(self, mock_context):
        """Should force accept plan on FORCE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "FORCE accept the current plan"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "force" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_accepts_plan_on_accept(self, mock_context):
        """Should accept plan on ACCEPT."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "ACCEPT"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_retries_replan_on_guidance(self, mock_context):
        """Should retry replan with guidance on GUIDANCE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "GUIDANCE: Focus on single wavelength first"},
            "replan_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert result["replan_count"] == 0

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"



class TestHandleReviewAndFailureLimits:
    """Tests for discrete review/execution/physics limit handlers."""

    def test_handle_code_review_limit_hint(self, mock_state, mock_result):
        """Should reset counter and add feedback on hint."""
        user_input = {"q1": "PROVIDE_HINT: Use a loop instead."}

        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")

        assert result_has_value(mock_result, "code_revision_count", 0)
        assert "Use a loop" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_code_review_limit_skip(self, mock_update, mock_state, mock_result):
        """Should skip stage and continue."""
        user_input = {"q1": "SKIP"}

        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_with(mock_state, "stage1", "blocked", summary=ANY)

    def test_handle_code_review_limit_stop(self, mock_state, mock_result):
        """Should stop workflow."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_code_review_limit_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown input."""
        user_input = {"q1": "Just keep going"}
        trigger_handlers.handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_design_review_limit_hint(self, mock_state, mock_result):
        """Should reset counter and add feedback."""
        user_input = {"q1": "PROVIDE_HINT: Fix the dimensions."}
        trigger_handlers.handle_design_review_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "design_revision_count", 0)
        assert "Fix the dimensions" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_execution_failure_limit_retry(self, mock_state, mock_result):
        """Should reset counter and add guidance."""
        user_input = {"q1": "RETRY_WITH_GUIDANCE: Check memory."}
        trigger_handlers.handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "execution_failure_count", 0)
        assert "Check memory" in mock_result["supervisor_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_physics_failure_limit_accept_partial(self, mock_update, mock_state, mock_result):
        """Should mark partial success."""
        user_input = {"q1": "ACCEPT_PARTIAL results."}
        trigger_handlers.handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_with(mock_state, "stage1", "completed_partial", summary=ANY)

    def test_handle_replan_limit_force(self, mock_state, mock_result):
        """Should force accept plan."""
        user_input = {"q1": "FORCE accept"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_replan_limit_guidance(self, mock_state, mock_result):
        """Should retry with guidance."""
        user_input = {"q1": "GUIDANCE: Try this."}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "replan_count", 0)
        assert mock_result["supervisor_verdict"] == "replan_needed"

    def test_handle_replan_limit_unknown(self, mock_state, mock_result):
        """Should ask user on unknown input."""
        user_input = {"q1": "Unknown"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
