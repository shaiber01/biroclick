"""Supervisor tests for code/design/execution/physics/replan limit triggers.

This module tests the supervisor_node handling of various limit-related triggers:
- code_review_limit
- design_review_limit
- execution_failure_limit
- physics_failure_limit
- replan_limit
- llm_error
- clarification
- critical error triggers (missing_paper_text, missing_stage_id, progress_init_failed)
- planning error triggers (no_stages_available, invalid_backtrack_target, backtrack_target_not_found)
- backtrack_limit
- invalid_backtrack_decision
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.supervision import supervisor_node


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
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "reviewer_feedback" in result
        assert "User hint:" in result["reviewer_feedback"]
        assert "supervisor_feedback" in result
        assert "Retrying code generation" in result["supervisor_feedback"]
        assert result.get("should_stop") is not True  # Should not stop

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
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert result.get("should_stop") is not True  # Should not stop
        mock_update.assert_called_once_with(
            state,
            "stage1",
            "blocked",
            summary="Skipped by user due to code review issues"
        )

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
        assert result["ask_user_trigger"] is None  # Should be cleared

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
        assert result["ask_user_trigger"] == "code_review_limit"  # Preserved for clarification
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "PROVIDE_HINT" in result["pending_user_questions"][0] or "clarify" in result["pending_user_questions"][0].lower()
        assert result.get("should_stop") is not True  # Should not stop

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "code_review_limit"  # Preserved for clarification
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_current_stage_id_on_skip(self, mock_context):
        """Should handle None current_stage_id when skipping."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_hint(self, mock_context):
        """Should handle case-insensitive HINT keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "hint: lowercase test"},
            "code_revision_count": 5,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_code_revision_count(self, mock_context):
        """Should handle missing code_revision_count field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "PROVIDE_HINT: test"},
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_multiple_responses_takes_last(self, mock_context):
        """Should use the last response when multiple responses exist."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {
                "Question1": "SKIP",
                "Question2": "PROVIDE_HINT: final answer",
            },
            "code_revision_count": 3,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint
        assert "final answer" in result["reviewer_feedback"]


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
        assert result["supervisor_verdict"] == "retry_design"  # Retry with hint
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "reviewer_feedback" in result
        assert "User hint:" in result["reviewer_feedback"]
        assert result.get("should_stop") is not True  # Should not stop

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
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert result.get("should_stop") is not True  # Should not stop
        mock_update.assert_called_once_with(
            state,
            "stage1",
            "blocked",
            summary="Skipped by user due to design review issues"
        )

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
        assert result["ask_user_trigger"] is None  # Should be cleared

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "design_review_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_current_stage_id_on_skip(self, mock_context):
        """Should handle None current_stage_id when skipping."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_design_revision_count(self, mock_context):
        """Should handle missing design_revision_count field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Question": "HINT: test"},
        }

        result = supervisor_node(state)

        assert result["design_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_design"  # Retry with hint


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
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with guidance
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "supervisor_feedback" in result
        # User guidance is stored in execution_feedback, supervisor_feedback has generic message
        assert "more memory" in result["execution_feedback"]
        assert result.get("should_stop") is not True  # Should not stop

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
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with guidance
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "supervisor_feedback" in result
        # User guidance is stored in execution_feedback
        assert "reduce resolution" in result["execution_feedback"]
        assert result.get("should_stop") is not True  # Should not stop

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
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert result.get("should_stop") is not True  # Should not stop
        mock_update.assert_called_once_with(
            state,
            "stage1",
            "blocked",
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
        assert result["ask_user_trigger"] is None  # Should be cleared

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "execution_failure_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_current_stage_id_on_skip(self, mock_context):
        """Should handle None current_stage_id when skipping."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_execution_failure_count(self, mock_context):
        """Should handle missing execution_failure_count field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "RETRY"},
        }

        result = supervisor_node(state)

        assert result["execution_failure_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with guidance

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unclear_response(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Question": "maybe"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "execution_failure_limit"  # Preserved for clarification
        assert "pending_user_questions" in result


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
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with guidance
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "supervisor_feedback" in result
        assert result.get("should_stop") is not True  # Should not stop

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

        assert result["supervisor_verdict"] == "retry_analyze"  # Accept partial → analyze
        mock_update.assert_called_with(
            state,
            "stage1",
            "completed_partial",
            summary="Accepted as partial by user despite physics issues",
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

        assert result["supervisor_verdict"] == "retry_analyze"  # Accept partial → analyze
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert result.get("should_stop") is not True  # Should not stop
        mock_update.assert_called_once_with(
            state,
            "stage1",
            "completed_partial",
            summary="Accepted as partial by user despite physics issues"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "physics_failure_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_current_stage_id_on_accept(self, mock_context):
        """Should handle None current_stage_id when accepting partial."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "ACCEPT"},
            "current_stage_id": None,
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "retry_analyze"  # Accept partial → analyze
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_physics_failure_count(self, mock_context):
        """Should handle missing physics_failure_count field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "RETRY"},
        }

        result = supervisor_node(state)

        assert result["physics_failure_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with guidance

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handles_skip_without_stage_id(self, mock_update, mock_context):
        """Should handle SKIP without current_stage_id."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        # Should not call update_progress_stage_status when current_stage_id is None
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unclear_response(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Question": "not sure"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "physics_failure_limit"  # Preserved for clarification
        assert "pending_user_questions" in result


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
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert result.get("should_stop") is not True  # Should not stop

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
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "supervisor_feedback" in result
        assert "force-accepted" in result["supervisor_feedback"].lower() or "accepted" in result["supervisor_feedback"].lower()
        assert result.get("should_stop") is not True  # Should not stop

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

        assert result["supervisor_verdict"] == "replan_with_guidance"
        assert result["replan_count"] == 0
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "planner_feedback" in result
        assert "User guidance:" in result["planner_feedback"]
        assert result.get("should_stop") is not True  # Should not stop

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
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None  # Should be cleared

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "replan_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_replan_count(self, mock_context):
        """Should handle missing replan_count field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "GUIDANCE: test"},
        }

        result = supervisor_node(state)

        assert result["replan_count"] == 0
        assert result["supervisor_verdict"] == "replan_with_guidance"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unclear_response(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "maybe"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "replan_limit"  # Preserved for clarification
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_force(self, mock_context):
        """Should handle case-insensitive FORCE keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "force"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "force" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_accept(self, mock_context):
        """Should handle case-insensitive ACCEPT keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "accept"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "accepted" in result["supervisor_feedback"].lower() or "force-accepted" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_guidance(self, mock_context):
        """Should handle case-insensitive GUIDANCE keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "guidance: test"},
            "replan_count": 5,
        }

        result = supervisor_node(state)

        assert result["replan_count"] == 0
        assert result["supervisor_verdict"] == "replan_with_guidance"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handles_update_progress_exception(self, mock_update, mock_context):
        """Should handle exceptions from update_progress_stage_status gracefully."""
        mock_context.return_value = None
        mock_update.side_effect = Exception("Database error")
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }

        # Should not raise exception, but may log error
        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_invalid_user_responses_type(self, mock_context):
        """Should handle invalid user_responses type gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": "not a dict",  # Invalid type
        }

        # Should handle gracefully, not crash
        result = supervisor_node(state)

        # Should either handle it or set ask_user
        assert "supervisor_verdict" in result
        # If verdict is ask_user, trigger is preserved; otherwise it's cleared
        if result.get("supervisor_verdict") == "ask_user":
            assert result["ask_user_trigger"] == "code_review_limit"
        else:
            assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_user_responses(self, mock_context):
        """Should handle None user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": None,
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "code_review_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_whitespace_only_response(self, mock_context):
        """Should handle whitespace-only responses."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "   \n\t  "},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "code_review_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_mixed_case_keywords(self, mock_context):
        """Should handle mixed case keywords correctly."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "PrOvIdE_hInT: test"},
            "code_revision_count": 3,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_partial_keyword_match(self, mock_context):
        """Should handle partial keyword matches correctly."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "I will PROVIDE_HINT now"},
            "code_revision_count": 2,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_string_response(self, mock_context):
        """Should handle empty string responses."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": ""},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "code_review_limit"  # Preserved for clarification


class TestLlmErrorTrigger:
    """Tests for llm_error trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_retries_on_retry(self, mock_context):
        """Should continue workflow on RETRY."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "RETRY"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert "supervisor_feedback" in result
        assert "retry" in result["supervisor_feedback"].lower()
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage on SKIP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert result.get("should_stop") is not True
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][1] == "stage1"
        assert call_args[0][2] == "blocked"
        assert "llm error" in call_args[1]["summary"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow on STOP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "maybe"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "llm_error"  # Preserved for clarification
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        question = result["pending_user_questions"][0]
        assert "RETRY" in question or "SKIP" in question or "STOP" in question

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "llm_error"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_current_stage_id_on_skip(self, mock_context):
        """Should handle None current_stage_id when skipping."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_retry(self, mock_context):
        """Should handle case-insensitive RETRY keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "retry"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "retry" in result["supervisor_feedback"].lower()


class TestClarificationTrigger:
    """Tests for clarification trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_continues_with_clarification(self, mock_context):
        """Should continue workflow with user clarification."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "clarification",
            "user_responses": {"Question": "The wavelength should be 550nm"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert "supervisor_feedback" in result
        assert "User clarification" in result["supervisor_feedback"]
        # The clarification text is stored in user_context, not supervisor_feedback
        assert "550nm" in result["user_context"][-1]
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_continues_with_empty_clarification(self, mock_context):
        """Should continue even with empty clarification."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "clarification",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert "supervisor_feedback" in result
        assert "No clarification" in result["supervisor_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_preserves_clarification_text(self, mock_context):
        """Should preserve full clarification text in user_context."""
        mock_context.return_value = None
        clarification = "Use FDTD with PML boundaries and mesh size 10nm"
        state = {
            "ask_user_trigger": "clarification",
            "user_responses": {"Question": clarification},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        # Clarification is stored in user_context, not supervisor_feedback
        assert clarification in result["user_context"][-1]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_whitespace_only_clarification(self, mock_context):
        """Should handle whitespace-only clarification."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "clarification",
            "user_responses": {"Question": "   "},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None


class TestCriticalErrorTriggers:
    """Tests for critical error triggers (missing_paper_text, missing_stage_id, progress_init_failed)."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_missing_paper_text_retry(self, mock_context):
        """Should continue workflow on RETRY for missing_paper_text."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "missing_paper_text",
            "user_responses": {"Question": "RETRY"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert "supervisor_feedback" in result
        assert "critical error" in result["supervisor_feedback"].lower()
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_missing_paper_text_stop(self, mock_context):
        """Should stop workflow on STOP for missing_paper_text."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "missing_paper_text",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_missing_paper_text_unclear(self, mock_context):
        """Should ask clarification on unclear response for missing_paper_text."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "missing_paper_text",
            "user_responses": {"Question": "maybe"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "missing_paper_text"  # Preserved for clarification
        assert "pending_user_questions" in result
        question = result["pending_user_questions"][0]
        assert "RETRY" in question or "STOP" in question

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_missing_stage_id_retry(self, mock_context):
        """Should continue workflow on RETRY for missing_stage_id."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "missing_stage_id",
            "user_responses": {"Question": "RETRY"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_missing_stage_id_stop(self, mock_context):
        """Should stop workflow on STOP for missing_stage_id."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "missing_stage_id",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_progress_init_failed_retry(self, mock_context):
        """Should continue workflow on RETRY for progress_init_failed."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "progress_init_failed",
            "user_responses": {"Question": "RETRY"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_progress_init_failed_stop(self, mock_context):
        """Should stop workflow on STOP for progress_init_failed."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "progress_init_failed",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_critical_error_empty_responses(self, mock_context):
        """Should ask clarification on empty responses for critical errors."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "missing_paper_text",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "missing_paper_text"  # Preserved for clarification
        assert "pending_user_questions" in result


class TestPlanningErrorTriggers:
    """Tests for planning error triggers (no_stages_available, invalid_backtrack_target, backtrack_target_not_found)."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_no_stages_available_replan(self, mock_context):
        """Should trigger replan on REPLAN for no_stages_available."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "no_stages_available",
            "user_responses": {"Question": "REPLAN"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"
        assert result["ask_user_trigger"] is None
        assert "planner_feedback" in result
        assert "REPLAN" in result["planner_feedback"]
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_no_stages_available_stop(self, mock_context):
        """Should stop workflow on STOP for no_stages_available."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "no_stages_available",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_no_stages_available_unclear(self, mock_context):
        """Should ask clarification on unclear response for no_stages_available."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "no_stages_available",
            "user_responses": {"Question": "unsure"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "no_stages_available"  # Preserved for clarification
        assert "pending_user_questions" in result
        question = result["pending_user_questions"][0]
        assert "REPLAN" in question or "STOP" in question

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_invalid_backtrack_target_replan(self, mock_context):
        """Should trigger replan on REPLAN for invalid_backtrack_target."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_target",
            "user_responses": {"Question": "REPLAN"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_invalid_backtrack_target_stop(self, mock_context):
        """Should stop workflow on STOP for invalid_backtrack_target."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_target",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_backtrack_target_not_found_replan(self, mock_context):
        """Should trigger replan on REPLAN for backtrack_target_not_found."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_target_not_found",
            "user_responses": {"Question": "REPLAN"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_backtrack_target_not_found_stop(self, mock_context):
        """Should stop workflow on STOP for backtrack_target_not_found."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_target_not_found",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_planning_error_empty_responses(self, mock_context):
        """Should ask clarification on empty responses for planning errors."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "no_stages_available",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "no_stages_available"  # Preserved for clarification
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_planning_error_case_insensitive_replan(self, mock_context):
        """Should handle case-insensitive REPLAN keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "no_stages_available",
            "user_responses": {"Question": "replan"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"


class TestBacktrackLimitTrigger:
    """Tests for backtrack_limit trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_force_continues_on_force(self, mock_context):
        """Should continue workflow on FORCE."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_limit",
            "user_responses": {"Question": "FORCE_CONTINUE"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert "supervisor_feedback" in result
        assert "backtrack limit" in result["supervisor_feedback"].lower()
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_force_continues_on_continue(self, mock_context):
        """Should continue workflow on CONTINUE."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_limit",
            "user_responses": {"Question": "CONTINUE"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow on STOP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_limit",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_limit",
            "user_responses": {"Question": "maybe"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "backtrack_limit"  # Preserved for clarification
        assert "pending_user_questions" in result
        question = result["pending_user_questions"][0]
        assert "FORCE" in question or "CONTINUE" in question or "STOP" in question

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_limit",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "backtrack_limit"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_force(self, mock_context):
        """Should handle case-insensitive FORCE keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_limit",
            "user_responses": {"Question": "force"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"


class TestInvalidBacktrackDecisionTrigger:
    """Tests for invalid_backtrack_decision trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_continues_on_continue(self, mock_context):
        """Should continue workflow on CONTINUE and clear invalid decision."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_decision",
            "user_responses": {"Question": "CONTINUE"},
            "backtrack_decision": {"target_stage_id": "invalid_stage"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        assert result["backtrack_decision"] is None  # Should clear invalid decision
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop workflow on STOP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_decision",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_decision",
            "user_responses": {"Question": "maybe"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"  # Preserved for clarification
        assert "pending_user_questions" in result
        question = result["pending_user_questions"][0]
        assert "CONTINUE" in question or "STOP" in question

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_decision",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"  # Preserved for clarification

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_backtrack_decision(self, mock_context):
        """Should handle missing backtrack_decision when continuing."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_decision",
            "user_responses": {"Question": "CONTINUE"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["backtrack_decision"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_case_insensitive_continue(self, mock_context):
        """Should handle case-insensitive CONTINUE keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "invalid_backtrack_decision",
            "user_responses": {"Question": "continue"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"


class TestTriggerHandlerIntegration:
    """Integration tests verifying trigger handler integration."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_all_limit_triggers_clear_trigger(self, mock_context):
        """All limit triggers should clear the ask_user_trigger."""
        mock_context.return_value = None
        
        triggers = [
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "replan_limit",
            "llm_error",
            "clarification",
            "missing_paper_text",
            "missing_stage_id",
            "progress_init_failed",
            "no_stages_available",
            "invalid_backtrack_target",
            "backtrack_target_not_found",
            "backtrack_limit",
            "invalid_backtrack_decision",
        ]
        
        for trigger in triggers:
            state = {
                "ask_user_trigger": trigger,
                "user_responses": {"Question": "STOP"},  # Universal response
                "progress": {"stages": [], "user_interactions": []},
            }
            
            result = supervisor_node(state)
            
            assert result["ask_user_trigger"] is None, f"Trigger {trigger} did not clear ask_user_trigger"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stop_response_always_stops_workflow(self, mock_context):
        """STOP response should always stop workflow for all triggers."""
        mock_context.return_value = None
        
        triggers = [
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "replan_limit",
            "llm_error",
            "missing_paper_text",
            "missing_stage_id",
            "progress_init_failed",
            "no_stages_available",
            "invalid_backtrack_target",
            "backtrack_target_not_found",
            "backtrack_limit",
            "invalid_backtrack_decision",
        ]
        
        for trigger in triggers:
            state = {
                "ask_user_trigger": trigger,
                "user_responses": {"Question": "STOP"},
                "progress": {"stages": [], "user_interactions": []},
            }
            
            result = supervisor_node(state)
            
            assert result["supervisor_verdict"] == "all_complete", \
                f"Trigger {trigger} did not set all_complete verdict on STOP"
            assert result["should_stop"] is True, \
                f"Trigger {trigger} did not set should_stop on STOP"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_skip_response_skips_stage(self, mock_context):
        """SKIP response should skip stage for applicable triggers."""
        mock_context.return_value = None
        
        triggers_with_skip = [
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "llm_error",
        ]
        
        for trigger in triggers_with_skip:
            state = {
                "ask_user_trigger": trigger,
                "user_responses": {"Question": "SKIP"},
                "current_stage_id": "stage1",
                "progress": {"stages": [], "user_interactions": []},
            }
            
            with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
                result = supervisor_node(state)
                
                assert result["supervisor_verdict"] == "ok_continue", \
                    f"Trigger {trigger} did not set ok_continue verdict on SKIP"
                assert result.get("should_stop") is not True, \
                    f"Trigger {trigger} set should_stop on SKIP"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_for_all_triggers(self, mock_context):
        """All triggers should log user interaction."""
        mock_context.return_value = None
        
        triggers = [
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "replan_limit",
            "llm_error",
            "clarification",
            "backtrack_limit",
            "invalid_backtrack_decision",
        ]
        
        for trigger in triggers:
            state = {
                "ask_user_trigger": trigger,
                "user_responses": {"Question": "STOP"},
                "pending_user_questions": ["Question?"],
                "current_stage_id": "stage1",
                "progress": {"stages": [], "user_interactions": []},
            }
            
            result = supervisor_node(state)
            
            assert "progress" in result, f"Trigger {trigger} did not return progress"
            assert "user_interactions" in result["progress"], \
                f"Trigger {trigger} did not log user interaction"
            assert len(result["progress"]["user_interactions"]) == 1, \
                f"Trigger {trigger} logged wrong number of interactions"
            interaction = result["progress"]["user_interactions"][0]
            assert interaction["interaction_type"] == trigger, \
                f"Trigger {trigger} logged wrong interaction type"


class TestUserResponseVariations:
    """Tests for various user response formats and edge cases."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_response_with_extra_text(self, mock_context):
        """Should handle response with extra text before/after keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "Please PROVIDE_HINT: Try using numpy arrays for better performance"},
            "code_revision_count": 3,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint
        assert "numpy arrays" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_response_with_special_characters(self, mock_context):
        """Should handle response with special characters."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": "HINT: Use np.array([1,2,3])!"},
            "code_revision_count": 2,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint
        assert "np.array" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_response_with_newlines(self, mock_context):
        """Should handle response with newlines."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Question": "GUIDANCE:\n1. Step one\n2. Step two"},
            "replan_count": 3,
        }

        result = supervisor_node(state)

        assert result["replan_count"] == 0
        assert result["supervisor_verdict"] == "replan_with_guidance"
        assert "Step one" in result["planner_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unicode_response(self, mock_context):
        """Should handle unicode characters in response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "clarification",
            "user_responses": {"Question": "Use λ=550nm for the wavelength"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        # Clarification is stored in user_context, not supervisor_feedback
        assert "λ=550nm" in result["user_context"][-1]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_very_long_response(self, mock_context):
        """Should handle very long response."""
        mock_context.return_value = None
        long_hint = "HINT: " + "A" * 10000
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": long_hint},
            "code_revision_count": 2,
        }

        result = supervisor_node(state)

        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_generate_code"  # Retry with hint
        assert "User hint:" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_numeric_response_value(self, mock_context):
        """Should handle numeric response value gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Question": 42},  # Non-string value
            "progress": {"stages": [], "user_interactions": []},
        }

        # Should handle gracefully
        result = supervisor_node(state)

        # Should ask for clarification since "42" doesn't match any keyword
        assert result["supervisor_verdict"] == "ask_user"
        assert result["ask_user_trigger"] == "code_review_limit"  # Preserved for clarification

