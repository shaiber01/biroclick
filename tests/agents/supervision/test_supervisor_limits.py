"""Supervisor tests for code/design/execution/physics/replan limit triggers."""

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
        assert result["supervisor_verdict"] == "ok_continue"
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
        assert result["ask_user_trigger"] is None  # Should be cleared
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
        assert result["ask_user_trigger"] is None
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
        assert result["supervisor_verdict"] == "ok_continue"

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
        assert result["supervisor_verdict"] == "ok_continue"

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
        assert result["supervisor_verdict"] == "ok_continue"
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
        assert result["supervisor_verdict"] == "ok_continue"
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
        assert result["ask_user_trigger"] is None

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
        assert result["supervisor_verdict"] == "ok_continue"


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
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "supervisor_feedback" in result
        assert "User guidance:" in result["supervisor_feedback"]
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
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "supervisor_feedback" in result
        assert "User guidance:" in result["supervisor_feedback"]
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
        assert result["ask_user_trigger"] is None

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
        assert result["supervisor_verdict"] == "ok_continue"

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
        assert result["ask_user_trigger"] is None
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
        assert result["supervisor_verdict"] == "ok_continue"
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

        assert result["supervisor_verdict"] == "ok_continue"
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

        assert result["supervisor_verdict"] == "ok_continue"
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
        assert result["ask_user_trigger"] is None

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

        assert result["supervisor_verdict"] == "ok_continue"
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
        assert result["supervisor_verdict"] == "ok_continue"

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
        assert result["ask_user_trigger"] is None
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

        assert result["supervisor_verdict"] == "replan_needed"
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
        assert result["ask_user_trigger"] is None

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
        assert result["supervisor_verdict"] == "replan_needed"

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
        assert result["ask_user_trigger"] is None
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
        assert result["supervisor_verdict"] == "replan_needed"


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
        assert result["ask_user_trigger"] is None

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
        assert result["ask_user_trigger"] is None

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
        assert result["supervisor_verdict"] == "ok_continue"

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
        assert result["supervisor_verdict"] == "ok_continue"

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
        assert result["ask_user_trigger"] is None

