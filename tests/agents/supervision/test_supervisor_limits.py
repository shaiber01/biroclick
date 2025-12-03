"""Supervisor tests for code/design/execution/physics/replan limit triggers."""

from unittest.mock import patch

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

