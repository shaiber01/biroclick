"""Supervisor tests for material checkpoint trigger."""

from unittest.mock import patch

from src.agents.supervision import supervisor_node


class TestMaterialCheckpointTrigger:
    """Tests for material_checkpoint trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_approves_materials_on_approve(self, mock_archive, mock_update, mock_context):
        """Should approve materials when user says APPROVE."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "APPROVE"},
            "pending_validated_materials": [{"material_id": "gold", "name": "Gold"}],
            "current_stage_id": "stage0",
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["validated_materials"] == [{"material_id": "gold", "name": "Gold"}]
        assert result["pending_validated_materials"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handles_change_database(self, mock_update, mock_context):
        """Should trigger replan on CHANGE_DATABASE response."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "CHANGE_DATABASE to custom"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "current_stage_id": "stage0",
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"
        assert "database" in result["planner_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handles_change_material(self, mock_update, mock_context):
        """Should trigger replan on CHANGE_MATERIAL response."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "CHANGE_MATERIAL to silver"},
            "current_stage_id": "stage0",
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"
        assert "material" in result["planner_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_need_help(self, mock_context):
        """Should ask for more details on NEED_HELP response."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "NEED_HELP"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert "details" in result["pending_user_questions"][0].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unclear_response(self, mock_context):
        """Should ask for clarification on unclear response."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "maybe"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert "unclear" in result["pending_user_questions"][0].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_rejection_without_specifics(self, mock_context):
        """Should ask for clarification on rejection without specifying what to change."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "NO"},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_pending_materials_on_approve(self, mock_context):
        """Should ask user when approving but no materials pending."""
        mock_context.return_value = None

        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "APPROVE"},
            "pending_validated_materials": [],
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"

