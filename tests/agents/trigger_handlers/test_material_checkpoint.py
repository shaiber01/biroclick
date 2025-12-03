"""Tests for material checkpoint trigger handler flows."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers

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



class TestHandleMaterialCheckpoint:
    """Unit tests for handle_material_checkpoint helper."""

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_approve(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should approve materials, archive stage, and update status."""
        user_input = {"q1": "I APPROVE these materials."}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "approved" in mock_result["supervisor_feedback"]
        assert mock_result["validated_materials"] == ["mat1", "mat2"]
        assert mock_result["pending_validated_materials"] == []

        mock_archive.assert_called_once_with(mock_state, "stage1")
        mock_update.assert_called_once_with(mock_state, "stage1", "completed_success")

    def test_handle_material_checkpoint_approve_no_pending(self, mock_state, mock_result):
        """Should fail approval if no pending materials exist."""
        mock_state["pending_validated_materials"] = []
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ask_user"
        assert any("No materials" in q for q in mock_result["pending_user_questions"])

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_database(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle database change request."""
        user_input = {"q1": "REJECT, please CHANGE_DATABASE."}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database change" in mock_result["planner_feedback"]
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

        mock_update.assert_called_once()
        args = mock_update.call_args[0]
        assert args[2] == "needs_rerun"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_material(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle material change request."""
        user_input = {"q1": "Please CHANGE_MATERIAL, this is wrong."}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "wrong material" in mock_result["planner_feedback"]

        mock_update.assert_called_once()
        args = mock_update.call_args[0]
        assert args[2] == "needs_rerun"

    def test_handle_material_checkpoint_need_help(self, mock_state, mock_result):
        """Should ask for clarification on HELP."""
        user_input = {"q1": "I NEED_HELP understanding this."}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_ambiguous_rejection(self, mock_state, mock_result):
        """Should ask user to specify what to change on generic rejection."""
        user_input = {"q1": "REJECT this."}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "specify what to change" in mock_result["pending_user_questions"][0]

    def test_handle_material_checkpoint_unclear(self, mock_state, mock_result):
        """Should ask user to clarify on unclear input."""
        user_input = {"q1": "foobar"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_approve_and_reject_mixed(self, mock_state, mock_result):
        """Should prioritize rejection/change over approval if mixed."""
        user_input = {
            "q1": "I APPROVE the first part but REJECT and want to CHANGE_MATERIAL"
        }
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "replan_needed"
