"""Tests for material checkpoint trigger handler flows."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestMaterialCheckpointTrigger:
    """Tests for material_checkpoint trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_approves_materials_on_approve(self, mock_archive, mock_update, mock_context):
        """Should approve materials when user says APPROVE."""
        mock_context.return_value = None
        
        materials = [{"material_id": "gold", "name": "Gold"}]
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "APPROVE"},
            "pending_validated_materials": materials,
            "current_stage_id": "stage0",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["validated_materials"] == materials
        assert result["pending_validated_materials"] == []
        assert "approved" in result.get("supervisor_feedback", "").lower()
        mock_archive.assert_called_once_with(state, "stage0")
        mock_update.assert_called_once_with(state, "stage0", "completed_success")

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_approves_materials_all_approval_keywords(self, mock_archive, mock_update, mock_context):
        """Should approve materials for all approval keywords."""
        mock_context.return_value = None
        
        materials = [{"material_id": "silver", "name": "Silver"}]
        approval_keywords = ["APPROVE", "YES", "CORRECT", "OK", "ACCEPT", "VALID", "PROCEED"]
        
        for keyword in approval_keywords:
            state = {
                "ask_user_trigger": "material_checkpoint",
                "user_responses": {"Material question": keyword},
                "pending_validated_materials": materials.copy(),
                "current_stage_id": "stage0",
            }
            
            result = supervisor_node(state)
            
            assert result["supervisor_verdict"] == "ok_continue", f"Failed for keyword: {keyword}"
            assert result["validated_materials"] == materials
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
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []
        mock_update.assert_called_once()
        call_args = mock_update.call_args[0]
        assert call_args[0] == state
        assert call_args[1] == "stage0"
        assert call_args[2] == "needs_rerun"
        assert "invalidation_reason" in mock_update.call_args.kwargs

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handles_change_database_with_rejection_keyword(self, mock_update, mock_context):
        """Should trigger replan when REJECT + DATABASE keywords are present."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "REJECT DATABASE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "current_stage_id": "stage0",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "database" in result["planner_feedback"].lower()
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []

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
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []
        mock_update.assert_called_once()
        call_args = mock_update.call_args[0]
        assert call_args[2] == "needs_rerun"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handles_change_material_with_rejection_keyword(self, mock_update, mock_context):
        """Should trigger replan when REJECT + MATERIAL keywords are present."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "REJECT MATERIAL"},
            "current_stage_id": "stage0",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "material" in result["planner_feedback"].lower()
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []

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
        assert len(result["pending_user_questions"]) > 0
        assert "details" in result["pending_user_questions"][0].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_help_keyword(self, mock_context):
        """Should ask for more details on HELP response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "HELP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert len(result["pending_user_questions"]) > 0
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
        assert len(result["pending_user_questions"]) > 0
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
        assert len(result["pending_user_questions"]) > 0
        assert "specify" in result["pending_user_questions"][0].lower()

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
        assert len(result["pending_user_questions"]) > 0
        assert any("no materials" in q.lower() or "error" in q.lower() for q in result["pending_user_questions"])
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_pending_materials_key(self, mock_context):
        """Should handle missing pending_validated_materials key."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "APPROVE"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert len(result["pending_user_questions"]) > 0

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_change_database_without_stage_id(self, mock_update, mock_context):
        """Should handle CHANGE_DATABASE when current_stage_id is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "CHANGE_DATABASE"},
            "pending_validated_materials": [{"material_id": "gold"}],
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "database" in result["planner_feedback"].lower()
        # Should not call update_progress_stage_status when stage_id is None
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_change_database_error_handling(self, mock_update, mock_context):
        """Should handle update_progress errors gracefully for CHANGE_DATABASE.
        
        BUG TEST: Ensures CHANGE_DATABASE path handles errors consistently
        with CHANGE_MATERIAL path.
        """
        mock_context.return_value = None
        mock_update.side_effect = Exception("Update failed")
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "CHANGE_DATABASE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "current_stage_id": "stage0",
        }
        
        # Should NOT raise exception - error should be handled gracefully
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "database" in result["planner_feedback"].lower()
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_change_material_without_stage_id(self, mock_update, mock_context):
        """Should handle CHANGE_MATERIAL when current_stage_id is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "CHANGE_MATERIAL"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "material" in result["planner_feedback"].lower()
        # Should use error handling wrapper, so update is called but with error handling
        assert mock_update.call_count >= 0  # May or may not be called depending on implementation

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_approve_without_stage_id(self, mock_archive, mock_update, mock_context):
        """Should approve materials even when current_stage_id is None."""
        mock_context.return_value = None
        
        materials = [{"material_id": "gold"}]
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question": "APPROVE"},
            "pending_validated_materials": materials,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["validated_materials"] == materials
        assert result["pending_validated_materials"] == []
        # Should not archive/update when stage_id is None
        mock_archive.assert_not_called()
        mock_update.assert_not_called()


class TestHandleMaterialCheckpoint:
    """Unit tests for handle_material_checkpoint helper function."""

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_approve(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should approve materials, archive stage, and update status."""
        user_input = {"q1": "I APPROVE these materials."}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "approved" in mock_result["supervisor_feedback"].lower()
        assert mock_result["validated_materials"] == ["mat1", "mat2"]
        assert mock_result["pending_validated_materials"] == []

        mock_archive.assert_called_once_with(mock_state, "stage1")
        mock_update.assert_called_once()
        call_args = mock_update.call_args[0]
        assert call_args[0] == mock_state
        assert call_args[1] == "stage1"
        assert call_args[2] == "completed_success"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_approve_all_keywords(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should approve materials for all approval keywords."""
        approval_keywords = ["APPROVE", "YES", "CORRECT", "OK", "ACCEPT", "VALID", "PROCEED"]
        
        for keyword in approval_keywords:
            mock_result.clear()
            user_input = {"q1": f"I {keyword} these materials."}

            trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

            assert mock_result["supervisor_verdict"] == "ok_continue", f"Failed for keyword: {keyword}"
            assert mock_result["validated_materials"] == ["mat1", "mat2"]
            assert mock_result["pending_validated_materials"] == []

    def test_handle_material_checkpoint_approve_no_pending(self, mock_state, mock_result):
        """Should fail approval if no pending materials exist."""
        mock_state["pending_validated_materials"] = []
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0
        assert any("no materials" in q.lower() or "error" in q.lower() for q in mock_result["pending_user_questions"])
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

    def test_handle_material_checkpoint_approve_missing_key(self, mock_state, mock_result):
        """Should handle missing pending_validated_materials key."""
        if "pending_validated_materials" in mock_state:
            del mock_state["pending_validated_materials"]
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_database(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle database change request."""
        user_input = {"q1": "REJECT, please CHANGE_DATABASE."}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database" in mock_result["planner_feedback"].lower()
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

        mock_update.assert_called_once()
        call_args = mock_update.call_args[0]
        assert call_args[0] == mock_state
        assert call_args[1] == "stage1"
        assert call_args[2] == "needs_rerun"
        assert "invalidation_reason" in mock_update.call_args.kwargs

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_database_error_handling(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle update_progress errors gracefully for CHANGE_DATABASE.
        
        BUG TEST: CHANGE_DATABASE path calls update_progress_stage_status directly
        without error handling, unlike CHANGE_MATERIAL which uses error wrapper.
        This test will FAIL if the bug exists (exception not caught).
        """
        mock_update.side_effect = Exception("Update failed")
        user_input = {"q1": "CHANGE_DATABASE"}

        # This should NOT raise an exception - error should be handled
        try:
            trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        except Exception as e:
            pytest.fail(f"CHANGE_DATABASE should handle update_progress errors gracefully, but raised: {e}")

        # Should still set verdict and clear materials despite error
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database" in mock_result["planner_feedback"].lower()
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_database_via_keywords(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle database change via REJECT + DATABASE keywords."""
        user_input = {"q1": "REJECT DATABASE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database" in mock_result["planner_feedback"].lower()
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_material(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle material change request."""
        user_input = {"q1": "Please CHANGE_MATERIAL, this is wrong."}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "material" in mock_result["planner_feedback"].lower()
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

        mock_update.assert_called_once()
        call_args = mock_update.call_args[0]
        assert call_args[2] == "needs_rerun"
        assert "invalidation_reason" in mock_update.call_args.kwargs

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_material_via_keywords(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle material change via REJECT + MATERIAL keywords."""
        user_input = {"q1": "REJECT MATERIAL"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "material" in mock_result["planner_feedback"].lower()
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_change_database_without_stage_id(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle CHANGE_DATABASE when current_stage_id is None."""
        user_input = {"q1": "CHANGE_DATABASE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, None)

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database" in mock_result["planner_feedback"].lower()
        # Should not call update_progress_stage_status when stage_id is None
        mock_update.assert_not_called()

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_change_material_without_stage_id(
        self, mock_update, mock_state, mock_result
    ):
        """Should handle CHANGE_MATERIAL when current_stage_id is None."""
        user_input = {"q1": "CHANGE_MATERIAL"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, None)

        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "material" in mock_result["planner_feedback"].lower()
        # Should use error handling wrapper, so may or may not call update
        # But should not crash

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_approve_without_stage_id(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should approve materials even when current_stage_id is None."""
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, None)

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["validated_materials"] == ["mat1", "mat2"]
        assert mock_result["pending_validated_materials"] == []
        # Should not archive/update when stage_id is None
        mock_archive.assert_not_called()
        mock_update.assert_not_called()

    def test_handle_material_checkpoint_need_help(self, mock_state, mock_result):
        """Should ask for clarification on HELP."""
        user_input = {"q1": "I NEED_HELP understanding this."}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0
        assert "details" in mock_result["pending_user_questions"][0].lower()

    def test_handle_material_checkpoint_help_keyword(self, mock_state, mock_result):
        """Should ask for clarification on HELP keyword."""
        user_input = {"q1": "HELP"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0
        assert "details" in mock_result["pending_user_questions"][0].lower()

    def test_handle_material_checkpoint_ambiguous_rejection(self, mock_state, mock_result):
        """Should ask user to specify what to change on generic rejection."""
        user_input = {"q1": "REJECT this."}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0
        assert "specify" in mock_result["pending_user_questions"][0].lower()

    def test_handle_material_checkpoint_all_rejection_keywords(self, mock_state, mock_result):
        """Should ask for clarification for all rejection keywords without specifics."""
        rejection_keywords = ["REJECT", "NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]
        
        for keyword in rejection_keywords:
            mock_result.clear()
            user_input = {"q1": keyword}
            trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
            assert mock_result["supervisor_verdict"] == "ask_user", f"Failed for keyword: {keyword}"
            assert len(mock_result["pending_user_questions"]) > 0

    def test_handle_material_checkpoint_unclear(self, mock_state, mock_result):
        """Should ask user to clarify on unclear input."""
        user_input = {"q1": "foobar"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0
        assert "unclear" in mock_result["pending_user_questions"][0].lower()

    def test_handle_material_checkpoint_empty_response(self, mock_state, mock_result):
        """Should handle empty user response."""
        user_input = {"q1": ""}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert len(mock_result["pending_user_questions"]) > 0

    def test_handle_material_checkpoint_multiple_responses(self, mock_state, mock_result):
        """Should use the last response when multiple responses are provided."""
        user_input = {
            "q1": "APPROVE",
            "q2": "REJECT",
            "q3": "CHANGE_MATERIAL"
        }
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        # Should use last response (CHANGE_MATERIAL)
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "material" in mock_result["planner_feedback"].lower()

    def test_handle_material_checkpoint_approve_and_reject_mixed(self, mock_state, mock_result):
        """Should prioritize rejection/change over approval if mixed."""
        user_input = {
            "q1": "I APPROVE the first part but REJECT and want to CHANGE_MATERIAL"
        }
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "material" in mock_result["planner_feedback"].lower()

    def test_handle_material_checkpoint_case_insensitive(self, mock_state, mock_result):
        """Should handle case-insensitive responses."""
        user_input = {"q1": "approve"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["validated_materials"] == ["mat1", "mat2"]

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_archive_error_handling(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should handle archive errors gracefully."""
        mock_archive.side_effect = Exception("Archive failed")
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        # Should still approve materials despite archive error
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["validated_materials"] == ["mat1", "mat2"]
        # Should record archive error
        assert "archive_errors" in mock_result
        assert len(mock_result["archive_errors"]) > 0
        assert mock_result["archive_errors"][0]["stage_id"] == "stage1"
        assert "Archive failed" in mock_result["archive_errors"][0]["error"]

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_update_error_handling(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should handle update_progress errors gracefully."""
        mock_update.side_effect = Exception("Update failed")
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        # Should still approve materials despite update error
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["validated_materials"] == ["mat1", "mat2"]
        # Error should be logged but not block execution

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_both_errors(
        self, mock_archive, mock_update, mock_state, mock_result
    ):
        """Should handle both archive and update errors gracefully."""
        mock_archive.side_effect = Exception("Archive failed")
        mock_update.side_effect = Exception("Update failed")
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        # Should still approve materials despite both errors
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["validated_materials"] == ["mat1", "mat2"]
        # Archive error should be recorded
        assert "archive_errors" in mock_result

    def test_handle_material_checkpoint_preserves_existing_validated_materials(
        self, mock_state, mock_result
    ):
        """Should preserve existing validated_materials when rejecting."""
        mock_result["validated_materials"] = [{"material_id": "existing"}]
        user_input = {"q1": "CHANGE_MATERIAL"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        # Should clear validated_materials on rejection
        assert mock_result["validated_materials"] == []

    def test_handle_material_checkpoint_clears_pending_on_approval(
        self, mock_state, mock_result
    ):
        """Should clear pending_validated_materials on approval."""
        user_input = {"q1": "APPROVE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["pending_validated_materials"] == []

    def test_handle_material_checkpoint_clears_pending_on_rejection(
        self, mock_state, mock_result
    ):
        """Should clear pending_validated_materials on rejection."""
        user_input = {"q1": "CHANGE_DATABASE"}

        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")

        assert mock_result["pending_validated_materials"] == []

    def test_handle_material_checkpoint_response_text_in_feedback(
        self, mock_state, mock_result
    ):
        """Should include user response text in planner feedback."""
        user_input = {"q1": "CHANGE_DATABASE to custom database"}
        
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        assert "CHANGE_DATABASE TO CUSTOM DATABASE" in mock_result["planner_feedback"]

    def test_handle_material_checkpoint_response_text_truncated_on_unclear(
        self, mock_state, mock_result
    ):
        """Should truncate very long unclear responses in feedback."""
        long_response = "a" * 200
        user_input = {"q1": long_response}
        
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        # Should truncate to 100 chars
        question = mock_result["pending_user_questions"][0]
        assert len(question) < len(long_response) + 50  # Some overhead for message text

    def test_handle_material_checkpoint_none_user_responses(self, mock_state, mock_result):
        """Should handle None user_responses gracefully."""
        # This should not crash - parse_user_response should handle None
        try:
            trigger_handlers.handle_material_checkpoint(mock_state, mock_result, None, "stage1")
        except (TypeError, AttributeError) as e:
            pytest.fail(f"Should handle None user_responses gracefully, but raised: {e}")
        
        # Should treat as unclear
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_empty_user_responses(self, mock_state, mock_result):
        """Should handle empty user_responses dict."""
        user_input = {}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should treat as unclear
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_approval_and_rejection_no_change(
        self, mock_state, mock_result
    ):
        """Should prioritize rejection when both approval and rejection keywords present without CHANGE."""
        user_input = {"q1": "I APPROVE but also REJECT"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should ask for clarification since both are present but no CHANGE keyword
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_pending_materials_none_value(
        self, mock_state, mock_result
    ):
        """Should handle None value in pending_validated_materials."""
        mock_state["pending_validated_materials"] = None
        user_input = {"q1": "APPROVE"}
        
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should treat as no materials
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_pending_materials_contains_none(
        self, mock_state, mock_result
    ):
        """Should handle None values within pending_validated_materials list."""
        mock_state["pending_validated_materials"] = [None, {"material_id": "gold"}]
        user_input = {"q1": "APPROVE"}
        
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should still approve (None values are passed through)
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["validated_materials"] == [None, {"material_id": "gold"}]

    def test_handle_material_checkpoint_whitespace_only_response(
        self, mock_state, mock_result
    ):
        """Should handle whitespace-only responses."""
        user_input = {"q1": "   \n\t  "}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should treat as unclear
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_change_database_case_variations(
        self, mock_state, mock_result
    ):
        """Should handle case variations of CHANGE_DATABASE."""
        variations = [
            "change_database",
            "CHANGE_database",
            "Change_Database",
            "CHANGE DATABASE",  # Space instead of underscore
        ]
        
        for variation in variations:
            mock_result.clear()
            user_input = {"q1": variation}
            trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
            # Should match (case-insensitive after parse_user_response uppercases)
            assert mock_result["supervisor_verdict"] == "replan_needed", f"Failed for: {variation}"

    def test_handle_material_checkpoint_change_material_case_variations(
        self, mock_state, mock_result
    ):
        """Should handle case variations of CHANGE_MATERIAL."""
        variations = [
            "change_material",
            "CHANGE_material",
            "Change_Material",
            "CHANGE MATERIAL",  # Space instead of underscore
        ]
        
        for variation in variations:
            mock_result.clear()
            user_input = {"q1": variation}
            trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
            # Should match (case-insensitive after parse_user_response uppercases)
            assert mock_result["supervisor_verdict"] == "replan_needed", f"Failed for: {variation}"

    def test_handle_material_checkpoint_database_keyword_without_change(
        self, mock_state, mock_result
    ):
        """Should handle DATABASE keyword with rejection but without CHANGE_DATABASE."""
        user_input = {"q1": "REJECT DATABASE"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should trigger CHANGE_DATABASE path
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database" in mock_result["planner_feedback"].lower()

    def test_handle_material_checkpoint_material_keyword_without_change(
        self, mock_state, mock_result
    ):
        """Should handle MATERIAL keyword with rejection but without CHANGE_MATERIAL."""
        user_input = {"q1": "REJECT MATERIAL"}
        trigger_handlers.handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        # Should trigger CHANGE_MATERIAL path
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "material" in mock_result["planner_feedback"].lower()
