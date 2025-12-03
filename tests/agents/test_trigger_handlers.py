"""Unit tests for src/agents/supervision/trigger_handlers.py"""

import pytest
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any

from src.agents.supervision.trigger_handlers import (
    handle_material_checkpoint,
    handle_code_review_limit,
    handle_design_review_limit,
    handle_execution_failure_limit,
    handle_physics_failure_limit,
    handle_context_overflow,
    handle_replan_limit,
    handle_backtrack_approval,
    handle_deadlock_detected,
    handle_llm_error,
    handle_clarification,
    handle_critical_error_retry,
    handle_planning_error_retry,
    handle_backtrack_limit,
    handle_invalid_backtrack_decision,
    handle_trigger
)

class TestTriggerHandlers:
    """Tests for all trigger handlers in trigger_handlers.py."""

    @pytest.fixture
    def mock_state(self):
        return {
            "paper_text": "Short paper text",
            "pending_validated_materials": ["mat1", "mat2"],
            "plan": {"stages": {}},
            "backtrack_decision": {"target_stage_id": "stage0"}
        }

    @pytest.fixture
    def mock_result(self):
        return {}

    # --- Material Checkpoint Tests ---

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    @patch("src.agents.supervision.trigger_handlers.archive_stage_outputs_to_progress")
    def test_handle_material_checkpoint_approve(self, mock_archive, mock_update, mock_state, mock_result):
        """Should approve materials, archive stage, and update status."""
        user_input = {"q1": "I APPROVE these materials."}
        
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
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
        
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert any("No materials" in q for q in mock_result["pending_user_questions"])

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_database(self, mock_update, mock_state, mock_result):
        """Should handle database change request."""
        user_input = {"q1": "REJECT, please CHANGE_DATABASE."}
        
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "database change" in mock_result["planner_feedback"]
        assert mock_result["validated_materials"] == []
        assert mock_result["pending_validated_materials"] == []
        
        mock_update.assert_called_once()
        args = mock_update.call_args[0]
        assert args[2] == "needs_rerun"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_material_checkpoint_reject_material(self, mock_update, mock_state, mock_result):
        """Should handle material change request."""
        user_input = {"q1": "Please CHANGE_MATERIAL, this is wrong."}
        
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "wrong material" in mock_result["planner_feedback"]
        
        mock_update.assert_called_once()
        args = mock_update.call_args[0]
        assert args[2] == "needs_rerun"

    def test_handle_material_checkpoint_need_help(self, mock_state, mock_result):
        """Should ask for clarification on HELP."""
        user_input = {"q1": "I NEED_HELP understanding this."}
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_ambiguous_rejection(self, mock_state, mock_result):
        """Should ask user to specify what to change on generic rejection."""
        user_input = {"q1": "REJECT this."}
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "specify what to change" in mock_result["pending_user_questions"][0]

    def test_handle_material_checkpoint_unclear(self, mock_state, mock_result):
        """Should ask user to clarify on unclear input."""
        user_input = {"q1": "foobar"}
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_material_checkpoint_approve_and_reject_mixed(self, mock_state, mock_result):
        """Should prioritize rejection/change over approval if mixed."""
        user_input = {"q1": "I APPROVE the first part but REJECT and want to CHANGE_MATERIAL"}
        handle_material_checkpoint(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "replan_needed"

    # --- Code Review Limit Tests ---

    def test_handle_code_review_limit_hint(self, mock_state, mock_result):
        """Should reset counter and add feedback on hint."""
        user_input = {"q1": "PROVIDE_HINT: Use a loop instead."}
        
        handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert result_has_value(mock_result, "code_revision_count", 0)
        assert "Use a loop" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_code_review_limit_skip(self, mock_update, mock_state, mock_result):
        """Should skip stage and continue."""
        user_input = {"q1": "SKIP"}
        
        handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_with(mock_state, "stage1", "blocked", summary=ANY)

    def test_handle_code_review_limit_stop(self, mock_state, mock_result):
        """Should stop workflow."""
        user_input = {"q1": "STOP"}
        handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_code_review_limit_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown input."""
        user_input = {"q1": "Just keep going"}
        handle_code_review_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    # --- Design Review Limit Tests ---

    def test_handle_design_review_limit_hint(self, mock_state, mock_result):
        """Should reset counter and add feedback."""
        user_input = {"q1": "PROVIDE_HINT: Fix the dimensions."}
        handle_design_review_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "design_revision_count", 0)
        assert "Fix the dimensions" in mock_result["reviewer_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    # --- Execution Failure Limit Tests ---

    def test_handle_execution_failure_limit_retry(self, mock_state, mock_result):
        """Should reset counter and add guidance."""
        user_input = {"q1": "RETRY_WITH_GUIDANCE: Check memory."}
        handle_execution_failure_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "execution_failure_count", 0)
        assert "Check memory" in mock_result["supervisor_feedback"]
        assert mock_result["supervisor_verdict"] == "ok_continue"

    # --- Physics Failure Limit Tests ---

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_physics_failure_limit_accept_partial(self, mock_update, mock_state, mock_result):
        """Should mark partial success."""
        user_input = {"q1": "ACCEPT_PARTIAL results."}
        handle_physics_failure_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_with(mock_state, "stage1", "completed_partial", summary=ANY)

    # --- Context Overflow Tests ---

    def test_handle_context_overflow_summarize(self, mock_state, mock_result):
        """Should apply summarization."""
        user_input = {"q1": "SUMMARIZE"}
        handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "summarization" in mock_result["supervisor_feedback"]

    def test_handle_context_overflow_truncate_long(self, mock_state, mock_result):
        """Should truncate long paper text."""
        long_text = "a" * 25000
        mock_state["paper_text"] = long_text
        user_input = {"q1": "TRUNCATE"}
        
        handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "TRUNCATED" in mock_result["paper_text"]
        assert len(mock_result["paper_text"]) < 25000
        assert mock_result["paper_text"].startswith(long_text[:15000])
        assert mock_result["paper_text"].endswith(long_text[-5000:])

    def test_handle_context_overflow_truncate_short(self, mock_state, mock_result):
        """Should not truncate if text is short."""
        short_text = "a" * 100
        mock_state["paper_text"] = short_text
        user_input = {"q1": "TRUNCATE"}
        
        handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "paper_text" not in mock_result # Should not be modified if not truncated
        assert "short enough" in mock_result["supervisor_feedback"]

    def test_handle_context_overflow_unknown(self, mock_state, mock_result):
        """Should continue on unknown input (default behavior)."""
        # NOTE: Current implementation defaults to continue. This test verifies that behavior.
        # If this is considered a bug, the test should expect 'ask_user' and the code should be fixed.
        # Given "tests that FAIL when bugs exist", assuming silence on unclear input is bad UX.
        # Updating expectation to fail if strict, but let's verify current behavior first or fix it.
        # I will assert 'ask_user' and expect it to fail if code continues.
        user_input = {"q1": "What is this?"}
        handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        # Strict expectation: Should ask for clarification
        assert mock_result["supervisor_verdict"] == "ask_user"

    # --- Replan Limit Tests ---

    def test_handle_replan_limit_force(self, mock_state, mock_result):
        """Should force accept plan."""
        user_input = {"q1": "FORCE accept"}
        handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_replan_limit_guidance(self, mock_state, mock_result):
        """Should retry with guidance."""
        user_input = {"q1": "GUIDANCE: Try this."}
        handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        assert result_has_value(mock_result, "replan_count", 0)
        assert mock_result["supervisor_verdict"] == "replan_needed"

    def test_handle_replan_limit_unknown(self, mock_state, mock_result):
        """Should ask user on unknown input."""
        user_input = {"q1": "Unknown"}
        handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        # Expectation: ask user, not continue blindly
        assert mock_result["supervisor_verdict"] == "ask_user"

    # --- Backtrack Approval Tests ---

    def test_handle_backtrack_approval_approve(self, mock_state, mock_result):
        """Should calculate dependents and set verdict to backtrack."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=["stage1", "stage2"])
        
        handle_backtrack_approval(mock_state, mock_result, user_input, "stage1", mock_get_dependents)
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert mock_result["backtrack_decision"]["stages_to_invalidate"] == ["stage1", "stage2"]
        mock_get_dependents.assert_called_once()

    def test_handle_backtrack_approval_reject(self, mock_state, mock_result):
        """Should clear suggestion and continue."""
        user_input = {"q1": "REJECT"}
        handle_backtrack_approval(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["backtrack_suggestion"] is None

    # --- Deadlock Tests ---

    def test_handle_deadlock_report(self, mock_state, mock_result):
        """Should stop and generate report."""
        user_input = {"q1": "GENERATE_REPORT"}
        handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_replan(self, mock_state, mock_result):
        """Should request replan."""
        user_input = {"q1": "REPLAN"}
        handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "replan_needed"

    def test_handle_deadlock_unknown(self, mock_state, mock_result):
        """Should ask for clarification."""
        user_input = {"q1": "Help"}
        handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    # --- LLM Error Tests ---

    def test_handle_llm_error_retry(self, mock_state, mock_result):
        """Should retry."""
        user_input = {"q1": "RETRY"}
        handle_llm_error(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_handle_llm_error_skip(self, mock_update, mock_state, mock_result):
        """Should skip stage."""
        user_input = {"q1": "SKIP"}
        
        handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        mock_update.assert_called_with(mock_state, "stage1", "blocked", summary=ANY)
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_llm_error_unknown(self, mock_state, mock_result):
        """Should ask for clarification."""
        user_input = {"q1": "What?"}
        handle_llm_error(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ask_user"

    # --- General Dispatcher Test ---

    def test_handle_trigger_dispatcher(self, mock_state, mock_result):
        """Should dispatch to correct handler."""
        # Test one path via dispatcher
        user_input = {"q1": "STOP"}
        handle_trigger("code_review_limit", mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "all_complete"

    def test_handle_trigger_unknown(self, mock_state, mock_result):
        """Should handle unknown trigger gracefully."""
        user_input = {"q1": "foo"}
        handle_trigger("unknown_trigger_xyz", mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "unknown trigger" in mock_result["supervisor_feedback"]

def result_has_value(result, key, value):
    """Helper to check value in result dict."""
    return result.get(key) == value
