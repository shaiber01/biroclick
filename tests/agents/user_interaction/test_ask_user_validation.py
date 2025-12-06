"""Validation and error handling tests for ask_user_node.

These tests use the LangGraph interrupt() mocking pattern since the implementation
uses interrupt() for human-in-the-loop workflows rather than CLI input.

NOTE: ask_user_node has simplified validation - it only checks for empty responses.
Keyword validation is handled by supervisor's trigger handlers, not here.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.agents.user_interaction import ask_user_node


class TestEmptyResponseHandling:
    """Tests for empty response handling (the only validation in ask_user_node)."""

    @patch("src.agents.user_interaction.interrupt")
    def test_rejects_empty_response(self, mock_interrupt):
        """Should reject empty response and ask user to retry."""
        mock_interrupt.return_value = ""
        
        state = {
            "pending_user_questions": ["Material checkpoint: APPROVE or REJECT?"],
            "ask_user_trigger": "material_checkpoint",
        }
        
        result = ask_user_node(state)
        
        # Should return with ask_user_trigger set and error message
        assert result.get("ask_user_trigger") is not None
        assert "empty" in result["pending_user_questions"][0].lower()
        assert result["ask_user_trigger"] == "material_checkpoint"
        # Should NOT have user_responses since response was empty
        assert "user_responses" not in result

    @patch("src.agents.user_interaction.interrupt")
    def test_rejects_whitespace_only_response(self, mock_interrupt):
        """Should reject whitespace-only response."""
        mock_interrupt.return_value = "   \n\t  "
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is not None
        assert "empty" in result["pending_user_questions"][0].lower()

    @patch("src.agents.user_interaction.interrupt")
    def test_accepts_any_non_empty_response(self, mock_interrupt):
        """Should accept any non-empty response (validation is done by supervisor)."""
        mock_interrupt.return_value = "option b - Adjust γX to 1.116×10¹⁴ rad/s"
        
        state = {
            "pending_user_questions": ["Choose option a, b, or c"],
            "ask_user_trigger": "reviewer_escalation",
        }
        
        result = ask_user_node(state)
        
        # Should accept any non-empty response
        assert result.get("ask_user_trigger") is None
        assert "user_responses" in result
        assert result["user_responses"]["Choose option a, b, or c"] == "option b - Adjust γX to 1.116×10¹⁴ rad/s"


class TestResponseStorage:
    """Tests for response storage."""

    @patch("src.agents.user_interaction.interrupt")
    def test_stores_response_correctly(self, mock_interrupt):
        """Should store response with question as key."""
        mock_interrupt.return_value = "User's detailed response"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert result["user_responses"]["Question?"] == "User's detailed response"

    @patch("src.agents.user_interaction.interrupt")
    def test_merges_with_existing_user_responses(self, mock_interrupt):
        """Should merge new response with existing user_responses."""
        mock_interrupt.return_value = "NewResponse"
        
        state = {
            "pending_user_questions": ["NewQuestion?"],
            "ask_user_trigger": "test",
            "user_responses": {
                "OldQuestion?": "OldResponse",
            },
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert len(result["user_responses"]) == 2
        assert result["user_responses"]["OldQuestion?"] == "OldResponse"
        assert result["user_responses"]["NewQuestion?"] == "NewResponse"

    @patch("src.agents.user_interaction.interrupt")
    def test_handles_none_user_responses(self, mock_interrupt):
        """Should handle None user_responses in state gracefully."""
        mock_interrupt.return_value = "Response"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "user_responses": None,  # Explicitly None
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert result["user_responses"]["Question?"] == "Response"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("src.agents.user_interaction.interrupt")
    def test_empty_questions_list(self, mock_interrupt):
        """Should handle empty questions list gracefully."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert result["workflow_phase"] == "awaiting_user"
        # Should not call interrupt when no questions
        mock_interrupt.assert_not_called()

    @patch("src.agents.user_interaction.interrupt")
    def test_missing_ask_user_trigger(self, mock_interrupt):
        """Should handle missing ask_user_trigger with safety net."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": ["Original question without options"],
            # Missing ask_user_trigger - safety net will set "unknown_escalation"
        }
        
        result = ask_user_node(state)
        
        # Safety net should set "unknown_escalation" as trigger
        assert result.get("ask_user_trigger") is None
        assert result["ask_user_trigger"] == "unknown_escalation"
        
        # interrupt should be called with regenerated questions containing WORKFLOW RECOVERY
        mock_interrupt.assert_called_once()
        interrupt_payload = mock_interrupt.call_args[0][0]
        assert interrupt_payload["trigger"] == "unknown_escalation"
        # Questions should be regenerated with WORKFLOW RECOVERY format
        regenerated_questions = interrupt_payload["questions"]
        assert len(regenerated_questions) == 1
        assert "WORKFLOW RECOVERY" in regenerated_questions[0]
        # Original context should be preserved
        assert "Original question without options" in regenerated_questions[0]

    @patch("src.agents.user_interaction.interrupt")
    def test_missing_trigger_strips_old_options(self, mock_interrupt):
        """When safety net triggers, old Options: section should be stripped."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": ["Question with old options\n\nOptions:\n- PROVIDE_GUIDANCE\n- OLD_OPTION"],
            # Missing ask_user_trigger
        }
        
        result = ask_user_node(state)
        
        mock_interrupt.assert_called_once()
        interrupt_payload = mock_interrupt.call_args[0][0]
        regenerated_questions = interrupt_payload["questions"]
        
        # Should contain WORKFLOW RECOVERY and new options
        assert "WORKFLOW RECOVERY" in regenerated_questions[0]
        # Original question context should be preserved (before Options:)
        assert "Question with old options" in regenerated_questions[0]
        # Old options should be stripped and NOT appear
        assert "OLD_OPTION" not in regenerated_questions[0]
        assert "PROVIDE_GUIDANCE" not in regenerated_questions[0]


class TestStateClearing:
    """Tests for state clearing on successful response."""

    @patch("src.agents.user_interaction.interrupt")
    def test_clears_pending_questions(self, mock_interrupt):
        """Should clear pending_user_questions on successful response."""
        mock_interrupt.return_value = "APPROVE"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
        }
        
        result = ask_user_node(state)
        
        assert result["pending_user_questions"] == []
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.user_interaction.interrupt")
    def test_clears_original_user_questions(self, mock_interrupt):
        """Should clear original_user_questions on successful response."""
        mock_interrupt.return_value = "Response"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "original_user_questions": ["Original question"],
        }
        
        result = ask_user_node(state)
        
        assert result["original_user_questions"] is None

    @patch("src.agents.user_interaction.interrupt")
    def test_sets_workflow_phase(self, mock_interrupt):
        """Should set workflow_phase to 'awaiting_user' on success."""
        mock_interrupt.return_value = "Response"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["workflow_phase"] == "awaiting_user"


class TestInterruptPayload:
    """Tests for interrupt() payload structure."""

    @patch("src.agents.user_interaction.interrupt")
    def test_interrupt_receives_correct_payload(self, mock_interrupt):
        """Should call interrupt with correct payload structure."""
        mock_interrupt.return_value = "APPROVE"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
            "paper_id": "test_paper_123",
        }
        
        ask_user_node(state)
        
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        
        assert payload["trigger"] == "material_checkpoint"
        assert payload["questions"] == ["Question?"]
        assert payload["paper_id"] == "test_paper_123"

    @patch("src.agents.user_interaction.interrupt")
    def test_interrupt_uses_unknown_paper_id_when_missing(self, mock_interrupt):
        """Should use 'unknown' as paper_id when not provided."""
        mock_interrupt.return_value = "APPROVE"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            # No paper_id
        }
        
        ask_user_node(state)
        
        payload = mock_interrupt.call_args[0][0]
        assert payload["paper_id"] == "unknown"


class TestErrorContextHelpers:
    """Tests for _infer_error_context and _generate_error_question helper functions."""

    def test_infer_error_context_physics_error(self):
        """Should return 'physics_error' when physics_verdict is None but execution_verdict exists."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "physics_verdict": None,
            "execution_verdict": "pass",  # Execution ran but physics didn't
        }
        
        result = _infer_error_context(state)
        assert result == "physics_error"

    def test_infer_error_context_execution_error(self):
        """Should return 'execution_error' when execution_verdict is None."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": None,
        }
        
        result = _infer_error_context(state)
        assert result == "execution_error"

    def test_infer_error_context_comparison_error(self):
        """Should return 'comparison_error' when comparison_verdict is None."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": None,
        }
        
        result = _infer_error_context(state)
        assert result == "comparison_error"

    def test_infer_error_context_code_review_error(self):
        """Should return 'code_review_error' when last_code_review_verdict is None."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": "pass",
            "last_code_review_verdict": None,
        }
        
        result = _infer_error_context(state)
        assert result == "code_review_error"

    def test_infer_error_context_design_review_error(self):
        """Should return 'design_review_error' when last_design_review_verdict is None."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": "pass",
            "last_code_review_verdict": "approve",
            "last_design_review_verdict": None,
        }
        
        result = _infer_error_context(state)
        assert result == "design_review_error"

    def test_infer_error_context_plan_review_error(self):
        """Should return 'plan_review_error' when last_plan_review_verdict is None."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": "pass",
            "last_code_review_verdict": "approve",
            "last_design_review_verdict": "approve",
            "last_plan_review_verdict": None,
        }
        
        result = _infer_error_context(state)
        assert result == "plan_review_error"

    def test_infer_error_context_stuck_awaiting_input(self):
        """Should return 'stuck_awaiting_input' when ask_user_trigger is set."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": "pass",
            "last_code_review_verdict": "approve",
            "last_design_review_verdict": "approve",
            "last_plan_review_verdict": "approve",
            "ask_user_trigger": "context_overflow",
        }
        
        result = _infer_error_context(state)
        assert result == "stuck_awaiting_input"

    def test_infer_error_context_unknown_error(self):
        """Should return 'unknown_error' when no specific error is detected."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": "pass",
            "last_code_review_verdict": "approve",
            "last_design_review_verdict": "approve",
            "last_plan_review_verdict": "approve",
            # No ask_user_trigger set, so should be unknown_error
        }
        
        result = _infer_error_context(state)
        assert result == "unknown_error"

    def test_infer_error_context_empty_state(self):
        """Should return 'execution_error' for empty state (execution_verdict is None)."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {}
        
        result = _infer_error_context(state)
        assert result == "execution_error"

    def test_generate_error_question_physics_error(self):
        """Should generate appropriate message for physics_error."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_1"}
        
        result = _generate_error_question("physics_error", state)
        
        assert "WORKFLOW RECOVERY" in result
        assert "Physics validation failed" in result
        assert "stage_1" in result

    def test_generate_error_question_execution_error(self):
        """Should generate appropriate message for execution_error."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_2"}
        
        result = _generate_error_question("execution_error", state)
        
        assert "WORKFLOW RECOVERY" in result
        assert "Execution validation failed" in result
        assert "stage_2" in result

    def test_generate_error_question_stuck_awaiting_input(self):
        """Should generate appropriate message for stuck_awaiting_input."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("stuck_awaiting_input", state)
        
        assert "WORKFLOW RECOVERY" in result
        assert "stuck" in result.lower() or "ask_user_trigger" in result

    def test_generate_error_question_unknown_error(self):
        """Should generate generic message for unknown_error."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("unknown_error", state)
        
        assert "WORKFLOW RECOVERY" in result
        assert "unexpected" in result.lower()

    def test_generate_error_question_unknown_context(self):
        """Should fall back to unknown_error message for unrecognized context."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("some_unrecognized_context", state)
        
        assert "WORKFLOW RECOVERY" in result
        assert "unexpected" in result.lower()

    def test_generate_error_question_uses_default_stage_id(self):
        """Should use 'unknown' as default stage_id when not in state."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}  # No current_stage_id
        
        result = _generate_error_question("physics_error", state)
        
        assert "unknown" in result
