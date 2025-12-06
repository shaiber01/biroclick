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
    def test_empty_questions_list_triggers_safety_net(self, mock_interrupt):
        """Safety net #1: Empty questions list should generate recovery questions.
        
        Gap #1 fix: When routers return "ask_user" due to errors (verdict is None,
        wrong type, etc.), they cannot set pending_user_questions. The safety net
        generates recovery questions so users are always prompted.
        """
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        # Safety net should call interrupt with generated recovery questions
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        
        # Should have WORKFLOW RECOVERY in generated questions
        assert "WORKFLOW RECOVERY" in payload["questions"][0]
        # Trigger should be overridden to unknown_escalation  
        assert payload["trigger"] == "unknown_escalation"
        assert result.get("ask_user_trigger") == "unknown_escalation"

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
    """Tests for _infer_error_context and _generate_error_question helper functions.
    
    The inference uses a priority-based lookup:
    1. last_node_before_ask_user - most specific, set when nodes escalate
    2. workflow_phase - indicates current pipeline position
    3. ask_user_trigger - detects stuck state
    4. Falls back to "unknown_error"
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Priority 1: last_node_before_ask_user tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_infer_from_last_node_plan_review(self):
        """Should return 'plan_review_error' when last_node is plan_review."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "plan_review"}
        assert _infer_error_context(state) == "plan_review_error"

    def test_infer_from_last_node_design_review(self):
        """Should return 'design_review_error' when last_node is design_review."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "design_review"}
        assert _infer_error_context(state) == "design_review_error"

    def test_infer_from_last_node_code_review(self):
        """Should return 'code_review_error' when last_node is code_review."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "code_review"}
        assert _infer_error_context(state) == "code_review_error"

    def test_infer_from_last_node_execution_check(self):
        """Should return 'execution_error' when last_node is execution_check."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "execution_check"}
        assert _infer_error_context(state) == "execution_error"

    def test_infer_from_last_node_physics_check(self):
        """Should return 'physics_error' when last_node is physics_check."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "physics_check"}
        assert _infer_error_context(state) == "physics_error"

    def test_infer_from_last_node_comparison_check(self):
        """Should return 'comparison_error' when last_node is comparison_check."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "comparison_check"}
        assert _infer_error_context(state) == "comparison_error"

    def test_infer_from_last_node_supervisor(self):
        """Should return 'supervisor_error' when last_node is supervisor."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "supervisor"}
        assert _infer_error_context(state) == "supervisor_error"

    def test_infer_from_last_node_handle_backtrack(self):
        """Should return 'backtrack_error' when last_node is handle_backtrack."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "handle_backtrack"}
        assert _infer_error_context(state) == "backtrack_error"

    def test_infer_from_last_node_material_checkpoint(self):
        """Should return 'material_checkpoint_error' when last_node is material_checkpoint."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "material_checkpoint"}
        assert _infer_error_context(state) == "material_checkpoint_error"

    def test_infer_from_last_node_unknown_node(self):
        """Should fall through to next priority when last_node is not recognized."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"last_node_before_ask_user": "some_unknown_node"}
        # No workflow_phase or trigger, so should return unknown_error
        assert _infer_error_context(state) == "unknown_error"

    # ═══════════════════════════════════════════════════════════════════════
    # Priority 2: workflow_phase fallback tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_infer_from_phase_planning(self):
        """Should return 'plan_review_error' when phase is planning."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "planning"}
        assert _infer_error_context(state) == "plan_review_error"

    def test_infer_from_phase_plan_review(self):
        """Should return 'plan_review_error' when phase is plan_review."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "plan_review"}
        assert _infer_error_context(state) == "plan_review_error"

    def test_infer_from_phase_design(self):
        """Should return 'design_review_error' when phase is design."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "design"}
        assert _infer_error_context(state) == "design_review_error"

    def test_infer_from_phase_design_review(self):
        """Should return 'design_review_error' when phase is design_review."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "design_review"}
        assert _infer_error_context(state) == "design_review_error"

    def test_infer_from_phase_code_generation(self):
        """Should return 'code_review_error' when phase is code_generation."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "code_generation"}
        assert _infer_error_context(state) == "code_review_error"

    def test_infer_from_phase_code_review(self):
        """Should return 'code_review_error' when phase is code_review."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "code_review"}
        assert _infer_error_context(state) == "code_review_error"

    def test_infer_from_phase_execution_validation(self):
        """Should return 'execution_error' when phase is execution_validation."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "execution_validation"}
        assert _infer_error_context(state) == "execution_error"

    def test_infer_from_phase_physics_validation(self):
        """Should return 'physics_error' when phase is physics_validation."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "physics_validation"}
        assert _infer_error_context(state) == "physics_error"

    def test_infer_from_phase_analysis(self):
        """Should return 'comparison_error' when phase is analysis."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "analysis"}
        assert _infer_error_context(state) == "comparison_error"

    def test_infer_from_phase_comparison_validation(self):
        """Should return 'comparison_error' when phase is comparison_validation."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "comparison_validation"}
        assert _infer_error_context(state) == "comparison_error"

    def test_infer_from_phase_unknown_phase(self):
        """Should fall through to next priority when phase is not recognized."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"workflow_phase": "some_unknown_phase"}
        # No trigger, so should return unknown_error
        assert _infer_error_context(state) == "unknown_error"

    # ═══════════════════════════════════════════════════════════════════════
    # Priority 3: stuck trigger test
    # ═══════════════════════════════════════════════════════════════════════

    def test_infer_stuck_when_trigger_set(self):
        """Should return 'stuck_awaiting_input' when ask_user_trigger is set."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {"ask_user_trigger": "some_trigger"}
        assert _infer_error_context(state) == "stuck_awaiting_input"

    # ═══════════════════════════════════════════════════════════════════════
    # Fallback test
    # ═══════════════════════════════════════════════════════════════════════

    def test_infer_unknown_when_no_info(self):
        """Should return 'unknown_error' when no useful state information."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {}
        assert _infer_error_context(state) == "unknown_error"

    # ═══════════════════════════════════════════════════════════════════════
    # Priority order tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_last_node_takes_priority_over_phase(self):
        """last_node_before_ask_user should take priority over workflow_phase."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "last_node_before_ask_user": "physics_check",
            "workflow_phase": "design",  # Would return design_review_error
        }
        assert _infer_error_context(state) == "physics_error"

    def test_last_node_takes_priority_over_trigger(self):
        """last_node_before_ask_user should take priority over ask_user_trigger."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "last_node_before_ask_user": "code_review",
            "ask_user_trigger": "some_trigger",  # Would return stuck_awaiting_input
        }
        assert _infer_error_context(state) == "code_review_error"

    def test_phase_takes_priority_over_trigger(self):
        """workflow_phase should take priority over ask_user_trigger."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "workflow_phase": "code_review",
            "ask_user_trigger": "some_trigger",  # Would return stuck_awaiting_input
        }
        assert _infer_error_context(state) == "code_review_error"

    def test_all_three_priorities_last_node_wins(self):
        """When all three are set, last_node_before_ask_user should win."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "last_node_before_ask_user": "supervisor",
            "workflow_phase": "physics_validation",
            "ask_user_trigger": "context_overflow",
        }
        assert _infer_error_context(state) == "supervisor_error"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge cases: Empty strings, None values, case sensitivity
    # ═══════════════════════════════════════════════════════════════════════

    def test_empty_string_last_node_falls_through(self):
        """Empty string last_node should fall through to workflow_phase."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "last_node_before_ask_user": "",  # Empty string is truthy but not in mapping
            "workflow_phase": "physics_validation",
        }
        # Empty string should NOT match any node, so should fall through to phase
        assert _infer_error_context(state) == "physics_error"

    def test_empty_string_workflow_phase_falls_through(self):
        """Empty string workflow_phase should fall through to trigger check."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "workflow_phase": "",  # Empty string is truthy but not in mapping
            "ask_user_trigger": "some_trigger",
        }
        # Empty string should NOT match any phase, so should fall through to trigger
        assert _infer_error_context(state) == "stuck_awaiting_input"

    def test_none_last_node_falls_through(self):
        """Explicit None last_node should fall through to workflow_phase."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "last_node_before_ask_user": None,
            "workflow_phase": "execution_validation",
        }
        assert _infer_error_context(state) == "execution_error"

    def test_none_workflow_phase_falls_through(self):
        """Explicit None workflow_phase should fall through to trigger check."""
        from src.agents.user_interaction import _infer_error_context
        
        state = {
            "last_node_before_ask_user": None,
            "workflow_phase": None,
            "ask_user_trigger": "context_overflow",
        }
        assert _infer_error_context(state) == "stuck_awaiting_input"

    def test_case_sensitive_last_node_no_match(self):
        """Node names are case-sensitive - wrong case should not match."""
        from src.agents.user_interaction import _infer_error_context
        
        # These should NOT match because case is wrong
        test_cases = [
            ("Physics_check", "unknown_error"),  # Capital P
            ("PHYSICS_CHECK", "unknown_error"),  # All caps
            ("PhysicsCheck", "unknown_error"),   # CamelCase
            ("physics_Check", "unknown_error"),  # Mixed case
        ]
        for wrong_case_node, expected in test_cases:
            state = {"last_node_before_ask_user": wrong_case_node}
            result = _infer_error_context(state)
            assert result == expected, f"'{wrong_case_node}' should return '{expected}', got '{result}'"

    def test_case_sensitive_workflow_phase_no_match(self):
        """Workflow phases are case-sensitive - wrong case should not match."""
        from src.agents.user_interaction import _infer_error_context
        
        # These should NOT match because case is wrong
        test_cases = [
            ("Physics_validation", "unknown_error"),  # Capital P
            ("PHYSICS_VALIDATION", "unknown_error"),  # All caps
            ("PhysicsValidation", "unknown_error"),   # CamelCase
        ]
        for wrong_case_phase, expected in test_cases:
            state = {"workflow_phase": wrong_case_phase}
            result = _infer_error_context(state)
            assert result == expected, f"'{wrong_case_phase}' should return '{expected}', got '{result}'"

    def test_whitespace_in_last_node_no_match(self):
        """Whitespace in node name should not match."""
        from src.agents.user_interaction import _infer_error_context
        
        test_cases = [
            " physics_check",    # Leading space
            "physics_check ",    # Trailing space
            " physics_check ",   # Both
            "physics _check",    # Internal space
        ]
        for bad_node in test_cases:
            state = {"last_node_before_ask_user": bad_node}
            result = _infer_error_context(state)
            assert result == "unknown_error", f"'{bad_node}' should return 'unknown_error', got '{result}'"

    # ═══════════════════════════════════════════════════════════════════════
    # _generate_error_question tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_generate_error_question_physics_error(self):
        """Should generate appropriate message for physics_error with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_1"}
        
        result = _generate_error_question("physics_error", state)
        
        # Check message structure
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content from the error message
        assert "Physics validation failed to run for stage 'stage_1'" in result
        assert "validation node was skipped or encountered an error" in result
        assert "physics checks were not performed" in result

    def test_generate_error_question_execution_error(self):
        """Should generate appropriate message for execution_error with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_2"}
        
        result = _generate_error_question("execution_error", state)
        
        # Check message structure
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content from the error message
        assert "Execution validation failed to run for stage 'stage_2'" in result
        assert "simulation may not have completed properly" in result

    def test_generate_error_question_comparison_error(self):
        """Should generate appropriate message for comparison_error."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_3"}
        
        result = _generate_error_question("comparison_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        assert "Comparison check failed to run for stage 'stage_3'" in result
        assert "Results analysis may not have completed" in result

    def test_generate_error_question_code_review_error(self):
        """Should generate appropriate message for code_review_error."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_1"}
        
        result = _generate_error_question("code_review_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        assert "Code review failed to run for stage 'stage_1'" in result
        assert "generated code may not have been reviewed" in result

    def test_generate_error_question_design_review_error(self):
        """Should generate appropriate message for design_review_error."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_1"}
        
        result = _generate_error_question("design_review_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        assert "Design review failed to run for stage 'stage_1'" in result
        assert "simulation design may not have been validated" in result

    def test_generate_error_question_plan_review_error(self):
        """Should generate appropriate message for plan_review_error (no stage_id)."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_1"}  # Should NOT appear in message
        
        result = _generate_error_question("plan_review_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        assert "Plan review failed to run" in result
        assert "reproduction plan may not have been validated" in result
        # plan_review_error message does NOT include stage_id
        assert "stage_1" not in result

    def test_generate_error_question_stuck_awaiting_input(self):
        """Should generate appropriate message for stuck_awaiting_input with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("stuck_awaiting_input", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content - must match the actual message
        assert "unprocessed ask_user_trigger" in result
        assert "previous user interaction wasn't properly completed" in result
        assert "system will attempt to recover" in result

    def test_generate_error_question_supervisor_error(self):
        """Should generate appropriate message for supervisor_error with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("supervisor_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content - must match the actual message
        assert "supervisor node encountered an issue" in result
        assert "problem with workflow orchestration" in result

    def test_generate_error_question_backtrack_error(self):
        """Should generate appropriate message for backtrack_error with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("backtrack_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content - must match the actual message
        assert "Backtracking encountered an issue" in result
        assert "trouble returning to a previous stage" in result

    def test_generate_error_question_material_checkpoint_error(self):
        """Should generate appropriate message for material_checkpoint_error with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "stage_0"}
        
        result = _generate_error_question("material_checkpoint_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content - must match the actual message
        assert "Material checkpoint validation encountered an issue for stage 'stage_0'" in result
        assert "Material validation results may need user review" in result

    def test_generate_error_question_unknown_error(self):
        """Should generate generic message for unknown_error with exact content."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("unknown_error", state)
        
        assert result.startswith("WORKFLOW RECOVERY NEEDED\n\n")
        # Check specific content - must match the actual message
        assert "unexpected workflow error occurred" in result
        assert "Unable to determine the specific cause" in result

    def test_generate_error_question_unknown_context_falls_back(self):
        """Should fall back to unknown_error message for unrecognized context."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}
        
        result = _generate_error_question("some_unrecognized_context", state)
        
        # Should get the SAME message as unknown_error
        expected = _generate_error_question("unknown_error", state)
        assert result == expected, f"Unrecognized context should produce same message as 'unknown_error'"

    def test_generate_error_question_uses_default_stage_id(self):
        """Should use 'unknown' as default stage_id when not in state."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {}  # No current_stage_id
        
        result = _generate_error_question("physics_error", state)
        
        # Should interpolate 'unknown' into the message
        assert "for stage 'unknown'" in result

    def test_generate_error_question_stage_id_in_all_relevant_messages(self):
        """Messages that should include stage_id must interpolate it correctly."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "test_stage_xyz"}
        
        # These contexts should include the stage_id in the message
        contexts_with_stage_id = [
            "physics_error",
            "execution_error", 
            "comparison_error",
            "code_review_error",
            "design_review_error",
            "material_checkpoint_error",
        ]
        
        for context in contexts_with_stage_id:
            result = _generate_error_question(context, state)
            assert "test_stage_xyz" in result, f"Context '{context}' should include stage_id in message"

    def test_generate_error_question_stage_id_not_in_plan_review(self):
        """plan_review_error message should NOT include stage_id."""
        from src.agents.user_interaction import _generate_error_question
        
        state = {"current_stage_id": "test_stage_xyz"}
        
        result = _generate_error_question("plan_review_error", state)
        
        # plan_review is about the overall plan, not a specific stage
        assert "test_stage_xyz" not in result, "plan_review_error should not include stage_id"


class TestSafetyNetEmptyQuestions:
    """Tests for safety net #1: generating recovery questions when questions are empty.
    
    Gap #1 fix: When routers return "ask_user" due to errors (verdict is None,
    wrong type, or unrecognized), they cannot set pending_user_questions because
    routers can only return route names. The safety net generates recovery questions
    so users are always prompted.
    
    There are NO legitimate cases where ask_user_node should receive empty questions:
    - If ask_user_trigger is set, the node that set it also sets questions
    - If routed via error paths, questions aren't set (this safety net catches it)
    - The material_checkpoint "passthrough" routes to select_stage, not ask_user
    """

    @patch("src.agents.user_interaction.interrupt")
    def test_infers_execution_error_from_workflow_phase(self, mock_interrupt):
        """Should infer execution_error when workflow_phase is execution_validation."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "some_trigger",
            "workflow_phase": "execution_validation",  # Priority 2: workflow_phase
            "current_stage_id": "stage_1",
        }
        
        ask_user_node(state)
        
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        questions = payload["questions"][0]
        
        # Should mention execution validation failure
        assert "WORKFLOW RECOVERY" in questions
        assert "Execution validation failed" in questions
        assert "stage_1" in questions

    @patch("src.agents.user_interaction.interrupt")
    def test_infers_physics_error_from_last_node(self, mock_interrupt):
        """Should infer physics_error when last_node_before_ask_user is physics_check."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "some_trigger",
            "last_node_before_ask_user": "physics_check",  # Priority 1: last_node
            "current_stage_id": "stage_2",
        }
        
        ask_user_node(state)
        
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        questions = payload["questions"][0]
        
        # Should mention physics validation failure
        assert "WORKFLOW RECOVERY" in questions
        assert "Physics validation failed" in questions
        assert "stage_2" in questions

    @patch("src.agents.user_interaction.interrupt")
    def test_generated_questions_have_unknown_escalation_options(self, mock_interrupt):
        """Generated questions should include unknown_escalation options (RETRY, SKIP_STAGE, STOP)."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        ask_user_node(state)
        
        payload = mock_interrupt.call_args[0][0]
        questions = payload["questions"][0]
        
        # Should have options from unknown_escalation trigger
        # These are defined in user_options.py
        assert "RETRY" in questions or "retry" in questions.lower()

    @patch("src.agents.user_interaction.interrupt")
    def test_user_can_respond_to_recovery_questions(self, mock_interrupt):
        """User should be able to respond to recovery questions."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        # Should have stored the user's response
        assert "user_responses" in result
        # Response should be mapped to the generated question
        assert len(result["user_responses"]) == 1
        # The first (only) value should be the user's response
        assert list(result["user_responses"].values())[0] == "RETRY"

    @patch("src.agents.user_interaction.interrupt")
    def test_safety_net_sets_trigger_for_supervisor(self, mock_interrupt):
        """Safety net should set trigger so supervisor knows how to handle the response."""
        mock_interrupt.return_value = "SKIP_STAGE"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "original_trigger",  # Will be overridden
        }
        
        result = ask_user_node(state)
        
        # Trigger should be set to unknown_escalation for supervisor
        assert result.get("ask_user_trigger") == "unknown_escalation"

    @patch("src.agents.user_interaction.interrupt")
    def test_empty_list_vs_none_both_trigger_safety_net(self, mock_interrupt):
        """Both empty list [] and None should trigger safety net."""
        mock_interrupt.return_value = "RETRY"
        
        # Test with empty list
        state1 = {"pending_user_questions": [], "ask_user_trigger": "test"}
        result1 = ask_user_node(state1)
        assert result1.get("ask_user_trigger") == "unknown_escalation"
        
        mock_interrupt.reset_mock()
        
        # Test with None
        state2 = {"pending_user_questions": None, "ask_user_trigger": "test"}
        result2 = ask_user_node(state2)
        assert result2.get("ask_user_trigger") == "unknown_escalation"
        
        # Both should have called interrupt
        assert mock_interrupt.call_count == 1  # Only second call after reset

    @patch("src.agents.user_interaction.interrupt")
    def test_safety_net_clears_pending_questions_after_response(self, mock_interrupt):
        """After user responds to recovery questions, pending_user_questions should be cleared."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        # Pending questions should be cleared after successful response
        assert result.get("pending_user_questions") == []

    @patch("src.agents.user_interaction.interrupt")
    def test_safety_net_logs_warning(self, mock_interrupt, caplog):
        """Safety net should log a warning when triggered."""
        import logging
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        with caplog.at_level(logging.WARNING):
            ask_user_node(state)
        
        # Should have logged a warning about no pending questions
        assert any("no pending_user_questions" in record.message for record in caplog.records)
