"""reviewer_escalation trigger tests."""

from unittest.mock import patch, MagicMock

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestReviewerEscalationTrigger:
    """Tests for reviewer_escalation trigger handling via supervisor_node."""

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_guidance_routes_to_llm(self, mock_context, mock_route_llm):
        """Should route PROVIDE_GUIDANCE to LLM for intelligent decision."""
        mock_context.return_value = None
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Use the Drude model"
        mock_route_llm.side_effect = set_verdict
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "PROVIDE_GUIDANCE: Use the Drude model"},
        }
        
        result = supervisor_node(state)
        
        # LLM should have been called
        mock_route_llm.assert_called_once()
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None
        # Verify reviewer feedback contains the guidance
        assert "reviewer_feedback" in result
        assert "Use the Drude model" in result["reviewer_feedback"]
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_guidance_lowercase_routes_to_llm(self, mock_context, mock_route_llm):
        """Should route lowercase provide_guidance to LLM."""
        mock_context.return_value = None
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: try a different approach"
        mock_route_llm.side_effect = set_verdict
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "provide_guidance: try a different approach"},
        }
        
        result = supervisor_node(state)
        
        mock_route_llm.assert_called_once()
        assert "try a different approach" in result["reviewer_feedback"]

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_guidance_alias_routes_to_llm(self, mock_context, mock_route_llm):
        """Should route GUIDANCE alias to LLM."""
        mock_context.return_value = None
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Use tabulated data"
        mock_route_llm.side_effect = set_verdict
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "GUIDANCE: Use tabulated data"},
        }
        
        result = supervisor_node(state)
        
        mock_route_llm.assert_called_once()
        assert "Use tabulated data" in result["reviewer_feedback"]

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_answer_alias_routes_to_llm(self, mock_context, mock_route_llm):
        """Should route ANSWER alias to LLM."""
        mock_context.return_value = None
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: The spacing is 20nm period"
        mock_route_llm.side_effect = set_verdict
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "ANSWER: The spacing is 20nm period"},
        }
        
        result = supervisor_node(state)
        
        mock_route_llm.assert_called_once()
        assert "The spacing is 20nm period" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skip_marks_stage_blocked(self, mock_update, mock_context):
        """Should mark stage as blocked when user says SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("ask_user_trigger") is None
        # Verify update_progress_stage_status was called with correct args
        mock_update.assert_called_once_with(
            state, "stage1", "blocked", summary="Skipped by user due to reviewer escalation"
        )
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skip_does_not_call_update_when_no_stage_id(self, mock_update, mock_context):
        """Should not call update_progress_stage_status when current_stage_id is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": None,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stop_ends_workflow(self, mock_context):
        """Should stop workflow when user says STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_freeform_response_routes_to_llm(self, mock_context, mock_route_llm):
        """Should route free-form responses to LLM for decision."""
        mock_context.return_value = None
        # Mock LLM to return ok_continue
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: I think we should use the FWHM interpretation"
        mock_route_llm.side_effect = set_verdict
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "I think we should use the FWHM interpretation"},
        }
        
        result = supervisor_node(state)
        
        # LLM should have been called for free-form response
        mock_route_llm.assert_called_once()
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None
        # Verify reviewer feedback contains the guidance
        assert "reviewer_feedback" in result
        assert "I think we should use the FWHM interpretation" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_empty_response_asks_clarification(self, mock_context):
        """Should ask for clarification on empty response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": ""},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_state_not_mutated(self, mock_context, mock_route_llm):
        """Should not mutate the input state dict."""
        mock_context.return_value = None
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
        mock_route_llm.side_effect = set_verdict
        
        original_state = {
            "ask_user_trigger": "reviewer_escalation",
            "user_responses": {"Question": "PROVIDE_GUIDANCE: Test guidance"},
            "other_field": "should_not_change",
        }
        state_copy = original_state.copy()
        
        result = supervisor_node(state_copy)
        
        # Verify state was not mutated
        assert state_copy["ask_user_trigger"] == "reviewer_escalation"  # Not cleared in state
        assert state_copy["other_field"] == "should_not_change"
        # But result should have the changes
        assert result.get("ask_user_trigger") is None


class TestHandleReviewerEscalation:
    """Direct handler tests for reviewer escalation."""

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_provide_guidance_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route PROVIDE_GUIDANCE to LLM for intelligent decision."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Use the measured optical constants"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "PROVIDE_GUIDANCE: Use the measured optical constants"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # LLM should have been called
        mock_route_llm.assert_called_once()
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_provide_guidance_with_guidance_alias_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route GUIDANCE alias to LLM."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Check the boundary conditions"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "GUIDANCE: Check the boundary conditions"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        mock_route_llm.assert_called_once()

    def test_provide_guidance_empty_user_responses(self, mock_state, mock_result):
        """Should handle empty user_responses dict gracefully."""
        user_input = {}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Should fall through to ask_user
        assert mock_result["supervisor_verdict"] == "ask_user"

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_provide_guidance_none_stage_id_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route guidance to LLM even when current_stage_id is None."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Test"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "PROVIDE_GUIDANCE: Test"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, None)
        
        mock_route_llm.assert_called_once()

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_skip_marks_stage_blocked(self, mock_update, mock_state, mock_result):
        """Should handle SKIP response correctly."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Verify _update_progress_with_error_handling was called with correct arguments
        mock_update.assert_called_once_with(
            mock_state, mock_result, "stage1", "blocked", summary="Skipped by user due to reviewer escalation"
        )
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_skip_none_stage_id(self, mock_update, mock_state, mock_result):
        """Should not call _update_progress_with_error_handling when stage_id is None."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    def test_stop_ends_workflow(self, mock_state, mock_result):
        """Should handle STOP response correctly."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_freeform_response_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route free-form responses to LLM for decision."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Just keep going and use the default parameters"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "Just keep going and use the default parameters"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # LLM should have been called for free-form response
        mock_route_llm.assert_called_once()
        # Verify should_stop is NOT set
        assert mock_result.get("should_stop") is not True

    def test_case_insensitive_fast_path_keywords(self, mock_state, mock_result):
        """Should match fast path keywords case-insensitively."""
        # Only test fast-path keywords (SKIP, STOP, APPROVE)
        # APPROVE keywords must be at START of response
        test_cases = [
            ("skip", "ok_continue", False),
            ("SKIP", "ok_continue", False),
            ("stop", "all_complete", True),
            ("STOP", "all_complete", True),
            ("approve", "ok_continue", False),
            ("APPROVE", "ok_continue", False),
            ("APPROVE:", "ok_continue", False),  # With colon
            ("APPROVE: looks good", "ok_continue", False),  # With text after
            ("proceed", "ok_continue", False),
            ("PROCEED", "ok_continue", False),
        ]
        
        for user_input_text, expected_verdict, should_stop in test_cases:
            mock_result.clear()
            user_input = {"q1": user_input_text}
            
            trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == expected_verdict, f"Failed for input: {user_input_text}"
            if should_stop:
                assert mock_result.get("should_stop") is True

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_approve_keyword_not_matched_in_middle(self, mock_route_llm, mock_state, mock_result):
        """Should NOT match approve keywords when they appear in the middle of guidance."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "retry_generate_code"
            result["reviewer_feedback"] = "User guidance"
        mock_route_llm.side_effect = set_verdict
        
        # These should NOT trigger fast path - they have keywords in the middle
        test_cases = [
            "I accept that we should use 66 meV",
            "Please proceed with fixing the gamma value",
            "I approve of using the corrected values",
        ]
        
        for user_input_text in test_cases:
            mock_result.clear()
            mock_route_llm.reset_mock()
            user_input = {"q1": user_input_text}
            
            trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
            
            # Should have called LLM (not fast path)
            mock_route_llm.assert_called_once(), f"Fast path incorrectly triggered for: {user_input_text}"

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_state_not_mutated(self, mock_route_llm, mock_state, mock_result):
        """Should not mutate the input state dict."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
        mock_route_llm.side_effect = set_verdict
        
        original_state = mock_state.copy()
        user_input = {"q1": "PROVIDE_GUIDANCE: Test"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Verify state was not mutated
        assert mock_state == original_state

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_guidance_routes_to_llm_with_full_text(self, mock_route_llm, mock_state, mock_result):
        """Should route guidance to LLM with full text."""
        captured_args = {}
        def set_verdict(state, result, user_responses, stage_id):
            captured_args["user_responses"] = user_responses
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: This is detailed guidance with multiple words"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "PROVIDE_GUIDANCE: This is detailed guidance with multiple words"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # LLM should have been called with the full user responses
        mock_route_llm.assert_called_once()
        assert captured_args["user_responses"] == user_input

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_guidance_with_special_characters_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route guidance with special characters to LLM."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: Use λ = 500nm and ε = -5.0+0.3i"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "PROVIDE_GUIDANCE: Use λ = 500nm and ε = -5.0+0.3i"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        mock_route_llm.assert_called_once()

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_natural_physics_response_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route natural physics response to LLM (bug report scenario)."""
        # This is the exact scenario from the bug report - user responds naturally
        # to a question about TDBC linewidth without using any keyword
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "retry_generate_code"  # LLM decides code needs to change
            result["reviewer_feedback"] = "User guidance: Assume the paper meant FWHM and then adjust γX"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "Assume the paper meant FWHM and then adjust γX. For secondary issue: you are supposed to have Palik Al data available to you"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # LLM should have been called for free-form response
        mock_route_llm.assert_called_once()
        # LLM should decide to regenerate code (not just ok_continue)
        assert mock_result["supervisor_verdict"] == "retry_generate_code"

    @patch("src.agents.supervision.trigger_handlers._route_with_llm")
    def test_response_with_numbers_routes_to_llm(self, mock_route_llm, mock_state, mock_result):
        """Should route short numeric response to LLM."""
        def set_verdict(state, result, user_responses, stage_id):
            result["supervisor_verdict"] = "ok_continue"
            result["reviewer_feedback"] = "User guidance: 2"
        mock_route_llm.side_effect = set_verdict
        
        user_input = {"q1": "2"}  # User selecting option 2
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # LLM should have been called
        mock_route_llm.assert_called_once()

    def test_whitespace_only_asks_clarification(self, mock_state, mock_result):
        """Should ask for clarification on whitespace-only response."""
        user_input = {"q1": "   \n\t  "}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        # Whitespace-only should be treated as empty
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result


class TestLLMRoutingIntegration:
    """Tests for LLM-based routing decision making."""

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_for_code_fix_request(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM should return retry_generate_code when user requests parameter fix."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "retry_generate_code",
            "summary": "User requested gamma parameter fix, regenerating code."
        }
        
        # Simulate the exact bug scenario from logs
        mock_state["last_node_before_ask_user"] = "physics_check"
        mock_state["ask_user_trigger"] = "reviewer_escalation"
        mock_state["pending_user_questions"] = ["Which TDBC linewidth: 16 meV or 66 meV?"]
        
        user_input = {"q1": "fix γ so that the result aligns with the paper"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage0")
        
        # LLM should have been called
        mock_llm.assert_called_once()
        # Should return retry_generate_code, NOT ok_continue
        assert mock_result["supervisor_verdict"] == "retry_generate_code"
        # Should set reviewer_feedback for code generator
        assert "reviewer_feedback" in mock_result
        assert "fix γ" in mock_result["reviewer_feedback"]

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_for_design_change(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM should return retry_design when user requests model change."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "retry_design",
            "summary": "User requested different material model."
        }
        
        mock_state["last_node_before_ask_user"] = "physics_check"
        user_input = {"q1": "use a Drude-Lorentz model instead of Lorentzian"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage0")
        
        assert mock_result["supervisor_verdict"] == "retry_design"
        assert "reviewer_feedback" in mock_result

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_sets_planner_feedback_for_replan(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM should set planner_feedback when returning replan_needed."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "replan_needed",
            "summary": "User wants to restructure the plan."
        }
        
        user_input = {"q1": "let's add a new stage for polarization study"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        # Should set planner_feedback, not reviewer_feedback
        assert "planner_feedback" in mock_result

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_handles_backtrack(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM should set backtrack_decision when returning backtrack_to_stage."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "backtrack_to_stage",
            "summary": "User wants to redo stage 0.",
            "backtrack_decision": {
                "target_stage_id": "stage0",
                "reason": "Material was wrong"
            }
        }
        
        user_input = {"q1": "go back to stage 0 and fix the material"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage2")
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in mock_result
        assert mock_result["backtrack_decision"]["target_stage_id"] == "stage0"

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_backtrack_without_target_asks_user(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM returning backtrack_to_stage without target should ask user for clarification."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "backtrack_to_stage",
            "summary": "User wants to go back.",
            # No backtrack_decision provided
        }
        
        # Set up progress with completed stages
        mock_state["progress"] = {
            "stages": [
                {"stage_id": "stage0", "status": "completed_success"},
                {"stage_id": "stage1", "status": "completed_partial"},
                {"stage_id": "stage2", "status": "in_progress"},
            ]
        }
        
        user_input = {"q1": "go back to an earlier stage"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage2")
        
        # Should override verdict to ask_user
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert "Which stage should we backtrack to" in mock_result["pending_user_questions"][0]
        # Should list completed stages
        assert "stage0" in mock_result["pending_user_questions"][0]
        assert "stage1" in mock_result["pending_user_questions"][0]
        assert mock_result.get("ask_user_trigger") is not None

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_backtrack_with_empty_target_asks_user(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM returning backtrack_to_stage with empty target should ask user."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "backtrack_to_stage",
            "summary": "User wants to go back.",
            "backtrack_decision": {
                "target_stage_id": "",  # Empty target
                "reason": "Some reason"
            }
        }
        
        user_input = {"q1": "go back"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage2")
        
        # Should override verdict to ask_user
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_failure_defaults_to_ok_continue(self, mock_prompt, mock_llm, mock_state, mock_result):
        """Should default to ok_continue with guidance if LLM fails."""
        mock_prompt.return_value = "system prompt"
        mock_llm.side_effect = Exception("LLM unavailable")
        
        user_input = {"q1": "fix the gamma parameter"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage0")
        
        # Should default to ok_continue
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Should still set reviewer_feedback with guidance
        assert "reviewer_feedback" in mock_result
        assert "fix the gamma parameter" in mock_result["reviewer_feedback"]

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_ask_user_sets_question(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM returning ask_user should set pending_user_questions."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "ask_user",
            "summary": "Response unclear, need clarification.",
            "user_question": "Did you mean to fix the gamma parameter or change the material model?"
        }
        
        user_input = {"q1": "change it"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage0")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        assert "gamma parameter" in mock_result["pending_user_questions"][0]
        assert mock_result.get("ask_user_trigger") is not None

    @patch("src.agents.supervision.trigger_handlers.call_agent_with_metrics")
    @patch("src.agents.supervision.trigger_handlers.build_agent_prompt")
    def test_llm_routing_ask_user_fallback_question(self, mock_prompt, mock_llm, mock_state, mock_result):
        """LLM returning ask_user without user_question should use fallback."""
        mock_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "verdict": "ask_user",
            "summary": "Response unclear."
            # No user_question provided
        }
        
        user_input = {"q1": "maybe"}
        
        trigger_handlers.handle_reviewer_escalation(mock_state, mock_result, user_input, "stage0")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        # Should contain fallback text with original response
        assert "unclear" in mock_result["pending_user_questions"][0].lower()
        assert "maybe" in mock_result["pending_user_questions"][0]

