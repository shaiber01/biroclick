"""replan_limit trigger tests."""

from unittest.mock import patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from tests.agents.trigger_handlers.shared import result_has_value


class TestReplanLimitTrigger:
    """Tests for replan_limit trigger handling via supervisor_node."""

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
        assert result.get("ask_user_trigger") is None
        assert result.get("should_stop") is not True
        assert "replan_count" not in result

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
        assert result.get("supervisor_feedback") == "Plan force-accepted by user."
        assert result.get("ask_user_trigger") is None
        assert result.get("should_stop") is not True

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
        assert "planner_feedback" in result
        assert "Focus on single wavelength first" in result["planner_feedback"]
        assert result["planner_feedback"].startswith("User guidance:")
        assert result.get("ask_user_trigger") is None

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
        assert result.get("ask_user_trigger") is None


class TestHandleReplanLimit:
    """Direct handler tests for replan limit."""

    def test_handle_replan_limit_force(self, mock_state, mock_result):
        """Test FORCE keyword handling."""
        user_input = {"q1": "FORCE accept"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("supervisor_feedback") == "Plan force-accepted by user."
        assert mock_result.get("should_stop") is not True
        assert "replan_count" not in mock_result
        assert "planner_feedback" not in mock_result
        assert "pending_user_questions" not in mock_result

    def test_handle_replan_limit_accept(self, mock_state, mock_result):
        """Test ACCEPT keyword handling."""
        user_input = {"q1": "ACCEPT"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("supervisor_feedback") == "Plan force-accepted by user."
        assert mock_result.get("should_stop") is not True

    def test_handle_replan_limit_guidance(self, mock_state, mock_result):
        """Test GUIDANCE keyword handling."""
        user_input = {"q1": "GUIDANCE: Try this."}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_with_guidance"
        assert mock_result["replan_count"] == 0
        assert "planner_feedback" in mock_result
        assert mock_result["planner_feedback"] == "User guidance: Try this."
        assert mock_result.get("should_stop") is not True
        assert "supervisor_feedback" not in mock_result

    def test_handle_replan_limit_stop(self, mock_state, mock_result):
        """Test STOP keyword handling."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True
        assert "replan_count" not in mock_result
        assert "planner_feedback" not in mock_result

    def test_handle_replan_limit_unknown(self, mock_state, mock_result):
        """Test unknown response handling."""
        user_input = {"q1": "Unknown"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert isinstance(mock_result["pending_user_questions"], list)
        assert len(mock_result["pending_user_questions"]) == 1
        # Clarification message shows display names from user_options.py
        assert "APPROVE_PLAN" in mock_result["pending_user_questions"][0]
        assert "GUIDANCE" in mock_result["pending_user_questions"][0]
        assert "STOP" in mock_result["pending_user_questions"][0]
        assert mock_result.get("should_stop") is not True
        assert "replan_count" not in mock_result

    def test_handle_replan_limit_case_insensitive_force(self, mock_state, mock_result):
        """Test case-insensitive FORCE matching (whole word only)."""
        test_cases = [
            "force",
            "Force",
            "FORCE",
            "FoRcE",
            "FORCE_ACCEPT",  # Compound keyword
        ]
        
        for response in test_cases:
            mock_result.clear()
            user_input = {"q1": response}
            trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue", \
                f"Failed for response: {response}"
            assert mock_result.get("supervisor_feedback") == "Plan force-accepted by user."

    def test_handle_replan_limit_case_insensitive_accept(self, mock_state, mock_result):
        """Test case-insensitive ACCEPT matching (whole word only)."""
        test_cases = [
            "accept",
            "Accept",
            "ACCEPT",
            "AcCePt",
        ]
        
        for response in test_cases:
            mock_result.clear()
            user_input = {"q1": response}
            trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue", \
                f"Failed for response: {response}"
            assert mock_result.get("supervisor_feedback") == "Plan force-accepted by user."

    def test_handle_replan_limit_case_insensitive_guidance(self, mock_state, mock_result):
        """Test case-insensitive GUIDANCE matching (whole word only)."""
        test_cases = [
            "guidance",
            "Guidance",
            "GUIDANCE",
            "GuIdAnCe",
            "guidance: test",
        ]
        
        for response in test_cases:
            mock_result.clear()
            user_input = {"q1": response}
            trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "replan_with_guidance", \
                f"Failed for response: {response}"
            assert mock_result["replan_count"] == 0
            assert "planner_feedback" in mock_result

    def test_handle_replan_limit_case_insensitive_stop(self, mock_state, mock_result):
        """Test case-insensitive STOP matching (whole word only)."""
        test_cases = [
            "stop",
            "Stop",
            "STOP",
            "StOp",
        ]
        
        for response in test_cases:
            mock_result.clear()
            user_input = {"q1": response}
            trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "all_complete", \
                f"Failed for response: {response}"
            assert mock_result["should_stop"] is True

    def test_handle_replan_limit_uses_last_response(self, mock_state, mock_result):
        """Test that handler uses the last response when multiple responses exist."""
        user_input = {
            "q1": "FORCE",
            "q2": "GUIDANCE: First guidance",
            "q3": "STOP",
        }
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use last response (STOP)
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_replan_limit_guidance_extracts_raw_response(self, mock_state, mock_result):
        """Test that GUIDANCE extracts the raw response correctly."""
        user_input = {"q1": "GUIDANCE: This is detailed guidance with multiple words"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["planner_feedback"] == "User guidance: This is detailed guidance with multiple words"

    def test_handle_replan_limit_guidance_with_special_characters(self, mock_state, mock_result):
        """Test GUIDANCE with special characters in response."""
        user_input = {"q1": "GUIDANCE: Use @#$%^&*() and <tags>"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["replan_count"] == 0
        assert "@#$%^&*()" in mock_result["planner_feedback"]
        assert "<tags>" in mock_result["planner_feedback"]

    def test_handle_replan_limit_guidance_empty_raw_response(self, mock_state, mock_result):
        """Test GUIDANCE with empty raw response."""
        user_input = {"q1": "GUIDANCE:"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_with_guidance"
        assert mock_result["replan_count"] == 0
        assert mock_result["planner_feedback"] == "User guidance: "

    def test_handle_replan_limit_guidance_empty_user_responses(self, mock_state, mock_result):
        """Test GUIDANCE with empty user_responses dict."""
        user_input = {}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Empty dict should result in empty string from parse_user_response
        # This should not match GUIDANCE, so should go to else branch
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_guidance_none_user_responses(self, mock_state, mock_result):
        """Test GUIDANCE with None user_responses."""
        # parse_user_response should handle None and return empty string
        trigger_handlers.handle_replan_limit(mock_state, mock_result, None, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_guidance_whitespace_only(self, mock_state, mock_result):
        """Test GUIDANCE with whitespace-only response."""
        user_input = {"q1": "   "}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_guidance_empty_string(self, mock_state, mock_result):
        """Test GUIDANCE with empty string response."""
        user_input = {"q1": ""}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_guidance_none_value_in_responses(self, mock_state, mock_result):
        """Test GUIDANCE with None value in user_responses."""
        user_input = {"q1": None}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # parse_user_response converts None to string "None"
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_guidance_last_response_empty_string(self, mock_state, mock_result):
        """Test GUIDANCE when last response is empty string but earlier ones exist."""
        user_input = {
            "q1": "GUIDANCE: First guidance",
            "q2": "",
        }
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use last response (empty string), so should not match GUIDANCE
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_keyword_precedence_force_before_guidance(self, mock_state, mock_result):
        """Test keyword precedence: FORCE should be checked before GUIDANCE."""
        user_input = {"q1": "FORCE GUIDANCE: This should be forced"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # FORCE comes first in the if-elif chain, so should match FORCE
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("supervisor_feedback") == "Plan force-accepted by user."
        assert "planner_feedback" not in mock_result

    def test_handle_replan_limit_keyword_precedence_force_before_stop(self, mock_state, mock_result):
        """Test keyword precedence: FORCE should be checked before STOP."""
        user_input = {"q1": "FORCE STOP"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # FORCE comes first, so should match FORCE
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("should_stop") is not True

    def test_handle_replan_limit_keyword_precedence_accept_before_guidance(self, mock_state, mock_result):
        """Test keyword precedence: ACCEPT should be checked before GUIDANCE."""
        user_input = {"q1": "ACCEPT GUIDANCE: This should be accepted"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # ACCEPT comes first in the if condition, so should match ACCEPT
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("supervisor_feedback") == "Plan force-accepted by user."

    def test_handle_replan_limit_keyword_precedence_guidance_before_stop(self, mock_state, mock_result):
        """Test keyword precedence: GUIDANCE should be checked before STOP."""
        user_input = {"q1": "GUIDANCE: Stop this"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # GUIDANCE comes first, so should match GUIDANCE
        assert mock_result["supervisor_verdict"] == "replan_with_guidance"
        assert mock_result.get("should_stop") is not True

    def test_handle_replan_limit_resets_replan_count_on_guidance(self, mock_state, mock_result):
        """Test that replan_count is reset to 0 on GUIDANCE even if not in result."""
        # Set replan_count in result first
        mock_result["replan_count"] = 5
        user_input = {"q1": "GUIDANCE: Test"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["replan_count"] == 0

    def test_handle_replan_limit_resets_replan_count_on_guidance_when_not_set(self, mock_state, mock_result):
        """Test that replan_count is set to 0 on GUIDANCE even if not previously set."""
        # Don't set replan_count in result
        user_input = {"q1": "GUIDANCE: Test"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["replan_count"] == 0

    def test_handle_replan_limit_does_not_set_replan_count_on_force(self, mock_state, mock_result):
        """Test that replan_count is not set/modified on FORCE."""
        user_input = {"q1": "FORCE"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert "replan_count" not in mock_result

    def test_handle_replan_limit_does_not_set_replan_count_on_stop(self, mock_state, mock_result):
        """Test that replan_count is not set/modified on STOP."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert "replan_count" not in mock_result

    def test_handle_replan_limit_state_not_mutated(self, mock_state, mock_result):
        """Test that state dict is not mutated by handler."""
        original_state = mock_state.copy()
        user_input = {"q1": "FORCE"}
        
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # State should not be modified
        assert mock_state == original_state

    def test_handle_replan_limit_result_is_mutated(self, mock_state, mock_result):
        """Test that result dict is mutated by handler."""
        user_input = {"q1": "FORCE"}
        original_result_keys = set(mock_result.keys())
        
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Result should be modified
        assert "supervisor_verdict" in mock_result
        assert "supervisor_feedback" in mock_result
        assert len(mock_result.keys()) > len(original_result_keys)

    def test_handle_replan_limit_none_stage_id(self, mock_state, mock_result):
        """Test handler with None stage_id."""
        user_input = {"q1": "FORCE"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_replan_limit_empty_string_stage_id(self, mock_state, mock_result):
        """Test handler with empty string stage_id."""
        user_input = {"q1": "FORCE"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_replan_limit_guidance_with_multiple_responses_uses_last(self, mock_state, mock_result):
        """Test GUIDANCE with multiple responses uses the last one for planner_feedback."""
        user_input = {
            "q1": "GUIDANCE: First guidance",
            "q2": "GUIDANCE: Second guidance",
            "q3": "GUIDANCE: Third guidance",
        }
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use last response for planner_feedback
        assert mock_result["planner_feedback"] == "User guidance: Third guidance"

    def test_handle_replan_limit_guidance_with_empty_middle_response(self, mock_state, mock_result):
        """Test GUIDANCE with empty response in middle, should use last response."""
        user_input = {
            "q1": "GUIDANCE: First guidance",
            "q2": "",
            "q3": "GUIDANCE: Last guidance",
        }
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Should use last response (q3), which contains GUIDANCE
        assert mock_result["supervisor_verdict"] == "replan_with_guidance"
        assert mock_result["planner_feedback"] == "User guidance: Last guidance"

    def test_handle_replan_limit_partial_match_force_no_match(self, mock_state, mock_result):
        """Test partial match: words containing FORCE should NOT match (word boundary)."""
        user_input = {"q1": "FORCEFUL action"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: FORCEFUL does not match FORCE
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_partial_match_accept_no_match(self, mock_state, mock_result):
        """Test partial match: words containing ACCEPT should NOT match (word boundary)."""
        user_input = {"q1": "ACCEPTED proposal"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: ACCEPTED does not match ACCEPT
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_partial_match_guidance(self, mock_state, mock_result):
        """Test partial match: GUIDANCE as a keyword still works."""
        user_input = {"q1": "GUIDANCE: test"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # GUIDANCE is the keyword itself, not a partial match
        assert mock_result["supervisor_verdict"] == "replan_with_guidance"
        assert mock_result["replan_count"] == 0

    def test_handle_replan_limit_partial_match_stop_no_match(self, mock_state, mock_result):
        """Test partial match: words containing STOP should NOT match (word boundary)."""
        user_input = {"q1": "STOPPED process"}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        # Word boundary matching: STOPPED does not match STOP
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_replan_limit_unknown_empty_string(self, mock_state, mock_result):
        """Test unknown response with empty string."""
        user_input = {"q1": ""}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_replan_limit_unknown_whitespace_only(self, mock_state, mock_result):
        """Test unknown response with whitespace only."""
        user_input = {"q1": "   \t\n   "}
        trigger_handlers.handle_replan_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
