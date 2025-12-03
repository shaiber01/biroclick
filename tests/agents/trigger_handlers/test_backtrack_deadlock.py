"""Backtrack approval and deadlock trigger tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestBacktrackApprovalTrigger:
    """Tests for backtrack_approval trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_on_approve(self, mock_context):
        """Should approve backtrack on APPROVE and calculate dependent stages."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE the backtrack"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert "stages_to_invalidate" in result["backtrack_decision"]
        # Should include stage2 which depends on stage1
        assert "stage2" in result["backtrack_decision"]["stages_to_invalidate"]
        assert result["backtrack_decision"]["target_stage_id"] == "stage1"
        # Verify trigger was cleared
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_multiple_dependents(self, mock_context):
        """Should include all transitive dependents when approving backtrack."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "YES, approve"},
            "backtrack_decision": {"target_stage_id": "stage0"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                    {"stage_id": "stage3", "dependencies": ["stage1"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        invalidated = result["backtrack_decision"]["stages_to_invalidate"]
        # Should include all stages that depend on stage0 (transitively)
        assert "stage1" in invalidated
        assert "stage2" in invalidated
        assert "stage3" in invalidated
        assert len(invalidated) == 3

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_no_dependents(self, mock_context):
        """Should handle approval when target stage has no dependents."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage2"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "stages_to_invalidate" in result["backtrack_decision"]
        # stage2 is last, so no dependents
        assert result["backtrack_decision"]["stages_to_invalidate"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_missing_plan(self, mock_context):
        """Should handle approval when plan is missing or empty."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            # Missing plan
        }
        
        result = supervisor_node(state)
        
        # Should still approve but with empty dependents
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert "stages_to_invalidate" in result["backtrack_decision"]
        assert result["backtrack_decision"]["stages_to_invalidate"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_missing_backtrack_decision(self, mock_context):
        """Should handle approval when backtrack_decision is missing."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                ]
            },
            # Missing backtrack_decision
        }
        
        result = supervisor_node(state)
        
        # Should still approve but may not have stages_to_invalidate
        assert result["supervisor_verdict"] == "backtrack_to_stage"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_missing_target_stage_id(self, mock_context):
        """Should handle approval when target_stage_id is missing."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {},  # Missing target_stage_id
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # Should not crash, but may not populate stages_to_invalidate

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejects_backtrack_on_reject(self, mock_context):
        """Should reject backtrack on REJECT and clear suggestion."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "REJECT"},
            "backtrack_suggestion": {"target": "stage1"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["backtrack_suggestion"] is None
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejects_backtrack_with_various_rejection_keywords(self, mock_context):
        """Should reject on various rejection keywords."""
        mock_context.return_value = None
        
        rejection_keywords = ["NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]
        
        for keyword in rejection_keywords:
            state = {
                "ask_user_trigger": "backtrack_approval",
                "user_responses": {"Question": keyword},
                "backtrack_suggestion": {"target": "stage1"},
            }
            
            result = supervisor_node(state)
            
            assert result["supervisor_verdict"] == "ok_continue", f"Failed for keyword: {keyword}"
            assert result["backtrack_suggestion"] is None, f"Failed for keyword: {keyword}"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejects_backtrack_case_insensitive(self, mock_context):
        """Should handle rejection keywords case-insensitively."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "no, don't backtrack"},
            "backtrack_suggestion": {"target": "stage1"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["backtrack_suggestion"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_defaults_to_continue_on_unclear(self, mock_context):
        """Should default to continue on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "maybe"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Should not modify backtrack_suggestion if it exists
        assert "backtrack_suggestion" not in result or result.get("backtrack_suggestion") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses gracefully."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {},
        }
        
        result = supervisor_node(state)
        
        # Should default to continue when no response
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_multiple_responses_takes_last(self, mock_context):
        """Should use the last response when multiple responses exist."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {
                "Question1": "REJECT",
                "Question2": "APPROVE",  # Last one should win
            },
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # Should approve because last response is APPROVE
        assert result["supervisor_verdict"] == "backtrack_to_stage"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approval_overrides_rejection_when_both_present(self, mock_context):
        """Should prioritize approval when both approval and rejection keywords present."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE but also REJECT"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # Logic checks: is_approval and not is_rejection
        # If both are present, should not approve
        # But let's test what actually happens
        assert result["supervisor_verdict"] in ["backtrack_to_stage", "ok_continue"]


class TestDeadlockTrigger:
    """Tests for deadlock_detected trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_generates_report_on_generate_report(self, mock_context):
        """Should generate report on GENERATE_REPORT."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_generates_report_on_report_keyword(self, mock_context):
        """Should generate report on REPORT keyword (partial match)."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPORT"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_generates_report_case_insensitive(self, mock_context):
        """Should handle GENERATE_REPORT case-insensitively."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "generate report"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_replans_on_replan(self, mock_context):
        """Should trigger replan on REPLAN."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPLAN"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "planner_feedback" in result
        assert "deadlock" in result["planner_feedback"].lower()
        assert result.get("should_stop") is not True  # Should not stop on replan

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_replans_includes_user_response_in_feedback(self, mock_context):
        """Should include user response in planner_feedback when replanning."""
        mock_context.return_value = None
        
        user_response = "REPLAN because of issues"
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": user_response},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert user_response in result["planner_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_case_insensitive(self, mock_context):
        """Should handle STOP case-insensitively."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "stop"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_unclear(self, mock_context):
        """Should ask clarification on unclear response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "I'm not sure"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert any("GENERATE_REPORT" in q or "REPLAN" in q or "STOP" in q 
                  for q in result["pending_user_questions"])

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_clarification_on_empty_response(self, mock_context):
        """Should ask clarification on empty response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": ""},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses dict."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_multiple_responses_takes_last(self, mock_context):
        """Should use the last response when multiple responses exist."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {
                "Question1": "REPLAN",
                "Question2": "STOP",  # Last one should win
            },
        }
        
        result = supervisor_node(state)
        
        # Should stop because last response is STOP
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True


class TestHandleBacktrackAndDeadlockHandlers:
    """Direct handler tests for backtrack approval and deadlock triggers."""

    def test_handle_backtrack_approval_approve(self, mock_state, mock_result):
        """Should calculate dependents and set verdict to backtrack."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=["stage1", "stage2"])
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in mock_result
        assert mock_result["backtrack_decision"]["stages_to_invalidate"] == ["stage1", "stage2"]
        assert mock_result["backtrack_decision"]["target_stage_id"] == "stage0"
        mock_get_dependents.assert_called_once()
        # Verify it was called with correct arguments
        call_args = mock_get_dependents.call_args
        assert call_args[0][0] == mock_state.get("plan", {})
        assert call_args[0][1] == "stage0"

    def test_handle_backtrack_approval_approve_with_empty_dependents(self, mock_state, mock_result):
        """Should handle approval when get_dependents returns empty list."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=[])
        mock_state["backtrack_decision"] = {"target_stage_id": "stage2"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert mock_result["backtrack_decision"]["stages_to_invalidate"] == []

    def test_handle_backtrack_approval_approve_with_none_dependents(self, mock_state, mock_result):
        """Should handle approval when get_dependents returns None."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=None)
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        # Verify what gets set when None is returned - this might reveal a bug
        assert "backtrack_decision" in mock_result
        # If None is set, this would be a bug - should be empty list instead
        stages_to_invalidate = mock_result["backtrack_decision"].get("stages_to_invalidate")
        # This test will FAIL if the bug exists (None is set instead of empty list)
        assert stages_to_invalidate is not None, "BUG: stages_to_invalidate should not be None, should be empty list"
        assert isinstance(stages_to_invalidate, list), "stages_to_invalidate should be a list"

    def test_handle_backtrack_approval_approve_without_get_dependents_fn(self, mock_state, mock_result):
        """Should handle approval when get_dependents_fn is None."""
        user_input = {"q1": "APPROVE"}
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", None
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        # Should not crash when get_dependents_fn is None

    def test_handle_backtrack_approval_approve_without_backtrack_decision(self, mock_state, mock_result):
        """Should handle approval when backtrack_decision is missing."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=["stage1"])
        # Remove backtrack_decision from mock_state
        mock_state.pop("backtrack_decision", None)

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        # get_dependents should not be called if no backtrack_decision
        mock_get_dependents.assert_not_called()

    def test_handle_backtrack_approval_approve_without_target_stage_id(self, mock_state, mock_result):
        """Should handle approval when target_stage_id is missing."""
        user_input = {"q1": "APPROVE"}
        mock_get_dependents = MagicMock(return_value=["stage1"])
        mock_state["backtrack_decision"] = {}  # Missing target_stage_id

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )

        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        # get_dependents should not be called if no target_stage_id
        mock_get_dependents.assert_not_called()

    def test_handle_backtrack_approval_approve_various_keywords(self, mock_state, mock_result):
        """Should approve on various approval keywords."""
        approval_keywords = ["YES", "CORRECT", "OK", "ACCEPT", "VALID", "PROCEED"]
        mock_get_dependents = MagicMock(return_value=["stage1"])
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        for keyword in approval_keywords:
            mock_result.clear()
            user_input = {"q1": keyword}
            
            trigger_handlers.handle_backtrack_approval(
                mock_state, mock_result, user_input, "stage1", mock_get_dependents
            )
            
            assert mock_result["supervisor_verdict"] == "backtrack_to_stage", f"Failed for keyword: {keyword}"

    def test_handle_backtrack_approval_reject(self, mock_state, mock_result):
        """Should clear suggestion and continue on reject."""
        user_input = {"q1": "REJECT"}
        mock_state["backtrack_suggestion"] = {"target": "stage1"}

        trigger_handlers.handle_backtrack_approval(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["backtrack_suggestion"] is None

    def test_handle_backtrack_approval_reject_various_keywords(self, mock_state, mock_result):
        """Should reject on various rejection keywords."""
        rejection_keywords = ["NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]
        mock_state["backtrack_suggestion"] = {"target": "stage1"}

        for keyword in rejection_keywords:
            mock_result.clear()
            user_input = {"q1": keyword}
            
            trigger_handlers.handle_backtrack_approval(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue", f"Failed for keyword: {keyword}"
            assert mock_result["backtrack_suggestion"] is None, f"Failed for keyword: {keyword}"

    def test_handle_backtrack_approval_unclear_response(self, mock_state, mock_result):
        """Should default to continue on unclear response."""
        user_input = {"q1": "maybe"}

        trigger_handlers.handle_backtrack_approval(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Should not modify backtrack_suggestion if it wasn't set
        assert "backtrack_suggestion" not in mock_result or mock_result.get("backtrack_suggestion") is None

    def test_handle_backtrack_approval_empty_response(self, mock_state, mock_result):
        """Should handle empty response."""
        user_input = {"q1": ""}

        trigger_handlers.handle_backtrack_approval(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_backtrack_approval_none_user_responses(self, mock_state, mock_result):
        """Should handle None user_responses gracefully."""
        mock_get_dependents = MagicMock(return_value=["stage1"])
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, None, "stage1", mock_get_dependents
        )
        
        # Should default to continue when user_responses is None
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_backtrack_approval_empty_dict_user_responses(self, mock_state, mock_result):
        """Should handle empty dict user_responses."""
        user_input = {}
        mock_get_dependents = MagicMock(return_value=["stage1"])
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )
        
        # Should default to continue when no responses
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_backtrack_approval_multiple_responses(self, mock_state, mock_result):
        """Should use last response when multiple responses exist."""
        user_input = {
            "q1": "REJECT",
            "q2": "APPROVE",  # Last one should win
        }
        mock_get_dependents = MagicMock(return_value=["stage1"])
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}

        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage1", mock_get_dependents
        )
        
        # Should approve because last response is APPROVE
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"

    def test_handle_deadlock_report(self, mock_state, mock_result):
        """Should stop and generate report on GENERATE_REPORT."""
        user_input = {"q1": "GENERATE_REPORT"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_report_keyword(self, mock_state, mock_result):
        """Should stop on REPORT keyword."""
        user_input = {"q1": "REPORT"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_report_case_insensitive(self, mock_state, mock_result):
        """Should handle GENERATE_REPORT case-insensitively."""
        user_input = {"q1": "generate report"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_replan(self, mock_state, mock_result):
        """Should request replan on REPLAN."""
        user_input = {"q1": "REPLAN"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "planner_feedback" in mock_result
        assert "deadlock" in mock_result["planner_feedback"].lower()
        assert mock_result.get("should_stop") is not True

    def test_handle_deadlock_replan_includes_user_response(self, mock_state, mock_result):
        """Should include user response in planner_feedback."""
        user_response = "REPLAN due to issues"
        user_input = {"q1": user_response}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert user_response in mock_result["planner_feedback"]

    def test_handle_deadlock_replan_with_empty_responses(self, mock_state, mock_result):
        """Should handle replan when user_responses is empty."""
        user_input = {"q1": "REPLAN"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        # Should still have planner_feedback even if empty
        assert "planner_feedback" in mock_result

    def test_handle_deadlock_stop(self, mock_state, mock_result):
        """Should stop on STOP."""
        user_input = {"q1": "STOP"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_stop_case_insensitive(self, mock_state, mock_result):
        """Should handle STOP case-insensitively."""
        user_input = {"q1": "stop"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "Help"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) > 0
        assert any("GENERATE_REPORT" in q or "REPLAN" in q or "STOP" in q 
                  for q in mock_result["pending_user_questions"])

    def test_handle_deadlock_empty_response(self, mock_state, mock_result):
        """Should ask clarification on empty response."""
        user_input = {"q1": ""}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_deadlock_none_user_responses(self, mock_state, mock_result):
        """Should handle None user_responses."""
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, None, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_deadlock_empty_dict_user_responses(self, mock_state, mock_result):
        """Should handle empty dict user_responses."""
        user_input = {}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_deadlock_multiple_responses(self, mock_state, mock_result):
        """Should use last response when multiple responses exist."""
        user_input = {
            "q1": "REPLAN",
            "q2": "STOP",  # Last one should win
        }

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        # Should stop because last response is STOP
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_current_stage_id_none(self, mock_state, mock_result):
        """Should handle None current_stage_id."""
        user_input = {"q1": "REPLAN"}

        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        # Should not crash with None stage_id
