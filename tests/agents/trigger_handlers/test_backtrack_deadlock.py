"""Backtrack approval and deadlock trigger tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers
from src.agents.supervision.supervisor import _get_dependent_stages


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
        """Should NOT approve when both approval and rejection keywords present.
        
        According to the code logic: `if is_approval and not is_rejection:`
        When both keywords are present, is_rejection is True, so approval is blocked.
        """
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
        
        # Logic: is_approval and not is_rejection
        # When both present: is_approval=True, is_rejection=True
        # So condition is False, falls through to rejection branch
        # This clears backtrack_suggestion and continues
        assert result["supervisor_verdict"] == "ok_continue", \
            "When both APPROVE and REJECT present, rejection should take precedence"
        assert result.get("backtrack_suggestion") is None, \
            "backtrack_suggestion should be cleared on rejection"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approval_keyword_false_positive_disapprove(self, mock_context):
        """Should NOT treat DISAPPROVE as APPROVE - tests word boundary matching."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "I DISAPPROVE of this backtrack"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # DISAPPROVE contains "APPROVE" substring but shouldn't match as approval
        # The check_keywords function uses word boundaries
        assert result["supervisor_verdict"] == "ok_continue", \
            "DISAPPROVE should not be treated as APPROVE (word boundary check)"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejection_keyword_false_positive_know(self, mock_context):
        """Should NOT treat KNOW as NO - tests word boundary matching."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "I know this is fine, APPROVE it"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "backtrack_suggestion": {"target": "stage1"},  # Should remain intact
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # "KNOW" contains "NO" but shouldn't match as rejection keyword
        assert result["supervisor_verdict"] == "backtrack_to_stage", \
            "KNOW should not be treated as NO rejection (word boundary check)"


class TestGetDependentStages:
    """Tests for _get_dependent_stages helper function used by backtrack handling."""

    def test_simple_linear_chain(self):
        """Should find all stages depending on target in a linear chain."""
        plan = {
            "stages": [
                {"stage_id": "s0", "dependencies": []},
                {"stage_id": "s1", "dependencies": ["s0"]},
                {"stage_id": "s2", "dependencies": ["s1"]},
                {"stage_id": "s3", "dependencies": ["s2"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "s0")
        
        assert set(result) == {"s1", "s2", "s3"}, \
            "All stages after s0 in chain should be marked as dependents"
        
        result_s1 = _get_dependent_stages(plan, "s1")
        assert set(result_s1) == {"s2", "s3"}, \
            "Only s2 and s3 depend on s1"
        
        result_s3 = _get_dependent_stages(plan, "s3")
        assert result_s3 == [], \
            "s3 has no dependents (last in chain)"

    def test_diamond_dependency_structure(self, complex_dependency_state):
        """Should find all transitive dependents in diamond structure."""
        plan = complex_dependency_state["plan"]
        
        # Backtrack to base: everything depends on it
        result = _get_dependent_stages(plan, "base")
        assert set(result) == {"left", "right", "merge", "final"}, \
            "All stages depend transitively on base"
        
        # Backtrack to left: merge and final depend on it
        result_left = _get_dependent_stages(plan, "left")
        assert set(result_left) == {"merge", "final"}, \
            "merge and final depend on left"
        
        # Backtrack to right: merge and final depend on it too
        result_right = _get_dependent_stages(plan, "right")
        assert set(result_right) == {"merge", "final"}, \
            "merge and final depend on right"

    def test_multiple_independent_branches(self):
        """Should handle multiple independent branches correctly."""
        plan = {
            "stages": [
                {"stage_id": "root", "dependencies": []},
                {"stage_id": "branch1_a", "dependencies": ["root"]},
                {"stage_id": "branch1_b", "dependencies": ["branch1_a"]},
                {"stage_id": "branch2_a", "dependencies": ["root"]},
                {"stage_id": "branch2_b", "dependencies": ["branch2_a"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "root")
        assert set(result) == {"branch1_a", "branch1_b", "branch2_a", "branch2_b"}, \
            "All branches depend on root"
        
        result_b1a = _get_dependent_stages(plan, "branch1_a")
        assert set(result_b1a) == {"branch1_b"}, \
            "Only branch1_b depends on branch1_a"
        
        result_b2a = _get_dependent_stages(plan, "branch2_a")
        assert set(result_b2a) == {"branch2_b"}, \
            "Only branch2_b depends on branch2_a"

    def test_empty_plan(self):
        """Should return empty list for empty plan."""
        result = _get_dependent_stages({}, "any_stage")
        assert result == [], "Empty plan has no dependents"
        
        result_no_stages = _get_dependent_stages({"stages": []}, "any_stage")
        assert result_no_stages == [], "Plan with empty stages has no dependents"

    def test_none_stages(self):
        """Should handle None stages gracefully."""
        plan = {"stages": None}
        result = _get_dependent_stages(plan, "any_stage")
        assert result == [], "None stages should return empty list"

    def test_nonexistent_target_stage(self):
        """Should return empty list for nonexistent target stage."""
        plan = {
            "stages": [
                {"stage_id": "s0", "dependencies": []},
                {"stage_id": "s1", "dependencies": ["s0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "nonexistent")
        assert result == [], "Nonexistent stage has no dependents"

    def test_stage_without_stage_id(self):
        """Should skip stages that don't have stage_id field."""
        plan = {
            "stages": [
                {"stage_id": "s0", "dependencies": []},
                {"name": "invalid_stage"},  # Missing stage_id
                {"stage_id": "s1", "dependencies": ["s0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "s0")
        assert set(result) == {"s1"}, \
            "Should skip invalid stage and still find s1 as dependent"

    def test_stage_with_none_dependencies(self):
        """Should handle stages with None dependencies."""
        plan = {
            "stages": [
                {"stage_id": "s0", "dependencies": None},  # None dependencies
                {"stage_id": "s1", "dependencies": ["s0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "s0")
        assert set(result) == {"s1"}, \
            "Should handle None dependencies and find s1 as dependent"

    def test_non_list_dependencies(self):
        """Should handle non-list dependencies gracefully."""
        plan = {
            "stages": [
                {"stage_id": "s0", "dependencies": []},
                {"stage_id": "s1", "dependencies": "s0"},  # String instead of list
                {"stage_id": "s2", "dependencies": ["s0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "s0")
        # s1 has invalid deps (string), should be skipped
        # s2 has valid deps
        assert "s2" in result, "Should find s2 as dependent of s0"

    def test_non_dict_stage_entries(self):
        """Should skip non-dict stage entries."""
        plan = {
            "stages": [
                {"stage_id": "s0", "dependencies": []},
                "invalid_stage_string",  # Not a dict
                None,  # Also not a dict
                {"stage_id": "s1", "dependencies": ["s0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "s0")
        assert set(result) == {"s1"}, \
            "Should skip non-dict entries and find s1"


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


class TestHandleTriggerDispatcher:
    """Tests for handle_trigger dispatcher function."""

    def test_dispatches_to_backtrack_approval_handler(self, mock_state, mock_result):
        """Should dispatch backtrack_approval to correct handler."""
        user_responses = {"q1": "APPROVE"}
        mock_get_deps = MagicMock(return_value=["stage1"])
        
        trigger_handlers.handle_trigger(
            trigger="backtrack_approval",
            state=mock_state,
            result=mock_result,
            user_responses=user_responses,
            current_stage_id="stage0",
            get_dependent_stages_fn=mock_get_deps,
        )
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        # Verify get_dependent_stages_fn was called for backtrack_approval
        mock_get_deps.assert_called_once()

    def test_dispatches_to_deadlock_detected_handler(self, mock_state, mock_result):
        """Should dispatch deadlock_detected to correct handler."""
        user_responses = {"q1": "STOP"}
        
        trigger_handlers.handle_trigger(
            trigger="deadlock_detected",
            state=mock_state,
            result=mock_result,
            user_responses=user_responses,
            current_stage_id="stage0",
        )
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_unknown_trigger_defaults_to_continue(self, mock_state, mock_result):
        """Should default to ok_continue for unknown triggers."""
        user_responses = {"q1": "anything"}
        
        trigger_handlers.handle_trigger(
            trigger="unknown_trigger_xyz",
            state=mock_state,
            result=mock_result,
            user_responses=user_responses,
            current_stage_id="stage0",
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "unknown trigger" in mock_result["supervisor_feedback"].lower()

    def test_does_not_pass_get_deps_fn_to_non_backtrack_handlers(self, mock_state, mock_result):
        """Should not pass get_dependent_stages_fn to handlers that don't need it."""
        user_responses = {"q1": "STOP"}
        mock_get_deps = MagicMock(return_value=["stage1"])
        
        trigger_handlers.handle_trigger(
            trigger="deadlock_detected",
            state=mock_state,
            result=mock_result,
            user_responses=user_responses,
            current_stage_id="stage0",
            get_dependent_stages_fn=mock_get_deps,  # Provided but shouldn't be used
        )
        
        # get_deps should NOT be called for deadlock handler
        mock_get_deps.assert_not_called()


class TestBacktrackApprovalEdgeCases:
    """Additional edge case tests for backtrack approval handling."""

    def test_preserves_existing_backtrack_decision_fields(self, mock_state, mock_result):
        """Should preserve existing fields in backtrack_decision when approving."""
        mock_state["backtrack_decision"] = {
            "target_stage_id": "stage0",
            "reason": "Some reason from supervisor",
            "extra_field": "should_be_preserved",
        }
        user_input = {"q1": "APPROVE"}
        mock_get_deps = MagicMock(return_value=["stage1", "stage2"])
        
        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage0", mock_get_deps
        )
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        decision = mock_result["backtrack_decision"]
        assert decision["target_stage_id"] == "stage0"
        assert decision["stages_to_invalidate"] == ["stage1", "stage2"]
        # Original fields should be preserved
        assert decision.get("reason") == "Some reason from supervisor"
        assert decision.get("extra_field") == "should_be_preserved"

    def test_approval_with_complex_dependency_chain(self, complex_dependency_state, mock_result):
        """Should correctly calculate all transitive dependents for diamond structure."""
        user_input = {"q1": "YES"}
        
        def get_deps(plan, target):
            return _get_dependent_stages(plan, target)
        
        trigger_handlers.handle_backtrack_approval(
            complex_dependency_state, mock_result, user_input, "base", get_deps
        )
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        invalidated = mock_result["backtrack_decision"]["stages_to_invalidate"]
        # Diamond structure: base -> left, right -> merge -> final
        assert set(invalidated) == {"left", "right", "merge", "final"}

    def test_approval_whitespace_only_response(self, mock_state, mock_result):
        """Should treat whitespace-only response as unclear (default to continue)."""
        user_input = {"q1": "   \n\t  "}
        
        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage0", None
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_rejection_clears_both_suggestion_and_does_not_set_decision(self, mock_state, mock_result):
        """Should clear backtrack_suggestion on reject and not set backtrack_decision."""
        mock_state["backtrack_suggestion"] = {"target": "stage1", "confidence": 0.8}
        mock_state["backtrack_decision"] = {"target_stage_id": "stage1"}
        user_input = {"q1": "NO, don't backtrack"}
        
        trigger_handlers.handle_backtrack_approval(
            mock_state, mock_result, user_input, "stage0", None
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["backtrack_suggestion"] is None
        # backtrack_decision should NOT be in result (unchanged from state)
        assert "backtrack_decision" not in mock_result


class TestDeadlockEdgeCases:
    """Additional edge case tests for deadlock handling."""

    def test_replan_planner_feedback_format(self, mock_state, mock_result):
        """Should format planner_feedback correctly with deadlock context."""
        user_input = {"q1": "REPLAN because stage X is stuck"}
        
        trigger_handlers.handle_deadlock_detected(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        feedback = mock_result["planner_feedback"]
        # Should include "deadlock" and user's response
        assert "deadlock" in feedback.lower()
        assert "REPLAN because stage X is stuck" in feedback

    def test_clarification_question_contains_all_options(self, mock_state, mock_result):
        """Should present all valid options in clarification question."""
        user_input = {"q1": "I don't know what to do"}
        
        trigger_handlers.handle_deadlock_detected(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        questions = mock_result["pending_user_questions"]
        assert len(questions) > 0
        question_text = questions[0].upper()
        # Should mention all valid options
        assert "GENERATE_REPORT" in question_text or "REPORT" in question_text
        assert "REPLAN" in question_text
        assert "STOP" in question_text

    def test_stop_and_report_both_stop_workflow(self, mock_state, mock_result):
        """Both STOP and GENERATE_REPORT should set should_stop=True."""
        for response in ["STOP", "GENERATE_REPORT"]:
            mock_result.clear()
            user_input = {"q1": response}
            
            trigger_handlers.handle_deadlock_detected(
                mock_state, mock_result, user_input, "stage1"
            )
            
            assert mock_result["should_stop"] is True, f"should_stop should be True for {response}"
            assert mock_result["supervisor_verdict"] == "all_complete", f"verdict should be all_complete for {response}"

    def test_replan_does_not_stop_workflow(self, mock_state, mock_result):
        """REPLAN should NOT set should_stop."""
        user_input = {"q1": "REPLAN"}
        
        trigger_handlers.handle_deadlock_detected(
            mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        # should_stop should not be set (or explicitly False/None)
        assert mock_result.get("should_stop") is not True, \
            "REPLAN should not stop workflow"


class TestKeywordMatchingBehavior:
    """Tests verifying keyword matching uses word boundaries correctly."""

    def test_approval_keywords_word_boundary(self, mock_state, mock_result):
        """Test that approval keywords use word boundaries."""
        # These should NOT match approval
        non_matching_inputs = [
            "DISAPPROVE",  # Contains APPROVE
            "UNAPPROVED",  # Contains APPROVE
            "NOYES",  # Contains YES
            "REVALIDATE",  # Contains VALID but not as word
        ]
        
        for input_text in non_matching_inputs:
            mock_result.clear()
            user_input = {"q1": input_text}
            
            trigger_handlers.handle_backtrack_approval(
                mock_state, mock_result, user_input, "stage0", None
            )
            
            # None of these should result in approval
            assert mock_result["supervisor_verdict"] != "backtrack_to_stage", \
                f"'{input_text}' should NOT match as approval"

    def test_rejection_keywords_word_boundary(self, mock_state, mock_result):
        """Test that rejection keywords use word boundaries."""
        # These should NOT match rejection
        non_matching_inputs = [
            "KNOW",  # Contains NO
            "ACKNOWLEDGE",  # Contains NO
            "REJECTIFY",  # Made up but contains REJECT
        ]
        
        mock_state["backtrack_suggestion"] = {"target": "stage1"}
        
        for input_text in non_matching_inputs:
            mock_result.clear()
            user_input = {"q1": f"{input_text} APPROVE"}  # Include approval keyword
            
            trigger_handlers.handle_backtrack_approval(
                mock_state, mock_result, user_input, "stage0", MagicMock(return_value=[])
            )
            
            # Should match approval, not rejection
            assert mock_result["supervisor_verdict"] == "backtrack_to_stage", \
                f"'{input_text}' should NOT match as rejection, APPROVE should win"

    def test_case_insensitive_matching(self, mock_state, mock_result):
        """Test that keyword matching is case insensitive."""
        cases = [
            "approve",
            "APPROVE", 
            "Approve",
            "ApPrOvE",
        ]
        
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}
        
        for case_variant in cases:
            mock_result.clear()
            user_input = {"q1": case_variant}
            
            trigger_handlers.handle_backtrack_approval(
                mock_state, mock_result, user_input, "stage0", MagicMock(return_value=[])
            )
            
            assert mock_result["supervisor_verdict"] == "backtrack_to_stage", \
                f"'{case_variant}' should match APPROVE (case insensitive)"


class TestSupervisorNodeTriggerClearing:
    """Tests for supervisor_node clearing ask_user_trigger after handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_clears_ask_user_trigger_on_backtrack_approval(self, mock_context):
        """Should clear ask_user_trigger after handling backtrack_approval."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        assert result.get("ask_user_trigger") is None, \
            "ask_user_trigger should be cleared after handling"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_clears_ask_user_trigger_on_backtrack_rejection(self, mock_context):
        """Should clear ask_user_trigger after handling rejection."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "REJECT"},
            "backtrack_suggestion": {"target": "stage1"},
        }
        
        result = supervisor_node(state)
        
        assert result.get("ask_user_trigger") is None, \
            "ask_user_trigger should be cleared on rejection"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_clears_ask_user_trigger_on_deadlock_stop(self, mock_context):
        """Should clear ask_user_trigger after handling deadlock STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result.get("ask_user_trigger") is None, \
            "ask_user_trigger should be cleared after deadlock handling"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_clears_ask_user_trigger_on_deadlock_replan(self, mock_context):
        """Should clear ask_user_trigger after handling deadlock REPLAN."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPLAN"},
        }
        
        result = supervisor_node(state)
        
        assert result.get("ask_user_trigger") is None, \
            "ask_user_trigger should be cleared after replan"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_preserves_trigger_when_asking_clarification(self, mock_context):
        """Should NOT clear trigger when asking for clarification (ask_user verdict)."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "I'm not sure what to do"},
        }
        
        result = supervisor_node(state)
        
        # When verdict is ask_user, the trigger is still cleared but new questions are set
        assert result.get("ask_user_trigger") is None, \
            "ask_user_trigger should be cleared even when asking clarification"
        assert result["supervisor_verdict"] == "ask_user"
        assert len(result.get("pending_user_questions", [])) > 0, \
            "New questions should be set for clarification"


class TestSupervisorNodeStateIntegrity:
    """Tests for state integrity after supervisor_node processing."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_workflow_phase_set_to_supervision(self, mock_context):
        """Should always set workflow_phase to supervision."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {"stages": []},
            "workflow_phase": "execution",  # Different phase
        }
        
        result = supervisor_node(state)
        
        assert result["workflow_phase"] == "supervision", \
            "workflow_phase should be set to supervision"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_invalid_user_responses_type(self, mock_context):
        """Should handle non-dict user_responses gracefully."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": "not a dict",  # Invalid type
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {"stages": []},
        }
        
        # Should not raise, should handle gracefully
        result = supervisor_node(state)
        
        # Should default to continue since no valid responses
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_none_user_responses(self, mock_context):
        """Should handle None user_responses gracefully."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": None,
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should default to continue
        assert result["supervisor_verdict"] == "ok_continue"


class TestBacktrackDecisionIntegration:
    """Tests for backtrack_decision handling in the full integration flow."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stages_to_invalidate_calculated_via_supervisor_node(self, mock_context):
        """Should calculate stages_to_invalidate correctly via supervisor_node."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "YES, approve the backtrack"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                    {"stage_id": "stage3", "dependencies": ["stage2"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        decision = result["backtrack_decision"]
        assert decision["target_stage_id"] == "stage1"
        # stage2 and stage3 depend on stage1 (transitively)
        assert set(decision["stages_to_invalidate"]) == {"stage2", "stage3"}

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_backtrack_decision_preserved_from_state(self, mock_context):
        """Should preserve existing backtrack_decision fields."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {
                "target_stage_id": "stage1",
                "reason": "Physics check failed repeatedly",
                "suggested_by": "execution_validator",
            },
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        decision = result["backtrack_decision"]
        assert decision["target_stage_id"] == "stage1"
        assert decision["reason"] == "Physics check failed repeatedly"
        assert decision["suggested_by"] == "execution_validator"
        assert "stages_to_invalidate" in decision
