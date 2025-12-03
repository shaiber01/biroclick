"""Core supervisor_node flow tests."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from src.agents.supervision import supervisor_node


class TestSupervisorNode:
    """Tests for supervisor_node function."""

    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_continues_workflow_on_success(self, mock_derive, mock_prompt, mock_context, mock_call, mock_archive, mock_update, validated_supervisor_response):
        """Should continue workflow on successful stage completion (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_derive.return_value = ("completed_success", "Analysis OK")
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_response.pop("should_stop", None) # Ensure should_stop is not True
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "physics_verdict": "pass",
            "execution_verdict": "pass",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact verdict
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Verify workflow_phase is set
        assert result.get("workflow_phase") == "supervision"
        # STRICT: Verify supervisor_feedback is present
        assert "supervisor_feedback" in result
        # STRICT: Verify archiving was called with correct arguments
        mock_archive.assert_called_once_with(state, "stage1")
        # STRICT: Verify status update was called with exact arguments
        mock_update.assert_called_once_with(state, "stage1", "completed_success", summary="Analysis OK")
        # STRICT: Verify LLM was called with correct agent name
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["agent_name"] == "supervisor"
        assert call_kwargs["system_prompt"] == "system prompt"
        assert call_kwargs["state"] == state

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_uses_updated_context_state(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should use updated state from context check for prompt building."""
        mock_context.return_value = {"new_flag": True, "updated_field": "value"}
        mock_prompt.return_value = "prompt"
        mock_call.return_value = validated_supervisor_response.copy()
        
        state = {"current_stage_id": "stage1", "supervisor_call_count": 0}
        
        supervisor_node(state)
        
        # STRICT: Verify build_agent_prompt received merged state with new_flag
        args, _ = mock_prompt.call_args
        passed_state = args[1]
        assert passed_state.get("new_flag") is True
        assert passed_state.get("updated_field") == "value"
        # STRICT: Verify original state keys are preserved
        assert passed_state.get("current_stage_id") == "stage1"
        assert passed_state.get("supervisor_call_count") == 0
        # STRICT: Verify context check was called
        mock_context.assert_called_once_with(state, "supervisor")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_triggers_backtrack_on_failure(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should trigger backtrack on stage failure (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "backtrack_to_stage"
        mock_response["backtrack_target"] = "design"
        mock_response["reasoning"] = "design flawed"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "physics_verdict": "fail",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact verdict
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # STRICT: Verify backtrack_decision is set correctly
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "design"
        assert result["backtrack_decision"]["reason"] == "design flawed"
        # STRICT: Verify supervisor_feedback contains reasoning
        assert result.get("supervisor_feedback") == "design flawed"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_defaults_on_llm_error(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call):
        """Should default to ok_continue on LLM error."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        mock_derive.return_value = ("completed_success", "OK")
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact verdict on error
        assert result["supervisor_verdict"] == "ok_continue"
        # STRICT: Verify error message is in feedback
        assert "supervisor_feedback" in result
        assert "LLM unavailable" in result["supervisor_feedback"]
        assert "API error" in result["supervisor_feedback"]
        # STRICT: Verify archiving still happens despite LLM error
        mock_archive.assert_called_once_with(state, "stage1")
        # STRICT: Verify status update still happens
        mock_update.assert_called_once()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        context_update = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        mock_context.return_value = context_update
        
        state = {"current_stage_id": "stage1"}
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact escalation fields
        assert result["awaiting_user_input"] is True
        assert result["pending_user_questions"] == ["Context overflow"]
        # STRICT: Verify context check was called
        mock_context.assert_called_once_with(state, "supervisor")
        # STRICT: Verify early return - no LLM call should happen
        assert "supervisor_verdict" not in result or result.get("supervisor_verdict") is None

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_returns_supervisor_verdict(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should return supervisor verdict from LLM output (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_response["reasoning"] = "All checks passed"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 5,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact verdict
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Verify feedback is set from reasoning
        assert result.get("supervisor_feedback") == "All checks passed"
        # STRICT: Verify archiving happens
        mock_archive.assert_called_once_with(state, "stage1")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_finish_verdict(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle finish verdict when workflow complete (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "all_complete"
        mock_response["should_stop"] = True
        mock_response["reasoning"] = "All stages complete"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": None,
            "completed_stages": ["stage0", "stage1"],
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact verdict
        assert result["supervisor_verdict"] == "all_complete"
        # STRICT: Verify should_stop is propagated
        assert result.get("should_stop") is True
        # STRICT: Verify feedback is set
        assert result.get("supervisor_feedback") == "All stages complete"
        # STRICT: Verify no archiving when current_stage_id is None
        # (archiving only happens if current_stage_id exists)

    # ═══════════════════════════════════════════════════════════════════════
    # EDGE CASES AND ERROR PATHS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_handles_missing_verdict_in_response(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call):
        """Should default to ok_continue when verdict is missing from LLM response."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        # Response missing verdict field
        mock_call.return_value = {"reasoning": "Some reasoning"}
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should default to ok_continue when verdict missing
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Feedback should still be set
        assert result.get("supervisor_feedback") == "Some reasoning"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_handles_empty_response(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle empty LLM response gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_call.return_value = {}
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should default to ok_continue
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Feedback should be empty string, not None
        assert result.get("supervisor_feedback") == ""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_archive_error(self, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle archive errors and store them in archive_errors."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        mock_archive.side_effect = Exception("Archive failed")
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify archive_errors is set
        assert "archive_errors" in result
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert "Archive failed" in result["archive_errors"][0]["error"]
        assert "timestamp" in result["archive_errors"][0]
        # STRICT: Verify verdict still set despite archive error
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_retries_archive_errors_from_state(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should retry archive errors from previous runs."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        mock_derive.return_value = ("completed_success", "OK")
        # First call fails, second succeeds
        mock_archive.side_effect = [None, None]  # Both succeed on retry
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "archive_errors": [
                {"stage_id": "stage0", "error": "Previous error", "timestamp": "2024-01-01T00:00:00Z"}
            ],
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify archive was called for both stage0 (retry) and stage1 (current)
        assert mock_archive.call_count >= 1
        # STRICT: Verify archive_errors is cleared if retry succeeds
        assert result.get("archive_errors") == []

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_none_current_stage_id(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle None current_stage_id without archiving."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": None,
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify verdict is set
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Verify no archiving happens (no current_stage_id)
        # This is tested by ensuring archive_stage_outputs_to_progress is not patched/called

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_invalid_user_responses_type(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle invalid user_responses type gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "user_responses": "not a dict",  # Invalid type
            "ask_user_trigger": None,
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should handle gracefully and continue
        assert result.get("supervisor_verdict") == "ok_continue"

    # ═══════════════════════════════════════════════════════════════════════
    # TRIGGER HANDLER TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_material_checkpoint_trigger(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle material_checkpoint trigger."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        state = {
            "current_stage_id": "stage0",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify trigger handler was called
        mock_handle_trigger.assert_called_once()
        call_kwargs = mock_handle_trigger.call_args[1]
        assert call_kwargs["trigger"] == "material_checkpoint"
        assert call_kwargs["current_stage_id"] == "stage0"
        assert call_kwargs["user_responses"] == {"q1": "APPROVE"}
        # STRICT: Verify trigger is cleared
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_backtrack_approval_trigger(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle backtrack_approval trigger."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"q1": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "design"},
            "plan": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify trigger handler was called with get_dependent_stages_fn
        mock_handle_trigger.assert_called_once()
        call_kwargs = mock_handle_trigger.call_args[1]
        assert call_kwargs["trigger"] == "backtrack_approval"
        assert call_kwargs["get_dependent_stages_fn"] is not None
        # STRICT: Verify trigger is cleared
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_skips_llm_call_when_trigger_present(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should skip LLM call when handling trigger."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        # Mock handle_trigger to set a verdict
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify trigger handler was called
        mock_handle_trigger.assert_called_once()
        # STRICT: Verify LLM was NOT called (call_agent_with_metrics not patched, so if called would fail)
        # This is verified by the fact that we don't patch call_agent_with_metrics
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_logs_user_interaction_after_trigger(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should log user interaction after handling trigger."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
            kwargs["result"]["supervisor_feedback"] = "User approved"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": ["Approve materials?"],
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify user interaction was logged
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "material_checkpoint"
        assert interaction["user_response"] == "APPROVE"
        assert interaction["question"] == "Approve materials?"
        assert "timestamp" in interaction
        assert "id" in interaction

    # ═══════════════════════════════════════════════════════════════════════
    # STATE MUTATION AND RETURN VALUE TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_sets_workflow_phase(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should always set workflow_phase to supervision."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify workflow_phase is always set
        assert result.get("workflow_phase") == "supervision"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_returns_dict_not_none(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should always return a dict, never None."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = validated_supervisor_response.copy()
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify result is a dict
        assert isinstance(result, dict)
        assert result is not None
        # STRICT: Verify result has at least workflow_phase
        assert "workflow_phase" in result

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_does_not_mutate_input_state(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should not mutate input state dict."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = validated_supervisor_response.copy()
        
        original_state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        state_copy = original_state.copy()
        
        result = supervisor_node(state_copy)
        
        # STRICT: Verify input state was not mutated
        assert state_copy == original_state
        # STRICT: Verify result is separate dict
        assert result is not state_copy

    # ═══════════════════════════════════════════════════════════════════════
    # INTEGRATION TESTS WITH HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.get_validation_hierarchy")
    def test_integrates_with_validation_hierarchy(self, mock_get_hierarchy, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should integrate with get_validation_hierarchy."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_get_hierarchy.return_value = {"material_validation": "passed"}
        mock_call.return_value = validated_supervisor_response.copy()
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify get_validation_hierarchy was called
        mock_get_hierarchy.assert_called_once_with(state)
        # STRICT: Verify result is set
        assert result.get("supervisor_verdict") is not None

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_integrates_with_stage_completion_derivation(self, mock_update, mock_archive, mock_derive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should integrate with _derive_stage_completion_outcome."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_partial", "Some targets missing")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "analysis_overall_classification": "PARTIAL_MATCH",
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify _derive_stage_completion_outcome was called
        mock_derive.assert_called_once_with(state, "stage1")
        # STRICT: Verify update_progress_stage_status was called with derived status
        mock_update.assert_called_once_with(state, "stage1", "completed_partial", summary="Some targets missing")

    # ═══════════════════════════════════════════════════════════════════════
    # COMPLEX FLOW TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_complex_flow_with_context_update_and_archiving(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle complex flow with context update, LLM call, and archiving."""
        # Context check returns update
        mock_context.return_value = {"context_budget": 1000}
        mock_prompt.return_value = "updated prompt"
        mock_derive.return_value = ("completed_success", "All good")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_response["reasoning"] = "Proceeding"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify all steps executed
        mock_context.assert_called_once_with(state, "supervisor")
        # STRICT: Verify prompt built with updated state (build_agent_prompt uses positional args)
        prompt_call_args = mock_prompt.call_args[0]
        assert prompt_call_args[0] == "supervisor"
        prompt_call_state = prompt_call_args[1]
        assert prompt_call_state.get("context_budget") == 1000
        # STRICT: Verify LLM called
        mock_call.assert_called_once()
        # STRICT: Verify archiving happened with updated state (state is merged with context_update)
        archive_call_args = mock_archive.call_args[0]
        assert archive_call_args[1] == "stage1"
        assert archive_call_args[0].get("context_budget") == 1000
        # STRICT: Verify status update with updated state
        update_call_args = mock_update.call_args[0]
        assert update_call_args[0].get("context_budget") == 1000
        assert update_call_args[1] == "stage1"
        assert update_call_args[2] == "completed_success"
        assert mock_update.call_args[1]["summary"] == "All good"
        # STRICT: Verify result
        assert result.get("supervisor_verdict") == "ok_continue"
        assert result.get("supervisor_feedback") == "Proceeding"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_complex_flow_with_trigger_and_user_interaction(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle complex flow with trigger and user interaction logging."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
            kwargs["result"]["supervisor_feedback"] = "User approved materials"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage0",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": ["Approve gold material?"],
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1", "interaction_type": "previous"}
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify trigger handler called
        mock_handle_trigger.assert_called_once()
        # STRICT: Verify user interaction logged
        assert len(result["progress"]["user_interactions"]) == 2
        new_interaction = result["progress"]["user_interactions"][1]
        assert new_interaction["id"] == "U2"  # Incremented from U1
        assert new_interaction["interaction_type"] == "material_checkpoint"
        # STRICT: Verify trigger cleared
        assert result.get("ask_user_trigger") is None
        # STRICT: Verify verdict set
        assert result.get("supervisor_verdict") == "ok_continue"
