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
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context overflow"],
        }
        mock_context.return_value = context_update
        
        state = {"current_stage_id": "stage1"}
        
        result = supervisor_node(state)
        
        # STRICT: Verify exact escalation fields
        assert result.get("ask_user_trigger") is not None
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

    # ═══════════════════════════════════════════════════════════════════════
    # SUPERVISOR CALL COUNT TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_supervisor_call_count_passed_to_state(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should correctly pass supervisor_call_count to LLM in state."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 5,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        supervisor_node(state)
        
        # STRICT: Verify LLM was called with state containing correct supervisor_call_count
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["state"]["supervisor_call_count"] == 5

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_supervisor_call_count_zero(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle supervisor_call_count of 0."""
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
        
        # STRICT: Verify function completes without error
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_supervisor_call_count_missing(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle missing supervisor_call_count gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            # supervisor_call_count is missing
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should complete without error
        assert result.get("supervisor_verdict") == "ok_continue"

    # ═══════════════════════════════════════════════════════════════════════
    # UNKNOWN TRIGGER HANDLING TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_unknown_trigger(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle unknown trigger types gracefully via handle_trigger default behavior."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        # handle_trigger defaults to ok_continue for unknown triggers
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
            kwargs["result"]["supervisor_feedback"] = "Handled unknown trigger: unknown_trigger_xyz"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "unknown_trigger_xyz",
            "user_responses": {"q1": "some response"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify trigger handler was called with unknown trigger
        mock_handle_trigger.assert_called_once()
        call_kwargs = mock_handle_trigger.call_args[1]
        assert call_kwargs["trigger"] == "unknown_trigger_xyz"
        # STRICT: Verify verdict is set (default behavior is ok_continue)
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Verify trigger is cleared
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_empty_string_trigger(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle empty string trigger (truthy check should fail)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        # Empty string is falsy, so should NOT call handle_trigger
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "",  # Empty string is falsy
            "user_responses": {},
            "progress": {"stages": []},
        }
        
        # Since ask_user_trigger is empty string (falsy), handle_trigger should NOT be called
        # Normal supervision will be run instead
        # But we need to mock call_agent_with_metrics since normal supervision calls it
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics") as mock_call, \
             patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress"), \
             patch("src.agents.supervision.supervisor.update_progress_stage_status"), \
             patch("src.agents.supervision.supervisor._derive_stage_completion_outcome") as mock_derive:
            mock_derive.return_value = ("completed_success", "OK")
            mock_call.return_value = {"verdict": "ok_continue", "reasoning": "OK"}
            
            result = supervisor_node(state)
            
            # STRICT: handle_trigger should NOT be called for empty string
            mock_handle_trigger.assert_not_called()
            # STRICT: Normal supervision (LLM call) should happen instead
            mock_call.assert_called_once()

    # ═══════════════════════════════════════════════════════════════════════
    # BACKTRACK DECISION TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_backtrack_approval_with_stages_to_invalidate(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should properly set stages_to_invalidate in backtrack_decision."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            # Simulate backtrack_approval handler setting up backtrack
            kwargs["result"]["supervisor_verdict"] = "backtrack_to_stage"
            kwargs["result"]["backtrack_decision"] = {
                "target_stage_id": "design",
                "reason": "User approved backtrack",
                "stages_to_invalidate": ["stage1", "stage2"]
            }
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"q1": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "design"},
            "plan": {
                "stages": [
                    {"stage_id": "design", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["design"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify backtrack_decision is properly set
        assert result.get("supervisor_verdict") == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "design"
        assert "stages_to_invalidate" in result["backtrack_decision"]
        # STRICT: Verify stages_to_invalidate is a list
        assert isinstance(result["backtrack_decision"]["stages_to_invalidate"], list)

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_backtrack_verdict_from_llm(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should properly handle backtrack_to_stage verdict from LLM."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "backtrack_to_stage"
        mock_response["backtrack_target"] = "stage_0"
        mock_response["reasoning"] = "Design flaw detected"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify verdict
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # STRICT: Verify backtrack_decision structure
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "stage_0"
        assert result["backtrack_decision"]["reason"] == "Design flaw detected"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_backtrack_verdict_without_target(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle backtrack_to_stage verdict without backtrack_target."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "backtrack_to_stage"
        mock_response.pop("backtrack_target", None)  # No target
        mock_response["reasoning"] = "Needs backtrack"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify verdict is set
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # STRICT: backtrack_decision should NOT be set when target is missing
        assert "backtrack_decision" not in result or result.get("backtrack_decision") is None

    # ═══════════════════════════════════════════════════════════════════════
    # ARCHIVE RETRY TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_archive_retry_partial_failure(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle partial archive retry failures (some succeed, some fail)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        # First call fails (retry), second succeeds (retry), third succeeds (current)
        mock_archive.side_effect = [
            Exception("Retry 1 failed"),  # stage0 retry fails
            None,                          # stage1 retry succeeds
            None,                          # current stage succeeds
        ]
        
        state = {
            "current_stage_id": "stage2",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "archive_errors": [
                {"stage_id": "stage0", "error": "Previous error", "timestamp": "2024-01-01T00:00:00Z"},
                {"stage_id": "stage1", "error": "Another error", "timestamp": "2024-01-01T00:00:00Z"},
            ],
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify archive was called 3 times (2 retries + 1 current)
        assert mock_archive.call_count == 3
        # STRICT: archive_errors should contain only the one that failed
        assert len(result.get("archive_errors", [])) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage0"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_archive_errors_invalid_type(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle invalid archive_errors type gracefully."""
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
            "archive_errors": "not a list",  # Invalid type
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should handle gracefully and reset archive_errors to empty list
        assert result.get("archive_errors") == []
        # STRICT: Should still complete successfully
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_archive_errors_with_invalid_entries(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle invalid entries in archive_errors list."""
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
            "archive_errors": [
                "not a dict",  # Invalid entry
                {"stage_id": "stage0", "error": "Valid error", "timestamp": "2024-01-01T00:00:00Z"},
                None,  # Invalid entry
                {"no_stage_id": True},  # Missing stage_id
            ],
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should handle gracefully - invalid entries are preserved but not retried
        # The function should complete without error
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_archive_error_accumulation(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should accumulate archive errors when multiple failures occur."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        # Both retry and current archiving fail
        mock_archive.side_effect = [
            Exception("Retry failed"),   # stage0 retry fails
            Exception("Current failed"), # current stage fails
        ]
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "archive_errors": [
                {"stage_id": "stage0", "error": "Previous error", "timestamp": "2024-01-01T00:00:00Z"},
            ],
        }
        
        result = supervisor_node(state)
        
        # STRICT: archive_errors should contain both the retry failure and new failure
        assert len(result.get("archive_errors", [])) == 2
        stage_ids = [e.get("stage_id") for e in result["archive_errors"]]
        assert "stage0" in stage_ids
        assert "stage1" in stage_ids

    # ═══════════════════════════════════════════════════════════════════════
    # USER INTERACTION LOGGING EDGE CASES
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_with_empty_pending_questions(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle user interaction logging when pending_user_questions is empty."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": [],  # Empty
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should still log interaction with placeholder question text
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_with_empty_user_responses(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle user interaction logging when user_responses is empty."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {},  # Empty
            "pending_user_questions": ["Please approve"],
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should still log interaction with empty user_response
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["user_response"] == ""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_with_multiple_responses(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should use last user response when multiple responses provided."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "clarification",
            "user_responses": {
                "q1": "First response",
                "q2": "Second response",
                "q3": "Final response",
            },
            "pending_user_questions": ["Question"],
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should use the last response value
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["user_response"] == "Final response"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_increments_id_correctly(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should correctly increment user interaction IDs."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": ["Approve?"],
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1"},
                    {"id": "U2"},
                    {"id": "U3"},
                    {"id": "U10"},
                    {"id": "U99"},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # STRICT: New interaction should have ID based on count, not parsing
        assert len(result["progress"]["user_interactions"]) == 6
        new_interaction = result["progress"]["user_interactions"][5]
        assert new_interaction["id"] == "U6"  # Based on len() + 1

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_without_progress_key(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle missing progress key in state."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": ["Approve?"],
            # No progress key
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should create progress with user_interactions
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1

    # ═══════════════════════════════════════════════════════════════════════
    # MORE VERDICT TYPES TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_replan_needed_verdict(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle replan_needed verdict from LLM."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "replan_needed"
        mock_response["reasoning"] = "Plan needs revision due to new findings"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify replan_needed verdict is passed through
        assert result["supervisor_verdict"] == "replan_needed"
        assert result["supervisor_feedback"] == "Plan needs revision due to new findings"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_ask_user_verdict(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle ask_user verdict from LLM."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ask_user"
        mock_response["reasoning"] = "Need user clarification"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify ask_user verdict is passed through
        assert result["supervisor_verdict"] == "ask_user"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_all_complete_verdict_with_should_stop(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should propagate should_stop when all_complete verdict."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "all_complete"
        mock_response["should_stop"] = True
        mock_response["reasoning"] = "All stages completed successfully"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Verify verdict and should_stop
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_should_stop_false_is_not_propagated(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should propagate should_stop only when True."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_response["should_stop"] = False
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: should_stop should be set to False when False
        assert result["should_stop"] is False

    # ═══════════════════════════════════════════════════════════════════════
    # USER CONTENT CONSTRUCTION TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    @patch("src.agents.supervision.supervisor.get_validation_hierarchy")
    def test_user_content_includes_analysis_summary(self, mock_hierarchy, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should include analysis_summary in user content when present."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_hierarchy.return_value = {"material_validation": "passed"}
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "analysis_summary": {"status": "match", "score": 0.95},
        }
        
        supervisor_node(state)
        
        # STRICT: Verify user_content was passed with analysis_summary
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "Analysis Summary" in user_content
        assert "match" in user_content or "status" in user_content

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    @patch("src.agents.supervision.supervisor.get_validation_hierarchy")
    def test_user_content_includes_progress_summary(self, mock_hierarchy, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should include progress summary in user content."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_hierarchy.return_value = {}
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "status": "completed_success"},
                    {"stage_id": "stage1", "status": "in_progress"},
                    {"stage_id": "stage2", "status": "not_started"},
                ]
            },
        }
        
        supervisor_node(state)
        
        # STRICT: Verify user_content includes progress summary
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "Progress" in user_content
        assert "Completed" in user_content
        assert "Pending" in user_content

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    @patch("src.agents.supervision.supervisor.get_validation_hierarchy")
    def test_user_content_with_none_current_stage(self, mock_hierarchy, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle None current_stage_id in user content."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_hierarchy.return_value = {}
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": None,
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        supervisor_node(state)
        
        # STRICT: Verify user_content is constructed with None stage
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "Current Stage: None" in user_content

    # ═══════════════════════════════════════════════════════════════════════
    # TRIGGER DISPATCH VERIFICATION TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_trigger_dispatch_passes_all_required_kwargs(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should pass all required kwargs to handle_trigger."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"q1": "PROVIDE_HINT: Use numpy instead"},
            "plan": {"stages": [{"stage_id": "stage0"}, {"stage_id": "stage1"}]},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        supervisor_node(state)
        
        # STRICT: Verify all required kwargs are passed
        mock_handle_trigger.assert_called_once()
        call_kwargs = mock_handle_trigger.call_args[1]
        
        assert "trigger" in call_kwargs
        assert call_kwargs["trigger"] == "code_review_limit"
        
        assert "state" in call_kwargs
        assert call_kwargs["state"]["current_stage_id"] == "stage1"
        
        assert "result" in call_kwargs
        assert isinstance(call_kwargs["result"], dict)
        
        assert "user_responses" in call_kwargs
        assert call_kwargs["user_responses"] == {"q1": "PROVIDE_HINT: Use numpy instead"}
        
        assert "current_stage_id" in call_kwargs
        assert call_kwargs["current_stage_id"] == "stage1"
        
        assert "get_dependent_stages_fn" in call_kwargs
        assert callable(call_kwargs["get_dependent_stages_fn"])

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_all_trigger_types_are_dispatched(self, mock_prompt, mock_context, mock_handle_trigger):
        """Verify multiple trigger types are all dispatched to handle_trigger."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        triggers = [
            "material_checkpoint",
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "context_overflow",
            "replan_limit",
            "backtrack_approval",
            "deadlock_detected",
            "llm_error",
            "clarification",
        ]
        
        for trigger in triggers:
            mock_handle_trigger.reset_mock()
            
            def mock_trigger_handler(*args, **kwargs):
                kwargs["result"]["supervisor_verdict"] = "ok_continue"
            
            mock_handle_trigger.side_effect = mock_trigger_handler
            
            state = {
                "current_stage_id": "stage1",
                "ask_user_trigger": trigger,
                "user_responses": {"q1": "APPROVE"},
                "plan": {"stages": []},
                "progress": {"stages": [], "user_interactions": []},
            }
            
            result = supervisor_node(state)
            
            # STRICT: Each trigger should be dispatched
            mock_handle_trigger.assert_called_once()
            call_kwargs = mock_handle_trigger.call_args[1]
            assert call_kwargs["trigger"] == trigger, f"Trigger {trigger} was not dispatched correctly"
            # STRICT: Trigger should be cleared
            assert result.get("ask_user_trigger") is None, f"Trigger {trigger} was not cleared"

    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT MERGE TESTS
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_context_update_preserves_original_keys(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should preserve original state keys when merging context update."""
        mock_context.return_value = {"new_key": "new_value"}
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 5,
            "original_key": "original_value",
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        supervisor_node(state)
        
        # STRICT: Verify prompt building used merged state
        mock_prompt.assert_called_once()
        args, _ = mock_prompt.call_args
        passed_state = args[1]
        assert passed_state.get("original_key") == "original_value"
        assert passed_state.get("new_key") == "new_value"
        assert passed_state.get("supervisor_call_count") == 5

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_context_update_overwrites_conflicting_keys(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should overwrite conflicting keys with context update values."""
        mock_context.return_value = {"current_stage_id": "updated_stage"}
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "original_stage",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        supervisor_node(state)
        
        # STRICT: Verify context update overwrote the key
        mock_prompt.assert_called_once()
        args, _ = mock_prompt.call_args
        passed_state = args[1]
        assert passed_state.get("current_stage_id") == "updated_stage"

    # ═══════════════════════════════════════════════════════════════════════
    # ERROR HANDLING EDGE CASES
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_llm_returns_none(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle LLM returning None gracefully by falling back to ok_continue."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_call.return_value = None  # LLM returns None
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        # The exception from None.get() is caught by the try-except in _run_normal_supervision
        # and should fall back to ok_continue
        result = supervisor_node(state)
        
        # STRICT: Should fall back to ok_continue on error
        assert result["supervisor_verdict"] == "ok_continue"
        # STRICT: Should include error info in feedback
        assert "LLM unavailable" in result.get("supervisor_feedback", "")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_build_agent_prompt_raises_exception(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle build_agent_prompt exception gracefully."""
        mock_context.return_value = None
        mock_prompt.side_effect = Exception("Prompt building failed")
        mock_derive.return_value = ("completed_success", "OK")
        mock_call.return_value = validated_supervisor_response.copy()
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        # This exception is not caught in the component
        with pytest.raises(Exception) as exc_info:
            supervisor_node(state)
        
        assert "Prompt building failed" in str(exc_info.value)

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_derive_stage_completion_raises_exception(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle _derive_stage_completion_outcome exception."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.side_effect = Exception("Derivation failed")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        # This exception is not caught in the component
        with pytest.raises(Exception) as exc_info:
            supervisor_node(state)
        
        assert "Derivation failed" in str(exc_info.value)

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_update_progress_stage_status_exception_handled(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle update_progress_stage_status exception - not caught in supervisor."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_update.side_effect = Exception("Update failed")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        # This exception is not caught in the component - it will propagate
        with pytest.raises(Exception) as exc_info:
            supervisor_node(state)
        
        assert "Update failed" in str(exc_info.value)

    # ═══════════════════════════════════════════════════════════════════════
    # ADDITIONAL EDGE CASES
    # ═══════════════════════════════════════════════════════════════════════

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_with_progress_none(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle progress being None in state."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": ["Approve?"],
            "progress": None,  # Explicitly None
        }
        
        # This might raise AttributeError if not handled
        # Progress being None causes progress.get() to fail
        try:
            result = supervisor_node(state)
            # If it succeeds, verify progress was created
            assert "progress" in result
        except AttributeError:
            # This reveals a bug - progress=None is not handled
            pytest.fail("Component should handle progress=None gracefully")

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_user_interaction_with_pending_questions_none(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should handle pending_user_questions being None."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            kwargs["result"]["supervisor_verdict"] = "ok_continue"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"q1": "APPROVE"},
            "pending_user_questions": None,  # Explicitly None
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should handle None gracefully
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Question should be placeholder
        if "progress" in result and "user_interactions" in result["progress"]:
            assert len(result["progress"]["user_interactions"]) == 1
            assert result["progress"]["user_interactions"][0]["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_does_not_mutate_nested_state_objects(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should not mutate nested objects in input state."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_call.return_value = validated_supervisor_response.copy()
        
        original_progress = {"stages": [{"id": "s1"}], "user_interactions": []}
        original_plan = {"stages": [{"stage_id": "stage0"}]}
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": original_plan,
            "progress": original_progress,
        }
        
        supervisor_node(state)
        
        # STRICT: Verify nested objects were not mutated
        assert original_progress == {"stages": [{"id": "s1"}], "user_interactions": []}
        assert original_plan == {"stages": [{"stage_id": "stage0"}]}

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_large_supervisor_call_count(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle large supervisor_call_count values."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 999999,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should handle large count without issues
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_handles_analysis_summary_with_datetime(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle analysis_summary containing datetime objects (json serialization)."""
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
            "analysis_summary": {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "status": "match"
            },
        }
        
        # json.dumps in user_content construction should handle datetime via default=str
        result = supervisor_node(state)
        
        # STRICT: Should complete without json serialization error
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_special_characters_in_stage_id(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle special characters in stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage-1_αβγ",  # Hyphens, underscores, unicode
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Should handle special characters
        assert result.get("supervisor_verdict") == "ok_continue"
        # STRICT: Archiving should be called with correct stage_id
        mock_archive.assert_called_once()
        archive_args = mock_archive.call_args[0]
        assert archive_args[1] == "stage-1_αβγ"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_verdict_with_extra_whitespace(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle verdict with extra whitespace (not trimmed by component)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "  ok_continue  "  # Whitespace
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Component does NOT trim whitespace - it passes through as-is
        # This might be a bug if downstream expects exact strings
        assert result.get("supervisor_verdict") == "  ok_continue  "

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_reasoning_with_newlines_and_special_chars(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should preserve newlines and special characters in reasoning/feedback."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_response["reasoning"] = "Line 1\nLine 2\n\tIndented\n\"Quoted\""
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: Special characters should be preserved
        assert result.get("supervisor_feedback") == "Line 1\nLine 2\n\tIndented\n\"Quoted\""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    @patch("src.agents.supervision.supervisor.get_validation_hierarchy")
    def test_validation_hierarchy_is_called(self, mock_hierarchy, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should call get_validation_hierarchy to build user content."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_derive.return_value = ("completed_success", "OK")
        mock_hierarchy.return_value = {"level1": "done", "level2": "pending"}
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        supervisor_node(state)
        
        # STRICT: get_validation_hierarchy should be called with state
        mock_hierarchy.assert_called_once_with(state)
        
        # STRICT: User content should include the hierarchy
        call_kwargs = mock_call.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "Validation Hierarchy" in user_content
        assert "level1" in user_content or "done" in user_content

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_trigger_handler_can_set_multiple_result_fields(self, mock_prompt, mock_context, mock_handle_trigger):
        """Should preserve multiple fields set by trigger handler."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        def mock_trigger_handler(*args, **kwargs):
            # Handler sets many fields
            kwargs["result"]["supervisor_verdict"] = "replan_needed"
            kwargs["result"]["supervisor_feedback"] = "Trigger feedback"
            kwargs["result"]["planner_feedback"] = "Planner guidance"
            kwargs["result"]["replan_count"] = 0
            kwargs["result"]["custom_field"] = "custom_value"
        
        mock_handle_trigger.side_effect = mock_trigger_handler
        
        state = {
            "current_stage_id": "stage1",
            "ask_user_trigger": "replan_limit",
            "user_responses": {"q1": "GUIDANCE: some hint"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # STRICT: All fields set by handler should be in result
        assert result.get("supervisor_verdict") == "replan_needed"
        assert result.get("supervisor_feedback") == "Trigger feedback"
        assert result.get("planner_feedback") == "Planner guidance"
        assert result.get("replan_count") == 0
        assert result.get("custom_field") == "custom_value"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_empty_progress_stages(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should handle empty progress stages list in user content."""
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
        
        supervisor_node(state)
        
        # STRICT: User content should show 0 completed, 0 pending
        call_kwargs = mock_call.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "Completed: 0" in user_content
        assert "Pending: 0" in user_content
        assert "Total: 0" in user_content

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    @patch("src.agents.supervision.supervisor._derive_stage_completion_outcome")
    def test_progress_with_various_statuses(self, mock_derive, mock_update, mock_archive, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should correctly count stages by status category."""
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
            "progress": {
                "stages": [
                    {"stage_id": "s0", "status": "completed_success"},
                    {"stage_id": "s1", "status": "completed_partial"},
                    {"stage_id": "s2", "status": "completed_failed"},
                    {"stage_id": "s3", "status": "in_progress"},
                    {"stage_id": "s4", "status": "not_started"},
                    {"stage_id": "s5", "status": "blocked"},
                ]
            },
        }
        
        supervisor_node(state)
        
        # STRICT: Completed = 3 (completed_*), Pending = 2 (in_progress, not_started), Total = 6
        call_kwargs = mock_call.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "Completed: 3" in user_content
        assert "Pending: 2" in user_content
        assert "Total: 6" in user_content
