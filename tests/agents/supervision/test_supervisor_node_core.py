"""Core supervisor_node flow tests."""

from unittest.mock import MagicMock, patch

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
        
        assert result.get("supervisor_verdict") == "ok_continue"
        
        # Check archiving and status update
        mock_archive.assert_called_once_with(state, "stage1")
        mock_update.assert_called_once_with(state, "stage1", "completed_success", summary="Analysis OK")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_uses_updated_context_state(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should use updated state from context check for prompt building."""
        mock_context.return_value = {"new_flag": True}
        mock_prompt.return_value = "prompt"
        mock_call.return_value = validated_supervisor_response.copy()
        
        state = {"current_stage_id": "stage1", "supervisor_call_count": 0}
        
        supervisor_node(state)
        
        # Verify build_agent_prompt received state with new_flag
        args, _ = mock_prompt.call_args
        passed_state = args[1]
        assert passed_state.get("new_flag") is True

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_triggers_backtrack_on_failure(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should trigger backtrack on stage failure (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "backtrack_to_stage"
        mock_response["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "design",
            "stages_to_invalidate": ["stage1"],
            "reason": "design flawed"
        }
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "physics_verdict": "fail",
            "supervisor_call_count": 0,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "backtrack_to_stage"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_defaults_on_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should default to ok_continue on LLM error."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 0,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1"}
        
        result = supervisor_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_returns_supervisor_verdict(self, mock_prompt, mock_context, mock_call, validated_supervisor_response):
        """Should return supervisor verdict from LLM output (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_supervisor_response.copy()
        mock_response["verdict"] = "ok_continue"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "supervisor_call_count": 5,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Supervisor returns verdict
        assert result.get("supervisor_verdict") == "ok_continue"

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
        mock_response["stop_reason"] = "Done"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": None,
            "completed_stages": ["stage0", "stage1"],
            "supervisor_call_count": 0,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
