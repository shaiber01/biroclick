"""Unit tests for src/agents/execution.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.execution import (
    execution_validator_node,
    physics_sanity_node,
)


class TestExecutionValidatorNode:
    """Tests for execution_validator_node function."""

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_validates_successful_execution(self, mock_prompt, mock_context, mock_call):
        """Should validate successful execution."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "verdict": "pass",
            "issues": [],
            "summary": "Execution completed successfully",
        }
        
        state = {
            "current_stage_id": "stage1",
            "run_result": {
                "success": True,
                "output_files": ["spectrum.csv"],
                "stdout": "Simulation complete",
            },
        }
        
        result = execution_validator_node(state)
        
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_verdict"] == "pass"

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_fails_on_execution_error(self, mock_prompt, mock_context, mock_call):
        """Should fail validation on execution error."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "issues": [{"severity": "critical", "description": "Segmentation fault"}],
            "summary": "Execution crashed",
            "feedback": "Check memory allocation",
        }
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},  # Uses stage_outputs not run_result
            "run_error": "Segmentation fault",  # Error is in run_error
            "execution_failure_count": 0,  # Uses execution_failure_count
        }
        
        result = execution_validator_node(state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_auto_passes_on_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should auto-pass when LLM call fails."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "run_result": {"success": True},
        }
        
        result = execution_validator_node(state)
        
        assert result["execution_verdict"] == "pass"

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_handles_timeout_via_flag(self, mock_prompt, mock_context, mock_call):
        """Should handle timeout via state flag."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        # With skip_with_warning strategy
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"timeout_exceeded": True},
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "fallback_strategy": "skip_with_warning"}
                ]
            }
        }
        
        result = execution_validator_node(state)
        # The node returns a dict with execution_verdict, but not necessarily summary in top level
        # We need to verify the verdict is "pass"
        assert result["execution_verdict"] == "pass"

        # With default strategy (fail)
        state["plan"]["stages"][0]["fallback_strategy"] = "ask_user"
        result = execution_validator_node(state)
        assert result["execution_verdict"] == "fail"

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow.
        
        Note: Patches base.py because execution_validator_node uses @with_context_check decorator.
        """
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "run_result": {}}
        
        result = execution_validator_node(state)
        
        assert result["awaiting_user_input"] is True


class TestPhysicsSanityNode:
    """Tests for physics_sanity_node function."""

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_passes_physically_valid_results(self, mock_prompt, mock_context, mock_call):
        """Should pass physically valid results."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "verdict": "pass",
            "issues": [],
            "summary": "Results are physically reasonable",
            "backtrack_suggestion": {"suggest_backtrack": False},
        }
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": ["spectrum.csv"]},  # Uses stage_outputs
            "design_description": {},
        }
        
        result = physics_sanity_node(state)
        
        assert result["workflow_phase"] == "physics_validation"  # Uses physics_validation not physics_sanity
        assert result["physics_verdict"] == "pass"

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_fails_unphysical_results(self, mock_prompt, mock_context, mock_call):
        """Should fail unphysical results."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "issues": [{"severity": "critical", "description": "Negative absorption"}],
            "summary": "Results violate physics",
            "feedback": "Check energy conservation",
            "backtrack_suggestion": {"suggest_backtrack": False},
        }
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},
            "physics_failure_count": 0,  # Uses physics_failure_count not physics_revision_count
        }
        
        result = physics_sanity_node(state)
        
        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_auto_passes_on_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should auto-pass when LLM call fails."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "run_result": {"success": True},
        }
        
        result = physics_sanity_node(state)
        
        assert result["physics_verdict"] == "pass"

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow.
        
        Note: Patches base.py because physics_sanity_node uses @with_context_check decorator.
        """
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "run_result": {}}
        
        result = physics_sanity_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.execution.call_agent_with_metrics")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.build_agent_prompt")
    def test_respects_max_failures(self, mock_prompt, mock_context, mock_call):
        """Should not exceed max failures."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "fail",
            "issues": [],
            "summary": "Still unphysical",
            "backtrack_suggestion": {"suggest_backtrack": False},
        }
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},
            "physics_failure_count": 3,
            "runtime_config": {"max_physics_failures": 3},
        }
        
        result = physics_sanity_node(state)
        
        # Should not increment past max
        assert result["physics_failure_count"] == 3

