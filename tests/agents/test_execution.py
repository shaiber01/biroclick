"""
Tests for Execution Agents (ExecutionValidatorAgent, PhysicsSanityAgent).
"""

import pytest
import json
from unittest.mock import patch, MagicMock, ANY
from src.agents.execution import execution_validator_node, physics_sanity_node

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Base state for execution tests."""
    return {
        "paper_id": "test_paper",
        "current_stage_id": "stage_1_sim",
        "stage_outputs": {
            "stdout": "Simulation running...",
            "stderr": "",
            "exit_code": 0,
            "files": ["output.csv"],
            "timeout_exceeded": False
        },
        "run_error": None,
        "execution_failure_count": 0,
        "physics_failure_count": 0,
        "design_revision_count": 0,
        "design_description": {
            "parameters": {"p1": 10}
        },
        "runtime_config": {
            "max_execution_failures": 3,
            "max_physics_failures": 3,
            "max_design_revisions": 3
        }
    }

# ═══════════════════════════════════════════════════════════════════════
# execution_validator_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExecutionValidatorNode:
    """Tests for execution_validator_node."""

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_pass(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator passing a successful run."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_llm.return_value = {
            "verdict": "pass",
            "summary": "Execution looks good."
        }
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Strict: Must return feedback/summary even on pass for logging/history
        assert result["execution_feedback"] == "Execution looks good."
        
        # strict assertion on LLM call inputs
        mock_llm.assert_called_once()
        args, kwargs = mock_llm.call_args
        assert kwargs["agent_name"] == "execution_validator"
        assert "Simulation running..." in kwargs["user_content"]
        assert "output.csv" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_fail_and_increment(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator failing a run increments counter and returns feedback."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Missing output file."
        }
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        assert result["execution_feedback"] == "Missing output file."
        assert result["total_execution_failures"] == 1

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_fail(self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state):
        """Test validator failing on timeout (default fallback)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "ask_user"
        
        base_state["stage_outputs"]["timeout_exceeded"] = True
        base_state["run_error"] = "Timeout exceeded"
        
        result = execution_validator_node(base_state)
        
        # Should fail without calling LLM
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        # Must provide feedback for timeout
        assert "Timeout exceeded" in result["execution_feedback"]
        
        # Ensure LLM was NOT called
        mock_llm.assert_not_called()

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    def test_validator_timeout_skip(self, mock_check, mock_prompt, mock_get_spec, base_state):
        """Test validator skipping on timeout if configured."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "skip_with_warning"
        
        base_state["stage_outputs"]["timeout_exceeded"] = True
        base_state["run_error"] = "Timeout exceeded"
        
        result = execution_validator_node(base_state)
        
        # Should pass with warning
        assert result["execution_verdict"] == "pass"
        # Should NOT increment failure count
        assert "execution_failure_count" not in result
        assert "Timeout exceeded" in result["execution_feedback"]

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_no_error(self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state):
        """Test validator timeout with no explicit run_error."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "ask_user"
        
        base_state["stage_outputs"]["timeout_exceeded"] = True
        base_state["run_error"] = None
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Execution timed out" in result["execution_feedback"]
        # It might say "None" or just exist. Just check it runs.
        mock_llm.assert_not_called()

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_max_failures_ask_user(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test asking user when max execution failures reached."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Still failing."
        }
        
        base_state["execution_failure_count"] = 3 # Already at max
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 3 # Stays at max
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["pending_user_questions"] is not None
        assert "failing" in result["execution_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_llm_exception(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator handling LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = execution_validator_node(base_state)
        
        # Should default to pass on error (auto-approve) per implementation
        assert result["execution_verdict"] == "pass"
        assert "API Error" in result["execution_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_empty_outputs(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with empty stage outputs."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        # Empty stage outputs
        base_state["stage_outputs"] = {}
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        # Verify JSON string in user_content is empty dict
        args, kwargs = mock_llm.call_args
        assert "{}" in kwargs["user_content"]

# ═══════════════════════════════════════════════════════════════════════
# physics_sanity_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPhysicsSanityNode:
    """Tests for physics_sanity_node."""

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_pass(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics pass."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "pass",
            "summary": "Physics OK."
        }
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Physics OK."
        
        # Verify context passed
        args, kwargs = mock_llm.call_args
        assert "p1" in kwargs["user_content"] # Design parameters should be in context

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_fail_increments_physics_count(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics fail increments physics_failure_count."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Unphysical energy."
        }
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1
        assert "design_revision_count" not in result
        assert result["physics_feedback"] == "Unphysical energy."

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_design_flaw_increments_design_count(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test design_flaw increments design_revision_count."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "design_flaw",
            "summary": "Impossible geometry."
        }
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 1
        assert "physics_failure_count" not in result
        # Should return design_feedback for design flaws
        assert result["design_feedback"] == "Impossible geometry." 

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_backtrack_suggestion(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test capturing backtrack suggestion."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Backtrack needed.",
            "backtrack_suggestion": {"suggest_backtrack": True, "reason": "Bad materials"}
        }
        
        result = physics_sanity_node(base_state)
        
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["physics_feedback"] == "Backtrack needed."

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_llm_exception(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics handling LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        assert "API Error" in result["physics_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_no_design(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics sanity with missing design."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["design_description"] = None
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        # Verify design section is missing or handled
        args, kwargs = mock_llm.call_args
        # Should not have "## Design Spec" if design is None/empty?
        # Let's check the code: if design: ...
        assert "## Design Spec" not in kwargs["user_content"]
