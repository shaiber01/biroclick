"""
Tests for Execution Agents (ExecutionValidatorAgent, PhysicsSanityAgent).
"""

import pytest
import json
from unittest.mock import patch, MagicMock, ANY
from src.agents.execution import execution_validator_node, physics_sanity_node


class NonSerializable:
    """Helper object to verify serialization fallback within prompts."""

    def __str__(self):
        return "NON_SERIALIZABLE_OBJECT"

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
        assert result["workflow_phase"] == "execution_validation"
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
        assert result["workflow_phase"] == "execution_validation"
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
        assert result["workflow_phase"] == "execution_validation"
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
        assert result["workflow_phase"] == "execution_validation"
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
        base_state["total_execution_failures"] = 4
        base_state["run_error"] = "Disk failure"
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 3 # Stays at max
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["pending_user_questions"] is not None
        assert "Disk failure" in result["pending_user_questions"][0]
        assert "failing" in result["execution_feedback"]
        assert result["total_execution_failures"] == 5

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
        assert result["workflow_phase"] == "execution_validation"
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
        assert result["workflow_phase"] == "execution_validation"
        # Verify JSON string in user_content is empty dict
        args, kwargs = mock_llm.call_args
        assert "{}" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_includes_run_error_in_prompts(self, mock_llm, mock_check, mock_prompt, base_state):
        """Validate run_error is propagated to both prompts for diagnostics."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "Recovered"}
        base_state["run_error"] = "Runtime log corruption"
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        args, kwargs = mock_llm.call_args
        assert "Runtime log corruption" in kwargs["system_prompt"]
        assert "CONTEXT: The previous execution failed" in kwargs["system_prompt"]
        assert "Runtime log corruption" in kwargs["user_content"]

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_detected_via_run_error_string(
        self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state
    ):
        """Run errors mentioning timeout should avoid unnecessary LLM calls."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "ask_user"
        base_state["stage_outputs"]["timeout_exceeded"] = False
        base_state["run_error"] = "Execution exceeded timeout window"
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert "timeout window" in result["execution_feedback"]
        mock_llm.assert_not_called()

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_detection_case_insensitive(
        self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state
    ):
        """Run errors mentioning timeout should be detected case-insensitively."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "ask_user"
        base_state["stage_outputs"]["timeout_exceeded"] = False
        base_state["run_error"] = "Execution Exceeded Timeout window" # Mixed case
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert "Exceeded Timeout" in result["execution_feedback"]
        mock_llm.assert_not_called()

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_serializes_non_jsonable_stage_outputs(
        self, mock_llm, mock_check, mock_prompt, base_state
    ):
        """Stage outputs with custom objects should still be presented to LLM."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        base_state["stage_outputs"] = {"obj": NonSerializable()}
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        args, kwargs = mock_llm.call_args
        assert "NON_SERIALIZABLE_OBJECT" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_stage_outputs_none_crash(self, mock_llm, mock_check, mock_prompt, base_state):
        """
        If stage_outputs is explicitly None (not empty dict), it should not crash.
        Current implementation might crash at stage_outputs.get().
        """
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = None
        
        # This is expected to fail if code doesn't handle None
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        # Verify user_content handles None or empty
        args, kwargs = mock_llm.call_args
        assert "null" in kwargs["user_content"] or "{}" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_llm_missing_verdict(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator handling LLM output missing 'verdict' key."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        # Missing verdict
        mock_llm.return_value = {
            "summary": "I forgot the verdict."
        }
        
        # This should probably raise KeyError in current code or handle it
        # Expectation: It should fail gracefully (maybe default to fail?) or raise a specific error
        # For now, let's assert it raises KeyError so we confirm the bug, 
        # BUT user wants tests that fail until fixed. 
        # So I assert it HANDLES it (e.g. treats as fail or pass).
        
        try:
            result = execution_validator_node(base_state)
            # If it survives, check reasonable default
            assert result["execution_verdict"] in ["pass", "fail"] 
        except KeyError:
            pytest.fail("execution_validator_node crashed due to missing 'verdict' key")

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
        assert "backtrack_suggestion" not in result
        
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
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 1
        assert "design_revision_count" not in result
        assert result["physics_feedback"] == "Unphysical energy."
        assert "backtrack_suggestion" not in result

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
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1
        assert "physics_failure_count" not in result
        # Should return design_feedback for design flaws
        assert result["design_feedback"] == "Impossible geometry." 
        assert result["physics_feedback"] == "Impossible geometry."

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
        
        assert result["workflow_phase"] == "physics_validation"
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
        assert result["workflow_phase"] == "physics_validation"
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
        assert result["workflow_phase"] == "physics_validation"
        # Verify design section is missing or handled
        args, kwargs = mock_llm.call_args
        # Should not have "## Design Spec" if design is None/empty?
        # Let's check the code: if design: ...
        assert "## Design Spec" not in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_includes_string_design_spec(self, mock_llm, mock_check, mock_prompt, base_state):
        """Design specs provided as strings must be passed through verbatim."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "warning", "summary": "Manual inspection suggested."}
        base_state["design_description"] = "Design v1 (text-only)"
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "warning"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Manual inspection suggested."
        args, kwargs = mock_llm.call_args
        assert "\n## Design Spec\nDesign v1 (text-only)" in kwargs["user_content"]
        assert "\n## Design Spec\n```json" not in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_failure_count_respects_runtime_max(self, mock_llm, mock_check, mock_prompt, base_state):
        """Physics failure counter should not exceed configured max."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Still broken."}
        base_state["physics_failure_count"] = 2
        base_state["runtime_config"]["max_physics_failures"] = 2
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 2
        assert result["physics_feedback"] == "Still broken."
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_design_revision_count_respects_runtime_max(self, mock_llm, mock_check, mock_prompt, base_state):
        """Design revision counter should clamp at runtime limit."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "design_flaw", "summary": "Needs redesign."}
        base_state["design_revision_count"] = 1
        base_state["runtime_config"]["max_design_revisions"] = 1
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1
        assert result["design_feedback"] == "Needs redesign."
        assert result["physics_feedback"] == "Needs redesign."
        assert "physics_failure_count" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_stage_outputs_none_crash(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of None stage_outputs."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = None
        
        try:
            result = physics_sanity_node(base_state)
        except AttributeError:
             pytest.fail("physics_sanity_node crashed due to None stage_outputs")

        assert result["physics_verdict"] == "pass"

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_llm_missing_verdict(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM response missing verdict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"summary": "Oops no verdict"}
        
        try:
            result = physics_sanity_node(base_state)
            assert result["physics_verdict"] in ["pass", "fail", "warning", "design_flaw"]
        except KeyError:
             pytest.fail("physics_sanity_node crashed due to missing 'verdict' key")

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_backtrack_malformed(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of malformed backtrack suggestion (not a dict)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        # backtrack_suggestion should be a dict, but LLM might return string or list
        mock_llm.return_value = {
            "verdict": "fail", 
            "summary": "Backtrack", 
            "backtrack_suggestion": "Yes, please backtrack."
        }
        
        try:
            result = physics_sanity_node(base_state)
            # Should not crash and ideally not return the malformed suggestion if it expects a dict
        except AttributeError:
             pytest.fail("physics_sanity_node crashed due to malformed backtrack_suggestion")

        # Check if it handled it gracefully (e.g. ignored it or wrapped it)
        # The code expects .get() so likely crashed.
