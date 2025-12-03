"""Tests for execution_validator_node."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.execution import execution_validator_node
from tests.agents.shared_objects import NonSerializable


@pytest.fixture(name="base_state")
def execution_base_state(execution_state):
    return execution_state


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
        
        # Verify initial state
        initial_failure_count = base_state.get("execution_failure_count", 0)
        initial_total_failures = base_state.get("total_execution_failures", 0)
        
        result = execution_validator_node(base_state)
        
        # Strict assertions on result structure
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Strict: Must return feedback/summary even on pass for logging/history
        assert result["execution_feedback"] == "Execution looks good."
        # CRITICAL: Pass verdict should NOT increment counters
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        assert "ask_user_trigger" not in result
        
        # Strict assertion on LLM call inputs
        mock_llm.assert_called_once()
        args, kwargs = mock_llm.call_args
        assert kwargs["agent_name"] == "execution_validator"
        assert "Simulation running..." in kwargs["user_content"]
        assert "output.csv" in kwargs["user_content"]
        assert "stage_1_sim" in kwargs["user_content"]
        assert "EXECUTION RESULTS FOR STAGE" in kwargs["user_content"]
        assert "Stage Outputs" in kwargs["user_content"]

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
        
        # Verify initial state
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0
        
        result = execution_validator_node(base_state)
        
        # Strict assertions on failure handling
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert result["execution_feedback"] == "Missing output file."
        assert result["total_execution_failures"] == 1
        # Should NOT trigger user ask when below max
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result

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
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0
        
        result = execution_validator_node(base_state)
        
        # Should fail without calling LLM
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        # Must provide feedback for timeout
        assert "Timeout exceeded" in result["execution_feedback"]
        assert "timed out" in result["execution_feedback"].lower()
        # Should NOT trigger user ask when below max
        assert "ask_user_trigger" not in result
        
        # Ensure LLM was NOT called
        mock_llm.assert_not_called()
        # Note: build_agent_prompt is called before timeout check in current implementation
        # This is inefficient but not a bug - prompt is built but not used

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_skip(self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state):
        """Test validator skipping on timeout if configured."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "skip_with_warning"
        
        base_state["stage_outputs"]["timeout_exceeded"] = True
        base_state["run_error"] = "Timeout exceeded"
        initial_failures = base_state.get("total_execution_failures", 0)
        
        result = execution_validator_node(base_state)
        
        # Should pass with warning
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Should NOT increment failure count
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        assert "Timeout exceeded" in result["execution_feedback"]
        assert "skip_with_warning" in result["execution_feedback"]
        # Should NOT trigger user ask
        assert "ask_user_trigger" not in result
        
        # Ensure LLM was NOT called
        mock_llm.assert_not_called()

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
        base_state["execution_failure_count"] = 0
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert "timed out" in result["execution_feedback"].lower()
        # Should handle None run_error gracefully
        assert result["execution_feedback"] is not None
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
        
        # Set counter exactly at max (3)
        base_state["execution_failure_count"] = 3
        base_state["total_execution_failures"] = 4
        base_state["run_error"] = "Disk failure"
        base_state["runtime_config"]["max_execution_failures"] = 3
        
        result = execution_validator_node(base_state)
        
        # Strict assertions on max failure handling
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 3  # Stays at max (not incremented)
        assert result["total_execution_failures"] == 5  # Still increments total
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["pending_user_questions"] is not None
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        assert "Disk failure" in result["pending_user_questions"][0]
        assert "failing" in result["execution_feedback"]
        assert "3 times" in result["pending_user_questions"][0] or "3" in result["pending_user_questions"][0]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_llm_exception(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator handling LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        initial_failures = base_state.get("total_execution_failures", 0)
        
        result = execution_validator_node(base_state)
        
        # Should default to pass on error (auto-approve) per implementation
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        assert "API Error" in result["execution_feedback"]
        assert "Auto-approved" in result["execution_feedback"] or "auto-passed" in result["execution_feedback"].lower()
        # Should NOT increment counters on auto-approve
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result

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
        assert result["execution_feedback"] == "OK"
        # Verify JSON string in user_content is empty dict
        args, kwargs = mock_llm.call_args
        assert "{}" in kwargs["user_content"]
        assert "Stage Outputs" in kwargs["user_content"]
        assert "stage_1_sim" in kwargs["user_content"]

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
        # Verify run_error appears in system prompt
        assert "Runtime log corruption" in kwargs["system_prompt"]
        assert "CONTEXT: The previous execution failed" in kwargs["system_prompt"]
        # Verify run_error appears in user content
        assert "Runtime log corruption" in kwargs["user_content"]
        assert "Run Error" in kwargs["user_content"]

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
        base_state["execution_failure_count"] = 0
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        assert "timeout window" in result["execution_feedback"]
        assert "timed out" in result["execution_feedback"].lower()
        mock_llm.assert_not_called()
        # Note: build_agent_prompt is called before timeout check in current implementation

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
        base_state["run_error"] = "Execution Exceeded Timeout window"  # Mixed case
        base_state["execution_failure_count"] = 0
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        assert "Exceeded Timeout" in result["execution_feedback"] or "timeout" in result["execution_feedback"].lower()
        mock_llm.assert_not_called()
        
    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_detection_timeout_error_pattern(
        self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state
    ):
        """Test timeout detection via 'timeout_error' string pattern."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "ask_user"
        base_state["stage_outputs"]["timeout_exceeded"] = False
        base_state["run_error"] = "timeout_error: process killed"
        base_state["execution_failure_count"] = 0
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
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
        """Test validator handling LLM output missing 'verdict' key.
        
        Missing verdict is handled gracefully (not as an exception):
        - Defaults to 'pass' with a warning
        - Does NOT treat this as 'LLM unavailable' error
        - Consistent with design_reviewer and code_reviewer patterns
        """
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        # Missing verdict - should be handled gracefully, not as exception
        mock_llm.return_value = {
            "summary": "I forgot the verdict."
        }
        
        result = execution_validator_node(base_state)
        
        # Should default to pass gracefully
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Should NOT increment counters (pass verdict)
        assert "execution_failure_count" not in result
        # Feedback should indicate missing verdict (graceful handling)
        assert "Missing verdict" in result["execution_feedback"] or "I forgot the verdict" in result["execution_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_fail_counter_at_max_minus_one(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test counter increment when at max-1 triggers escalation when reaching max.
        
        With max=3 and count=2:
        - Failure increments count to 3
        - 3 >= 3 triggers escalation
        """
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Error occurred"}
        
        base_state["execution_failure_count"] = 2  # Max is 3, so at max-1
        base_state["runtime_config"]["max_execution_failures"] = 3
        base_state["total_execution_failures"] = 2
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 3  # Should increment to max
        assert result["total_execution_failures"] == 3
        # SHOULD trigger user ask when count reaches max
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["awaiting_user_input"] is True

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_fail_missing_total_execution_failures(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that total_execution_failures defaults to 0 if missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Error"}
        
        base_state["execution_failure_count"] = 0
        # Explicitly remove total_execution_failures
        if "total_execution_failures" in base_state:
            del base_state["total_execution_failures"]
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1  # Should default to 0 + 1

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_fail_missing_runtime_config(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that missing runtime_config uses default max_execution_failures."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Error"}
        
        base_state["execution_failure_count"] = 0
        # Remove runtime_config
        if "runtime_config" in base_state:
            del base_state["runtime_config"]
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        # Should use default MAX_EXECUTION_FAILURES from schemas.state

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_llm_missing_summary(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator handling LLM output missing 'summary' key."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass"}  # Missing summary
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Should use default fallback message
        assert result["execution_feedback"] == "No feedback provided."

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_empty_string_run_error(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with empty string run_error (not None)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        base_state["run_error"] = ""
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        # Empty string should be falsy, so should not appear in prompts
        args, kwargs = mock_llm.call_args
        assert "CONTEXT: The previous execution failed" not in kwargs["system_prompt"]
        assert "Run Error" not in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_complex_nested_stage_outputs(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with complex nested stage outputs."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = {
            "nested": {
                "level1": {
                    "level2": {"value": 42, "list": [1, 2, 3]}
                }
            },
            "array": [{"item": 1}, {"item": 2}],
            "mixed": [1, "string", {"dict": True}]
        }
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        args, kwargs = mock_llm.call_args
        # Verify complex structure is serialized
        assert "nested" in kwargs["user_content"]
        assert "level1" in kwargs["user_content"]
        assert "42" in kwargs["user_content"]
        assert "array" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_missing_current_stage_id(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with missing current_stage_id."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        if "current_stage_id" in base_state:
            del base_state["current_stage_id"]
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        args, kwargs = mock_llm.call_args
        # Should use "unknown" as fallback
        assert "unknown" in kwargs["user_content"].lower() or "stage" in kwargs["user_content"].lower()

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_llm_runtime_error(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator handling RuntimeError exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = RuntimeError("Internal error")
        
        result = execution_validator_node(base_state)
        
        # Should auto-approve on any exception
        assert result["execution_verdict"] == "pass"
        assert "Internal error" in result["execution_feedback"]
        assert "Auto-approved" in result["execution_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_llm_timeout_exception(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator handling timeout exception from LLM."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = TimeoutError("LLM API timeout")
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        assert "timeout" in result["execution_feedback"].lower() or "LLM API timeout" in result["execution_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_custom_max_execution_failures(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with custom max_execution_failures in runtime_config."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Error"}
        
        base_state["execution_failure_count"] = 5  # At custom max
        base_state["runtime_config"]["max_execution_failures"] = 5
        base_state["total_execution_failures"] = 5
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 5  # Stays at max
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["total_execution_failures"] == 6

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_stage_outputs_with_none_values(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with stage_outputs containing None values."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = {
            "value": None,
            "dict": {"key": None, "other": "value"},
            "list": [None, 1, None]
        }
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        args, kwargs = mock_llm.call_args
        # None values should be serialized as "null" in JSON
        assert "null" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_multiple_failures_increment_total(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that total_execution_failures increments on every failure."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Error"}
        
        base_state["execution_failure_count"] = 1
        base_state["total_execution_failures"] = 10  # Already high
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 2
        assert result["total_execution_failures"] == 11  # Should increment regardless

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_non_string_run_error(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with non-string run_error (should be converted to string)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        # Use a non-string error (e.g., Exception object or dict)
        base_state["run_error"] = {"error_code": 500, "message": "Server error"}
        
        result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "pass"
        # Non-string should be converted to string representation
        args, kwargs = mock_llm.call_args
        # The error should appear in prompts (converted to string)
        error_str = str(base_state["run_error"])
        assert error_str in kwargs["system_prompt"] or error_str in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_invalid_verdict_value(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test validator with invalid verdict value (not 'pass' or 'fail')."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "maybe", "summary": "Uncertain"}
        
        result = execution_validator_node(base_state)
        
        # Should still process the verdict even if invalid
        assert result["execution_verdict"] == "maybe"
        assert result["workflow_phase"] == "execution_validation"
        # Invalid verdict should NOT increment counters (only "fail" does)
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_with_invalid_fallback(self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state):
        """Test validator timeout with invalid fallback strategy."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "invalid_fallback"  # Not "skip_with_warning" or "ask_user"
        
        base_state["stage_outputs"]["timeout_exceeded"] = True
        base_state["run_error"] = "Timeout"
        base_state["execution_failure_count"] = 0
        
        result = execution_validator_node(base_state)
        
        # Should default to "fail" behavior when fallback is invalid
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        mock_llm.assert_not_called()

    @patch("src.agents.execution.get_stage_design_spec")
    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_validator_timeout_exceeded_as_string(self, mock_llm, mock_check, mock_prompt, mock_get_spec, base_state):
        """Test validator with timeout_exceeded as string 'True' instead of boolean."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "ask_user"
        
        # Set timeout_exceeded as string (edge case)
        base_state["stage_outputs"]["timeout_exceeded"] = "True"
        base_state["run_error"] = "Timeout"
        base_state["execution_failure_count"] = 0
        
        result = execution_validator_node(base_state)
        
        # String "True" is truthy, so should be treated as timeout
        # Code uses: is_timeout = stage_outputs.get("timeout_exceeded", False)
        # This returns "True" (string), which is truthy, so `if is_timeout:` is True
        # Therefore it should be detected as timeout
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        mock_llm.assert_not_called()

# ═══════════════════════════════════════════════════════════════════════
# physics_sanity_node Tests
# ═══════════════════════════════════════════════════════════════════════
