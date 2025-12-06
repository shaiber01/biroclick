"""Execution validator and physics sanity integration tests."""

from unittest.mock import patch, MagicMock

import pytest

from tests.integration.helpers.agent_responses import execution_verdict_response


class TestWithContextCheckDecorator:
    """Test the @with_context_check decorator behavior applied to execution nodes."""

    def test_execution_validator_returns_empty_when_trigger_set(self, base_state):
        """When ask_user_trigger is set, execution_validator_node should return empty dict immediately."""
        from src.agents.execution import execution_validator_node

        base_state["ask_user_trigger"] = "some_trigger"  # Trigger set = skip node
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_call:
            result = execution_validator_node(base_state)

        # Should return empty dict without calling LLM
        assert result == {}, f"Expected empty dict when ask_user_trigger set, got: {result}"
        mock_call.assert_not_called()

    def test_physics_sanity_returns_empty_when_trigger_set(self, base_state):
        """When ask_user_trigger is set, physics_sanity_node should return empty dict immediately."""
        from src.agents.execution import physics_sanity_node

        base_state["ask_user_trigger"] = "some_trigger"  # Trigger set = skip node
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_call:
            result = physics_sanity_node(base_state)

        # Should return empty dict without calling LLM
        assert result == {}, f"Expected empty dict when ask_user_trigger set, got: {result}"
        mock_call.assert_not_called()

    def test_execution_validator_processes_normally_when_no_trigger(self, base_state):
        """When ask_user_trigger is not set, should process normally."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")
        base_state["ask_user_trigger"] = None  # No trigger = process normally
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        # Should process normally
        assert result["execution_verdict"] == "pass"
        mock_call.assert_called_once()


class TestExecutionValidatorBehavior:
    """Execution validator specific behaviors."""

    def test_execution_validator_returns_verdict_from_llm(self, base_state):
        """Test execution validator returns correct verdict and sets all required fields."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(
            verdict="pass",
            summary="Execution completed successfully"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}
        # Ensure counters are initialized
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        # Strict assertions on all returned fields
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_feedback"] == "Execution completed successfully"
        # Pass verdict should NOT increment counters
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result
        assert "last_node_before_ask_user" not in result
        
        # Verify LLM was called with correct parameters
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["agent_name"] == "execution_validator"
        assert "stage_0" in call_kwargs["user_content"]
        assert "output.csv" in call_kwargs["user_content"]
        assert "EXECUTION RESULTS FOR STAGE" in call_kwargs["user_content"]
        assert "Stage Outputs" in call_kwargs["user_content"]
        # Verify state was passed
        assert call_kwargs["state"] is not None

    def test_execution_validator_with_warning_verdict(self, base_state):
        """Test execution validator handles warning verdict correctly."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(
            verdict="warning",
            summary="Execution completed with minor issues"
        )

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "warning"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_feedback"] == "Execution completed with minor issues"
        # Warning should NOT increment failure counters
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        # Warning should NOT trigger user escalation
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result

    def test_execution_validator_with_missing_stage_outputs(self, base_state):
        """Test execution validator handles missing stage_outputs gracefully."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = None  # Missing stage_outputs

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_feedback"] == "OK"
        # Should still call LLM with empty stage_outputs
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert "Stage Outputs" in call_kwargs["user_content"]
        # When stage_outputs is None, it becomes {} (empty dict) from `or {}` logic
        assert "{}" in call_kwargs["user_content"]

    def test_execution_validator_with_empty_stage_outputs(self, base_state):
        """Test execution validator handles empty stage_outputs dict."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert "{}" in call_kwargs["user_content"] or "Stage Outputs" in call_kwargs["user_content"]

    def test_execution_validator_with_missing_current_stage_id(self, base_state):
        """Test execution validator handles missing current_stage_id by using 'unknown' fallback."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state.pop("current_stage_id", None)
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        assert mock_call.called
        # Should use "unknown" as fallback in the prompt
        call_kwargs = mock_call.call_args.kwargs
        assert "EXECUTION RESULTS FOR STAGE: unknown" in call_kwargs["user_content"], (
            f"Expected 'unknown' as stage_id fallback, got: {call_kwargs['user_content'][:200]}"
        )

    def test_execution_validator_includes_run_error_in_prompt(self, base_state):
        """Test execution validator includes run_error in both system prompt and user content."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Error detected")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["run_error"] = "Segmentation fault at line 42"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_feedback"] == "Error detected"
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        
        # run_error should be appended to system_prompt as context
        system_prompt = call_kwargs.get("system_prompt", "")
        assert "Segmentation fault at line 42" in system_prompt, (
            f"run_error should be in system_prompt, got: {system_prompt[-300:]}"
        )
        assert "CONTEXT: The previous execution failed with error:" in system_prompt
        
        # run_error should also be in user_content under "Run Error" section
        user_content = call_kwargs.get("user_content", "")
        assert "## Run Error" in user_content, (
            f"'Run Error' section should be in user_content, got: {user_content[:500]}"
        )
        assert "Segmentation fault at line 42" in user_content


class TestValidatorVerdicts:
    """Test various validator verdict scenarios."""

    def test_physics_sanity_returns_design_flaw(self, base_state):
        """Test physics sanity correctly handles design_flaw verdict."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Simulation parameters inconsistent with physics",
            design_issues=["Wavelength range too narrow"],
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        # Strict assertions on design_flaw handling
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Simulation parameters inconsistent with physics"
        assert result["design_feedback"] == "Simulation parameters inconsistent with physics"
        # design_flaw should increment design_revision_count, NOT physics_failure_count
        assert result["design_revision_count"] == 1
        assert "physics_failure_count" not in result

    def test_execution_validator_returns_fail(self, base_state):
        """Test execution validator correctly handles fail verdict and increments counters."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Simulation crashed",
            error_analysis="Memory allocation failure",
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {
            "files": [],
            "exit_code": 1,
            "stderr": "Segmentation fault",
        }
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        # Strict assertions on fail handling
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_feedback"] == "Simulation crashed"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        # Should NOT trigger user ask when below max
        assert "ask_user_trigger" not in result

    def test_execution_validator_fail_increments_existing_counter(self, base_state):
        """Test execution validator increments existing failure counter correctly."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Still failing")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["execution_failure_count"] = 1
        base_state["total_execution_failures"] = 2

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 2
        assert result["total_execution_failures"] == 3

    def test_execution_validator_fail_at_max_triggers_user_ask(self, base_state):
        """Test execution validator triggers user ask when max failures reached."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Max failures reached")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["execution_failure_count"] = 2  # At max (MAX_EXECUTION_FAILURES = 2)
        base_state["total_execution_failures"] = 3
        base_state["run_error"] = "Persistent error"
        base_state.setdefault("runtime_config", {})["max_execution_failures"] = 2

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        # Counter should stay at max (not increment beyond)
        assert result["execution_failure_count"] == 2, (
            f"Counter should stay at max (2), got: {result['execution_failure_count']}"
        )
        # total_execution_failures should still increment
        assert result["total_execution_failures"] == 4
        # Should trigger user ask
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["pending_user_questions"] is not None
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        # Verify question contains all expected components
        question = result["pending_user_questions"][0]
        assert "Persistent error" in question
        assert "2/2" in question, f"Question should show failure count 2/2, got: {question}"
        assert "RETRY_WITH_GUIDANCE" in question
        assert "SKIP_STAGE" in question
        assert "STOP" in question
        # Should set ask_user_trigger for user interaction routing
        assert result.get("ask_user_trigger") is not None
        # Should set last_node_before_ask_user
        assert result["last_node_before_ask_user"] == "execution_check"

    def test_execution_validator_fail_with_custom_max_from_runtime_config(self, base_state):
        """Test execution validator respects custom max_execution_failures from runtime_config."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Failing")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["execution_failure_count"] = 4  # At custom max
        base_state["total_execution_failures"] = 5
        base_state["run_error"] = "Custom error"
        base_state.setdefault("runtime_config", {})["max_execution_failures"] = 4

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 4  # Stays at custom max
        assert result["total_execution_failures"] == 6
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert "Custom error" in result["pending_user_questions"][0]
        assert "4/4" in result["pending_user_questions"][0], (
            f"Question should show 4/4 failures, got: {result['pending_user_questions'][0]}"
        )
        assert result.get("ask_user_trigger") is not None
        assert result["last_node_before_ask_user"] == "execution_check"

    def test_execution_validator_fail_under_max_does_not_trigger_user_ask(self, base_state):
        """Test execution validator does NOT trigger user ask when below max failures."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="First failure")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["execution_failure_count"] = 0  # Below max
        base_state["total_execution_failures"] = 0
        base_state["run_error"] = "Some error"
        base_state.setdefault("runtime_config", {})["max_execution_failures"] = 3

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        # Should NOT trigger user ask
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result
        assert "last_node_before_ask_user" not in result


class TestPhysicsFailureLimit:
    """Test physics_sanity_node failure limit handling and user escalation."""

    def test_physics_sanity_fail_at_max_triggers_user_ask(self, base_state):
        """Test physics sanity triggers user ask when max failures reached."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Physics check failed repeatedly"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 2  # At max (MAX_PHYSICS_FAILURES = 2)
        base_state.setdefault("runtime_config", {})["max_physics_failures"] = 2

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        # Counter should stay at max
        assert result["physics_failure_count"] == 2, (
            f"Counter should stay at max (2), got: {result['physics_failure_count']}"
        )
        # Should trigger user ask
        assert result["ask_user_trigger"] == "physics_failure_limit"
        assert result["pending_user_questions"] is not None
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        # Verify question content
        question = result["pending_user_questions"][0]
        assert "2/2" in question, f"Question should show failure count 2/2, got: {question}"
        assert "RETRY" in question or "ACCEPT" in question
        assert "SKIP_STAGE" in question
        assert "STOP" in question
        # Should set ask_user_trigger for user interaction routing
        assert result.get("ask_user_trigger") is not None
        # Should set last_node_before_ask_user
        assert result["last_node_before_ask_user"] == "physics_check"

    def test_physics_sanity_fail_under_max_does_not_trigger_user_ask(self, base_state):
        """Test physics sanity does NOT trigger user ask when below max failures."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="First physics failure"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 0  # Below max
        base_state.setdefault("runtime_config", {})["max_physics_failures"] = 3

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1
        # Should NOT trigger user ask
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result
        assert "last_node_before_ask_user" not in result

    def test_physics_sanity_fail_with_custom_max_from_runtime_config(self, base_state):
        """Test physics sanity respects custom max_physics_failures from runtime_config."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Physics keeps failing"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 5  # At custom max
        base_state.setdefault("runtime_config", {})["max_physics_failures"] = 5

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 5  # Stays at custom max
        assert result["ask_user_trigger"] == "physics_failure_limit"
        assert "5/5" in result["pending_user_questions"][0], (
            f"Question should show 5/5 failures, got: {result['pending_user_questions'][0]}"
        )
        assert result.get("ask_user_trigger") is not None
        assert result["last_node_before_ask_user"] == "physics_check"


class TestTimeoutHandling:
    """Test timeout handling in execution_validator_node."""

    def test_timeout_with_ask_user_fallback(self, base_state):
        """Test timeout handling with default ask_user fallback."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"timeout_exceeded": True}
        base_state["run_error"] = "Execution exceeded timeout of 300 seconds"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="ask_user"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        # Should fail without calling LLM
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        # Feedback should contain timeout info
        assert "timed out" in result["execution_feedback"].lower(), (
            f"Expected 'timed out' in feedback, got: {result['execution_feedback']}"
        )
        # Original run_error should be in feedback
        assert "Execution exceeded timeout of 300 seconds" in result["execution_feedback"]
        # LLM should NOT be called for timeout
        mock_llm.assert_not_called()

    def test_timeout_with_skip_with_warning_fallback(self, base_state):
        """Test timeout handling with skip_with_warning fallback converts to pass verdict."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"timeout_exceeded": True}
        base_state["run_error"] = "Timeout exceeded"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="skip_with_warning"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        # Should pass (skip with warning)
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Feedback should indicate skip_with_warning strategy
        assert "skip_with_warning" in result["execution_feedback"], (
            f"Expected 'skip_with_warning' in feedback, got: {result['execution_feedback']}"
        )
        assert "Timeout exceeded" in result["execution_feedback"]
        # Should NOT increment failure counters
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        # Should NOT trigger user ask
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result
        assert "awaiting_user_input" not in result
        # LLM should NOT be called for timeout
        mock_llm.assert_not_called()

    def test_timeout_detected_via_string_pattern(self, base_state):
        """Test timeout detection via 'exceeded timeout' string pattern in run_error."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}  # No timeout_exceeded flag
        base_state["run_error"] = "Execution exceeded timeout of 600 seconds"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="ask_user"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        # Should detect timeout via string pattern and fail
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        # Feedback should indicate timeout
        feedback_lower = result["execution_feedback"].lower()
        assert "timed out" in feedback_lower or "timeout" in feedback_lower, (
            f"Expected timeout indication in feedback, got: {result['execution_feedback']}"
        )
        # LLM should NOT be called - timeout is handled locally
        mock_llm.assert_not_called()

    def test_timeout_detected_via_timeout_error_string(self, base_state):
        """Test timeout detection via 'timeout_error' string pattern."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}
        base_state["run_error"] = "Timeout_error: Process killed"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="ask_user"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        # Should still increment counters
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1
        # LLM should NOT be called
        mock_llm.assert_not_called()

    def test_timeout_not_detected_without_pattern(self, base_state):
        """Test that non-timeout errors are NOT treated as timeouts."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Error analyzed")
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}  # No timeout_exceeded flag
        base_state["run_error"] = "Segmentation fault - memory corruption"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="ask_user"
        ), patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = execution_validator_node(base_state)

        # Should call LLM for non-timeout errors
        mock_llm.assert_called_once()
        assert result["execution_verdict"] == "fail"


class TestLLMErrorHandling:
    """Test LLM error handling in execution validator."""

    def test_execution_validator_handles_llm_exception(self, base_state):
        """Test execution validator handles LLM exceptions by auto-approving."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", side_effect=Exception("LLM API error")
        ):
            result = execution_validator_node(base_state)

        # Should auto-approve on LLM error
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Feedback should indicate auto-approval due to LLM error
        feedback_lower = result["execution_feedback"].lower()
        assert "auto" in feedback_lower or "llm" in feedback_lower, (
            f"Expected LLM error indication in feedback, got: {result['execution_feedback']}"
        )
        # Should NOT increment counters on auto-approve
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        # Should NOT trigger user ask
        assert "ask_user_trigger" not in result

    def test_execution_validator_handles_llm_timeout_exception(self, base_state):
        """Test execution validator handles LLM timeout exceptions."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", 
            side_effect=TimeoutError("Request timed out")
        ):
            result = execution_validator_node(base_state)

        # Should auto-approve on any exception
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"

    def test_execution_validator_handles_missing_verdict(self, base_state):
        """Test execution validator handles missing verdict in LLM response by defaulting to pass."""
        from src.agents.execution import execution_validator_node

        # Response missing verdict field
        mock_response = {"summary": "Some response without verdict"}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        # Should default to pass when verdict missing
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Feedback should indicate missing verdict AND include original summary
        assert "missing verdict" in result["execution_feedback"].lower()
        assert "defaulting to pass" in result["execution_feedback"].lower()
        assert "Some response without verdict" in result["execution_feedback"], (
            f"Original summary should be preserved, got: {result['execution_feedback']}"
        )
        # Should NOT increment counters
        assert "execution_failure_count" not in result

    def test_execution_validator_handles_missing_verdict_and_summary(self, base_state):
        """Test execution validator handles missing verdict and summary."""
        from src.agents.execution import execution_validator_node

        # Response missing both verdict and summary
        mock_response = {"some_other_field": "value"}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Feedback should indicate missing verdict
        assert "Missing verdict" in result["execution_feedback"]
        assert "defaulting to pass" in result["execution_feedback"].lower()

    def test_execution_validator_handles_empty_response(self, base_state):
        """Test execution validator handles empty LLM response."""
        from src.agents.execution import execution_validator_node

        # Empty response
        mock_response = {}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        # Feedback should indicate missing verdict
        assert "missing verdict" in result["execution_feedback"].lower()


class TestPhysicsSanityBehavior:
    """Additional coverage for physics_sanity_node."""

    def test_physics_sanity_passes(self, base_state):
        """Test physics sanity passes with correct field population."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="pass",
            summary="Physics checks passed",
            checks_performed=["energy_conservation", "value_ranges"],
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_description"] = {"geometry": "nanorod"}
        base_state["physics_failure_count"] = 0
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        # Strict assertions
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("agent_name") == "physics_sanity"
        assert call_kwargs.get("state") is not None
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Physics checks passed"
        # Pass should NOT increment any counters
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        # Pass should NOT set design_feedback
        assert "design_feedback" not in result
        # Pass should NOT trigger user ask
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result
        assert "awaiting_user_input" not in result
        # Should include design spec in user content
        assert "Design Spec" in call_kwargs["user_content"]
        assert "nanorod" in call_kwargs["user_content"]
        # Should include stage outputs
        assert "Stage Outputs" in call_kwargs["user_content"]
        assert "spectrum.csv" in call_kwargs["user_content"]

    def test_physics_sanity_passes_backtrack_suggestion(self, base_state):
        """Test physics sanity correctly handles backtrack suggestion with design_flaw verdict."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Fundamental issue with simulation setup",
            backtrack_suggestion={
                "suggest_backtrack": True,
                "target_stage_id": "stage_0",
                "reason": "Material properties need revalidation",
            },
        )

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}
        base_state["design_revision_count"] = 0
        base_state["physics_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        # Strict assertions on backtrack handling
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Fundamental issue with simulation setup"
        # design_flaw should set design_feedback
        assert result["design_feedback"] == "Fundamental issue with simulation setup"
        # Backtrack suggestion should be included
        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["backtrack_suggestion"]["target_stage_id"] == "stage_0"
        assert result["backtrack_suggestion"]["reason"] == "Material properties need revalidation"
        # design_flaw should increment design_revision_count
        assert result["design_revision_count"] == 1
        # design_flaw should NOT increment physics_failure_count
        assert "physics_failure_count" not in result

    def test_physics_sanity_without_backtrack_suggestion(self, base_state):
        """Test physics sanity handles missing backtrack_suggestion gracefully - not added to result."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="pass",
            summary="Physics OK"
        )
        # No backtrack_suggestion in response

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        # backtrack_suggestion should NOT be in result when suggest_backtrack is False
        assert "backtrack_suggestion" not in result, (
            f"backtrack_suggestion should not be in result for pass verdict, got: {result.get('backtrack_suggestion')}"
        )

    def test_physics_sanity_fail_verdict_increments_physics_failure_count(self, base_state):
        """Test physics sanity fail verdict increments physics_failure_count but not design_revision_count."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Physics check failed: values out of range"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 0
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Physics check failed: values out of range"
        # fail should increment physics_failure_count
        assert result["physics_failure_count"] == 1
        # fail should NOT increment design_revision_count
        assert "design_revision_count" not in result
        # fail should NOT set design_feedback
        assert "design_feedback" not in result
        # fail should NOT include backtrack_suggestion (unless explicitly provided)
        assert "backtrack_suggestion" not in result

    def test_physics_sanity_warning_verdict(self, base_state):
        """Test physics sanity handles warning verdict correctly - no counters incremented."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="warning",
            summary="Minor physics concerns but proceed"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 0
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "warning"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Minor physics concerns but proceed"
        # Warning should NOT increment any counters
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        # Warning should NOT set design_feedback
        assert "design_feedback" not in result
        # Warning should NOT trigger user ask
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result
        assert "awaiting_user_input" not in result
        # Warning should NOT include backtrack_suggestion
        assert "backtrack_suggestion" not in result

    def test_physics_sanity_handles_missing_design_description(self, base_state):
        """Test physics sanity handles missing design_description gracefully - no crash."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state.pop("design_description", None)

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        # Should still call LLM even without design_description
        assert "PHYSICS SANITY CHECK" in call_kwargs["user_content"]
        # Stage Outputs should still be present
        assert "Stage Outputs" in call_kwargs["user_content"]
        # Design Spec section should be omitted when design_description is None
        # (because design is None, not {})

    def test_physics_sanity_handles_llm_exception(self, base_state):
        """Test physics sanity handles LLM exceptions by auto-approving."""
        from src.agents.execution import physics_sanity_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 0
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", side_effect=Exception("LLM unavailable")
        ):
            result = physics_sanity_node(base_state)

        # Should auto-approve on LLM error
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        # Feedback should indicate auto-approval due to LLM error
        feedback_lower = result["physics_feedback"].lower()
        assert "auto" in feedback_lower or "llm" in feedback_lower, (
            f"Expected LLM error indication in feedback, got: {result['physics_feedback']}"
        )
        # Should NOT increment any counters on auto-approve
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        # Should NOT trigger user ask
        assert "ask_user_trigger" not in result

    def test_physics_sanity_handles_missing_verdict(self, base_state):
        """Test physics sanity handles missing verdict in LLM response by defaulting to pass."""
        from src.agents.execution import physics_sanity_node

        mock_response = {"summary": "Response without verdict"}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        # Feedback should indicate missing verdict AND include original summary
        assert "missing verdict" in result["physics_feedback"].lower()
        assert "defaulting to pass" in result["physics_feedback"].lower()
        assert "Response without verdict" in result["physics_feedback"], (
            f"Original summary should be preserved, got: {result['physics_feedback']}"
        )

    def test_physics_sanity_handles_empty_response(self, base_state):
        """Test physics sanity handles empty LLM response."""
        from src.agents.execution import physics_sanity_node

        mock_response = {}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        # Feedback should indicate missing verdict
        assert "missing verdict" in result["physics_feedback"].lower()


class TestCounterEdgeCases:
    """Test counter incrementing edge cases and boundary conditions."""

    def test_execution_validator_fail_with_none_counter(self, base_state):
        """Test execution validator handles None counter value correctly - treated as 0."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Failed")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["execution_failure_count"] = None  # None counter
        base_state["total_execution_failures"] = None

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        # Should treat None as 0 and increment to 1
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1, (
            f"None counter should be treated as 0 and increment to 1, got: {result['execution_failure_count']}"
        )
        assert result["total_execution_failures"] == 1, (
            f"None counter should be treated as 0 and increment to 1, got: {result['total_execution_failures']}"
        )

    def test_execution_validator_fail_with_missing_counter(self, base_state):
        """Test execution validator handles missing counter key correctly - treated as 0."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Failed")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state.pop("execution_failure_count", None)
        base_state.pop("total_execution_failures", None)

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        # Should treat missing as 0 and increment to 1
        assert result["execution_verdict"] == "fail"
        assert result["workflow_phase"] == "execution_validation"
        assert result["execution_failure_count"] == 1, (
            f"Missing counter should default to 0 and increment to 1, got: {result['execution_failure_count']}"
        )
        assert result["total_execution_failures"] == 1, (
            f"Missing counter should default to 0 and increment to 1, got: {result['total_execution_failures']}"
        )

    def test_physics_sanity_design_flaw_increments_existing_counter(self, base_state):
        """Test physics sanity increments existing design_revision_count on design_flaw verdict."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Design issue"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_revision_count"] = 1
        base_state["physics_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 2, (
            f"design_revision_count should increment from 1 to 2, got: {result['design_revision_count']}"
        )
        # design_flaw should NOT increment physics_failure_count
        assert "physics_failure_count" not in result

    def test_physics_sanity_fail_increments_existing_counter(self, base_state):
        """Test physics sanity increments existing physics_failure_count on fail verdict."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Physics failed"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 1
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 2, (
            f"physics_failure_count should increment from 1 to 2, got: {result['physics_failure_count']}"
        )
        # fail should NOT increment design_revision_count
        assert "design_revision_count" not in result

    def test_physics_sanity_fail_with_none_counter(self, base_state):
        """Test physics sanity handles None physics_failure_count correctly."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="fail", summary="Failed")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = None

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1, (
            f"None counter should be treated as 0 and increment to 1, got: {result['physics_failure_count']}"
        )

    def test_physics_sanity_design_flaw_with_none_counter(self, base_state):
        """Test physics sanity handles None design_revision_count correctly."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="design_flaw", summary="Design issue")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_revision_count"] = None

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 1, (
            f"None counter should be treated as 0 and increment to 1, got: {result['design_revision_count']}"
        )


class TestDesignDescriptionHandling:
    """Test design_description handling in physics_sanity_node."""

    def test_physics_sanity_with_dict_design_description(self, base_state):
        """Test physics sanity handles dict design_description correctly - serialized as JSON."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_description"] = {
            "geometry": "nanorod",
            "dimensions": {"length": 100, "diameter": 40},
            "material": "gold"
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        call_kwargs = mock_call.call_args.kwargs
        user_content = call_kwargs["user_content"]
        # Should have Design Spec section
        assert "## Design Spec" in user_content
        # Should serialize dict as JSON with all fields
        assert "nanorod" in user_content
        assert "100" in user_content
        assert "40" in user_content
        assert "gold" in user_content

    def test_physics_sanity_with_string_design_description(self, base_state):
        """Test physics sanity handles string design_description correctly - included directly."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_description"] = "Gold nanorod with length 100nm"

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        call_kwargs = mock_call.call_args.kwargs
        user_content = call_kwargs["user_content"]
        # Should include string directly
        assert "Gold nanorod with length 100nm" in user_content
        assert "## Design Spec" in user_content

    def test_physics_sanity_with_empty_design_description(self, base_state):
        """Test physics sanity handles empty dict design_description - excludes section."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_description"] = {}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        call_kwargs = mock_call.call_args.kwargs
        user_content = call_kwargs["user_content"]
        # Empty dict should be excluded (no useful information to show)
        assert "## Design Spec" not in user_content

    def test_physics_sanity_with_none_design_description(self, base_state):
        """Test physics sanity handles None design_description - no Design Spec section."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_description"] = None

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        call_kwargs = mock_call.call_args.kwargs
        user_content = call_kwargs["user_content"]
        # When design is None, Design Spec section should NOT be added
        # (the code checks `if design is not None`)
        # This is different from empty dict {}
        # Verify basic structure is still there
        assert "PHYSICS SANITY CHECK" in user_content
        assert "Stage Outputs" in user_content


class TestBacktrackSuggestionEdgeCases:
    """Test backtrack_suggestion edge cases."""

    def test_physics_sanity_backtrack_with_false_suggest_backtrack(self, base_state):
        """Test physics sanity handles backtrack_suggestion with suggest_backtrack=False - not added to result."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Design issue",
            backtrack_suggestion={
                "suggest_backtrack": False,
                "target_stage_id": "stage_0",
                "reason": "Not needed"
            }
        )

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        # design_flaw should still increment design_revision_count
        assert result["design_revision_count"] == 1
        # Should NOT include backtrack_suggestion in result when suggest_backtrack is False
        assert "backtrack_suggestion" not in result, (
            f"backtrack_suggestion should not be in result when suggest_backtrack=False, got: {result.get('backtrack_suggestion')}"
        )

    def test_physics_sanity_backtrack_with_missing_fields(self, base_state):
        """Test physics sanity handles backtrack_suggestion with missing fields - still included."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Design issue",
            backtrack_suggestion={
                "suggest_backtrack": True
                # Missing target_stage_id and reason
            }
        )

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 1
        # Should still include backtrack_suggestion even with missing fields
        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        # Missing fields should not be present (not defaulted)
        assert "target_stage_id" not in result["backtrack_suggestion"] or result["backtrack_suggestion"].get("target_stage_id") is None

    def test_physics_sanity_backtrack_with_non_dict(self, base_state):
        """Test physics sanity handles non-dict backtrack_suggestion gracefully - ignored."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Design issue"
        )
        # Set backtrack_suggestion to non-dict (should be handled gracefully)
        mock_response["backtrack_suggestion"] = "invalid"

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1
        # Non-dict backtrack_suggestion should be ignored
        assert "backtrack_suggestion" not in result, (
            f"Non-dict backtrack_suggestion should not be in result, got: {result.get('backtrack_suggestion')}"
        )

    def test_physics_sanity_backtrack_with_none(self, base_state):
        """Test physics sanity handles None backtrack_suggestion gracefully."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Design issue"
        )
        mock_response["backtrack_suggestion"] = None

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 1
        # None backtrack_suggestion should be ignored
        assert "backtrack_suggestion" not in result


class TestPromptContentValidation:
    """Test that prompts contain all required information."""

    def test_execution_validator_prompt_includes_all_stage_output_fields(self, base_state):
        """Test execution validator prompt includes all stage output fields."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {
            "files": ["/tmp/output1.csv", "/tmp/output2.csv"],
            "exit_code": 0,
            "stdout": "Simulation complete",
            "execution_time": 42.5
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            execution_validator_node(base_state)

        call_kwargs = mock_call.call_args.kwargs
        user_content = call_kwargs["user_content"]
        # All stage_outputs fields should be in user_content
        assert "output1.csv" in user_content
        assert "output2.csv" in user_content
        assert "exit_code" in user_content
        assert "0" in user_content
        assert "stdout" in user_content
        assert "Simulation complete" in user_content
        assert "execution_time" in user_content
        assert "42.5" in user_content

    def test_physics_sanity_prompt_structure(self, base_state):
        """Test physics sanity prompt has correct structure with all sections."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"], "data": {"peak": 650}}
        base_state["design_description"] = {"geometry": "nanorod", "material": "gold"}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            physics_sanity_node(base_state)

        call_kwargs = mock_call.call_args.kwargs
        user_content = call_kwargs["user_content"]
        
        # Check overall structure
        assert "# PHYSICS SANITY CHECK FOR STAGE: stage_0" in user_content
        assert "## Stage Outputs" in user_content
        assert "## Design Spec" in user_content
        
        # Check stage outputs content
        assert "spectrum.csv" in user_content
        assert "peak" in user_content
        assert "650" in user_content
        
        # Check design spec content
        assert "nanorod" in user_content
        assert "gold" in user_content


class TestVerdictCaseSensitivity:
    """Test verdict handling is case-sensitive."""

    def test_execution_validator_verdict_case_sensitivity(self, base_state):
        """Test execution validator verdict is case-sensitive - must be lowercase."""
        from src.agents.execution import execution_validator_node

        # Test with uppercase verdict (should be passed through as-is)
        mock_response = {"verdict": "PASS", "summary": "OK"}
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        # Verdict should be passed through as-is (case-sensitive)
        assert result["execution_verdict"] == "PASS"

    def test_execution_validator_fail_verdict_case(self, base_state):
        """Test execution validator only increments counters for lowercase 'fail' verdict."""
        from src.agents.execution import execution_validator_node

        # Test with uppercase FAIL - should NOT trigger counter increment
        mock_response = {"verdict": "FAIL", "summary": "Failed"}
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        # Verdict is "FAIL" (uppercase), but counter increment logic checks for "fail" (lowercase)
        # So counters should NOT be incremented
        assert result["execution_verdict"] == "FAIL"
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result

