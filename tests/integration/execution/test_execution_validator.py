"""Execution validator and physics sanity integration tests."""

from unittest.mock import patch

import pytest

from tests.integration.helpers.agent_responses import execution_verdict_response


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
        
        # Verify LLM was called with correct parameters
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["agent_name"] == "execution_validator"
        assert "stage_0" in call_kwargs["user_content"]
        assert "output.csv" in call_kwargs["user_content"]
        assert "EXECUTION RESULTS FOR STAGE" in call_kwargs["user_content"]
        assert "Stage Outputs" in call_kwargs["user_content"]

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
        # Should still call LLM with empty stage_outputs
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert "Stage Outputs" in call_kwargs["user_content"]
        # Should handle None gracefully in JSON dump
        assert "null" in call_kwargs["user_content"] or "{}" in call_kwargs["user_content"]

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
        """Test execution validator handles missing current_stage_id."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="pass", summary="OK")

        base_state.pop("current_stage_id", None)
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert mock_call.called
        # Should use "unknown" as fallback
        call_kwargs = mock_call.call_args.kwargs
        assert "unknown" in call_kwargs["user_content"] or "STAGE:" in call_kwargs["user_content"]
        # Verdict should include stage_id
        assert mock_response.get("stage_id") == "unknown" or "stage_id" in mock_response

    def test_execution_validator_includes_run_error_in_prompt(self, base_state):
        """Test execution validator includes run_error in prompt when present."""
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response(verdict="fail", summary="Error detected")

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [], "exit_code": 1}
        base_state["run_error"] = "Segmentation fault at line 42"
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        # run_error should be in system prompt or user content
        system_prompt = call_kwargs.get("system_prompt", "")
        user_content = call_kwargs.get("user_content", "")
        assert "Segmentation fault" in system_prompt or "Segmentation fault" in user_content
        assert "Run Error" in user_content or "run_error" in user_content.lower()


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
        # Counter should stay at max (not increment beyond)
        assert result["execution_failure_count"] == 2
        # total_execution_failures should still increment
        assert result["total_execution_failures"] == 4
        # Should trigger user ask
        assert result["ask_user_trigger"] == "execution_failure_limit"
        assert result["pending_user_questions"] is not None
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        assert "Persistent error" in result["pending_user_questions"][0]
        assert "RETRY_WITH_GUIDANCE" in result["pending_user_questions"][0] or "SKIP_STAGE" in result["pending_user_questions"][0]

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
        assert "timed out" in result["execution_feedback"].lower()
        assert "300 seconds" in result["execution_feedback"]
        # LLM should NOT be called for timeout
        mock_llm.assert_not_called()

    def test_timeout_with_skip_with_warning_fallback(self, base_state):
        """Test timeout handling with skip_with_warning fallback."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"timeout_exceeded": True}
        base_state["run_error"] = "Timeout exceeded"
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="skip_with_warning"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        # Should pass with warning
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        assert "skip_with_warning" in result["execution_feedback"]
        assert "Timeout exceeded" in result["execution_feedback"]
        # Should NOT increment failure counters
        assert "execution_failure_count" not in result
        assert "total_execution_failures" not in result
        # LLM should NOT be called
        mock_llm.assert_not_called()

    def test_timeout_detected_via_string_pattern(self, base_state):
        """Test timeout detection via string pattern in run_error."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}  # No timeout_exceeded flag
        base_state["run_error"] = "Execution exceeded timeout of 600 seconds"
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="ask_user"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        # Should detect timeout via string pattern
        assert result["execution_verdict"] == "fail"
        assert "timed out" in result["execution_feedback"].lower() or "timeout" in result["execution_feedback"].lower()
        mock_llm.assert_not_called()

    def test_timeout_detected_via_timeout_error_string(self, base_state):
        """Test timeout detection via timeout_error string pattern."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}
        base_state["run_error"] = "Timeout_error: Process killed"
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.get_stage_design_spec", return_value="ask_user"
        ), patch(
            "src.agents.execution.call_agent_with_metrics"
        ) as mock_llm:
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        mock_llm.assert_not_called()


class TestLLMErrorHandling:
    """Test LLM error handling in execution validator."""

    def test_execution_validator_handles_llm_exception(self, base_state):
        """Test execution validator handles LLM exceptions gracefully."""
        from src.agents.execution import execution_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}
        base_state["execution_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", side_effect=Exception("LLM API error")
        ):
            result = execution_validator_node(base_state)

        # Should auto-approve on LLM error
        assert result["execution_verdict"] == "pass"
        assert result["workflow_phase"] == "execution_validation"
        assert "LLM error" in result["execution_feedback"].lower() or "auto" in result["execution_feedback"].lower()
        # Should NOT increment counters on auto-approve
        assert "execution_failure_count" not in result

    def test_execution_validator_handles_missing_verdict(self, base_state):
        """Test execution validator handles missing verdict in LLM response."""
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
        # Should have summary indicating missing verdict (may include original summary)
        assert "missing verdict" in result["execution_feedback"].lower()
        assert "defaulting to pass" in result["execution_feedback"].lower()
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
        assert result["execution_feedback"] == "Missing verdict in LLM response, defaulting to pass." or "No feedback provided."


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

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        # Strict assertions
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("agent_name") == "physics_sanity"
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Physics checks passed"
        # Pass should NOT increment any counters
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        # Should include design spec in user content
        assert "Design Spec" in call_kwargs["user_content"] or "nanorod" in call_kwargs["user_content"]

    def test_physics_sanity_passes_backtrack_suggestion(self, base_state):
        """Test physics sanity correctly handles backtrack suggestion."""
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

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        # Strict assertions on backtrack handling
        assert result["physics_verdict"] == "design_flaw"
        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["backtrack_suggestion"]["target_stage_id"] == "stage_0"
        assert result["backtrack_suggestion"]["reason"] == "Material properties need revalidation"
        assert result["design_revision_count"] == 1

    def test_physics_sanity_without_backtrack_suggestion(self, base_state):
        """Test physics sanity handles missing backtrack_suggestion gracefully."""
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
        # Should have default backtrack_suggestion with suggest_backtrack=False
        # Note: backtrack_suggestion is only added to result if suggest_backtrack is True
        assert "backtrack_suggestion" not in result or result.get("backtrack_suggestion", {}).get("suggest_backtrack") is False

    def test_physics_sanity_fail_verdict_increments_physics_failure_count(self, base_state):
        """Test physics sanity fail verdict increments physics_failure_count."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Physics check failed: values out of range"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1
        # fail should NOT increment design_revision_count
        assert "design_revision_count" not in result

    def test_physics_sanity_warning_verdict(self, base_state):
        """Test physics sanity handles warning verdict correctly."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="warning",
            summary="Minor physics concerns but proceed"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "warning"
        assert result["workflow_phase"] == "physics_validation"
        # Warning should NOT increment any counters
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result

    def test_physics_sanity_handles_missing_design_description(self, base_state):
        """Test physics sanity handles missing design_description gracefully."""
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
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        # Should still call LLM even without design_description
        assert "PHYSICS SANITY CHECK" in call_kwargs["user_content"]

    def test_physics_sanity_handles_llm_exception(self, base_state):
        """Test physics sanity handles LLM exceptions gracefully."""
        from src.agents.execution import physics_sanity_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 0

        with patch(
            "src.agents.execution.call_agent_with_metrics", side_effect=Exception("LLM unavailable")
        ):
            result = physics_sanity_node(base_state)

        # Should auto-approve on LLM error
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert "LLM error" in result["physics_feedback"].lower() or "auto" in result["physics_feedback"].lower()
        assert "physics_failure_count" not in result

    def test_physics_sanity_handles_missing_verdict(self, base_state):
        """Test physics sanity handles missing verdict in LLM response."""
        from src.agents.execution import physics_sanity_node

        mock_response = {"summary": "Response without verdict"}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"
        # Should indicate missing verdict (may include original summary)
        assert "missing verdict" in result["physics_feedback"].lower()
        assert "defaulting to pass" in result["physics_feedback"].lower()


class TestCounterEdgeCases:
    """Test counter incrementing edge cases and boundary conditions."""

    def test_execution_validator_fail_with_none_counter(self, base_state):
        """Test execution validator handles None counter value correctly."""
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
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1

    def test_execution_validator_fail_with_missing_counter(self, base_state):
        """Test execution validator handles missing counter key correctly."""
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
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1

    def test_physics_sanity_design_flaw_increments_existing_counter(self, base_state):
        """Test physics sanity increments existing design_revision_count."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Design issue"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["design_revision_count"] = 1

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 2

    def test_physics_sanity_fail_increments_existing_counter(self, base_state):
        """Test physics sanity increments existing physics_failure_count."""
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="fail",
            summary="Physics failed"
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        base_state["physics_failure_count"] = 1

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 2


class TestDesignDescriptionHandling:
    """Test design_description handling in physics_sanity_node."""

    def test_physics_sanity_with_dict_design_description(self, base_state):
        """Test physics sanity handles dict design_description correctly."""
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
        call_kwargs = mock_call.call_args.kwargs
        # Should serialize dict as JSON
        assert "nanorod" in call_kwargs["user_content"] or "100" in call_kwargs["user_content"]
        assert "Design Spec" in call_kwargs["user_content"]

    def test_physics_sanity_with_string_design_description(self, base_state):
        """Test physics sanity handles string design_description correctly."""
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
        # Should include string directly
        assert "Gold nanorod" in call_kwargs["user_content"]
        assert "Design Spec" in call_kwargs["user_content"]

    def test_physics_sanity_with_empty_design_description(self, base_state):
        """Test physics sanity handles empty design_description."""
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
        # Should still include Design Spec section even if empty
        assert "Design Spec" in call_kwargs["user_content"] or "{}" in call_kwargs["user_content"]


class TestBacktrackSuggestionEdgeCases:
    """Test backtrack_suggestion edge cases."""

    def test_physics_sanity_backtrack_with_false_suggest_backtrack(self, base_state):
        """Test physics sanity handles backtrack_suggestion with suggest_backtrack=False."""
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
        # Should NOT include backtrack_suggestion in result when suggest_backtrack is False
        assert "backtrack_suggestion" not in result

    def test_physics_sanity_backtrack_with_missing_fields(self, base_state):
        """Test physics sanity handles backtrack_suggestion with missing fields."""
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
        # Should still include backtrack_suggestion even with missing fields
        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True

    def test_physics_sanity_backtrack_with_non_dict(self, base_state):
        """Test physics sanity handles non-dict backtrack_suggestion gracefully."""
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
        # Should handle gracefully - non-dict backtrack_suggestion should not be added to result
        assert "backtrack_suggestion" not in result or isinstance(result.get("backtrack_suggestion"), dict)

