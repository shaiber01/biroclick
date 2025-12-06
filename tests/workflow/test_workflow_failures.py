"""
Tests for workflow failure handling in execution and physics validation nodes.

This module tests:
- execution_validator_node: Validates that simulations ran correctly
- physics_sanity_node: Validates physics plausibility of results

Both nodes handle failure modes, timeouts, and escalation to user.
"""
from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest

from src.agents import execution_validator_node, physics_sanity_node
from schemas.state import (
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_DESIGN_REVISIONS,
)

from tests.workflow.fixtures import MockResponseFactory


class TestWorkflowWithFailures:
    """Test workflow recovery from failures."""

    def test_execution_failure_recovery(self, base_state):
        """Test handling of execution failures.
        
        Validates that when execution fails:
        - execution_verdict is set to "fail"
        - execution_failure_count is incremented
        - workflow_phase is set correctly
        - execution_feedback contains the summary
        - total_execution_failures is incremented
        """
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["code"] = "# test code"
        base_state["execution_result"] = {
            "success": False,
            "error": "Simulation crashed",
            "output_files": [],
        }
        # Ensure counters are initialized
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_response = MockResponseFactory.execution_validator_fail()
            mock_llm.return_value = mock_response
            result = execution_validator_node(base_state)

        # Core assertions - verdict and counters
        assert result["execution_verdict"] == "fail", "Execution verdict should be 'fail'"
        assert result["execution_failure_count"] == 1, "Failure count should increment to 1"
        assert result["total_execution_failures"] == 1, "Total failures should increment to 1"
        
        # Workflow phase assertion
        assert result["workflow_phase"] == "execution_validation", "Should set workflow phase"
        
        # Feedback should contain the summary from mock response
        assert "execution_feedback" in result, "Should include execution feedback"
        assert result["execution_feedback"] == mock_response["summary"], "Feedback should match summary"

    def test_execution_failure_counter_increments_on_subsequent_failures(self, base_state):
        """Test that failure counter increments correctly on subsequent failures."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["execution_failure_count"] = 1  # Already had one failure
        base_state["total_execution_failures"] = 1

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            result = execution_validator_node(base_state)

        assert result["execution_failure_count"] == 2, "Failure count should increment from 1 to 2"
        assert result["total_execution_failures"] == 2, "Total failures should increment from 1 to 2"


class TestExecutionValidatorLogic:
    """Test execution validator logic."""

    def test_successful_execution_metrics(self, base_state):
        """Test validation of successful execution with metrics.
        
        Validates that on success:
        - execution_verdict is set to "pass"
        - workflow_phase is set to "execution_validation"
        - execution_feedback contains summary
        - execution_failure_count is NOT present (only set on failures)
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_result"] = {
            "success": True,
            "output_files": ["data.csv"],
            "runtime_seconds": 45.5,
        }

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_response = MockResponseFactory.execution_validator_pass()
            mock_llm.return_value = mock_response

            result = execution_validator_node(base_state)

            # Verify verdict
            assert result["execution_verdict"] == "pass", "Verdict should be 'pass'"
            
            # Verify workflow phase
            assert result["workflow_phase"] == "execution_validation", "Should set workflow phase"
            
            # Verify feedback
            assert "execution_feedback" in result, "Should include execution feedback"
            assert result["execution_feedback"] == mock_response["summary"], "Feedback should match summary"
            
            # Verify failure count is NOT in result (only set on failures)
            assert "execution_failure_count" not in result, (
                "execution_failure_count should not be set on success"
            )

    def test_execution_failure_handling(self, base_state):
        """Test validation of failed execution.
        
        Validates that on failure:
        - execution_verdict is set to "fail"
        - execution_failure_count is incremented
        - workflow_phase is set correctly
        - execution_feedback contains the error summary
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0
        base_state["execution_result"] = {
            "success": False,
            "error": "Timeout",
            "output_files": [],
        }

        error_msg = "Timeout occurred during simulation"
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_response = MockResponseFactory.execution_validator_fail(error_msg)
            mock_llm.return_value = mock_response

            result = execution_validator_node(base_state)

            # Core assertions
            assert result["execution_verdict"] == "fail", "Verdict should be 'fail'"
            assert result["execution_failure_count"] == 1, "Failure count should be 1"
            
            # Workflow phase
            assert result["workflow_phase"] == "execution_validation", "Should set workflow phase"
            
            # Feedback should contain the error
            assert "execution_feedback" in result, "Should include execution feedback"
            assert result["execution_feedback"] == error_msg, "Feedback should match error message"


class TestExecutionValidatorEdgeCases:
    """Test edge cases for execution validator."""

    def test_timeout_with_skip_with_warning_fallback(self, base_state):
        """Test timeout handling when fallback_strategy is 'skip_with_warning'.
        
        When a stage times out and has fallback_strategy='skip_with_warning',
        the execution should pass (not fail) with a warning message.
        
        Note: get_stage_design_spec looks for the key directly on the stage dict,
        not nested inside a "design_spec" object.
        """
        plan = MockResponseFactory.planner_response()
        # Add fallback_strategy directly to the stage (not nested in design_spec)
        plan["stages"][1]["fallback_strategy"] = "skip_with_warning"
        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"timeout_exceeded": True}
        base_state["run_error"] = "Execution exceeded timeout of 300s"

        # No LLM call should be made for timeout with skip_with_warning
        result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass", (
            "Timeout with skip_with_warning should pass"
        )
        assert "timeout" in result["execution_feedback"].lower(), (
            "Feedback should mention timeout"
        )
        assert "skip_with_warning" in result["execution_feedback"], (
            "Feedback should mention the fallback strategy"
        )

    def test_timeout_without_skip_fallback(self, base_state):
        """Test timeout handling when fallback_strategy is NOT 'skip_with_warning'.
        
        Normal timeout should result in a fail verdict.
        """
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"timeout_exceeded": True}
        base_state["run_error"] = "Execution exceeded timeout of 300s"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail", "Timeout without skip fallback should fail"
        assert "timeout" in result["execution_feedback"].lower(), (
            "Feedback should mention timeout"
        )
        assert result["execution_failure_count"] == 1, "Failure count should increment"

    def test_timeout_detected_from_run_error_string(self, base_state):
        """Test timeout detection via string pattern in run_error."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {}  # No timeout_exceeded flag
        base_state["run_error"] = "Process exceeded timeout limit"
        base_state["execution_failure_count"] = 0
        base_state["total_execution_failures"] = 0

        result = execution_validator_node(base_state)

        # Should detect timeout from string pattern
        assert result["execution_verdict"] == "fail", "Should detect timeout from error string"
        assert result["execution_failure_count"] == 1, "Failure count should increment"

    def test_max_execution_failures_triggers_user_escalation(self, base_state):
        """Test that hitting max execution failures triggers user escalation.
        
        When execution_failure_count reaches MAX_EXECUTION_FAILURES,
        the system should escalate to ask_user with appropriate trigger.
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_failure_count"] = MAX_EXECUTION_FAILURES - 1
        base_state["total_execution_failures"] = MAX_EXECUTION_FAILURES - 1
        base_state["run_error"] = "Repeated crash"

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            result = execution_validator_node(base_state)

        # Should escalate to user
        assert result["execution_verdict"] == "fail", "Verdict should be fail"
        assert result["execution_failure_count"] == MAX_EXECUTION_FAILURES, (
            f"Failure count should be {MAX_EXECUTION_FAILURES}"
        )
        assert result.get("ask_user_trigger") == "execution_failure_limit", (
            "Should trigger user escalation"
        )
        assert result.get("ask_user_trigger") is not None, (
            "Should set ask_user_trigger"
        )
        assert result.get("pending_user_questions"), (
            "Should have pending questions for user"
        )
        assert result.get("last_node_before_ask_user") == "execution_check", (
            "Should record which node triggered escalation"
        )

    def test_missing_verdict_defaults_to_pass(self, base_state):
        """Test that missing verdict in LLM response defaults to 'pass'.
        
        When the LLM response doesn't include a verdict field,
        the node should default to 'pass' with appropriate warning.
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            # Return response without verdict
            mock_llm.return_value = {
                "summary": "Analysis complete",
                "output_files_found": ["data.csv"],
            }
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass", (
            "Missing verdict should default to 'pass'"
        )
        assert "missing verdict" in result["execution_feedback"].lower(), (
            "Feedback should mention missing verdict"
        )

    def test_llm_call_failure_auto_approves(self, base_state):
        """Test that LLM call failure results in auto-approval.
        
        When the LLM call raises an exception, the node should
        auto-approve and continue rather than blocking.
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API connection failed")
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass", (
            "LLM failure should auto-approve"
        )
        assert "llm error" in result["execution_feedback"].lower() or \
               "auto-approved" in result["execution_feedback"].lower(), (
            "Feedback should mention LLM error or auto-approval"
        )

    def test_returns_empty_dict_when_trigger_set(self, base_state):
        """Test that node returns empty dict when ask_user_trigger is set.
        
        This prevents state modification while waiting for user input.
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["ask_user_trigger"] = "some_trigger"

        result = execution_validator_node(base_state)

        assert result == {}, "Should return empty dict when ask_user_trigger is set"

    def test_run_error_included_in_prompt(self, base_state):
        """Test that run_error is injected into the prompt for LLM analysis."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["run_error"] = "MemoryError: Unable to allocate array"
        base_state["stage_outputs"] = {"partial": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            execution_validator_node(base_state)

            # Check that the user_content includes the run_error
            call_args = mock_llm.call_args
            user_content = call_args.kwargs.get("user_content", "")
            assert "MemoryError" in user_content, (
                "run_error should be included in user content"
            )

    def test_stage_id_unknown_when_not_set(self, base_state):
        """Test handling when current_stage_id is not set."""
        base_state["current_stage_id"] = None
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_pass()
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass", "Should still process without stage_id"

    def test_custom_max_failures_from_runtime_config(self, base_state):
        """Test that max failures can be customized via runtime_config."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["runtime_config"] = {"max_execution_failures": 5}
        base_state["execution_failure_count"] = 4  # One below custom max
        base_state["total_execution_failures"] = 4

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            result = execution_validator_node(base_state)

        # Should hit max (5) and escalate
        assert result["execution_failure_count"] == 5, "Should increment to 5"
        assert result.get("ask_user_trigger") == "execution_failure_limit", (
            "Should escalate at custom max limit"
        )


class TestPhysicsSanityNode:
    """Test physics sanity validation node."""

    def test_physics_sanity_pass(self, base_state):
        """Test successful physics sanity check.
        
        Validates that on pass:
        - physics_verdict is set to "pass"
        - workflow_phase is set to "physics_validation"
        - physics_feedback contains summary
        - physics_failure_count is NOT incremented
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {
            "extinction_peak": 700,
            "extinction_values": [0.1, 0.5, 1.0, 0.5, 0.2],
        }
        base_state["design_description"] = {"simulation_type": "FDTD"}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_response = MockResponseFactory.physics_sanity_pass()
            mock_llm.return_value = mock_response
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass", "Verdict should be 'pass'"
        assert result["workflow_phase"] == "physics_validation", "Should set workflow phase"
        assert "physics_feedback" in result, "Should include physics feedback"
        assert result["physics_feedback"] == mock_response["summary"], (
            "Feedback should match summary"
        )
        assert "physics_failure_count" not in result, (
            "physics_failure_count should not be set on pass"
        )

    def test_physics_sanity_fail(self, base_state):
        """Test physics sanity check failure.
        
        Validates that on fail:
        - physics_verdict is set to "fail"
        - physics_failure_count is incremented
        - workflow_phase is set correctly
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["physics_failure_count"] = 0
        base_state["stage_outputs"] = {
            "extinction_peak": -500,  # Invalid negative peak
            "extinction_values": [-1, -2, -3],  # Invalid negative values
        }
        base_state["design_description"] = {"simulation_type": "FDTD"}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Negative extinction values are non-physical",
                "checks_performed": ["value_ranges"],
            }
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail", "Verdict should be 'fail'"
        assert result["physics_failure_count"] == 1, "Failure count should be 1"
        assert result["workflow_phase"] == "physics_validation", "Should set workflow phase"
        assert "physics_feedback" in result, "Should include physics feedback"

    def test_physics_sanity_design_flaw(self, base_state):
        """Test physics sanity check detecting design flaw.
        
        When verdict is 'design_flaw':
        - physics_verdict is set to "design_flaw"
        - design_revision_count is incremented
        - design_feedback is set
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = 0
        base_state["stage_outputs"] = {
            "resonance_wavelength": 1500,  # Far from expected ~700nm
        }
        base_state["design_description"] = {"geometry": "wrong_orientation"}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "design_flaw",
                "summary": "Resonance far from expected - likely incorrect geometry",
                "checks_performed": ["physics_plausibility"],
            }
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw", "Verdict should be 'design_flaw'"
        assert result["design_revision_count"] == 1, "Design revision count should be 1"
        assert "design_feedback" in result, "Should include design feedback"
        assert result["workflow_phase"] == "physics_validation", "Should set workflow phase"
        # physics_failure_count should NOT be incremented for design_flaw
        assert "physics_failure_count" not in result, (
            "physics_failure_count should not be set for design_flaw"
        )

    def test_physics_sanity_warning(self, base_state):
        """Test physics sanity check with warning verdict.
        
        Warning verdict should pass but note concerns.
        """
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {
            "extinction_peak": 720,  # Slightly off expected
        }

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "warning",
                "summary": "Peak slightly off expected but within tolerance",
            }
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "warning", "Verdict should be 'warning'"
        assert result["workflow_phase"] == "physics_validation"
        # Warning should NOT increment failure counts
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result

    def test_max_physics_failures_triggers_user_escalation(self, base_state):
        """Test that hitting max physics failures triggers user escalation."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["physics_failure_count"] = MAX_PHYSICS_FAILURES - 1
        base_state["stage_outputs"] = {"bad_data": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Physics check failed again",
            }
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == MAX_PHYSICS_FAILURES
        assert result.get("ask_user_trigger") == "physics_failure_limit", (
            "Should trigger physics_failure_limit escalation"
        )
        assert result.get("ask_user_trigger") is not None
        assert result.get("pending_user_questions"), "Should have pending questions"
        assert result.get("last_node_before_ask_user") == "physics_check"

    def test_backtrack_suggestion_handling(self, base_state):
        """Test handling of backtrack suggestion from physics agent."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Fundamental error requires earlier stage fix",
                "backtrack_suggestion": {
                    "suggest_backtrack": True,
                    "target_stage": "stage_0_materials",
                    "reason": "Material properties incorrect",
                },
            }
            result = physics_sanity_node(base_state)

        assert "backtrack_suggestion" in result, "Should include backtrack suggestion"
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["backtrack_suggestion"]["target_stage"] == "stage_0_materials"

    def test_backtrack_suggestion_not_included_when_false(self, base_state):
        """Test that backtrack_suggestion is not in result when suggest_backtrack=False."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "pass",
                "summary": "All checks passed",
                "backtrack_suggestion": {"suggest_backtrack": False},
            }
            result = physics_sanity_node(base_state)

        # backtrack_suggestion should NOT be in result when suggest_backtrack is False
        assert "backtrack_suggestion" not in result, (
            "backtrack_suggestion should not be included when suggest_backtrack=False"
        )

    def test_missing_verdict_defaults_to_pass(self, base_state):
        """Test that missing verdict in LLM response defaults to 'pass'."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "summary": "Analysis complete",
            }
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass", (
            "Missing verdict should default to 'pass'"
        )
        assert "missing verdict" in result["physics_feedback"].lower(), (
            "Feedback should mention missing verdict"
        )

    def test_llm_call_failure_auto_approves(self, base_state):
        """Test that LLM call failure results in auto-approval."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API timeout")
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass", "LLM failure should auto-approve"
        assert "llm error" in result["physics_feedback"].lower() or \
               "auto-approved" in result["physics_feedback"].lower(), (
            "Feedback should mention LLM error or auto-approval"
        )

    def test_returns_empty_dict_when_trigger_set(self, base_state):
        """Test that node returns empty dict when ask_user_trigger is set."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["ask_user_trigger"] = "some_trigger"

        result = physics_sanity_node(base_state)

        assert result == {}, "Should return empty dict when ask_user_trigger is set"

    def test_empty_design_description_handled(self, base_state):
        """Test handling when design_description is empty dict or None."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}
        base_state["design_description"] = {}  # Empty dict

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.physics_sanity_pass()
            
            # Should not raise an error
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"

    def test_none_design_description_handled(self, base_state):
        """Test handling when design_description is None."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}
        base_state["design_description"] = None

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.physics_sanity_pass()
            
            # Should not raise an error
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"

    def test_string_design_description_handled(self, base_state):
        """Test handling when design_description is a string instead of dict."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}
        base_state["design_description"] = "FDTD simulation of gold nanorod"

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.physics_sanity_pass()
            
            # Should not raise an error
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "pass"

    def test_custom_max_physics_failures_from_runtime_config(self, base_state):
        """Test that max physics failures can be customized via runtime_config."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["runtime_config"] = {"max_physics_failures": 5}
        base_state["physics_failure_count"] = 4  # One below custom max
        base_state["stage_outputs"] = {"bad": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            result = physics_sanity_node(base_state)

        assert result["physics_failure_count"] == 5
        assert result.get("ask_user_trigger") == "physics_failure_limit"

    def test_consecutive_physics_failures_increment_correctly(self, base_state):
        """Test that consecutive failures increment counter correctly."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["physics_failure_count"] = 1  # Already had one failure
        base_state["stage_outputs"] = {"bad": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed again"}
            result = physics_sanity_node(base_state)

        assert result["physics_failure_count"] == 2, (
            "Failure count should increment from 1 to 2"
        )

    def test_design_flaw_increments_design_revision_count(self, base_state):
        """Test that design_flaw verdict increments design_revision_count."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = 1  # Already had one revision
        base_state["stage_outputs"] = {"wrong": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "design_flaw",
                "summary": "Design needs revision",
            }
            result = physics_sanity_node(base_state)

        assert result["design_revision_count"] == 2, (
            "Design revision count should increment from 1 to 2"
        )


class TestExecutionValidatorPromptBuilding:
    """Test prompt building in execution validator."""

    def test_stage_outputs_included_in_prompt(self, base_state):
        """Test that stage_outputs are included in the prompt."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {
            "extinction_spectrum": [[400, 0.1], [700, 1.0], [900, 0.2]],
            "runtime_seconds": 120,
        }

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_pass()
            execution_validator_node(base_state)

            call_args = mock_llm.call_args
            user_content = call_args.kwargs.get("user_content", "")
            
            # Check that stage outputs are in the content
            assert "extinction_spectrum" in user_content, (
                "Stage outputs should be in user content"
            )
            assert "stage_1_extinction" in user_content, (
                "Stage ID should be in user content"
            )


class TestPhysicsSanityPromptBuilding:
    """Test prompt building in physics sanity node."""

    def test_design_spec_included_in_prompt(self, base_state):
        """Test that design_description is included in the prompt."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"data": [1, 2, 3]}
        base_state["design_description"] = {
            "simulation_type": "FDTD",
            "geometry": {"type": "nanorod", "length": 100},
        }

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.physics_sanity_pass()
            physics_sanity_node(base_state)

            call_args = mock_llm.call_args
            user_content = call_args.kwargs.get("user_content", "")
            
            # Check that design spec is in the content
            assert "FDTD" in user_content or "simulation_type" in user_content, (
                "Design spec should be in user content"
            )
            assert "nanorod" in user_content or "geometry" in user_content, (
                "Geometry should be in user content"
            )


class TestFailureCounterEdgeCases:
    """Test edge cases for failure counters."""

    def test_none_execution_failure_count_handled(self, base_state):
        """Test handling when execution_failure_count is None."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_failure_count"] = None  # Explicitly None
        base_state["total_execution_failures"] = None

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            result = execution_validator_node(base_state)

        # Should handle None by treating as 0
        assert result["execution_failure_count"] == 1, (
            "Should handle None as 0 and increment to 1"
        )
        assert result["total_execution_failures"] == 1

    def test_none_physics_failure_count_handled(self, base_state):
        """Test handling when physics_failure_count is None."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["physics_failure_count"] = None
        base_state["stage_outputs"] = {"bad": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            result = physics_sanity_node(base_state)

        assert result["physics_failure_count"] == 1, (
            "Should handle None as 0 and increment to 1"
        )

    def test_none_design_revision_count_handled(self, base_state):
        """Test handling when design_revision_count is None."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = None
        base_state["stage_outputs"] = {"wrong": True}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "design_flaw", "summary": "Flaw"}
            result = physics_sanity_node(base_state)

        assert result["design_revision_count"] == 1, (
            "Should handle None as 0 and increment to 1"
        )


class TestStateIntegrity:
    """Test state integrity and required fields."""

    def test_execution_validator_returns_required_fields(self, base_state):
        """Test that execution_validator always returns required fields."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_pass()
            result = execution_validator_node(base_state)

        # Required fields on every call
        required_fields = ["workflow_phase", "execution_verdict", "execution_feedback"]
        for field in required_fields:
            assert field in result, f"Required field '{field}' missing from result"

    def test_physics_sanity_returns_required_fields(self, base_state):
        """Test that physics_sanity always returns required fields."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {}

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.physics_sanity_pass()
            result = physics_sanity_node(base_state)

        # Required fields on every call
        required_fields = ["workflow_phase", "physics_verdict", "physics_feedback"]
        for field in required_fields:
            assert field in result, f"Required field '{field}' missing from result"

    def test_execution_validator_does_not_mutate_input_state(self, base_state):
        """Test that execution_validator doesn't mutate the input state."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_failure_count"] = 0
        original_count = base_state["execution_failure_count"]

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            execution_validator_node(base_state)

        # Input state should not be mutated
        assert base_state["execution_failure_count"] == original_count, (
            "Input state should not be mutated"
        )

    def test_physics_sanity_does_not_mutate_input_state(self, base_state):
        """Test that physics_sanity doesn't mutate the input state."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["physics_failure_count"] = 0
        base_state["stage_outputs"] = {"data": [1]}
        original_count = base_state["physics_failure_count"]

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            physics_sanity_node(base_state)

        assert base_state["physics_failure_count"] == original_count, (
            "Input state should not be mutated"
        )
