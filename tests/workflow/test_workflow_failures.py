from copy import deepcopy
from unittest.mock import patch

from src.agents import execution_validator_node

from tests.workflow.fixtures import MockResponseFactory


class TestWorkflowWithFailures:
    """Test workflow recovery from failures."""

    def test_execution_failure_recovery(self, base_state):
        """Test handling of execution failures."""
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

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1


class TestExecutionValidatorLogic:
    """Test execution validator logic."""

    def test_successful_execution_metrics(self, base_state):
        """Test validation of successful execution with metrics."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_result"] = {
            "success": True,
            "output_files": ["data.csv"],
            "runtime_seconds": 45.5,
        }

        # Mock pass response
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_pass()

            result = execution_validator_node(base_state)

            assert result["execution_verdict"] == "pass"

    def test_execution_failure_handling(self, base_state):
        """Test validation of failed execution."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["execution_result"] = {
            "success": False,
            "error": "Timeout",
            "output_files": [],
        }

        # Mock fail response
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail("Timeout occurred")

            result = execution_validator_node(base_state)

            assert result["execution_verdict"] == "fail"
            assert result["execution_failure_count"] == 1


