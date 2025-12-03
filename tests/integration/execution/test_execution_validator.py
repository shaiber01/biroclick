"""Execution validator and physics sanity integration tests."""

from unittest.mock import patch

from tests.integration.helpers.agent_responses import execution_verdict_response


class TestExecutionValidatorBehavior:
    """Execution validator specific behaviors."""

    def test_execution_validator_returns_verdict_from_llm(self, base_state):
        from src.agents.execution import execution_validator_node

        mock_response = execution_verdict_response()

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result.get("execution_verdict") == "pass"
        assert "workflow_phase" in result


class TestValidatorVerdicts:
    """Test various validator verdict scenarios."""

    def test_physics_sanity_returns_design_flaw(self, base_state):
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            verdict="design_flaw",
            summary="Simulation parameters inconsistent with physics",
            design_issues=["Wavelength range too narrow"],
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"

    def test_execution_validator_returns_fail(self, base_state):
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

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"


class TestPhysicsSanityBehavior:
    """Additional coverage for physics_sanity_node."""

    def test_physics_sanity_passes(self, base_state):
        from src.agents.execution import physics_sanity_node

        mock_response = execution_verdict_response(
            summary="Physics checks passed",
            checks_performed=["energy_conservation", "value_ranges"],
        )

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = physics_sanity_node(base_state)

        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("agent_name") == "physics_sanity"
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"

    def test_physics_sanity_passes_backtrack_suggestion(self, base_state):
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

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True

