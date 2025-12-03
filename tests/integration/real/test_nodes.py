"""Integration tests ensuring node functions execute with minimal state."""

from unittest.mock import patch


class TestNodeFunctionsCallable:
    """
    Test that node functions can be called with minimal state.

    Only mock the LLM call itself - everything else runs for real.
    These tests verify:
    1. The node doesn't crash.
    2. The node returns a dict with expected state keys.
    """

    def test_plan_node_returns_valid_plan(self, minimal_state, mock_llm_response):
        """plan_node must return a result containing a plan structure."""
        from src.agents.planning import plan_node

        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = plan_node(minimal_state)

        assert result is not None
        assert "plan" in result, "plan_node must return 'plan' key"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "planning"
        plan = result["plan"]
        assert "stages" in plan, "Plan must have stages"
        assert "targets" in plan, "Plan must have targets"

    def test_supervisor_node_returns_verdict(self, minimal_state, mock_llm_response):
        """supervisor_node must return a supervisor_verdict."""
        from src.agents.supervision.supervisor import supervisor_node

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            mock_llm_response,
        ):
            result = supervisor_node(minimal_state)

        assert result is not None
        assert "supervisor_verdict" in result, "supervisor_node must return supervisor_verdict"
        assert "supervisor_feedback" in result
        assert result["workflow_phase"] == "supervision"

    def test_report_node_completes_workflow(self, minimal_state, mock_llm_response):
        """generate_report_node must mark workflow as complete."""
        from src.agents.reporting import generate_report_node

        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)

        assert result is not None
        assert "workflow_complete" in result, "generate_report_node must set workflow_complete"
        assert result["workflow_complete"] is True
        assert "executive_summary" in result
        assert result["workflow_phase"] == "reporting"

    def test_design_node_returns_design(self, minimal_state):
        """simulation_designer_node must return design_description."""
        from src.agents.design import simulation_designer_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        design_response = {
            "design": {
                "stage_id": "stage_0",
                "geometry": [],
                "simulation_parameters": {},
            },
            "explanation": "Designed",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=design_response,
        ):
            result = simulation_designer_node(minimal_state)

        assert result is not None
        assert "design_description" in result, "simulation_designer_node must return design_description"
        assert result["design_description"]["design"]["stage_id"] == "stage_0"
        assert "workflow_phase" in result

    def test_code_generator_returns_code(self, minimal_state):
        """code_generator_node must return generated code."""
        from src.agents.code import code_generator_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        minimal_state["design_description"] = {
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design",
        }
        code_response = {
            "code": "import meep as mp\nprint('hello')",
            "explanation": "Generated",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)

        assert result is not None
        assert "code" in result, "code_generator_node must return code"
        assert "import meep" in result["code"]
        assert "workflow_phase" in result

    def test_execution_validator_node_returns_verdict(self, minimal_state, mock_llm_response):
        """execution_validator_node must return execution_verdict."""
        from src.agents.execution import execution_validator_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"stdout": "run complete", "stderr": ""}

        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = execution_validator_node(minimal_state)

        assert result is not None
        assert "execution_verdict" in result
        assert result["execution_verdict"] in ["pass", "fail"]

    def test_physics_sanity_node_returns_verdict(self, minimal_state, mock_llm_response):
        """physics_sanity_node must return physics_verdict."""
        from src.agents.execution import physics_sanity_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"files": ["spectrum.csv"]}

        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = physics_sanity_node(minimal_state)

        assert result is not None
        assert "physics_verdict" in result
        assert result["physics_verdict"] in ["pass", "fail", "warning", "design_flaw"]

    def test_prompt_adaptor_node_returns_adaptations(self, minimal_state, mock_llm_response):
        """prompt_adaptor_node must return prompt_adaptations."""
        from src.agents.planning import adapt_prompts_node

        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = adapt_prompts_node(minimal_state)

        assert result is not None
        assert "prompt_adaptations" in result
        assert isinstance(result["prompt_adaptations"], list)


