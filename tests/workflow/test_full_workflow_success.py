from copy import deepcopy
from unittest.mock import patch

from src.agents import (
    plan_node,
    plan_reviewer_node,
    select_stage_node,
    simulation_designer_node,
    design_reviewer_node,
    code_generator_node,
    code_reviewer_node,
)

from tests.workflow.fixtures import MockResponseFactory


class TestFullWorkflowSuccess:
    """Test complete successful workflow from paper to report."""

    def test_planning_phase(self, base_state):
        """Test planning phase produces valid plan with all required fields."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()

            result = plan_node(base_state)

            assert "plan" in result
            plan = result["plan"]

            # Verify plan structure
            assert plan["paper_id"] == "test_gold_nanorod"
            assert len(plan["stages"]) == 2
            assert plan["stages"][0]["stage_id"] == "stage_0_materials"
            assert plan["stages"][0]["stage_type"] == "MATERIAL_VALIDATION"

            # Verify progress initialization
            assert "progress" in result
            assert len(result["progress"]["stages"]) == 2
            assert result["progress"]["stages"][0]["status"] == "not_started"

            # Verify extracted parameters
            assert "extracted_parameters" in result
            assert len(result["extracted_parameters"]) == 2
            assert result["extracted_parameters"][0]["name"] == "length"

            assert result["workflow_phase"] == "planning"

    def test_plan_review_approve(self, base_state):
        """Test plan review approval."""
        # Set up state with plan
        base_state["plan"] = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()

            result = plan_reviewer_node(base_state)

            assert result["last_plan_review_verdict"] == "approve"
            assert result["workflow_phase"] == "plan_review"
            # Verify no unexpected issues
            assert (
                "planner_feedback" not in result or not result["planner_feedback"]
            )

    def test_stage_selection(self, base_state):
        """Test stage selection picks first available stage."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])

        result = select_stage_node(base_state)

        # Should select first stage with no dependencies
        selected = result.get("current_stage_id")
        assert selected == "stage_0_materials", "Should select materials stage first (no deps)"
        assert result.get("current_stage_type") == "MATERIAL_VALIDATION"

    def test_full_single_stage_workflow(self, base_state):
        """Test complete single-stage workflow."""
        # Setup: Plan with approved review
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        # Mock execution result (simulates code_runner)
        base_state["execution_result"] = {
            "success": True,
            "output_files": ["extinction.csv"],
            "runtime_seconds": 120,
        }
        base_state["stage_outputs"] = {
            "files": ["extinction.csv"],
            "stage_id": "stage_1_extinction",
        }

        workflow_states = []

        # Design phase
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)
            workflow_states.append(("design", result))

            assert "design_description" in result
            assert (
                result["design_description"]["geometry_definitions"][0]["material"]
                == "gold"
            )

        # Design review
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)
            workflow_states.append(("design_review", result))

        assert base_state["last_design_review_verdict"] == "approve"

        # Code generation
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)
            base_state.update(result)
            workflow_states.append(("code_gen", result))

        assert "code" in base_state
        assert "import meep" in base_state["code"]
        assert "STUB" not in base_state["code"]

        # Code review
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = code_reviewer_node(base_state)
            base_state.update(result)
            workflow_states.append(("code_review", result))

        assert base_state["last_code_review_verdict"] == "approve"


