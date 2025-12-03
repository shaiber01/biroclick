from copy import deepcopy
from unittest.mock import patch

from src.agents import (
    simulation_designer_node,
    design_reviewer_node,
    code_reviewer_node,
    code_generator_node,
)

from tests.workflow.fixtures import MockResponseFactory


class TestWorkflowWithRevisions:
    """Test workflow with revision cycles."""

    def test_design_revision_cycle(self, base_state):
        """Test design → review reject → revision → approve cycle."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]

        # First design attempt
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)

            # Verify no feedback initially
            assert "reviewer_feedback" not in base_state or not base_state["reviewer_feedback"]

        # Review rejects
        feedback_msg = "Add PML thickness"
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision(feedback_msg)
            result = design_reviewer_node(base_state)
            base_state.update(result)

        assert base_state["last_design_review_verdict"] == "needs_revision"
        assert base_state["design_revision_count"] == 1
        assert base_state["reviewer_feedback"] == feedback_msg

        # Second design attempt with feedback
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)

            # In a real scenario, we'd check that the feedback influenced the design
            # Here we just verify the flow continued
            assert result["workflow_phase"] == "design"

        # Review approves
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)

        assert base_state["last_design_review_verdict"] == "approve"

    def test_code_revision_with_max_limit(self, base_state):
        """Test code revision respects max limit."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Test design"
        base_state["code"] = "# test code"
        base_state["runtime_config"] = {"max_code_revisions": 2}

        # First revision
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)

        assert base_state["code_revision_count"] == 1

        # Second revision
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)

        assert base_state["code_revision_count"] == 2

        # Third revision - should not increment past max
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)

        # Count stays at max (2)
        assert base_state["code_revision_count"] == 2


class TestCodeGeneratorValidation:
    """Test code generator validation logic."""

    def test_missing_validated_materials(self, base_state):
        """Test code generation fails if materials not validated (for non-material stages)."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        # Make design description long enough (>50 chars)
        base_state["design_description"] = "Valid design " * 5
        base_state["validated_materials"] = []  # Empty

        result = code_generator_node(base_state)

        assert "run_error" in result
        assert "validated_materials is empty" in result["run_error"]

    def test_stub_detection_triggers_revision(self, base_state):
        """Test that stub markers in generated code trigger revision."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        # Make design description long enough (>50 chars)
        base_state["design_description"] = "Valid design " * 5

        # Mock response with stub
        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = "# TODO: Implement simulation"

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response

            result = code_generator_node(base_state)

            assert "reviewer_feedback" in result
            assert "stub" in result["reviewer_feedback"].lower()
            assert result["code_revision_count"] == 1


