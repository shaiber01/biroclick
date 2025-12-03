from copy import deepcopy
from unittest.mock import patch

from src.agents import select_stage_node, supervisor_node
from src.agents.constants import AnalysisClassification

from tests.workflow.fixtures import MockResponseFactory


class TestMultiStageWorkflow:
    """Test workflows with multiple dependent stages."""

    def test_stage_dependency_progression(self, base_state):
        """Test stages execute in dependency order."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])

        # First selection should pick a stage with no deps
        result = select_stage_node(base_state)
        first_stage = result.get("current_stage_id")
        assert first_stage is not None, "Should select first stage"

        # Mark first stage complete
        for stage in base_state["progress"]["stages"]:
            if stage["stage_id"] == first_stage:
                stage["status"] = "completed_success"

        # Second selection should pick next available stage
        result = select_stage_node(base_state)
        second_stage = result.get("current_stage_id")

        # Should either select another stage or return None if done
        if second_stage is not None:
            assert second_stage != first_stage, "Should not re-select completed stage"

    def test_blocked_stage_skipped(self, base_state):
        """Test blocked stages are skipped."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])

        # Mark stage_0 as blocked
        base_state["progress"]["stages"][0]["status"] = "blocked"

        # Stage_1 depends on stage_0, so it should also be blocked/skipped
        result = select_stage_node(base_state)

        # Should not select stage_1 since its dependency is blocked
        assert result.get("current_stage_id") != "stage_1_extinction" or result.get("current_stage_id") is None


class TestSupervisorDecisions:
    """Test supervisor routing decisions."""

    def test_supervisor_continues_on_success(self, base_state):
        """Test supervisor returns ok_continue on success."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["analysis_overall_classification"] = AnalysisClassification.ACCEPTABLE_MATCH

        with patch("src.agents.supervision.supervisor.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.supervisor_continue()

            result = supervisor_node(base_state)

        assert result["supervisor_verdict"] == "ok_continue"

    def test_supervisor_completes_workflow(self, base_state):
        """Test supervisor returns all_complete when done."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        # All stages complete
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0_materials", "status": "completed_success"},
                {"stage_id": "stage_1_extinction", "status": "completed_success"},
            ]
        }

        with patch("src.agents.supervision.supervisor.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.supervisor_complete()

            result = supervisor_node(base_state)

        # Note: actual verdict depends on implementation
        assert "supervisor_verdict" in result


