from unittest.mock import patch

from src.agents import plan_node, plan_reviewer_node

from tests.workflow.fixtures import MockResponseFactory


class TestPlannerFailureModes:
    """Test planner failure modes and edge cases."""

    def test_missing_paper_text(self, base_state):
        """Test planner handles missing paper text."""
        base_state["paper_text"] = ""

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) > 0

    def test_short_paper_text(self, base_state):
        """Test planner handles insufficient paper text."""
        base_state["paper_text"] = "Too short"

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert "too short" in result["pending_user_questions"][0].lower()


class TestPlanReviewerValidation:
    """Test plan reviewer validation logic."""

    def test_circular_dependency_detection(self, base_state):
        """Test detection of circular dependencies in plan."""
        plan = MockResponseFactory.planner_response()
        # Create circular dependency: stage0 -> stage1 -> stage0
        plan["stages"][0]["dependencies"] = ["stage_1_extinction"]
        plan["stages"][1]["dependencies"] = ["stage_0_materials"]

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_empty_stage_targets(self, base_state):
        """Test detection of stages without targets."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["targets"] = []

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no targets" in result.get("planner_feedback", "").lower()


