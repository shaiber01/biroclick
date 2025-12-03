"""Edge-case handling for planner entry points."""

from unittest.mock import patch

from schemas.state import create_initial_state


class TestPlanningEdgeCases:
    """Planning-specific edge cases that should escalate properly."""

    def test_plan_node_with_very_short_paper(self):
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="Short paper.",
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text" or result.get(
            "awaiting_user_input"
        ) is True

    def test_plan_node_handles_missing_paper_text(self):
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="",
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True

    def test_plan_node_handles_context_check_escalation(self, base_state):
        """If check_context_or_escalate returns early, plan_node should return it."""
        from src.agents.planning import plan_node

        escalation_response = {"awaiting_user_input": True, "reason": "Context too large"}

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=escalation_response
        ):
            result = plan_node(base_state)

        assert result == escalation_response


class TestPlannerErrorHandling:
    """LLM failure scenarios for planner and reviewer nodes."""

    def test_llm_error_triggers_user_escalation(self, base_state):
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_node(base_state)

        assert result.get("ask_user_trigger") == "llm_error"
        assert result.get("awaiting_user_input") is True

    def test_reviewer_llm_error_auto_approves(self, base_state, valid_plan):
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = valid_plan

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_reviewer_node(base_state)

        assert result.get("last_plan_review_verdict") == "approve"
        assert result.get("workflow_phase") == "plan_review"

