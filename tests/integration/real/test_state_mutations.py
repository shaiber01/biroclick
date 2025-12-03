"""Integration tests that verify nodes mutate state as expected."""

from unittest.mock import patch

from schemas.state import create_initial_state


class TestStateMutations:
    """Test that nodes mutate state correctly."""

    def test_plan_node_sets_plan_field(self):
        """plan_node must set state['plan'] with the LLM response."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "test",
            "title": "Gold Nanorod Study",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION"}],
            "targets": [],
            "extracted_parameters": [],
        }
        paper_text = (
            "We study the optical properties of gold nanorods with length 100nm and "
            "diameter 40nm. Using FDTD simulations, we calculate extinction spectra "
            "and near-field enhancement patterns. The localized surface plasmon "
            "resonance is observed at 650nm wavelength."
        )
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            result = plan_node(state)

        assert "plan" in result, "plan_node must return 'plan' in result"
        assert result["plan"]["title"] == "Gold Nanorod Study"

    def test_plan_node_rejects_short_paper_text(self):
        """plan_node must reject paper text that's too short."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "short", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result

    def test_plan_reviewer_sets_verdict_field(self):
        """plan_reviewer_node must set last_plan_review_verdict."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
        }
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert "last_plan_review_verdict" in result
        assert result["last_plan_review_verdict"] == "approve"

    def test_plan_reviewer_rejects_empty_plan(self):
        """plan_reviewer_node must reject plans with no stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {"title": "Test", "stages": [], "targets": []}

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_counter_increments_on_revision(self):
        """Revision counters must increment when verdict is needs_revision."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_design"] = {"stage_id": "stage_0"}
        state["design_revision_count"] = 0
        mock_response = {
            "verdict": "needs_revision",
            "issues": ["problem"],
            "summary": "Fix",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] == 1, "Counter should increment on needs_revision"


