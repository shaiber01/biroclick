from unittest.mock import patch

import pytest


class TestDesignRevisionCounters:
    """Verify design reviewer counters respect bounds and increments."""

    def test_design_revision_counter_bounded(self, base_state):
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": ["test"],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        max_revisions = 3
        base_state["design_revision_count"] = max_revisions
        base_state["runtime_config"] = {"max_design_revisions": max_revisions}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] == max_revisions

    def test_design_revision_counter_increments_under_max(self, base_state):
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": ["test"],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 1

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["design_revision_count"] == 2


class TestDesignReviewerCounterIncrements:
    """Verify design reviewer increments revision counters on rejection."""

    def test_design_reviewer_increments_counter_on_rejection(self, base_state):
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": ["test"],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result.get("design_revision_count") == 1


class TestDesignReviewerFeedback:
    """Verify design reviewer feedback fields are populated correctly."""

    def test_design_reviewer_populates_feedback_on_rejection(self, base_state):
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "major", "description": "Missing wavelength range"},
                {"severity": "minor", "description": "Consider adding symmetry"},
            ],
            "summary": "Design needs several improvements",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0", "geometry": []}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert "reviewer_feedback" in result
        feedback = result["reviewer_feedback"]
        assert len(feedback) > 20


class TestDesignReviewerOutputFields:
    """Verify design reviewer sets required fields on approval."""

    def test_design_reviewer_sets_all_fields_on_approve(self, base_state):
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Design looks good",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {
            "stage_id": "stage_0",
            "geometry": [{"type": "box"}],
        }
        base_state["design_revision_count"] = 2

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result.get("last_design_review_verdict") == "approve"
        assert "workflow_phase" in result
        assert "design_revision_count" in result


class TestSimulationDesignerCompleteness:
    """Verify simulation_designer_node produces complete designs."""

    def test_designer_returns_design_with_all_sections(self, base_state, valid_plan):
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Full FDTD simulation design",
            "geometry": [
                {"type": "cylinder", "radius": 20, "height": 100, "material": "gold"}
            ],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
            "materials": [{"material_id": "gold", "source": "Palik"}],
            "computational_domain": {"pml_layers": 1, "resolution": 32},
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        design = result.get("design_description", {})
        assert "geometry" in design
        assert "sources" in design
        assert "monitors" in design
        assert len(design.get("geometry", [])) > 0

    def test_simulation_designer_node_creates_design(self, base_state, valid_plan):
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation with gold nanorod...",
            "geometry": [{"type": "cylinder", "radius": 20, "material": "gold"}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
            "materials": [{"material_id": "gold", "source": "Johnson-Christy"}],
            "new_assumptions": {"sim_a1": "assuming periodic boundary"},
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        design = result["design_description"]
        assert design.get("stage_id") == "stage_0"
        assert design["geometry"][0].get("type") == "cylinder"
        assert design["materials"] == mock_response["materials"]


class TestDesignerFieldMapping:
    """Verify simulation designer mapping of geometry fields."""

    def test_designer_maps_geometry_correctly(self, base_state, valid_plan):
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Test design",
            "geometry": [
                {"type": "cylinder", "radius": 20, "height": 100, "material": "gold"},
                {"type": "box", "size": [500, 500, 200], "material": "water"},
            ],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        design = result.get("design_description", {})
        assert len(design.get("geometry", [])) == 2
        assert design["geometry"][0]["type"] == "cylinder"

