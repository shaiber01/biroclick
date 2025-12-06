"""
Integration tests for design agent nodes: simulation_designer_node, design_reviewer_node.

These tests verify the full behavior of design agents including:
- State field handling and output correctness
- Counter management and bounds
- Error handling paths
- Context escalation behavior
- Feedback population
- New assumptions extraction
"""

from unittest.mock import patch, MagicMock

import pytest

from schemas.state import MAX_DESIGN_REVISIONS


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN REVIEWER NODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignReviewerVerdictHandling:
    """Verify design reviewer handles different verdicts correctly."""

    def test_design_reviewer_approve_verdict_sets_all_required_fields(self, base_state):
        """Approval should set verdict, workflow_phase, and preserve counter."""
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
        base_state["design_description"] = {"stage_id": "stage_0", "geometry": []}
        base_state["design_revision_count"] = 2

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Verify all required fields are present with correct values
        assert result["last_design_review_verdict"] == "approve"
        assert result["workflow_phase"] == "design_review"
        assert result["design_revision_count"] == 2  # Counter preserved on approval
        assert result["reviewer_issues"] == []
        # Feedback should NOT be set on approval
        assert result.get("reviewer_feedback") is None or result.get("reviewer_feedback") == ""

    def test_design_reviewer_needs_revision_verdict_increments_counter(self, base_state):
        """Needs revision should increment counter and populate feedback."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Missing wavelength"}],
            "summary": "Design needs improvements",
            "feedback": "Please add wavelength range specification",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 1

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 2  # Incremented from 1
        assert "reviewer_feedback" in result
        # Feedback should contain the actual feedback content
        assert "wavelength" in result["reviewer_feedback"].lower()

    def test_design_reviewer_missing_verdict_defaults_to_needs_revision(self, base_state):
        """Missing verdict in LLM response should default to needs_revision (fail-closed)."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            # No "verdict" key
            "issues": [],
            "summary": "No verdict provided",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Fail-closed: missing verdict should default to needs_revision (safer)
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1  # Incremented for needs_revision


class TestDesignReviewerVerdictNormalization:
    """Verify verdict normalization handles various LLM response formats."""

    def test_pass_verdict_normalized_to_approve(self, base_state):
        """LLM returning 'pass' should be normalized to 'approve'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "pass",  # Alternative wording
            "issues": [],
            "summary": "All checks passed",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "approve"

    def test_approved_verdict_normalized_to_approve(self, base_state):
        """LLM returning 'approved' should be normalized to 'approve'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approved",  # Past tense
            "issues": [],
            "summary": "Design approved",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "approve"

    def test_accept_verdict_normalized_to_approve(self, base_state):
        """LLM returning 'accept' should be normalized to 'approve'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "accept",  # Synonym
            "issues": [],
            "summary": "Accepted",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "approve"

    def test_reject_verdict_normalized_to_needs_revision(self, base_state):
        """LLM returning 'reject' should be normalized to 'needs_revision'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "reject",  # Alternative rejection wording
            "issues": [{"severity": "major", "description": "Critical issue"}],
            "summary": "Design rejected",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1  # Should increment

    def test_revision_needed_verdict_normalized_to_needs_revision(self, base_state):
        """LLM returning 'revision_needed' should be normalized to 'needs_revision'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "revision_needed",  # Snake case alternative
            "issues": [{"severity": "minor", "description": "Minor fix needed"}],
            "summary": "Revisions needed",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "needs_revision"

    def test_needs_work_verdict_normalized_to_needs_revision(self, base_state):
        """LLM returning 'needs_work' should be normalized to 'needs_revision'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_work",  # Informal alternative
            "issues": [{"severity": "major", "description": "Needs work"}],
            "summary": "Design needs work",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 1

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 2

    def test_unknown_verdict_defaults_to_needs_revision(self, base_state):
        """Unknown verdict string should default to needs_revision (fail-closed safety)."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "maybe_good",  # Garbage verdict
            "issues": [],
            "summary": "Unclear verdict",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Fail-closed: unknown verdict should default to needs_revision (safer)
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1  # Incremented for needs_revision


class TestDesignRevisionCounters:
    """Verify design reviewer counters respect bounds and increments."""

    def test_design_revision_counter_bounded_at_max(self, base_state):
        """Counter should not exceed max_design_revisions."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "test issue"}],
            "summary": "Fix this",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        max_revisions = 3
        base_state["design_revision_count"] = max_revisions
        base_state["runtime_config"] = {"max_design_revisions": max_revisions}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Counter should stay at max, not increment
        assert result["design_revision_count"] == max_revisions

    def test_design_revision_counter_increments_from_zero(self, base_state):
        """Counter should increment from 0 to 1 on first rejection."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "test"}],
            "summary": "Minor fix needed",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["design_revision_count"] == 1

    def test_design_revision_counter_increments_under_max(self, base_state):
        """Counter should increment when below max."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "test"}],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 1

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["design_revision_count"] == 2

    def test_design_revision_counter_respects_custom_max_from_config(self, base_state):
        """Counter should respect runtime_config max_design_revisions."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "test"}],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 5
        base_state["runtime_config"] = {"max_design_revisions": 5}  # Custom max

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # At custom max (5), should not increment
        assert result["design_revision_count"] == 5


class TestDesignReviewerFeedback:
    """Verify design reviewer feedback fields are populated correctly."""

    def test_design_reviewer_populates_feedback_from_feedback_field(self, base_state):
        """Feedback should be extracted from 'feedback' field when present."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "major", "description": "Missing wavelength range"},
            ],
            "summary": "Design needs several improvements",
            "feedback": "The simulation design is missing the wavelength specification. Please add wavelength_range to sources.",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0", "geometry": []}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert "reviewer_feedback" in result
        # Should use feedback field, not summary
        assert "wavelength" in result["reviewer_feedback"].lower()
        assert "specification" in result["reviewer_feedback"].lower()

    def test_design_reviewer_uses_summary_when_no_feedback_field(self, base_state):
        """Feedback should fall back to summary when feedback field missing."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "major", "description": "Missing geometry"},
            ],
            "summary": "Design needs geometry specification",
            # No "feedback" key
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Should use summary as fallback
        assert "reviewer_feedback" in result
        assert "geometry" in result["reviewer_feedback"].lower()

    def test_design_reviewer_populates_issues_on_rejection(self, base_state):
        """reviewer_issues should contain all issues from LLM response."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "major", "description": "Missing wavelength range"},
                {"severity": "minor", "description": "Consider adding symmetry"},
                {"severity": "critical", "description": "No materials defined"},
            ],
            "summary": "Multiple issues found",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert "reviewer_issues" in result
        assert len(result["reviewer_issues"]) == 3
        # Check specific issue content
        descriptions = [i.get("description", "") for i in result["reviewer_issues"]]
        assert any("wavelength" in d.lower() for d in descriptions)
        assert any("materials" in d.lower() for d in descriptions)


class TestDesignReviewerLLMFailure:
    """Verify design reviewer handles LLM failures with fail-closed safety."""

    def test_design_reviewer_defaults_to_needs_revision_on_llm_exception(self, base_state):
        """LLM call failure should result in needs_revision (fail-closed safety)."""
        from src.agents.design import design_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 1

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=Exception("API timeout"),
        ):
            result = design_reviewer_node(base_state)

        # Fail-closed: LLM error should trigger needs_revision (safer than auto-approve)
        assert result["last_design_review_verdict"] == "needs_revision"
        # Counter should be incremented for needs_revision
        assert result["design_revision_count"] == 2
        # Should have some indication of the error
        assert len(result.get("reviewer_issues", [])) > 0


class TestDesignReviewerContextCheck:
    """Verify design reviewer handles context escalation."""

    def test_design_reviewer_returns_empty_when_awaiting_user_input(self, base_state):
        """Should return empty dict when already awaiting user input."""
        from src.agents.design import design_reviewer_node

        base_state["awaiting_user_input"] = True
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}

        # Should NOT call LLM when awaiting input
        with patch("src.agents.design.call_agent_with_metrics") as mock_call:
            result = design_reviewer_node(base_state)

        mock_call.assert_not_called()
        assert result == {}

    def test_design_reviewer_returns_escalation_on_context_overflow(self, base_state):
        """Should return escalation when context check fails."""
        from src.agents.design import design_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}

        escalation_result = {
            "ask_user_trigger": "context_overflow",
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context overflow. How to proceed?"],
        }

        with patch(
            "src.agents.base.check_context_or_escalate", return_value=escalation_result
        ):
            result = design_reviewer_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "context_overflow"


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION DESIGNER NODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulationDesignerCompleteness:
    """Verify simulation_designer_node produces complete designs."""

    def test_designer_returns_design_with_all_sections(self, base_state, valid_plan):
        """Design output should contain all required sections."""
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

        # Check workflow_phase is set
        assert result["workflow_phase"] == "design"

        # Check design is stored in design_description
        assert "design_description" in result
        design = result["design_description"]

        # Verify design structure matches LLM response
        assert design.get("stage_id") == "stage_0"
        assert "geometry" in design
        assert "sources" in design
        assert "monitors" in design
        assert len(design.get("geometry", [])) == 1
        assert design["geometry"][0]["type"] == "cylinder"

    def test_simulation_designer_node_creates_design(self, base_state, valid_plan):
        """Basic design creation with all fields mapped correctly."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation with gold nanorod...",
            "geometry": [{"type": "cylinder", "radius": 20, "material": "gold"}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
            "materials": [{"material_id": "gold", "source": "Johnson-Christy"}],
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
        """Multiple geometry objects should be preserved."""
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

        design = result["design_description"]
        assert len(design["geometry"]) == 2
        assert design["geometry"][0]["type"] == "cylinder"
        assert design["geometry"][1]["type"] == "box"

    def test_designer_maps_materials_with_source(self, base_state, valid_plan):
        """Material sources should be preserved in design."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Material test",
            "geometry": [{"type": "sphere", "radius": 50, "material": "silver"}],
            "sources": [{"type": "plane_wave"}],
            "monitors": [{"type": "flux", "name": "scattering"}],
            "materials": [
                {"material_id": "silver", "source": "Palik", "wavelength_range": [300, 1000]},
                {"material_id": "glass", "source": "BK7", "n": 1.52},
            ],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        design = result["design_description"]
        assert len(design["materials"]) == 2
        assert design["materials"][0]["source"] == "Palik"
        assert design["materials"][1]["material_id"] == "glass"


class TestSimulationDesignerMissingStageId:
    """Verify simulation designer handles missing current_stage_id."""

    def test_designer_escalates_when_stage_id_missing(self, base_state, valid_plan):
        """Missing stage_id should trigger user escalation."""
        from src.agents.design import simulation_designer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = None  # Missing!
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        # Should NOT call LLM
        with patch("src.agents.design.call_agent_with_metrics") as mock_call:
            result = simulation_designer_node(base_state)

        mock_call.assert_not_called()
        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert len(result.get("pending_user_questions", [])) > 0
        # Check the error message is informative
        assert "stage" in result["pending_user_questions"][0].lower()

    def test_designer_escalates_when_stage_id_empty_string(self, base_state, valid_plan):
        """Empty string stage_id should trigger user escalation."""
        from src.agents.design import simulation_designer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = ""  # Empty string - falsy
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch("src.agents.design.call_agent_with_metrics") as mock_call:
            result = simulation_designer_node(base_state)

        mock_call.assert_not_called()
        assert result.get("ask_user_trigger") is not None


class TestSimulationDesignerNewAssumptions:
    """Verify simulation designer handles new_assumptions extraction."""

    def test_designer_extracts_new_assumptions_list(self, base_state, valid_plan):
        """Valid new_assumptions list should be merged into state assumptions."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Design with assumptions",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
            "new_assumptions": [
                "Periodic boundary conditions assumed",
                "Uniform illumination assumed",
            ],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["assumptions"] = {"global_assumptions": ["Pre-existing assumption"]}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        # Should have updated assumptions
        assert "assumptions" in result
        global_assumptions = result["assumptions"]["global_assumptions"]
        assert len(global_assumptions) == 3  # 1 pre-existing + 2 new
        assert "Pre-existing assumption" in global_assumptions
        assert "Periodic boundary conditions assumed" in global_assumptions

    def test_designer_handles_empty_new_assumptions(self, base_state, valid_plan):
        """Empty new_assumptions list should not modify state."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Design without new assumptions",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
            "new_assumptions": [],  # Empty list
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["assumptions"] = {"global_assumptions": ["Existing"]}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        # assumptions should NOT be in result (no change)
        assert "assumptions" not in result

    def test_designer_handles_invalid_new_assumptions_type(self, base_state, valid_plan):
        """Non-list new_assumptions should be ignored with warning."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Design with invalid assumptions",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
            "new_assumptions": {"invalid": "dict"},  # Wrong type - should be list
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["assumptions"] = {"global_assumptions": ["Existing"]}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        # Should not crash, assumptions should not be updated
        assert "assumptions" not in result

    def test_designer_handles_no_prior_assumptions(self, base_state, valid_plan):
        """Should create assumptions structure when none exists."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "First design",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
            "new_assumptions": ["First assumption ever"],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["assumptions"] = None  # No prior assumptions

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        assert "assumptions" in result
        assert "global_assumptions" in result["assumptions"]
        assert "First assumption ever" in result["assumptions"]["global_assumptions"]


class TestSimulationDesignerLLMFailure:
    """Verify simulation designer handles LLM failures with user escalation."""

    def test_designer_escalates_on_llm_exception(self, base_state, valid_plan):
        """LLM failure should trigger user escalation (not auto-approve)."""
        from src.agents.design import simulation_designer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=Exception("API rate limit exceeded"),
        ):
            result = simulation_designer_node(base_state)

        # Designer should escalate to user (can't produce output without LLM)
        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "llm_error"
        assert len(result.get("pending_user_questions", [])) > 0
        # Error message should be informative
        assert "rate limit" in result["pending_user_questions"][0].lower() or "failed" in result["pending_user_questions"][0].lower()

    def test_designer_includes_workflow_phase_on_error(self, base_state, valid_plan):
        """Escalation result should include workflow_phase."""
        from src.agents.design import simulation_designer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=RuntimeError("Connection failed"),
        ):
            result = simulation_designer_node(base_state)

        assert result.get("workflow_phase") == "design"


class TestSimulationDesignerContextCheck:
    """Verify simulation designer handles context check and escalation."""

    def test_designer_escalates_on_context_overflow(self, base_state, valid_plan):
        """Context overflow should return escalation, not call LLM."""
        from src.agents.design import simulation_designer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"

        escalation = {
            "ask_user_trigger": "context_overflow",
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context too large. How to proceed?"],
        }

        # Patch where it's imported (in design.py), not where it's defined
        with patch(
            "src.agents.design.check_context_or_escalate", return_value=escalation
        ):
            result = simulation_designer_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "context_overflow"

    def test_designer_merges_context_state_updates(self, base_state, valid_plan):
        """State updates from context check should be merged before LLM call."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Test",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"

        # Context check returns state updates (not escalation)
        context_updates = {
            "metrics": {"context_tokens": 5000},
        }

        # Patch where it's imported (in design.py), not where it's defined
        with patch(
            "src.agents.design.check_context_or_escalate", return_value=context_updates
        ), patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = simulation_designer_node(base_state)

        # LLM should have been called
        mock_call.assert_called_once()
        # Design should be in result
        assert "design_description" in result


class TestSimulationDesignerRevisionFeedback:
    """Verify simulation designer includes revision feedback in prompt."""

    def test_designer_includes_feedback_in_system_prompt(self, base_state, valid_plan):
        """Reviewer feedback should be added to system prompt on revision."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Revised design",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["reviewer_feedback"] = "Please add wavelength range to sources."

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = simulation_designer_node(base_state)

        # Check that the system prompt included feedback
        call_kwargs = mock_call.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt", call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
        assert "REVISION FEEDBACK" in system_prompt
        assert "wavelength" in system_prompt.lower()


class TestSimulationDesignerComplexityClass:
    """Verify simulation designer includes complexity class from plan."""

    def test_designer_injects_complexity_class(self, base_state, valid_plan):
        """Complexity class from plan should be in system prompt."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Test",
            "geometry": [{"type": "sphere"}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
        }

        # Add complexity class to plan stage
        valid_plan["stages"][0]["complexity_class"] = "advanced"
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = simulation_designer_node(base_state)

        # Check that complexity class was included
        call_kwargs = mock_call.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt", call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
        assert "complexity" in system_prompt.lower()


class TestSimulationDesignerNonDictResponse:
    """Verify simulation designer handles non-dict LLM responses."""

    def test_designer_handles_string_response(self, base_state, valid_plan):
        """String response from LLM should be stored as design_description."""
        from src.agents.design import simulation_designer_node

        mock_response = "This is a plain text response, not JSON"

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["assumptions"] = {"global_assumptions": ["Existing"]}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        # Should not crash, design_description should contain the string
        assert "design_description" in result
        assert result["design_description"] == mock_response
        # Should NOT try to extract new_assumptions from string
        assert "assumptions" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES AND INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignAgentsEdgeCases:
    """Edge cases for both design agents."""

    def test_reviewer_with_none_design_description(self, base_state):
        """Reviewer should handle None design_description gracefully."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Design approved",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = None  # None design

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Should not crash
        assert result["last_design_review_verdict"] == "approve"

    def test_reviewer_with_empty_issues_list(self, base_state):
        """Reviewer should handle empty issues list correctly."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [],  # Empty but needs_revision
            "summary": "Needs work",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["reviewer_issues"] == []
        assert result["design_revision_count"] == 1  # Still incremented

    def test_reviewer_with_none_issues(self, base_state):
        """Reviewer should handle None issues gracefully."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": None,  # None instead of list
            "summary": "Approved",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Should handle None gracefully
        assert result["reviewer_issues"] == []

    def test_designer_with_no_plan(self, base_state):
        """Designer should work with minimal plan."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Minimal design",
            "geometry": [],
            "sources": [],
            "monitors": [],
        }

        base_state["plan"] = {}  # Empty plan
        base_state["current_stage_id"] = "stage_0"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        assert result["workflow_phase"] == "design"
        assert "design_description" in result

    def test_reviewer_counter_none_value(self, base_state):
        """Reviewer should handle None counter as 0."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "test"}],
            "summary": "Fix",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = None  # None counter

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        # Should treat None as 0 and increment to 1
        assert result["design_revision_count"] == 1


class TestDesignReviewerPlanStageReference:
    """Verify design reviewer includes plan stage in user content."""

    def test_reviewer_includes_plan_stage_spec(self, base_state, valid_plan):
        """User content should include plan stage specification for reference."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Design matches plan",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "geometry": [{"type": "cylinder"}],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = design_reviewer_node(base_state)

        # Check that plan stage was included in user content
        call_kwargs = mock_call.call_args
        user_content = call_kwargs.kwargs.get("user_content", call_kwargs.args[2] if len(call_kwargs.args) > 2 else "")
        assert "PLAN STAGE SPEC" in user_content
        assert "MATERIAL_VALIDATION" in user_content  # Stage type from valid_plan

    def test_reviewer_includes_previous_feedback_in_user_content(self, base_state):
        """User content should include previous reviewer feedback if present."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Issues fixed",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}
        base_state["reviewer_feedback"] = "Previous feedback: Add wavelength range to sources."

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = design_reviewer_node(base_state)

        # Check that feedback was included in user content
        call_kwargs = mock_call.call_args
        user_content = call_kwargs.kwargs.get("user_content", call_kwargs.args[2] if len(call_kwargs.args) > 2 else "")
        assert "REVISION FEEDBACK" in user_content
        assert "wavelength" in user_content.lower()


class TestDesignWorkflowPhase:
    """Verify workflow_phase is set correctly for both nodes."""

    def test_designer_sets_workflow_phase_design(self, base_state, valid_plan):
        """Designer should set workflow_phase to 'design'."""
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Test",
            "geometry": [],
            "sources": [],
            "monitors": [],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        assert result["workflow_phase"] == "design"

    def test_reviewer_sets_workflow_phase_design_review(self, base_state):
        """Reviewer should set workflow_phase to 'design_review'."""
        from src.agents.design import design_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Approved",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {"stage_id": "stage_0"}

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = design_reviewer_node(base_state)

        assert result["workflow_phase"] == "design_review"
