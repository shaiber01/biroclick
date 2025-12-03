"""Integration tests that verify nodes mutate state as expected."""

from unittest.mock import patch, MagicMock

from schemas.state import create_initial_state, MAX_REPLANS, MAX_DESIGN_REVISIONS, MAX_CODE_REVISIONS


class TestStateMutations:
    """Test that nodes mutate state correctly."""

    # ═══════════════════════════════════════════════════════════════════════
    # plan_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_plan_node_sets_plan_field(self):
        """plan_node must set state['plan'] with the LLM response."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "test",
            "title": "Gold Nanorod Study",
            "summary": "Study of gold nanorods",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
            "extracted_parameters": [{"name": "length", "value": 100, "unit": "nm"}],
            "paper_domain": "plasmonics",
            "planned_materials": [{"name": "gold", "source": "palik"}],
            "assumptions": {"global_assumptions": ["Assumption 1"]},
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
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Verify plan is set with all fields
        assert "plan" in result, "plan_node must return 'plan' in result"
        assert result["plan"]["title"] == "Gold Nanorod Study"
        assert result["plan"]["paper_id"] == "test"
        assert result["plan"]["summary"] == "Study of gold nanorods"
        assert len(result["plan"]["stages"]) == 1
        assert result["plan"]["stages"][0]["stage_id"] == "s1"
        assert len(result["plan"]["targets"]) == 1
        
        # Verify other state fields are set
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "planning"
        assert "paper_domain" in result
        assert result["paper_domain"] == "plasmonics"
        assert "planned_materials" in result
        assert len(result["planned_materials"]) == 1
        assert "assumptions" in result
        assert "global_assumptions" in result["assumptions"]
        
        # Verify progress is initialized
        assert "progress" in result
        assert "extracted_parameters" in result

    def test_plan_node_handles_missing_llm_fields(self):
        """plan_node must handle missing fields in LLM response gracefully."""
        from src.agents.planning import plan_node

        # LLM returns minimal response
        mock_plan = {
            "title": "Test",
            "stages": [],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should still set plan with defaults for missing fields
        assert "plan" in result
        assert result["plan"]["title"] == "Test"
        assert result["plan"].get("stages") == []
        assert result["plan"].get("targets") == []
        assert result["plan"].get("extracted_parameters") == []
        assert result["paper_domain"] == "other"  # Default when missing

    def test_plan_node_preserves_existing_paper_id(self):
        """plan_node must preserve paper_id from state if valid."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "different_id",
            "title": "Test",
            "stages": [],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("original_id", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should preserve original paper_id
        assert result["plan"]["paper_id"] == "original_id"

    def test_plan_node_rejects_short_paper_text(self):
        """plan_node must reject paper text that's too short."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "short", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0

    def test_plan_node_rejects_empty_paper_text(self):
        """plan_node must reject empty paper text."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result

    def test_plan_node_rejects_whitespace_only_paper_text(self):
        """plan_node must reject whitespace-only paper text."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "   \n\t   ", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result

    def test_plan_node_rejects_boundary_length_paper_text(self):
        """plan_node must reject paper text exactly at boundary (99 chars)."""
        from src.agents.planning import plan_node

        # Exactly 99 characters
        short_text = "a" * 99
        state = create_initial_state("test", short_text, "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result

    def test_plan_node_accepts_boundary_length_paper_text(self):
        """plan_node must accept paper text at minimum boundary (100 chars)."""
        from src.agents.planning import plan_node

        # Exactly 100 characters
        valid_text = "a" * 100
        mock_plan = {
            "paper_id": "test",
            "title": "Test",
            "stages": [],
        }
        state = create_initial_state("test", valid_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        assert "plan" in result
        assert "ask_user_trigger" not in result

    def test_plan_node_handles_progress_initialization_failure(self):
        """plan_node must handle progress initialization failures."""
        from src.agents.planning import plan_node
        from schemas.state import initialize_progress_from_plan

        # Use a plan structure that will cause initialize_progress_from_plan to raise an exception
        # We'll patch initialize_progress_from_plan to raise an exception to test the error handling
        mock_plan = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION"}],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                # Patch initialize_progress_from_plan to raise an exception
                with patch(
                    "src.agents.planning.initialize_progress_from_plan",
                    side_effect=ValueError("Invalid plan structure"),
                ):
                    result = plan_node(state)

        # Should mark plan for revision when progress initialization fails
        assert "last_plan_review_verdict" in result
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert "Progress initialization failed" in result["planner_feedback"]
        assert "replan_count" in result

    # ═══════════════════════════════════════════════════════════════════════
    # plan_reviewer_node tests
    # ═══════════════════════════════════════════════════════════════════════

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
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good", "feedback": "Looks good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert "last_plan_review_verdict" in result
        assert result["last_plan_review_verdict"] == "approve"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "plan_review"
        # Should not increment replan_count on approve
        assert "replan_count" not in result or result.get("replan_count") == state.get("replan_count", 0)

    def test_plan_reviewer_increments_replan_count_on_needs_revision(self):
        """plan_reviewer_node must increment replan_count when verdict is needs_revision."""
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
        state["replan_count"] = 1
        # Set max_replans higher than current count to allow incrementing
        state["runtime_config"] = {"max_replans": 3}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs work",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "replan_count" in result
        # Should increment from 1 to 2 for LLM rejections
        assert result["replan_count"] == 2, (
            f"BUG: replan_count should be 2 (was 1), but got {result['replan_count']}. "
            "plan_reviewer_node is not incrementing replan_count for LLM rejections."
        )
        assert "planner_feedback" in result

    def test_plan_reviewer_respects_max_replans(self):
        """plan_reviewer_node must not increment replan_count beyond max_replans."""
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
        state["replan_count"] = MAX_REPLANS
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs work",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["replan_count"] == MAX_REPLANS  # Should not exceed max

    def test_plan_reviewer_rejects_empty_plan(self):
        """plan_reviewer_node must reject plans with no stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {"title": "Test", "stages": [], "targets": []}

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert len(result.get("planner_feedback", "")) > 0

    def test_plan_reviewer_rejects_none_plan(self):
        """plan_reviewer_node must handle None plan."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = None

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_rejects_plan_with_none_stages(self):
        """plan_reviewer_node must reject plans with None stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {"title": "Test", "stages": None, "targets": []}

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_rejects_plan_with_missing_stage_id(self):
        """plan_reviewer_node must reject plans with stages missing stage_id."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_type": "MATERIAL_VALIDATION",
                    "description": "Test",
                    "targets": ["Fig1"],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "PLAN_ISSUE" in result.get("planner_feedback", "")

    def test_plan_reviewer_rejects_plan_with_duplicate_stage_ids(self):
        """plan_reviewer_node must reject plans with duplicate stage IDs."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_0",  # Duplicate
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": [],
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "duplicate" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_missing_dependencies(self):
        """plan_reviewer_node must reject plans with dependencies on non-existent stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],  # stage_0 doesn't exist
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "missing stage" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_circular_dependencies(self):
        """plan_reviewer_node must reject plans with circular dependencies."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_1"],  # Circular
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_0"],  # Circular
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_self_dependency(self):
        """plan_reviewer_node must reject plans with self-dependencies."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],  # Self-dependency
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "depends on itself" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_stages_missing_targets(self):
        """plan_reviewer_node must reject plans with stages that have no targets."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": [],  # No targets
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no targets" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_invalid_stage_structure(self):
        """plan_reviewer_node must reject plans with invalid stage structures."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": ["not_a_dict", {"stage_id": "stage_0"}],  # Invalid structure
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "invalid stage structure" in result.get("planner_feedback", "").lower()

    # ═══════════════════════════════════════════════════════════════════════
    # design_reviewer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_counter_increments_on_revision(self):
        """Revision counters must increment when verdict is needs_revision."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0"}
        state["design_revision_count"] = 0
        mock_response = {
            "verdict": "needs_revision",
            "issues": ["problem"],
            "summary": "Fix",
            "feedback": "Needs improvement",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] == 1, "Counter should increment on needs_revision"
        assert "last_design_review_verdict" in result
        assert result["last_design_review_verdict"] == "needs_revision"
        assert "reviewer_feedback" in result
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design_review"

    def test_design_reviewer_counter_does_not_increment_on_approve(self):
        """design_reviewer_node must not increment counter when verdict is approve."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0"}
        state["design_revision_count"] = 5
        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Good",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["design_revision_count"] == 5, "Counter should not increment on approve"
        assert result["last_design_review_verdict"] == "approve"
        assert "reviewer_feedback" not in result

    def test_design_reviewer_respects_max_revisions(self):
        """design_reviewer_node must not increment counter beyond max_design_revisions."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0"}
        state["design_revision_count"] = MAX_DESIGN_REVISIONS
        mock_response = {
            "verdict": "needs_revision",
            "issues": ["problem"],
            "summary": "Fix",
            "feedback": "Still needs work",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS, "Counter should not exceed max"

    def test_design_reviewer_handles_missing_verdict(self):
        """design_reviewer_node must handle missing verdict gracefully."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0"}
        mock_response = {
            "issues": [],
            "summary": "No verdict",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        # Should default to approve
        assert result["last_design_review_verdict"] == "approve"

    def test_design_reviewer_includes_reviewer_issues(self):
        """design_reviewer_node must include reviewer_issues in result."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0"}
        mock_response = {
            "verdict": "approve",
            "issues": [{"severity": "minor", "description": "Minor issue"}],
            "summary": "Good",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert "reviewer_issues" in result
        assert len(result["reviewer_issues"]) == 1

    # ═══════════════════════════════════════════════════════════════════════
    # simulation_designer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_simulation_designer_sets_design_description(self):
        """simulation_designer_node must set design_description."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ]
        }
        mock_response = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40},
            "materials": ["gold"],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "design_description" in result
        assert result["design_description"] == mock_response
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design"

    def test_simulation_designer_rejects_missing_stage_id(self):
        """simulation_designer_node must reject when current_stage_id is missing."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = None

        result = simulation_designer_node(state)

        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "design_description" not in result

    def test_simulation_designer_handles_new_assumptions(self):
        """simulation_designer_node must handle new_assumptions from LLM."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = {"global_assumptions": ["Existing assumption"]}
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ]
        }
        mock_response = {
            "geometry": "nanorod",
            "new_assumptions": ["New assumption 1", "New assumption 2"],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "assumptions" in result
        assert "global_assumptions" in result["assumptions"]
        assert len(result["assumptions"]["global_assumptions"]) == 3
        assert "New assumption 1" in result["assumptions"]["global_assumptions"]
        assert "New assumption 2" in result["assumptions"]["global_assumptions"]

    # ═══════════════════════════════════════════════════════════════════════
    # code_generator_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_code_generator_sets_code_field(self):
        """code_generator_node must set code field."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100},
        }
        mock_code = "import meep as mp\n# Simulation code here"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": mock_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code" in result
        assert result["code"] == mock_code
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "code_generation"

    def test_code_generator_rejects_missing_stage_id(self):
        """code_generator_node must reject when current_stage_id is missing."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = None

        result = code_generator_node(state)

        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "code" not in result

    def test_code_generator_rejects_stub_design(self):
        """code_generator_node must reject stub design descriptions."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = "STUB: This would be generated"

        result = code_generator_node(state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] > 0
        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower()

    def test_code_generator_rejects_empty_design(self):
        """code_generator_node must reject empty design descriptions."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = ""

        result = code_generator_node(state)

        assert "design_revision_count" in result
        assert "reviewer_feedback" in result

    def test_code_generator_rejects_missing_validated_materials_for_stage_1_plus(self):
        """code_generator_node must reject missing validated_materials for Stage 1+."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_1"
        state["current_stage_type"] = "SINGLE_STRUCTURE"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100},
        }
        state["validated_materials"] = []  # Empty

        result = code_generator_node(state)

        assert "run_error" in result
        assert "validated_materials" in result["run_error"].lower()
        assert "code_revision_count" in result

    def test_code_generator_rejects_stub_code(self):
        """code_generator_node must reject stub code from LLM."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100},
        }
        stub_code = "STUB: Code would be generated here"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": stub_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code_revision_count" in result
        assert result["code_revision_count"] > 0
        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower()

    # ═══════════════════════════════════════════════════════════════════════
    # code_reviewer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_code_reviewer_sets_verdict_field(self):
        """code_reviewer_node must set last_code_review_verdict."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Good code",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert "last_code_review_verdict" in result
        assert result["last_code_review_verdict"] == "approve"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "code_review"

    def test_code_reviewer_increments_counter_on_needs_revision(self):
        """code_reviewer_node must increment code_revision_count on needs_revision."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["code_revision_count"] = 1
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs fixes",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 2
        assert "reviewer_feedback" in result

    def test_code_reviewer_respects_max_revisions(self):
        """code_reviewer_node must not increment counter beyond max_code_revisions."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["code_revision_count"] = MAX_CODE_REVISIONS
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs fixes",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result.get("ask_user_trigger") == "code_review_limit"
        assert result.get("awaiting_user_input") is True

    def test_code_reviewer_defaults_to_needs_revision_on_missing_verdict(self):
        """code_reviewer_node must default to needs_revision when verdict is missing."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        mock_response = {
            "issues": [],
            "summary": "No verdict",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "needs_revision"

    # ═══════════════════════════════════════════════════════════════════════
    # execution_validator_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_execution_validator_sets_verdict_field(self):
        """execution_validator_node must set execution_verdict."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "pass",
            "summary": "Execution successful",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert "execution_verdict" in result
        assert result["execution_verdict"] == "pass"
        assert "execution_feedback" in result
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "execution_validation"

    def test_execution_validator_increments_counter_on_fail(self):
        """execution_validator_node must increment execution_failure_count on fail."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}
        state["run_error"] = "Simulation failed"
        state["execution_failure_count"] = 0
        mock_response = {
            "verdict": "fail",
            "summary": "Execution failed",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1

    def test_execution_validator_handles_timeout_with_skip_warning(self):
        """execution_validator_node must handle timeout with skip_with_warning fallback."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"timeout_exceeded": True, "files": []}
        state["run_error"] = "Execution exceeded timeout"
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "fallback_strategy": "skip_with_warning",
                }
            ]
        }

        result = execution_validator_node(state)

        assert result["execution_verdict"] == "pass"
        assert "timeout" in result["execution_feedback"].lower()

    # ═══════════════════════════════════════════════════════════════════════
    # physics_sanity_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_physics_sanity_sets_verdict_field(self):
        """physics_sanity_node must set physics_verdict."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "pass",
            "summary": "Physics looks good",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert "physics_verdict" in result
        assert result["physics_verdict"] == "pass"
        assert "physics_feedback" in result
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "physics_validation"

    def test_physics_sanity_increments_physics_failure_count_on_fail(self):
        """physics_sanity_node must increment physics_failure_count on fail verdict."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["physics_failure_count"] = 0
        mock_response = {
            "verdict": "fail",
            "summary": "Physics check failed",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1

    def test_physics_sanity_increments_design_revision_count_on_design_flaw(self):
        """physics_sanity_node must increment design_revision_count on design_flaw verdict."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["design_revision_count"] = 2
        mock_response = {
            "verdict": "design_flaw",
            "summary": "Design flaw detected",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 3
        assert "design_feedback" in result


