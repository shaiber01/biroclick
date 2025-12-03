import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from schemas.state import create_initial_state


class TestPlannerLLMCalls:
    """Verify planner-related nodes call the LLM with correct parameters."""

    def test_plan_node_calls_llm_with_correct_agent_name(self, base_state):
        """plan_node must call LLM with agent_name='planner'."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "planner", \
            f"Expected agent_name='planner', got '{call_kwargs.get('agent_name')}'"

    def test_plan_node_passes_figures_and_text_to_llm(self, base_state):
        """plan_node must include paper text and figures in user_content."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
            "extracted_parameters": [{"name": "length", "value": 10}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content")
        content_str = user_content if isinstance(user_content, str) else json.dumps(
            user_content, default=str
        )

        assert "Extinction spectrum" in content_str, \
            "user_content should reference figure descriptions so LLM can plan against them"
        assert "gold nanorods" in content_str.lower(), \
            "user_content should include actual paper text, not just metadata"
        assert base_state["paper_text"][:40].strip().split()[0].lower() in content_str.lower(), \
            "user_content should include actual paper text chunk"
        state_payload = call_kwargs.get("state", {})
        assert state_payload.get("paper_id") == base_state["paper_id"], \
            "State forwarded to LLM should still reference the same paper identifier"


class TestPlanReviewerLLMCalls:
    """Verify plan reviewer LLM interactions."""

    def test_plan_reviewer_node_calls_llm_with_correct_agent_name(self, base_state):
        """plan_reviewer_node must call LLM with agent_name='plan_reviewer'."""
        from src.agents.planning import plan_reviewer_node

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_reviewer_node(base_state)

        call_kwargs = mock.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        assert len(system_prompt) > 100, \
            f"System prompt too short ({len(system_prompt)} chars)"
        assert call_kwargs.get("agent_name") == "plan_reviewer", \
            f"Expected agent_name='plan_reviewer', got '{call_kwargs.get('agent_name')}'"


class TestOutputStructure:
    """Verify node outputs have correct structure."""

    def test_plan_node_output_has_required_fields(self, base_state):
        """plan_node output must have workflow_phase and plan."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
            "extracted_parameters": [{"name": "p1", "value": 10}],
            "planned_materials": ["Au"],
            "assumptions": {"a1": "test assumption"},
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        assert result["workflow_phase"] == "planning"
        assert "plan" in result, f"Missing plan. Got keys: {result.keys()}"

        plan = result["plan"]
        assert "stages" in plan and plan["stages"], "Plan missing stages"
        first_stage = plan["stages"][0]
        assert first_stage["stage_id"] == "s1"
        assert first_stage["stage_type"] == "MATERIAL_VALIDATION"
        assert isinstance(first_stage.get("dependencies"), list), \
            "Stage dependencies should be a list"

        assert result["planned_materials"] == mock_response["planned_materials"]
        assert result["assumptions"] == mock_response["assumptions"]
        assert result["paper_domain"] == mock_response["paper_domain"]
        assert result["extracted_parameters"], "extracted_parameters should be populated"

        progress = result.get("progress")
        assert progress and progress["stages"], "Progress not initialized"
        stage_entry = progress["stages"][0]
        assert stage_entry["stage_id"] == "s1"
        assert stage_entry["status"] == "not_started"
        assert "stage_type" in stage_entry
        deps = stage_entry.get("dependencies")
        assert deps in (None, []) or isinstance(deps, list), \
            f"Progress dependencies should round-trip list semantics, got {deps}"


class TestBusinessLogic:
    """Test that planning business logic rules are enforced."""

    def test_plan_reviewer_rejects_empty_stages(self, base_state):
        """plan_reviewer_node should reject plans with no stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Empty Plan",
            "stages": [],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "")
        assert "PLAN_ISSUE" in feedback

    def test_plan_reviewer_rejects_stages_without_targets(self, base_state):
        """plan_reviewer_node should reject stages without targets."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Bad Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": [],
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "")
        assert "Stage 's1' has no targets" in feedback


class TestCircularDependencyDetection:
    """Verify plan_reviewer correctly detects dependency cycles."""

    def test_plan_reviewer_detects_simple_cycle(self, base_state):
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Cyclic Plan",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_b"],
                },
                {
                    "stage_id": "stage_b",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_a"],
                },
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "")
        assert "circular" in feedback.lower() or "cycle" in feedback.lower()

    def test_plan_reviewer_detects_self_dependency(self, base_state):
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Self-Dependent Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_detects_transitive_cycle(self, base_state):
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Transitive Cycle",
            "stages": [
                {
                    "stage_id": "a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["c"],
                },
                {
                    "stage_id": "b",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig1"],
                    "dependencies": ["a"],
                },
                {
                    "stage_id": "c",
                    "stage_type": "ARRAY_SYSTEM",
                    "targets": ["Fig2"],
                    "dependencies": ["b"],
                },
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"


class TestStateIsolation:
    """Verify plan_node doesn't mutate input state."""

    def test_plan_node_doesnt_mutate_input(self, base_state):
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ],
            "targets": [],
            "extracted_parameters": [],
        }

        original_state = deepcopy(base_state)

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            plan_node(base_state)

        assert base_state.get("paper_id") == original_state.get("paper_id")
        assert base_state.get("paper_text") == original_state.get("paper_text")


class TestPlanningEdgeCases:
    """Planning-specific edge cases that should escalate properly."""

    def test_plan_node_with_very_short_paper(self):
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="Short paper.",
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text" or \
            result.get("awaiting_user_input") is True

    def test_plan_node_handles_missing_paper_text(self):
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="",
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True


class TestFieldMapping:
    """Verify planner output fields are correctly mapped to state."""

    def test_planner_maps_stages_correctly(self, base_state):
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test_paper",
            "title": "Test Title",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "FDTD_DIRECT",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_0"],
                },
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
            "extracted_parameters": [{"name": "wavelength", "value": 500}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert len(plan.get("stages", [])) == 2
        assert plan["stages"][0]["stage_id"] == "stage_0"
        assert plan["stages"][1]["dependencies"] == ["stage_0"]


class TestProgressInitialization:
    """Verify progress is correctly initialized from plan."""

    def test_plan_node_initializes_progress(self, base_state):
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "FDTD_DIRECT",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],
                },
            ],
            "targets": [{"figure_id": "Fig1"}],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        progress = result.get("progress")
        assert progress and len(progress.get("stages", [])) == 2
        for stage in progress["stages"]:
            assert stage["status"] == "not_started"


class TestPlanNodeEdgeCases:
    """Verify plan_node handles failure paths gracefully."""

    def test_plan_node_handles_progress_init_failure(self, base_state):
        from src.agents.planning import plan_node

        mock_response = {
            "stages": [{"stage_id": "s1"}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.initialize_progress_from_plan",
                side_effect=ValueError("Init failed"),
            ):
                result = plan_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Progress initialization failed" in result["planner_feedback"]

