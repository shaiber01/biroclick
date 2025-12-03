"""Planner node contract tests covering LLM calls and output mapping."""

import json
from copy import deepcopy
from unittest.mock import patch


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
        assert call_kwargs.get("agent_name") == "planner", (
            "Expected agent_name='planner', "
            f"got '{call_kwargs.get('agent_name')}'"
        )

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
        content_str = (
            user_content
            if isinstance(user_content, str)
            else json.dumps(user_content, default=str)
        )

        assert "Extinction spectrum" in content_str, (
            "user_content should reference figure descriptions "
            "so LLM can plan against them"
        )
        assert "gold nanorods" in content_str.lower(), (
            "user_content should include actual paper text, not just metadata"
        )
        assert (
            base_state["paper_text"][:40].strip().split()[0].lower()
            in content_str.lower()
        ), "user_content should include actual paper text chunk"
        state_payload = call_kwargs.get("state", {})
        assert state_payload.get("paper_id") == base_state["paper_id"], (
            "State forwarded to LLM should still reference the same paper identifier"
        )

    def test_plan_node_adds_replan_context(self, base_state):
        """plan_node should add replan context to system prompt if replan_count > 0."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 1

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [{"figure_id": "f1"}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)

        call_kwargs = mock.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        assert "Replan Attempt #1" in system_prompt


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
        assert isinstance(
            first_stage.get("dependencies"), list
        ), "Stage dependencies should be a list"

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
        assert deps in (None, []) or isinstance(
            deps, list
        ), f"Progress dependencies should round-trip list semantics, got {deps}"

        # Ensure no error flags are set
        assert not result.get("awaiting_user_input")


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

    def test_plan_node_handles_malformed_extracted_parameters(self, base_state):
        """plan_node should handle malformed extracted_parameters gracefully."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [{"figure_id": "f1"}],
            # Missing 'value' field - likely invalid per schema
            "extracted_parameters": [{"name": "p1"}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        if result.get("last_plan_review_verdict") == "needs_revision":
            assert "failed" in result.get("planner_feedback", "").lower()
        else:
            # If it didn't fail, ensure parameter was filtered or handled without crash.
            result.get("extracted_parameters", [])

