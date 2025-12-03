"""Planner node contract tests covering LLM calls and output mapping."""

import json
from copy import deepcopy
from unittest.mock import patch, MagicMock
import pytest


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

    def test_plan_node_adds_replan_context_multiple_attempts(self, base_state):
        """plan_node should add correct replan context for multiple replan attempts."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 3

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
        assert "Replan Attempt #3" in system_prompt
        assert "Replan Attempt #1" not in system_prompt or system_prompt.count("Replan Attempt #3") == 1

    def test_plan_node_no_replan_context_when_zero(self, base_state):
        """plan_node should not add replan context when replan_count is 0."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 0

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
        assert "Replan Attempt" not in system_prompt

    def test_plan_node_passes_state_to_llm(self, base_state):
        """plan_node must pass state dict to LLM call."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)

        call_kwargs = mock.call_args.kwargs
        assert "state" in call_kwargs, "LLM call must include state parameter"
        state_passed = call_kwargs["state"]
        assert isinstance(state_passed, dict), "State must be a dict"
        assert state_passed.get("paper_id") == base_state["paper_id"]


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

        # Check all key fields are unchanged
        assert base_state.get("paper_id") == original_state.get("paper_id")
        assert base_state.get("paper_text") == original_state.get("paper_text")
        assert base_state.get("paper_figures") == original_state.get("paper_figures")
        assert base_state.get("paper_domain") == original_state.get("paper_domain")
        assert base_state.get("replan_count") == original_state.get("replan_count")
        # Ensure no new keys were added to input state
        assert set(base_state.keys()) == set(original_state.keys()), (
            "plan_node should not add keys to input state"
        )


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
        assert len(plan.get("stages", [])) == 2, "Plan should have 2 stages"
        assert plan["stages"][0]["stage_id"] == "stage_0"
        assert plan["stages"][0]["stage_type"] == "MATERIAL_VALIDATION"
        assert plan["stages"][1]["stage_id"] == "stage_1"
        assert plan["stages"][1]["stage_type"] == "FDTD_DIRECT"
        # Dependencies are in plan, not progress
        assert plan["stages"][1]["dependencies"] == ["stage_0"]
        # Verify all fields are mapped
        # paper_id is preserved from state if valid, not from agent_output
        assert plan.get("paper_id") == base_state.get("paper_id"), (
            "paper_id should be preserved from state when valid"
        )
        assert plan.get("title") == "Test Title"
        assert len(plan.get("targets", [])) == 2
        assert len(plan.get("extracted_parameters", [])) == 1


class TestProgressInitialization:
    """Verify progress is correctly initialized from plan.
    
    Note on dependencies: Dependencies are stored in plan, not progress, per the design 
    in schemas/state.py. Progress only stores EXECUTION STATE fields (status, outputs, 
    discrepancies, etc.). Design specs (like dependencies, targets, expected_outputs) 
    stay in plan. This is INTENDED behavior - use get_plan_stage() to look up 
    dependencies when needed. The test verifies dependencies are in plan as expected.
    """

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
        assert progress is not None, "Progress must be initialized"
        assert isinstance(progress, dict), "Progress must be a dict"
        assert "stages" in progress, "Progress must have stages key"
        assert len(progress.get("stages", [])) == 2, "Progress should have 2 stages"
        
        # Verify each stage is properly initialized
        stage_ids = {s["stage_id"] for s in progress["stages"]}
        assert "stage_0" in stage_ids
        assert "stage_1" in stage_ids
        
        for stage in progress["stages"]:
            assert stage["status"] == "not_started", f"Stage {stage.get('stage_id')} should be not_started"
            assert "stage_id" in stage
            assert "stage_type" in stage
            # Dependencies are in plan, not progress (per schema design)
            # Verify dependencies are in plan instead
        plan = result.get("plan", {})
        plan_stage_1 = next((s for s in plan.get("stages", []) if s.get("stage_id") == "stage_1"), None)
        assert plan_stage_1 is not None
        assert plan_stage_1.get("dependencies") == ["stage_0"], (
            "Dependencies should be in plan, not progress"
        )


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


class TestPaperTextValidation:
    """Verify plan_node validates paper_text correctly."""

    def test_plan_node_rejects_empty_paper_text(self, base_state):
        """plan_node should reject empty paper_text and request user input."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = ""

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert "paper text" in result.get("pending_user_questions", [""])[0].lower()
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "planning"

    def test_plan_node_rejects_too_short_paper_text(self, base_state):
        """plan_node should reject paper_text shorter than 100 characters."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = "Short text"  # Less than 100 chars

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert "too short" in result.get("pending_user_questions", [""])[0].lower()

    def test_plan_node_rejects_none_paper_text(self, base_state):
        """plan_node should handle None paper_text."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = None

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "missing_paper_text"

    def test_plan_node_rejects_whitespace_only_paper_text(self, base_state):
        """plan_node should reject paper_text that is only whitespace."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = "   \n\t   "  # Only whitespace

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "missing_paper_text"


class TestLLMErrorHandling:
    """Verify plan_node handles LLM errors correctly."""

    def test_plan_node_handles_llm_exception(self, base_state):
        """plan_node should handle LLM call exceptions gracefully."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("LLM call failed")
        ):
            result = plan_node(base_state)

        # Should return error escalation state
        assert "workflow_phase" in result
        assert result.get("awaiting_user_input") is True or "error" in str(result).lower()
        # Should have error information
        assert "ask_user_trigger" in result or "planner_feedback" in result

    def test_plan_node_handles_llm_timeout(self, base_state):
        """plan_node should handle LLM timeout exceptions."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=TimeoutError("Request timed out")
        ):
            result = plan_node(base_state)

        assert result.get("awaiting_user_input") is True or "error" in str(result).lower()


class TestPaperIdPreservation:
    """Verify plan_node preserves paper_id correctly.
    
    Note: This is INTENDED behavior per the component design. The component
    preserves paper_id from state if it's valid (not None and not "unknown"),
    because the paper_id from state is authoritative (set by paper_loader).
    The LLM's paper_id is only used as fallback.
    """

    def test_plan_node_preserves_valid_paper_id_from_state(self, base_state):
        """plan_node should preserve paper_id from state if valid."""
        from src.agents.planning import plan_node

        base_state["paper_id"] = "preserved_id"

        mock_response = {
            "paper_id": "different_id",  # LLM returns different ID
            "title": "Test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("paper_id") == "preserved_id", (
            "Should preserve paper_id from state when valid"
        )

    def test_plan_node_uses_agent_paper_id_when_state_invalid(self, base_state):
        """plan_node should use agent paper_id when state paper_id is 'unknown'."""
        from src.agents.planning import plan_node

        base_state["paper_id"] = "unknown"

        mock_response = {
            "paper_id": "agent_provided_id",
            "title": "Test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("paper_id") == "agent_provided_id", (
            "Should use agent paper_id when state paper_id is 'unknown'"
        )

    def test_plan_node_handles_missing_paper_id(self, base_state):
        """plan_node should handle missing paper_id in state."""
        from src.agents.planning import plan_node

        if "paper_id" in base_state:
            del base_state["paper_id"]

        mock_response = {
            "paper_id": "agent_provided_id",
            "title": "Test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("paper_id") == "agent_provided_id", (
            "Should use agent paper_id when state paper_id is missing"
        )


class TestProgressInitializationEdgeCases:
    """Verify progress initialization handles edge cases."""

    def test_plan_node_handles_empty_stages_list(self, base_state):
        """plan_node should handle empty stages list."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [],  # Empty stages
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        # Should not crash, but progress may be None or empty
        plan = result.get("plan", {})
        assert plan.get("stages") == []
        # Progress may not be initialized if no stages
        progress = result.get("progress")
        if progress:
            assert progress.get("stages", []) == []

    def test_plan_node_handles_missing_stages_key(self, base_state):
        """plan_node should handle missing stages key in response."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            # Missing "stages" key
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("stages", []) == []
        # Progress should not be initialized if no stages
        progress = result.get("progress")
        assert progress is None or progress.get("stages", []) == []

    def test_plan_node_handles_sync_extracted_parameters_failure(self, base_state):
        """plan_node should handle sync_extracted_parameters failure."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [{"name": "p1", "value": 10}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.sync_extracted_parameters",
                side_effect=ValueError("Sync failed")
            ):
                result = plan_node(base_state)

        # Should mark plan for revision if sync fails
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "failed" in result.get("planner_feedback", "").lower()


class TestReplanCountLogic:
    """Verify replan_count increment and bounds checking."""

    def test_plan_node_increments_replan_count_on_progress_failure(self, base_state):
        """plan_node should increment replan_count when progress initialization fails."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 0

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.initialize_progress_from_plan",
                side_effect=ValueError("Init failed")
            ):
                result = plan_node(base_state)

        assert result.get("replan_count") == 1, (
            "Should increment replan_count from 0 to 1"
        )

    def test_plan_node_respects_max_replans_limit(self, base_state):
        """plan_node should not increment replan_count beyond max_replans."""
        from src.agents.planning import plan_node
        from schemas.state import MAX_REPLANS

        base_state["replan_count"] = MAX_REPLANS
        base_state["runtime_config"] = {"max_replans": MAX_REPLANS}

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.initialize_progress_from_plan",
                side_effect=ValueError("Init failed")
            ):
                result = plan_node(base_state)

        assert result.get("replan_count") == MAX_REPLANS, (
            f"Should not increment beyond MAX_REPLANS ({MAX_REPLANS})"
        )

    def test_plan_node_respects_custom_max_replans(self, base_state):
        """plan_node should respect custom max_replans from runtime_config."""
        from src.agents.planning import plan_node

        custom_max = 5
        base_state["replan_count"] = custom_max
        base_state["runtime_config"] = {"max_replans": custom_max}

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.initialize_progress_from_plan",
                side_effect=ValueError("Init failed")
            ):
                result = plan_node(base_state)

        assert result.get("replan_count") == custom_max, (
            f"Should not increment beyond custom max_replans ({custom_max})"
        )


class TestFieldMappingCompleteness:
    """Verify all fields from agent_output are correctly mapped."""

    def test_plan_node_maps_all_plan_fields(self, base_state):
        """plan_node should map all plan fields from agent_output."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "paper_domain": "metamaterials",
            "title": "Test Title",
            "summary": "Test summary text",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [{"figure_id": "f1"}],
            "extracted_parameters": [{"name": "p1", "value": 10}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        # paper_id is preserved from state if valid, not from agent_output
        assert plan.get("paper_id") == base_state.get("paper_id"), (
            "paper_id should be preserved from state when valid"
        )
        assert plan.get("paper_domain") == "metamaterials"
        assert plan.get("title") == "Test Title"
        assert plan.get("summary") == "Test summary text"
        assert len(plan.get("stages", [])) == 1
        assert len(plan.get("targets", [])) == 1
        assert len(plan.get("extracted_parameters", [])) == 1

    def test_plan_node_maps_planned_materials(self, base_state):
        """plan_node should map planned_materials from agent_output."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
            "planned_materials": ["Au", "Ag", "Cu"],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        assert result.get("planned_materials") == ["Au", "Ag", "Cu"]

    def test_plan_node_maps_assumptions(self, base_state):
        """plan_node should map assumptions from agent_output."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
            "assumptions": {
                "geometry": "spherical",
                "medium": "water",
            },
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        assumptions = result.get("assumptions", {})
        assert assumptions.get("geometry") == "spherical"
        assert assumptions.get("medium") == "water"

    def test_plan_node_handles_missing_optional_fields(self, base_state):
        """plan_node should handle missing optional fields gracefully."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            # Missing: planned_materials, assumptions, summary
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        # Should use defaults for missing fields
        assert result.get("planned_materials", []) == []
        assert result.get("assumptions", {}) == {}
        plan = result.get("plan", {})
        assert plan.get("summary", "") == ""


class TestContextCheckIntegration:
    """Verify context check integration."""

    def test_plan_node_handles_context_escalation(self, base_state):
        """plan_node should handle context check escalation."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        escalation_state = {
            "awaiting_user_input": True,
            "ask_user_trigger": "context_budget_exceeded",
            "pending_user_questions": ["Context budget exceeded"],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.check_context_or_escalate",
                return_value=escalation_state
            ):
                result = plan_node(base_state)

        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "context_budget_exceeded"

    def test_plan_node_continues_when_context_check_returns_metrics(self, base_state):
        """plan_node should continue when context check returns only metrics."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        metrics_only = {
            "metrics": {"tokens_used": 1000},
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            with patch(
                "src.agents.planning.check_context_or_escalate",
                return_value=metrics_only
            ):
                result = plan_node(base_state)

        # Should proceed with LLM call
        assert mock_llm.called
        assert result.get("workflow_phase") == "planning"


class TestUserContentBuilding:
    """Verify user content is built correctly with all required information."""

    def test_plan_node_includes_replan_feedback_in_user_content(self, base_state):
        """plan_node should include replan feedback in user_content when present."""
        from src.agents.planning import plan_node

        base_state["planner_feedback"] = "Previous plan had issues with material selection"

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        assert "Previous plan had issues" in user_content or "REVISION FEEDBACK" in user_content

    def test_plan_node_builds_user_content_with_figures(self, base_state):
        """plan_node should include figure information in user_content."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        # Should include figure descriptions
        assert "Fig1" in user_content or "Extinction spectrum" in user_content


class TestExtractedParametersHandling:
    """Verify extracted_parameters are handled correctly."""

    def test_plan_node_syncs_extracted_parameters(self, base_state):
        """plan_node should sync extracted_parameters after progress initialization."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [
                {"name": "wavelength", "value": 650, "unit": "nm"},
                {"name": "length", "value": 100, "unit": "nm"},
            ],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        # extracted_parameters should be synced
        assert "extracted_parameters" in result
        assert len(result.get("extracted_parameters", [])) == 2
        assert result["extracted_parameters"][0]["name"] == "wavelength"

    def test_plan_node_handles_empty_extracted_parameters(self, base_state):
        """plan_node should handle empty extracted_parameters list."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        assert result.get("extracted_parameters", []) == []

    def test_plan_node_handles_missing_extracted_parameters(self, base_state):
        """plan_node should handle missing extracted_parameters key."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            # Missing extracted_parameters
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        # Should default to empty list
        assert result.get("extracted_parameters", []) == []


class TestPlanDataStructure:
    """Verify plan data structure is correctly formed."""

    def test_plan_node_includes_all_plan_fields(self, base_state):
        """plan_node should include all required plan fields."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test Title",
            "summary": "Test summary",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [{"figure_id": "f1"}],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert "paper_id" in plan
        assert "paper_domain" in plan
        assert "title" in plan
        assert "summary" in plan
        assert "stages" in plan
        assert "targets" in plan
        assert "extracted_parameters" in plan

    def test_plan_node_handles_none_summary(self, base_state):
        """plan_node should convert None summary to empty string."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
            "summary": None,
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        # None should be converted to empty string
        assert plan.get("summary") == "", (
            f"summary should be empty string when None, got {plan.get('summary')}"
        )

    def test_plan_node_handles_none_title(self, base_state):
        """plan_node should convert None title to empty string."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
            "title": None,
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("title") == "", (
            f"title should be empty string when None, got {plan.get('title')}"
        )

    def test_plan_node_handles_none_paper_domain(self, base_state):
        """plan_node should convert None paper_domain to 'other'."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
            "paper_domain": None,
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("paper_domain") == "other", (
            f"paper_domain should be 'other' when None, got {plan.get('paper_domain')}"
        )
        # Also check top-level paper_domain
        assert result.get("paper_domain") == "other"

    def test_plan_node_handles_none_targets(self, base_state):
        """plan_node should convert None targets to empty list."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": None,
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("targets") == [], (
            f"targets should be empty list when None, got {plan.get('targets')}"
        )

    def test_plan_node_handles_none_extracted_parameters(self, base_state):
        """plan_node should convert None extracted_parameters to empty list."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": None,
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("extracted_parameters") == [], (
            f"extracted_parameters should be empty list when None, got {plan.get('extracted_parameters')}"
        )

    def test_plan_node_handles_none_stages(self, base_state):
        """plan_node should convert None stages to empty list."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": None,  # None stages
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("stages") == [], (
            f"stages should be empty list when None, got {plan.get('stages')}"
        )


class TestWorkflowPhase:
    """Verify workflow_phase is set correctly."""

    def test_plan_node_sets_workflow_phase_to_planning(self, base_state):
        """plan_node should always set workflow_phase to 'planning'."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"

    def test_plan_node_sets_workflow_phase_even_on_error(self, base_state):
        """plan_node should set workflow_phase even when paper_text is invalid."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = ""  # Invalid

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"

