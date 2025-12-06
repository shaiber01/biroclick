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
    """Verify plan_node doesn't mutate input state.
    
    LangGraph pattern: nodes return dicts that are merged into state.
    Nodes must NOT mutate the input state directly.
    """

    def test_plan_node_doesnt_mutate_input(self, base_state):
        """plan_node must not mutate input state dict."""
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

    def test_plan_node_doesnt_mutate_nested_objects(self, base_state):
        """plan_node must not mutate nested objects like paper_figures."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        original_figures = deepcopy(base_state.get("paper_figures", []))

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            plan_node(base_state)

        # Nested objects should be unchanged
        assert base_state.get("paper_figures") == original_figures, (
            "paper_figures should not be mutated"
        )

    def test_plan_node_doesnt_mutate_input_on_error(self, base_state):
        """plan_node must not mutate input state even when LLM call fails."""
        from src.agents.planning import plan_node

        original_state = deepcopy(base_state)

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("LLM failed")
        ):
            plan_node(base_state)

        # State should be unchanged even after error
        assert base_state == original_state, (
            "Input state should not be mutated on error"
        )

    def test_plan_node_doesnt_mutate_input_on_paper_text_error(self, base_state):
        """plan_node must not mutate state when paper_text validation fails."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = ""
        original_state = deepcopy(base_state)

        plan_node(base_state)

        # State should be unchanged
        assert base_state == original_state


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
        """plan_node should handle malformed extracted_parameters with missing value field.
        
        The sync_extracted_parameters function provides defaults for missing fields,
        so parameters with missing 'value' should still be synced with value=None.
        """
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [{"figure_id": "f1"}],
            # Missing 'value' field - sync_extracted_parameters handles this
            "extracted_parameters": [{"name": "p1"}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        # Plan node should succeed since sync_extracted_parameters provides defaults
        assert "workflow_phase" in result, "Result must have workflow_phase"
        assert result["workflow_phase"] == "planning"
        
        # Verify extracted_parameters was processed (sync provides defaults for missing fields)
        extracted = result.get("extracted_parameters", [])
        assert isinstance(extracted, list), "extracted_parameters must be a list"
        if extracted:  # If parameters were synced
            assert extracted[0]["name"] == "p1", "Parameter name should be preserved"
            # value defaults to None when missing
            assert "value" in extracted[0], "Parameter should have value field (even if None)"


class TestPaperTextValidation:
    """Verify plan_node validates paper_text correctly.
    
    The validation rule is: paper_text.strip() must be >= 100 characters.
    """

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

        assert result.get("ask_user_trigger") is not None, "Should await user input for empty paper_text"
        assert result.get("ask_user_trigger") == "missing_paper_text", (
            f"Expected trigger 'missing_paper_text', got '{result.get('ask_user_trigger')}'"
        )
        questions = result.get("pending_user_questions", [])
        assert len(questions) >= 1, "Should have at least one pending question"
        assert "paper text" in questions[0].lower(), "Question should mention paper text"
        assert "0 characters" in questions[0] or "0 char" in questions[0].lower(), (
            "Question should indicate 0 characters"
        )
        assert result["workflow_phase"] == "planning"

    def test_plan_node_rejects_too_short_paper_text(self, base_state):
        """plan_node should reject paper_text shorter than 100 characters."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = "Short text"  # 10 chars - clearly too short

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "missing_paper_text"
        questions = result.get("pending_user_questions", [])
        assert len(questions) >= 1, "Should have at least one pending question"
        # Check the message mentions the short length
        assert "10 character" in questions[0] or "too short" in questions[0].lower(), (
            f"Question should mention short length or 'too short': {questions[0]}"
        )

    def test_plan_node_rejects_exactly_99_characters(self, base_state):
        """plan_node should reject paper_text with exactly 99 characters (boundary test)."""
        from src.agents.planning import plan_node

        # Create exactly 99 characters of content
        base_state["paper_text"] = "x" * 99
        assert len(base_state["paper_text"]) == 99, "Test setup: must be exactly 99 chars"

        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        # 99 chars is < 100, so should be rejected
        assert result.get("ask_user_trigger") is not None, (
            "99 characters should be rejected (< 100 threshold)"
        )
        assert result.get("ask_user_trigger") == "missing_paper_text"

    def test_plan_node_accepts_exactly_100_characters(self, base_state):
        """plan_node should accept paper_text with exactly 100 characters (boundary test)."""
        from src.agents.planning import plan_node

        # Create exactly 100 characters of content
        base_state["paper_text"] = "x" * 100
        assert len(base_state["paper_text"]) == 100, "Test setup: must be exactly 100 chars"

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)

        # 100 chars is >= 100, so should be accepted
        assert result.get("ask_user_trigger") is None, (
            "100 characters should be accepted (>= 100 threshold)"
        )
        assert "plan" in result, "Should return plan when paper_text is valid"

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

        assert result.get("ask_user_trigger") is not None
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

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "missing_paper_text"


class TestLLMErrorHandling:
    """Verify plan_node handles LLM errors correctly.
    
    When LLM calls fail, plan_node should use create_llm_error_escalation
    which returns a specific structure with ask_user_trigger="llm_error".
    """

    def test_plan_node_handles_llm_exception(self, base_state):
        """plan_node should handle LLM call exceptions with proper escalation."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("LLM call failed")
        ):
            result = plan_node(base_state)

        # Verify all required escalation fields
        assert result.get("workflow_phase") == "planning", (
            "Should preserve workflow_phase as 'planning'"
        )
        assert result.get("ask_user_trigger") is not None, (
            "LLM errors should escalate to user"
        )
        assert result.get("ask_user_trigger") == "llm_error", (
            f"Expected trigger 'llm_error', got '{result.get('ask_user_trigger')}'"
        )
        
        # Verify pending_user_questions contains error info
        questions = result.get("pending_user_questions", [])
        assert len(questions) >= 1, "Should have at least one pending question"
        assert "failed" in questions[0].lower() or "planner" in questions[0].lower(), (
            f"Question should mention failure: {questions[0]}"
        )
        assert "LLM call failed" in questions[0] or "api" in questions[0].lower(), (
            "Question should include error context"
        )

    def test_plan_node_handles_llm_timeout(self, base_state):
        """plan_node should handle LLM timeout exceptions with proper escalation."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=TimeoutError("Request timed out")
        ):
            result = plan_node(base_state)

        # Verify escalation structure matches create_llm_error_escalation output
        assert result.get("ask_user_trigger") is not None, "Should await user input"
        assert result.get("ask_user_trigger") == "llm_error", (
            "Timeout should also use 'llm_error' trigger"
        )
        assert result.get("workflow_phase") == "planning"
        
        questions = result.get("pending_user_questions", [])
        assert len(questions) >= 1
        assert "timed out" in questions[0].lower() or "timeout" in questions[0].lower(), (
            f"Question should mention timeout: {questions[0]}"
        )

    def test_plan_node_handles_generic_exception(self, base_state):
        """plan_node should handle generic Exception type."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=Exception("Generic error")
        ):
            result = plan_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "llm_error"
        assert result.get("workflow_phase") == "planning"

    def test_plan_node_handles_connection_error(self, base_state):
        """plan_node should handle network-related errors."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=ConnectionError("Network unavailable")
        ):
            result = plan_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "llm_error"
        questions = result.get("pending_user_questions", [])
        assert len(questions) >= 1
        assert "network" in questions[0].lower() or "unavailable" in questions[0].lower(), (
            f"Question should include connection error info: {questions[0]}"
        )


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
    """Verify progress initialization handles edge cases.
    
    Progress initialization happens after plan is created.
    It calls initialize_progress_from_plan followed by sync_extracted_parameters.
    """

    def test_plan_node_handles_empty_stages_list(self, base_state):
        """plan_node should handle empty stages list without crashing."""
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

        # Verify plan structure is correct
        plan = result.get("plan", {})
        assert plan.get("stages") == [], "Empty stages should be preserved"
        
        # Progress should not be initialized if no stages (line 205: if stages)
        progress = result.get("progress")
        assert progress is None, (
            "Progress should be None when there are no stages to initialize"
        )
        
        # Should still have workflow_phase
        assert result.get("workflow_phase") == "planning"

    def test_plan_node_handles_missing_stages_key(self, base_state):
        """plan_node should handle missing stages key in response."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "title": "Test",
            # Missing "stages" key - defaults to []
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("stages") == [], "Missing stages should default to empty list"
        
        # Progress should not be initialized if no stages
        progress = result.get("progress")
        assert progress is None, "Progress should be None when no stages"

    def test_plan_node_handles_sync_extracted_parameters_failure(self, base_state):
        """plan_node should mark plan for revision when sync fails."""
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
        assert result.get("last_plan_review_verdict") == "needs_revision", (
            "Sync failure should trigger needs_revision"
        )
        assert "failed" in result.get("planner_feedback", "").lower(), (
            "Feedback should mention failure"
        )
        assert "sync" in result.get("planner_feedback", "").lower() or "progress" in result.get("planner_feedback", "").lower(), (
            "Feedback should indicate what failed"
        )

    def test_plan_node_resets_existing_progress_on_replan(self, base_state):
        """plan_node should reset existing progress when replanning."""
        from src.agents.planning import plan_node

        # Set up state with existing progress from previous plan
        base_state["progress"] = {
            "stages": [{"stage_id": "old_stage", "status": "completed"}],
        }
        base_state["replan_count"] = 1

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "new_stage", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        progress = result.get("progress")
        assert progress is not None, "Should have new progress"
        assert len(progress.get("stages", [])) == 1
        assert progress["stages"][0]["stage_id"] == "new_stage", (
            "Progress should have new stage, not old one"
        )
        assert progress["stages"][0]["status"] == "not_started", (
            "New stage should be not_started"
        )

    def test_plan_node_calls_init_and_sync_in_order(self, base_state):
        """plan_node should call initialize_progress_from_plan then sync_extracted_parameters."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [{"name": "p1", "value": 10}],
        }

        call_order = []
        
        def track_init(state):
            call_order.append("init")
            return state
        
        def track_sync(state):
            call_order.append("sync")
            return state

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            with patch("src.agents.planning.initialize_progress_from_plan", side_effect=track_init):
                with patch("src.agents.planning.sync_extracted_parameters", side_effect=track_sync):
                    plan_node(base_state)

        assert call_order == ["init", "sync"], (
            f"Should call init then sync in order, got {call_order}"
        )


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
    """Verify context check integration.
    
    The plan_node calls check_context_or_escalate before making LLM calls.
    If context is exceeded, escalation is returned immediately.
    If only metrics are returned, processing continues.
    """

    def test_plan_node_handles_context_escalation(self, base_state):
        """plan_node should return context escalation immediately without LLM call."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        escalation_state = {
            "ask_user_trigger": "context_overflow",
            "ask_user_trigger": "context_budget_exceeded",
            "pending_user_questions": ["Context budget exceeded - tokens: 50000"],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            with patch(
                "src.agents.planning.check_context_or_escalate",
                return_value=escalation_state
            ):
                result = plan_node(base_state)

        # Should return escalation without calling LLM
        assert mock_llm.call_count == 0, (
            "LLM should not be called when context check returns escalation"
        )
        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "context_budget_exceeded"
        assert result.get("pending_user_questions") == ["Context budget exceeded - tokens: 50000"]
        
        # Should NOT have plan data since LLM wasn't called
        assert "plan" not in result, "Should not have plan when escalation returned"

    def test_plan_node_continues_when_context_check_returns_metrics(self, base_state):
        """plan_node should continue processing when context check returns only metrics."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        metrics_only = {
            "metrics": {"tokens_used": 1000, "estimated_cost": 0.05},
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
        assert mock_llm.called, "LLM should be called when no escalation"
        assert mock_llm.call_count == 1
        assert result.get("workflow_phase") == "planning"
        assert "plan" in result, "Should have plan when context check passes"

    def test_plan_node_handles_context_check_returning_none(self, base_state):
        """plan_node should continue when context check returns None."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            with patch(
                "src.agents.planning.check_context_or_escalate",
                return_value=None
            ):
                result = plan_node(base_state)

        # None means no escalation needed
        assert mock_llm.called, "LLM should be called when context check returns None"
        assert result.get("workflow_phase") == "planning"
        assert "plan" in result


class TestUserContentBuilding:
    """Verify user content is built correctly with all required information.
    
    User content is built by build_user_content_for_planner() and should include:
    - Paper text
    - Figure information
    - Replan feedback (if present)
    """

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
        # Must include either the feedback text or the section header
        assert "Previous plan had issues" in user_content or "REVISION FEEDBACK" in user_content, (
            f"Feedback should be included in user_content. Got: {user_content[:500]}"
        )

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
        # Should include figure section or figure descriptions
        assert "FIGURES" in user_content or "Fig" in user_content, (
            f"User content should include figure information: {user_content[:500]}"
        )

    def test_plan_node_builds_user_content_with_paper_text(self, base_state):
        """plan_node should include paper text in user_content."""
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
        
        # Should include actual paper text content
        assert "gold nanorods" in user_content.lower(), (
            "User content should include paper text about gold nanorods"
        )
        assert "FDTD" in user_content or "simulation" in user_content.lower(), (
            "User content should include simulation-related paper text"
        )

    def test_plan_node_user_content_has_no_feedback_when_not_replanning(self, base_state):
        """plan_node should not include revision feedback section when not replanning."""
        from src.agents.planning import plan_node

        # Ensure no planner_feedback is set
        base_state.pop("planner_feedback", None)

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
        
        # Should not have revision feedback section when not replanning
        assert "REVISION FEEDBACK" not in user_content, (
            "Should not include revision feedback section when not replanning"
        )


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


class TestMetricsLogging:
    """Verify plan_node logs metrics correctly."""

    def test_plan_node_calls_log_agent_call(self, base_state):
        """plan_node should call log_agent_call after successful LLM call."""
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
            with patch(
                "src.agents.planning.log_agent_call", return_value=lambda s, r: None
            ) as mock_log:
                plan_node(base_state)

        # log_agent_call should be called with agent name and operation
        assert mock_log.called, "log_agent_call should be called"
        call_args = mock_log.call_args
        # First arg is agent_name, second is operation
        assert "Planner" in call_args[0][0] or "planner" in call_args[0][0].lower(), (
            f"Should log for PlannerAgent, got {call_args[0]}"
        )


class TestOutputCompleteness:
    """Verify plan_node output has all expected fields on success."""

    def test_plan_node_success_output_has_all_required_keys(self, base_state):
        """plan_node successful output must have all required keys."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test Title",
            "summary": "Test summary",
            "stages": [{"stage_id": "s1", "targets": ["f1"], "stage_type": "FDTD_DIRECT"}],
            "targets": [{"figure_id": "f1"}],
            "extracted_parameters": [{"name": "p1", "value": 10}],
            "planned_materials": ["Au"],
            "assumptions": {"key": "value"},
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        # Required keys on success
        required_keys = [
            "workflow_phase",
            "plan",
            "planned_materials",
            "assumptions",
            "paper_domain",
            "progress",
            "extracted_parameters",
        ]
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Verify types
        assert isinstance(result["plan"], dict), "plan must be a dict"
        assert isinstance(result["planned_materials"], list), "planned_materials must be a list"
        assert isinstance(result["assumptions"], dict), "assumptions must be a dict"
        assert isinstance(result["progress"], dict), "progress must be a dict on success"
        assert isinstance(result["extracted_parameters"], list), "extracted_parameters must be a list"

    def test_plan_node_error_output_has_all_required_keys(self, base_state):
        """plan_node error output (paper_text validation) must have all required keys."""
        from src.agents.planning import plan_node

        base_state["paper_text"] = ""

        result = plan_node(base_state)

        # Required keys on error/escalation
        required_keys = [
            "workflow_phase",
            "awaiting_user_input",
            "ask_user_trigger",
            "pending_user_questions",
        ]
        
        for key in required_keys:
            assert key in result, f"Missing required key on error: {key}"

        # Verify types
        assert isinstance(result["pending_user_questions"], list), "pending_user_questions must be a list"
        assert len(result["pending_user_questions"]) >= 1, "Must have at least one question"


class TestStageStructureValidation:
    """Verify plan_node handles various stage structures correctly."""

    def test_plan_node_handles_stage_with_all_optional_fields(self, base_state):
        """plan_node should handle stages with all optional fields populated."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{
                "stage_id": "s1",
                "stage_type": "FDTD_DIRECT",
                "description": "Full description",
                "targets": ["f1", "f2"],
                "dependencies": [],
                "expected_outputs": ["spectrum.csv"],
                "runtime_budget_minutes": 30,
                "validation_criteria": {"rmse": 0.1},
            }],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert len(plan.get("stages", [])) == 1
        stage = plan["stages"][0]
        assert stage["stage_id"] == "s1"
        assert stage["stage_type"] == "FDTD_DIRECT"
        assert stage["description"] == "Full description"
        assert len(stage["targets"]) == 2

    def test_plan_node_handles_multiple_stages_with_dependencies(self, base_state):
        """plan_node should handle multiple stages with dependency chain."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["f1"], "dependencies": []},
                {"stage_id": "s2", "stage_type": "FDTD_DIRECT", "targets": ["f2"], "dependencies": ["s1"]},
                {"stage_id": "s3", "stage_type": "FDTD_DIRECT", "targets": ["f3"], "dependencies": ["s1", "s2"]},
            ],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        progress = result.get("progress", {})
        
        assert len(plan.get("stages", [])) == 3, "Should have 3 stages in plan"
        assert len(progress.get("stages", [])) == 3, "Should have 3 stages in progress"
        
        # Dependencies should be in plan, not progress
        plan_s3 = next(s for s in plan["stages"] if s["stage_id"] == "s3")
        assert plan_s3["dependencies"] == ["s1", "s2"], "Dependencies should be preserved in plan"

    def test_plan_node_handles_stage_with_minimal_fields(self, base_state):
        """plan_node should handle stages with only required fields."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],  # Minimal stage
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        assert len(plan.get("stages", [])) == 1
        
        progress = result.get("progress", {})
        assert len(progress.get("stages", [])) == 1
        
        # Progress stage should have default status
        assert progress["stages"][0]["status"] == "not_started"


class TestSystemPromptBuilding:
    """Verify system prompt is built correctly."""

    def test_plan_node_uses_build_agent_prompt(self, base_state):
        """plan_node should use build_agent_prompt for system prompt."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            with patch(
                "src.agents.planning.build_agent_prompt", return_value="Custom system prompt"
            ) as mock_build:
                plan_node(base_state)

        # build_agent_prompt should be called with "planner"
        assert mock_build.called, "build_agent_prompt should be called"
        assert mock_build.call_args[0][0] == "planner", (
            f"Should build prompt for 'planner', got {mock_build.call_args[0][0]}"
        )
        
        # System prompt should be passed to LLM
        call_kwargs = mock_llm.call_args.kwargs
        assert "Custom system prompt" in call_kwargs.get("system_prompt", ""), (
            "Built system prompt should be passed to LLM"
        )

    def test_plan_node_appends_replan_note_to_system_prompt(self, base_state):
        """plan_node should append replan note when replan_count > 0."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 2

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            with patch(
                "src.agents.planning.build_agent_prompt", return_value="Base prompt"
            ):
                plan_node(base_state)

        call_kwargs = mock_llm.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        
        # Should have base prompt + replan note
        assert "Base prompt" in system_prompt
        assert "Replan Attempt #2" in system_prompt
        assert "Previous plan was rejected" in system_prompt


class TestPromptAdaptationIntegration:
    """Verify prompt adaptations are used."""

    def test_plan_node_passes_state_with_prompt_adaptations(self, base_state):
        """plan_node should pass state to build_agent_prompt for adaptation."""
        from src.agents.planning import plan_node

        base_state["prompt_adaptations"] = [
            {"agent": "planner", "adaptation": "Focus on nanorods"}
        ]

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            with patch(
                "src.agents.planning.build_agent_prompt", return_value="Adapted prompt"
            ) as mock_build:
                plan_node(base_state)

        # State with adaptations should be passed to build_agent_prompt
        call_args = mock_build.call_args
        passed_state = call_args[0][1]  # Second positional arg
        assert "prompt_adaptations" in passed_state, (
            "State with prompt_adaptations should be passed to build_agent_prompt"
        )


class TestPlanNodeBehaviorWithAwaitingUserInput:
    """Verify plan_node behavior when awaiting_user_input is already True.
    
    Note: Unlike nodes with @with_context_check decorator, plan_node handles
    context checking manually and does NOT have an early return for
    awaiting_user_input=True.
    """

    def test_plan_node_processes_even_when_awaiting_user_input(self, base_state):
        """plan_node processes normally even if awaiting_user_input=True.
        
        This tests the CURRENT behavior - plan_node does not check for
        awaiting_user_input at entry. If this is a bug, the test will
        document the current behavior until it's fixed.
        """
        from src.agents.planning import plan_node

        base_state["awaiting_user_input"] = True
        base_state["ask_user_trigger"] = "some_previous_trigger"

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = plan_node(base_state)

        # Current behavior: plan_node processes despite awaiting_user_input
        # This documents the current behavior - if this changes, update the test
        assert mock_llm.called, (
            "CURRENT BEHAVIOR: plan_node calls LLM even when awaiting_user_input=True"
        )
        assert "plan" in result, "CURRENT BEHAVIOR: plan_node returns plan"


class TestProgressStageFieldsCompleteness:
    """Verify progress stage entries have all required fields."""

    def test_progress_stages_have_required_execution_fields(self, base_state):
        """Each progress stage must have required execution state fields."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["f1"]},
                {"stage_id": "s2", "stage_type": "FDTD_DIRECT", "targets": ["f2"], "dependencies": ["s1"]},
            ],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        progress = result.get("progress", {})
        assert "stages" in progress, "Progress must have stages key"
        
        required_fields = [
            "stage_id",
            "stage_type",
            "status",
        ]
        
        for stage in progress["stages"]:
            for field in required_fields:
                assert field in stage, f"Progress stage missing required field: {field}"
            
            # Verify status is a valid value
            assert stage["status"] in ["not_started", "in_progress", "completed", "failed"], (
                f"Invalid status value: {stage['status']}"
            )

    def test_progress_stages_reference_plan_by_stage_id(self, base_state):
        """Progress stages should reference plan stages by stage_id."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "unique_s1", "stage_type": "FDTD_DIRECT", "targets": ["f1"]},
            ],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        plan = result.get("plan", {})
        progress = result.get("progress", {})
        
        # Verify stage_ids match between plan and progress
        plan_ids = {s["stage_id"] for s in plan.get("stages", [])}
        progress_ids = {s["stage_id"] for s in progress.get("stages", [])}
        
        assert plan_ids == progress_ids, (
            f"Progress stage_ids {progress_ids} should match plan stage_ids {plan_ids}"
        )


class TestLLMCallParameters:
    """Verify all parameters passed to LLM call are correct."""

    def test_plan_node_passes_all_required_parameters_to_llm(self, base_state):
        """plan_node must pass all required parameters to call_agent_with_metrics."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            plan_node(base_state)

        # Verify all required kwargs are passed
        call_kwargs = mock_llm.call_args.kwargs
        
        required_params = ["agent_name", "system_prompt", "user_content", "state"]
        for param in required_params:
            assert param in call_kwargs, f"Missing required LLM parameter: {param}"
        
        # Verify parameter types
        assert isinstance(call_kwargs["agent_name"], str)
        assert isinstance(call_kwargs["system_prompt"], str)
        assert isinstance(call_kwargs["user_content"], str)
        assert isinstance(call_kwargs["state"], dict)

    def test_plan_node_passes_non_empty_user_content(self, base_state):
        """plan_node must pass non-empty user_content to LLM."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            plan_node(base_state)

        call_kwargs = mock_llm.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        
        assert len(user_content) > 100, (
            f"user_content should be substantial, got {len(user_content)} chars"
        )


class TestExtractedParametersSync:
    """Verify extracted_parameters sync behavior in detail."""

    def test_extracted_parameters_sync_includes_all_fields(self, base_state):
        """Synced parameters should include all standard fields."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [
                {
                    "name": "wavelength",
                    "value": 650,
                    "unit": "nm",
                    "source": "figure",
                    "location": "Figure 1",
                    "cross_checked": True,
                    "discrepancy_notes": "Minor difference",
                }
            ],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        params = result.get("extracted_parameters", [])
        assert len(params) == 1
        
        param = params[0]
        # Check all fields are preserved
        assert param["name"] == "wavelength"
        assert param["value"] == 650
        assert param["unit"] == "nm"
        assert param["source"] == "figure"
        assert param["location"] == "Figure 1"
        assert param["cross_checked"] is True
        assert param["discrepancy_notes"] == "Minor difference"

    def test_extracted_parameters_sync_provides_defaults(self, base_state):
        """Synced parameters should have defaults for missing optional fields."""
        from src.agents.planning import plan_node

        mock_response = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [],
            "extracted_parameters": [
                {"name": "length", "value": 100}  # Minimal parameter
            ],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_node(base_state)

        params = result.get("extracted_parameters", [])
        assert len(params) == 1
        
        param = params[0]
        # Required fields preserved
        assert param["name"] == "length"
        assert param["value"] == 100
        # Optional fields should have defaults
        assert "unit" in param  # Default is ""
        assert "source" in param  # Default is "inferred"
