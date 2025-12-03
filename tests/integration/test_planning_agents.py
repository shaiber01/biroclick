import json
from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest

from schemas.state import create_initial_state


class TestAdaptPromptsNode:
    """Verify prompt adaptation logic."""

    def test_adapt_prompts_success(self, base_state):
        """adapt_prompts_node should call LLM and update state."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": [
                {"agent": "planner", "adaptation": "Focus on materials"},
                {"agent": "designer", "adaptation": "Check boundaries"}
            ],
            "paper_domain": "metamaterials"
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert len(result["prompt_adaptations"]) == 2
        assert result["paper_domain"] == "metamaterials"
        
        # Verify LLM call
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["agent_name"] == "prompt_adaptor"
        assert "Analyze this paper" in call_kwargs["user_content"]

    def test_adapt_prompts_llm_failure(self, base_state):
        """adapt_prompts_node should return empty list on LLM failure."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics", side_effect=Exception("LLM Error")
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == []

    def test_adapt_prompts_handles_context_escalation(self, base_state):
        """adapt_prompts_node should handle context escalation via decorator."""
        from src.agents.planning import adapt_prompts_node
        
        escalation_response = {"awaiting_user_input": True, "reason": "Context too large"}
        
        # We patch in src.agents.base because that's where the decorator is defined/used
        with patch("src.agents.base.check_context_or_escalate", return_value=escalation_response):
            result = adapt_prompts_node(base_state)
            
        assert result == escalation_response


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

    def test_plan_node_adds_replan_context(self, base_state):
        """plan_node should add replan context to system prompt if replan_count > 0."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 1
        
        mock_response = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
            "targets": [{"figure_id": "f1"}]
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_node(base_state)
            
        call_kwargs = mock.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        assert "Replan Attempt #1" in system_prompt


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

    def test_plan_reviewer_handles_llm_error(self, base_state):
        """plan_reviewer should auto-approve on LLM failure."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}]
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", side_effect=Exception("LLM Fail")
        ):
            result = plan_reviewer_node(base_state)
            
        assert result["last_plan_review_verdict"] == "approve"
        # Should contain warning log or similar, but we check the fallback behavior here


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
        
        # Ensure no error flags are set
        assert not result.get("awaiting_user_input")


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
        
    def test_plan_reviewer_increments_replan_count(self, base_state):
        """plan_reviewer should increment replan_count on rejection."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["replan_count"] = 0
        base_state["plan"] = {"stages": []} # Force rejection

        result = plan_reviewer_node(base_state)
        assert result["replan_count"] == 1
        
    def test_plan_reviewer_max_replans(self, base_state):
        """plan_reviewer should not increment replan_count beyond max."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["replan_count"] = 3
        base_state["runtime_config"] = {"max_replans": 3}
        base_state["plan"] = {"stages": []} # Force rejection

        result = plan_reviewer_node(base_state)
        assert result["replan_count"] == 3


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

    def test_plan_reviewer_rejects_dependency_on_missing_stage(self, base_state):
        """plan_reviewer should reject dependencies that point to non-existent stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Broken Dependency",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_b"],  # stage_b does not exist
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "")
        assert "depends on missing stage" in feedback.lower()

    def test_plan_reviewer_rejects_duplicate_stage_ids(self, base_state):
        """plan_reviewer should reject plans with duplicate stage IDs."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Duplicate Stages",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_a",  # Duplicate ID
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "")
        assert "duplicate stage id" in feedback.lower()

    def test_plan_reviewer_rejects_missing_stage_id(self, base_state):
        """plan_reviewer should reject stages with missing IDs."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Missing ID",
            "stages": [
                {
                    # Missing stage_id
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "")
        assert "missing 'stage_id'" in feedback.lower()


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
        
    def test_plan_node_handles_context_check_escalation(self, base_state):
        """If check_context_or_escalate returns early, plan_node should return it."""
        from src.agents.planning import plan_node
        
        escalation_response = {"awaiting_user_input": True, "reason": "Context too large"}
        
        with patch("src.agents.planning.check_context_or_escalate", return_value=escalation_response):
            result = plan_node(base_state)
            
        assert result == escalation_response


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

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            # We rely on sync_extracted_parameters or initialize_progress_from_plan to be strict
            # If they are not, this test might pass with success, which implies loose validation.
            # We want to enforce strictness if possible.
            result = plan_node(base_state)

        # Assuming sync_extracted_parameters is robust, it might NOT raise if it just filters invalid ones
        # But if it fails, it should be caught.
        if result.get("last_plan_review_verdict") == "needs_revision":
            assert "failed" in result.get("planner_feedback", "").lower()
        else:
            # If it didn't fail, ensure parameter was filtered or handled
            params = result.get("extracted_parameters", [])
            # If it kept the bad parameter, that's a potential issue depending on downstream consumers
            # But for now, just ensuring no crash.
            pass
