"""Tests that enforce plan reviewer behavior and dependency validation."""

from unittest.mock import patch


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
        assert len(system_prompt) > 100, f"System prompt too short ({len(system_prompt)} chars)"
        assert call_kwargs.get("agent_name") == "plan_reviewer", (
            "Expected agent_name='plan_reviewer', "
            f"got '{call_kwargs.get('agent_name')}'"
        )

    def test_plan_reviewer_handles_llm_error(self, base_state):
        """plan_reviewer should auto-approve on LLM failure."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "stages": [{"stage_id": "s1", "targets": ["f1"]}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", side_effect=Exception("LLM Fail")
        ):
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve"


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
        base_state["plan"] = {"stages": []}  # Force rejection

        result = plan_reviewer_node(base_state)
        assert result["replan_count"] == 1

    def test_plan_reviewer_max_replans(self, base_state):
        """plan_reviewer should not increment replan_count beyond max."""
        from src.agents.planning import plan_reviewer_node

        base_state["replan_count"] = 3
        base_state["runtime_config"] = {"max_replans": 3}
        base_state["plan"] = {"stages": []}  # Force rejection

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
                    "dependencies": ["stage_b"],
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
                },
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

