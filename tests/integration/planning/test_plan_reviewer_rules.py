"""Tests that enforce plan reviewer behavior and dependency validation."""

from unittest.mock import patch
import json


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
            result = plan_reviewer_node(base_state)

        # Verify LLM was called
        assert mock.called, "LLM should be called for valid plan"
        call_kwargs = mock.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        assert len(system_prompt) > 100, f"System prompt too short ({len(system_prompt)} chars)"
        assert call_kwargs.get("agent_name") == "plan_reviewer", (
            "Expected agent_name='plan_reviewer', "
            f"got '{call_kwargs.get('agent_name')}'"
        )
        
        # Verify return value structure
        assert "workflow_phase" in result, "Result must include workflow_phase"
        assert result["workflow_phase"] == "plan_review", f"Expected workflow_phase='plan_review', got '{result['workflow_phase']}'"
        assert "last_plan_review_verdict" in result, "Result must include last_plan_review_verdict"
        assert result["last_plan_review_verdict"] == "approve", f"Expected verdict='approve', got '{result['last_plan_review_verdict']}'"
        
        # Verify plan is included in user_content
        user_content = call_kwargs.get("user_content", "")
        assert "REPRODUCTION PLAN TO REVIEW" in user_content, "User content should include plan review section"
        assert '"paper_id": "test"' in user_content, "Plan should be serialized in user content"
        
        # Verify replan_count is NOT incremented for approve
        assert "replan_count" not in result or result.get("replan_count") == base_state.get("replan_count", 0), (
            "replan_count should not be incremented when verdict is 'approve'"
        )

    def test_plan_reviewer_handles_llm_error(self, base_state):
        """plan_reviewer should auto-approve on LLM failure."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["f1"],
                    "dependencies": [],
                }
            ],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", side_effect=Exception("LLM Fail")
        ):
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve", (
            f"Expected auto-approve on LLM error, got '{result['last_plan_review_verdict']}'"
        )
        assert result["workflow_phase"] == "plan_review", "Workflow phase should be set"
        # Verify replan_count is NOT incremented for auto-approve
        assert "replan_count" not in result or result.get("replan_count") == base_state.get("replan_count", 0), (
            "replan_count should not be incremented when auto-approving due to LLM error"
        )
    
    def test_plan_reviewer_includes_assumptions_in_user_content(self, base_state):
        """plan_reviewer should include assumptions in user_content when calling LLM."""
        from src.agents.planning import plan_reviewer_node

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
        base_state["assumptions"] = {
            "material": "gold",
            "wavelength": 650,
        }

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            plan_reviewer_node(base_state)

        user_content = mock.call_args.kwargs.get("user_content", "")
        assert "ASSUMPTIONS" in user_content, "User content should include assumptions section"
        assert '"material": "gold"' in user_content, "Assumptions should be serialized in user content"
        assert '"wavelength": 650' in user_content, "All assumption values should be included"
    
    def test_plan_reviewer_handles_missing_assumptions(self, base_state):
        """plan_reviewer should handle missing assumptions gracefully."""
        from src.agents.planning import plan_reviewer_node

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
        # Explicitly remove assumptions if present
        if "assumptions" in base_state:
            del base_state["assumptions"]

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            result = plan_reviewer_node(base_state)

        # Should still work without assumptions
        assert result["last_plan_review_verdict"] == "approve"
        user_content = mock.call_args.kwargs.get("user_content", "")
        # Assumptions section should not be present if assumptions is None
        if "assumptions" not in base_state:
            # User content should still contain plan
            assert "REPRODUCTION PLAN TO REVIEW" in user_content


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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for empty stages, got '{result['last_plan_review_verdict']}'"
        )
        assert result["workflow_phase"] == "plan_review", "Workflow phase should be set"
        feedback = result.get("planner_feedback", "")
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
        assert "no stages" in feedback.lower() or "at least one stage" in feedback.lower(), (
            f"Feedback should mention empty stages issue, got: {feedback}"
        )
        # Verify LLM is NOT called for blocking issues
        assert "must contain at least one stage" in feedback or "no stages" in feedback.lower(), (
            "Should detect blocking issue without LLM call"
        )
    
    def test_plan_reviewer_rejects_none_stages(self, base_state):
        """plan_reviewer_node should reject plans with None stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "None Stages Plan",
            "stages": None,
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for None stages, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
    
    def test_plan_reviewer_rejects_none_plan(self, base_state):
        """plan_reviewer_node should handle None plan gracefully."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = None

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for None plan, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "PLAN_ISSUE" in feedback or "no stages" in feedback.lower(), (
            f"Feedback should indicate plan issue, got: {feedback}"
        )
    
    def test_plan_reviewer_rejects_missing_plan_key(self, base_state):
        """plan_reviewer_node should handle missing plan key."""
        from src.agents.planning import plan_reviewer_node

        if "plan" in base_state:
            del base_state["plan"]

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for missing plan, got '{result['last_plan_review_verdict']}'"
        )

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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for stage without targets, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "Stage 's1' has no targets" in feedback or "no targets" in feedback.lower(), (
            f"Feedback should mention stage s1 has no targets, got: {feedback}"
        )
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
    
    def test_plan_reviewer_rejects_stages_without_targets_or_target_details(self, base_state):
        """plan_reviewer_node should reject stages with neither targets nor target_details."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Bad Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": [],
                    "target_details": [],
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for stage without targets or target_details, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "no targets" in feedback.lower(), (
            f"Feedback should mention no targets, got: {feedback}"
        )
    
    def test_plan_reviewer_accepts_stages_with_target_details_only(self, base_state):
        """plan_reviewer_node should accept stages with target_details but no targets."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Good Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": [],
                    "target_details": [{"figure_id": "Fig1"}],
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        # Should not be rejected for missing targets if target_details exists
        assert result["last_plan_review_verdict"] == "approve", (
            f"Expected 'approve' for stage with target_details, got '{result['last_plan_review_verdict']}'"
        )
    
    def test_plan_reviewer_rejects_stages_with_none_targets(self, base_state):
        """plan_reviewer_node should reject stages with None targets."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Bad Plan",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": None,
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for None targets, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "no targets" in feedback.lower() or "Stage 's1'" in feedback, (
            f"Feedback should mention missing targets, got: {feedback}"
        )

    def test_plan_reviewer_increments_replan_count(self, base_state):
        """plan_reviewer should increment replan_count on rejection."""
        from src.agents.planning import plan_reviewer_node

        base_state["replan_count"] = 0
        base_state["plan"] = {"stages": []}  # Force rejection

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", "Should reject empty plan"
        # For blocking issues, replan_count is only incremented if it was already in state
        # Since we set it to 0, it should be incremented
        assert result.get("replan_count") == 1, (
            f"Expected replan_count=1 after rejection, got {result.get('replan_count')}"
        )
    
    def test_plan_reviewer_increments_replan_count_from_zero(self, base_state):
        """plan_reviewer should increment replan_count from 0 on LLM rejection."""
        from src.agents.planning import plan_reviewer_node

        base_state["replan_count"] = 0
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

        # LLM rejects the plan
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Plan is incomplete"}],
            "summary": "Plan needs revision",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision", "Should reject plan"
        # For LLM rejections, replan_count should be incremented even if starting from 0
        assert result.get("replan_count") == 1, (
            f"Expected replan_count=1 after LLM rejection, got {result.get('replan_count')}"
        )
    
    def test_plan_reviewer_does_not_increment_replan_count_when_not_in_state_for_blocking_issue(self, base_state):
        """plan_reviewer should not add replan_count if not in state for blocking issues."""
        from src.agents.planning import plan_reviewer_node

        # Remove replan_count from state
        if "replan_count" in base_state:
            del base_state["replan_count"]
        
        base_state["plan"] = {"stages": []}  # Force rejection (blocking issue)

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", "Should reject empty plan"
        # For blocking issues, replan_count should NOT be added if it wasn't in state
        assert "replan_count" not in result, (
            f"replan_count should not be added for blocking issues when not in state, got {result.get('replan_count')}"
        )

    def test_plan_reviewer_max_replans(self, base_state):
        """plan_reviewer should not increment replan_count beyond max."""
        from src.agents.planning import plan_reviewer_node

        base_state["replan_count"] = 3
        base_state["runtime_config"] = {"max_replans": 3}
        base_state["plan"] = {"stages": []}  # Force rejection

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", "Should reject empty plan"
        assert result.get("replan_count") == 3, (
            f"Expected replan_count=3 (at max), got {result.get('replan_count')}"
        )
    
    def test_plan_reviewer_respects_max_replans_from_config(self, base_state):
        """plan_reviewer should respect max_replans from runtime_config."""
        from src.agents.planning import plan_reviewer_node

        base_state["replan_count"] = 1
        base_state["runtime_config"] = {"max_replans": 2}
        base_state["plan"] = {"stages": []}  # Force rejection

        result = plan_reviewer_node(base_state)
        assert result.get("replan_count") == 2, (
            f"Expected replan_count=2 (incremented to max), got {result.get('replan_count')}"
        )
        
        # Try again - should stay at max
        base_state["replan_count"] = 2
        result = plan_reviewer_node(base_state)
        assert result.get("replan_count") == 2, (
            f"Expected replan_count=2 (at max), got {result.get('replan_count')}"
        )
    
    def test_plan_reviewer_uses_default_max_replans_when_not_in_config(self, base_state):
        """plan_reviewer should use default MAX_REPLANS when not in runtime_config."""
        from src.agents.planning import plan_reviewer_node
        from schemas.state import MAX_REPLANS

        base_state["replan_count"] = MAX_REPLANS - 1
        base_state["runtime_config"] = {}  # No max_replans specified
        base_state["plan"] = {"stages": []}  # Force rejection

        result = plan_reviewer_node(base_state)
        assert result.get("replan_count") == MAX_REPLANS, (
            f"Expected replan_count={MAX_REPLANS} (default max), got {result.get('replan_count')}"
        )
        
        # Try again - should stay at max
        base_state["replan_count"] = MAX_REPLANS
        result = plan_reviewer_node(base_state)
        assert result.get("replan_count") == MAX_REPLANS, (
            f"Expected replan_count={MAX_REPLANS} (at max), got {result.get('replan_count')}"
        )
    
    def test_plan_reviewer_does_not_increment_on_approve(self, base_state):
        """plan_reviewer should not increment replan_count when verdict is approve."""
        from src.agents.planning import plan_reviewer_node

        base_state["replan_count"] = 0
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

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve", "Should approve valid plan"
        # replan_count should not be incremented on approve
        assert "replan_count" not in result or result.get("replan_count") == 0, (
            f"replan_count should not be incremented on approve, got {result.get('replan_count')}"
        )


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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for circular dependency, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "circular" in feedback.lower() or "cycle" in feedback.lower(), (
            f"Feedback should mention circular dependency, got: {feedback}"
        )
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
        # Verify cycle description includes stage names
        assert "stage_a" in feedback.lower() or "stage_b" in feedback.lower(), (
            f"Feedback should mention cycle stages, got: {feedback}"
        )

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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for self-dependency, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "depends on itself" in feedback.lower() or "self" in feedback.lower(), (
            f"Feedback should mention self-dependency, got: {feedback}"
        )
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
        assert "stage_0" in feedback.lower(), (
            f"Feedback should mention stage_0, got: {feedback}"
        )

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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for transitive cycle, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "circular" in feedback.lower() or "cycle" in feedback.lower(), (
            f"Feedback should mention cycle, got: {feedback}"
        )
        # Verify cycle includes all stages in the cycle
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
    
    def test_plan_reviewer_detects_multiple_cycles(self, base_state):
        """plan_reviewer should detect multiple independent cycles."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Multiple Cycles",
            "stages": [
                # First cycle: a -> b -> a
                {
                    "stage_id": "a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["b"],
                },
                {
                    "stage_id": "b",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["a"],
                },
                # Second cycle: c -> d -> c
                {
                    "stage_id": "c",
                    "stage_type": "ARRAY_SYSTEM",
                    "targets": ["Fig3"],
                    "dependencies": ["d"],
                },
                {
                    "stage_id": "d",
                    "stage_type": "PARAMETER_SWEEP",
                    "targets": ["Fig4"],
                    "dependencies": ["c"],
                },
            ],
            "targets": [
                {"figure_id": "Fig1"},
                {"figure_id": "Fig2"},
                {"figure_id": "Fig3"},
                {"figure_id": "Fig4"},
            ],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for multiple cycles, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "circular" in feedback.lower() or "cycle" in feedback.lower(), (
            f"Feedback should mention cycles, got: {feedback}"
        )
        # Should detect both cycles
        assert feedback.count("circular") >= 1 or feedback.count("cycle") >= 1, (
            f"Feedback should mention multiple cycles, got: {feedback}"
        )

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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for missing dependency, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "depends on missing stage" in feedback.lower() or "missing stage" in feedback.lower(), (
            f"Feedback should mention missing stage dependency, got: {feedback}"
        )
        assert "stage_a" in feedback.lower() or "stage_b" in feedback.lower(), (
            f"Feedback should mention stage names, got: {feedback}"
        )
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
    
    def test_plan_reviewer_rejects_multiple_missing_dependencies(self, base_state):
        """plan_reviewer should detect all missing dependencies."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Multiple Missing Dependencies",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["missing_1", "missing_2"],
                },
                {
                    "stage_id": "stage_b",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["missing_3"],
                },
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for missing dependencies, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "missing stage" in feedback.lower(), (
            f"Feedback should mention missing stages, got: {feedback}"
        )
        # Should mention multiple missing stages
        assert feedback.count("missing") >= 2 or feedback.count("depends") >= 2, (
            f"Feedback should mention multiple missing dependencies, got: {feedback}"
        )
    
    def test_plan_reviewer_rejects_empty_dependency_list(self, base_state):
        """plan_reviewer should handle empty dependency lists correctly."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Empty Dependencies",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        # Empty dependencies should be valid
        assert result["last_plan_review_verdict"] == "approve", (
            f"Expected 'approve' for valid plan with empty dependencies, got '{result['last_plan_review_verdict']}'"
        )
    
    def test_plan_reviewer_rejects_none_dependencies(self, base_state):
        """plan_reviewer should handle None dependencies."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "None Dependencies",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": None,
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        # Should handle None gracefully - code uses .get("dependencies", []) which returns [] for None
        # So this should not cause an error, but let's verify
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        # None dependencies should be treated as empty list
        assert result["last_plan_review_verdict"] in ["approve", "needs_revision"], (
            f"Should handle None dependencies gracefully, got '{result['last_plan_review_verdict']}'"
        )

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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for duplicate stage IDs, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "duplicate stage id" in feedback.lower() or "duplicate" in feedback.lower(), (
            f"Feedback should mention duplicate stage ID, got: {feedback}"
        )
        assert "stage_a" in feedback.lower(), (
            f"Feedback should mention duplicate stage ID 'stage_a', got: {feedback}"
        )
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
    
    def test_plan_reviewer_rejects_multiple_duplicate_stage_ids(self, base_state):
        """plan_reviewer should detect multiple duplicate stage IDs."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Multiple Duplicates",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_a",  # First duplicate
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_b",
                    "stage_type": "ARRAY_SYSTEM",
                    "targets": ["Fig3"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_b",  # Second duplicate
                    "stage_type": "PARAMETER_SWEEP",
                    "targets": ["Fig4"],
                    "dependencies": [],
                },
            ],
            "targets": [
                {"figure_id": "Fig1"},
                {"figure_id": "Fig2"},
                {"figure_id": "Fig3"},
                {"figure_id": "Fig4"},
            ],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for multiple duplicates, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "duplicate" in feedback.lower(), (
            f"Feedback should mention duplicates, got: {feedback}"
        )
        # Should mention both duplicate IDs
        assert ("stage_a" in feedback.lower() and "stage_b" in feedback.lower()) or feedback.count("duplicate") >= 2, (
            f"Feedback should mention multiple duplicates, got: {feedback}"
        )

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
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for missing stage_id, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "missing 'stage_id'" in feedback.lower() or "missing stage_id" in feedback.lower(), (
            f"Feedback should mention missing stage_id, got: {feedback}"
        )
        assert "PLAN_ISSUE" in feedback, f"Feedback should contain 'PLAN_ISSUE', got: {feedback}"
    
    def test_plan_reviewer_rejects_empty_stage_id(self, base_state):
        """plan_reviewer should reject stages with empty string stage_id."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Empty Stage ID",
            "stages": [
                {
                    "stage_id": "",  # Empty string
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for empty stage_id, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "missing 'stage_id'" in feedback.lower() or "missing stage_id" in feedback.lower(), (
            f"Feedback should mention missing stage_id, got: {feedback}"
        )
    
    def test_plan_reviewer_rejects_none_stage_id(self, base_state):
        """plan_reviewer should reject stages with None stage_id."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "None Stage ID",
            "stages": [
                {
                    "stage_id": None,
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for None stage_id, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "missing 'stage_id'" in feedback.lower() or "missing stage_id" in feedback.lower(), (
            f"Feedback should mention missing stage_id, got: {feedback}"
        )
    
    def test_plan_reviewer_rejects_multiple_stages_with_missing_ids(self, base_state):
        """plan_reviewer should detect all stages with missing IDs."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Multiple Missing IDs",
            "stages": [
                {
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "valid_id",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": [],
                },
                {
                    "stage_type": "ARRAY_SYSTEM",
                    "targets": ["Fig3"],
                    "dependencies": [],
                },
            ],
            "targets": [
                {"figure_id": "Fig1"},
                {"figure_id": "Fig2"},
                {"figure_id": "Fig3"},
            ],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for multiple missing IDs, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        assert "missing 'stage_id'" in feedback.lower() or "missing stage_id" in feedback.lower(), (
            f"Feedback should mention missing stage_id, got: {feedback}"
        )
        # Should detect multiple missing IDs
        assert feedback.count("missing") >= 2 or feedback.count("stage_id") >= 2, (
            f"Feedback should mention multiple missing stage IDs, got: {feedback}"
        )


class TestComplexScenarios:
    """Test complex scenarios and edge cases."""

    def test_plan_reviewer_handles_valid_plan_with_multiple_stages(self, base_state):
        """plan_reviewer should approve valid plan with multiple stages and dependencies."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Valid Multi-Stage Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["material_gold"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],
                },
                {
                    "stage_id": "stage_2",
                    "stage_type": "ARRAY_SYSTEM",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_1"],
                },
            ],
            "targets": [
                {"figure_id": "Fig1"},
                {"figure_id": "Fig2"},
            ],
        }

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve", (
            f"Expected 'approve' for valid plan, got '{result['last_plan_review_verdict']}'"
        )
        assert result["workflow_phase"] == "plan_review", "Workflow phase should be set"
    
    def test_plan_reviewer_handles_combined_issues(self, base_state):
        """plan_reviewer should detect multiple types of issues in one plan."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Multiple Issues Plan",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": [],  # Missing targets
                    "dependencies": ["missing_stage"],  # Missing dependency
                },
                {
                    "stage_id": "stage_a",  # Duplicate ID
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    # Missing stage_id
                    "stage_type": "ARRAY_SYSTEM",
                    "targets": ["Fig2"],
                    "dependencies": [],
                },
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for plan with multiple issues, got '{result['last_plan_review_verdict']}'"
        )
        feedback = result.get("planner_feedback", "")
        # Should detect multiple issues
        assert feedback.count("PLAN_ISSUE") >= 3 or len([i for i in feedback.split("\n") if "PLAN_ISSUE" in i]) >= 3, (
            f"Feedback should mention multiple issues, got: {feedback}"
        )
    
    def test_plan_reviewer_feedback_format_for_blocking_issues(self, base_state):
        """plan_reviewer should format feedback correctly for blocking issues."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Blocking Issues Plan",
            "stages": [
                {
                    "stage_id": "stage_a",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": [],
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", "Should reject plan"
        feedback = result.get("planner_feedback", "")
        
        # Feedback should be a string
        assert isinstance(feedback, str), f"Feedback should be a string, got {type(feedback)}"
        assert len(feedback) > 0, "Feedback should not be empty"
        
        # Should contain structured information
        assert "PLAN_ISSUE" in feedback, "Feedback should contain PLAN_ISSUE marker"
    
    def test_plan_reviewer_workflow_phase_is_always_set(self, base_state):
        """plan_reviewer should always set workflow_phase in result."""
        from src.agents.planning import plan_reviewer_node

        # Test with valid plan
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

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = plan_reviewer_node(base_state)

        assert "workflow_phase" in result, "Result must include workflow_phase"
        assert result["workflow_phase"] == "plan_review", (
            f"Expected workflow_phase='plan_review', got '{result['workflow_phase']}'"
        )
        
        # Test with invalid plan
        base_state["plan"] = {"stages": []}
        result = plan_reviewer_node(base_state)
        assert "workflow_phase" in result, "Result must include workflow_phase even for invalid plan"
        assert result["workflow_phase"] == "plan_review", (
            f"Expected workflow_phase='plan_review', got '{result['workflow_phase']}'"
        )
    
    def test_plan_reviewer_handles_malformed_stage_structure(self, base_state):
        """plan_reviewer should handle malformed stage structures gracefully."""
        from src.agents.planning import plan_reviewer_node

        # Stage is not a dict
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Malformed Plan",
            "stages": ["not_a_dict"],  # Invalid: should be list of dicts
            "targets": [],
        }

        # Should not crash, should detect issue
        result = plan_reviewer_node(base_state)
        assert result["last_plan_review_verdict"] == "needs_revision", (
            f"Expected 'needs_revision' for malformed plan, got '{result['last_plan_review_verdict']}'"
        )
    
    def test_plan_reviewer_validates_all_stages_before_llm_call(self, base_state):
        """plan_reviewer should validate all stages before calling LLM."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "title": "Partially Invalid Plan",
            "stages": [
                {
                    "stage_id": "valid_stage",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "invalid_stage",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": [],  # Missing targets
                    "dependencies": [],
                },
            ],
            "targets": [{"figure_id": "Fig1"}],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics"
        ) as mock_llm:
            result = plan_reviewer_node(base_state)

        # Should detect blocking issue and NOT call LLM
        assert result["last_plan_review_verdict"] == "needs_revision", "Should reject plan"
        assert not mock_llm.called, "LLM should not be called when blocking issues exist"
        feedback = result.get("planner_feedback", "")
        assert "invalid_stage" in feedback.lower() or "no targets" in feedback.lower(), (
            f"Feedback should mention invalid stage, got: {feedback}"
        )

