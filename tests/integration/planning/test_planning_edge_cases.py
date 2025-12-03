"""Edge-case handling for planner entry points."""

from unittest.mock import patch, MagicMock
import pytest

from schemas.state import create_initial_state, MAX_REPLANS


class TestPlanningEdgeCases:
    """Planning-specific edge cases that should escalate properly."""

    def test_plan_node_with_very_short_paper(self):
        """Test that very short paper text triggers escalation."""
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="Short paper.",
        )

        result = plan_node(state)
        # Must have specific trigger and awaiting flag
        assert result.get("ask_user_trigger") == "missing_paper_text", (
            f"Expected ask_user_trigger='missing_paper_text', got {result.get('ask_user_trigger')}"
        )
        assert result.get("awaiting_user_input") is True, (
            f"Expected awaiting_user_input=True, got {result.get('awaiting_user_input')}"
        )
        assert result.get("workflow_phase") == "planning", (
            f"Expected workflow_phase='planning', got {result.get('workflow_phase')}"
        )
        assert isinstance(result.get("pending_user_questions"), list), (
            "Expected pending_user_questions to be a list"
        )
        assert len(result.get("pending_user_questions", [])) > 0, (
            "Expected at least one pending user question"
        )

    def test_plan_node_handles_missing_paper_text(self):
        """Test that empty paper text triggers escalation."""
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="",
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text", (
            f"Expected ask_user_trigger='missing_paper_text', got {result.get('ask_user_trigger')}"
        )
        assert result.get("awaiting_user_input") is True, (
            f"Expected awaiting_user_input=True, got {result.get('awaiting_user_input')}"
        )
        assert result.get("workflow_phase") == "planning"
        assert "paper text" in result.get("pending_user_questions", [""])[0].lower(), (
            "Expected error message to mention paper text"
        )

    def test_plan_node_handles_none_paper_text(self):
        """Test that None paper text triggers escalation."""
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text=None,
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert result.get("workflow_phase") == "planning"

    def test_plan_node_handles_whitespace_only_paper_text(self):
        """Test that whitespace-only paper text triggers escalation."""
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="   \n\t   ",
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True

    def test_plan_node_handles_paper_text_exactly_99_chars(self):
        """Test boundary condition: paper text exactly 99 chars (below threshold)."""
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="x" * 99,  # Exactly 99 chars, below 100 char threshold
        )

        result = plan_node(state)
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True

    def test_plan_node_handles_paper_text_exactly_100_chars(self):
        """Test boundary condition: paper text exactly 100 chars (at threshold)."""
        from src.agents.planning import plan_node

        state = create_initial_state(
            paper_id="test",
            paper_text="x" * 100,  # Exactly 100 chars, at threshold
        )

        # Should proceed (not escalate) if text is >= 100 chars
        with patch("src.agents.planning.check_context_or_escalate", return_value=None):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(state)
                # Should not have escalation trigger
                assert result.get("ask_user_trigger") != "missing_paper_text"
                assert result.get("workflow_phase") == "planning"

    def test_plan_node_handles_context_check_escalation(self, base_state):
        """If check_context_or_escalate returns escalation, plan_node should return it."""
        from src.agents.planning import plan_node

        escalation_response = {
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context too large"],
        }

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=escalation_response
        ):
            result = plan_node(base_state)

        # Should return escalation response exactly
        assert result == escalation_response, (
            f"Expected escalation response, got {result}"
        )
        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "context_overflow"

    def test_plan_node_handles_context_check_returns_metrics_only(self, base_state):
        """If check_context_or_escalate returns only metrics (no escalation), should continue."""
        from src.agents.planning import plan_node

        metrics_only = {"metrics": {"tokens": 1000}}  # No awaiting_user_input

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=metrics_only
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        # Should proceed with planning, not escalate
        assert result.get("awaiting_user_input") is not True
        assert result.get("workflow_phase") == "planning"

    def test_plan_node_handles_context_check_returns_none(self, base_state):
        """If check_context_or_escalate returns None, should proceed normally."""
        from src.agents.planning import plan_node

        with patch("src.agents.planning.check_context_or_escalate", return_value=None):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"
        assert "plan" in result


class TestPlannerErrorHandling:
    """LLM failure scenarios for planner and reviewer nodes."""

    def test_llm_error_triggers_user_escalation(self, base_state):
        """Test that LLM errors trigger proper escalation."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_node(base_state)

        assert result.get("ask_user_trigger") == "llm_error", (
            f"Expected ask_user_trigger='llm_error', got {result.get('ask_user_trigger')}"
        )
        assert result.get("awaiting_user_input") is True, (
            f"Expected awaiting_user_input=True, got {result.get('awaiting_user_input')}"
        )
        assert result.get("workflow_phase") == "planning", (
            f"Expected workflow_phase='planning', got {result.get('workflow_phase')}"
        )
        assert isinstance(result.get("pending_user_questions"), list), (
            "Expected pending_user_questions to be a list"
        )
        assert len(result.get("pending_user_questions", [])) > 0, (
            "Expected at least one pending user question"
        )
        assert "API Error" in result.get("pending_user_questions", [""])[0], (
            "Expected error message to include error details"
        )

    def test_llm_error_with_different_exception_types(self, base_state):
        """Test that different exception types are handled correctly."""
        from src.agents.planning import plan_node

        exception_types = [
            ValueError("Invalid input"),
            KeyError("Missing key"),
            ConnectionError("Network error"),
            TimeoutError("Request timeout"),
            Exception("Generic error"),
        ]

        for exc in exception_types:
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                side_effect=exc,
            ):
                result = plan_node(base_state)
                assert result.get("ask_user_trigger") == "llm_error"
                assert result.get("awaiting_user_input") is True
                assert result.get("workflow_phase") == "planning"

    def test_reviewer_llm_error_defaults_to_needs_revision(self, base_state, valid_plan):
        """Test that reviewer LLM errors result in needs_revision (fail-closed safety)."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = valid_plan
        initial_replan_count = base_state.get("replan_count", 0)

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_reviewer_node(base_state)

        # Fail-closed: LLM error should trigger needs_revision (safer than auto-approve)
        assert result.get("last_plan_review_verdict") == "needs_revision", (
            f"Expected verdict='needs_revision', got {result.get('last_plan_review_verdict')}"
        )
        assert result.get("workflow_phase") == "plan_review", (
            f"Expected workflow_phase='plan_review', got {result.get('workflow_phase')}"
        )
        # Counter should be incremented for needs_revision
        assert result.get("replan_count") == initial_replan_count + 1

    def test_reviewer_llm_error_increments_replan_count(self, base_state, valid_plan):
        """Test that reviewer needs_revision on LLM error increments replan_count."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = valid_plan
        base_state["replan_count"] = 2

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_reviewer_node(base_state)

        # Fail-closed: LLM error should trigger needs_revision
        assert result.get("last_plan_review_verdict") == "needs_revision"
        # Counter should be incremented for needs_revision
        assert result.get("replan_count") == 3


class TestPlanNodeMalformedData:
    """Test plan_node handling of malformed LLM responses."""

    def test_plan_node_handles_missing_stages_in_response(self, base_state):
        """Test handling when LLM response lacks stages."""
        from src.agents.planning import plan_node

        # Component preserves paper_id from state when it's valid (not "unknown")
        # So we should check that it uses the state's paper_id, not the LLM's
        expected_paper_id = base_state.get("paper_id")

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",  # LLM provides different ID, but should be ignored
                    "title": "Test",
                    # Missing "stages" key
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"
        plan = result.get("plan", {})
        assert plan.get("stages") == [], "Expected empty stages list when missing"
        # Component should preserve valid paper_id from state
        assert plan.get("paper_id") == expected_paper_id, (
            f"Expected paper_id from state ({expected_paper_id}), got {plan.get('paper_id')}"
        )

    def test_plan_node_handles_none_stages_in_response(self, base_state):
        """Test handling when LLM response has None stages."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": None,  # None instead of list
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"
        plan = result.get("plan", {})
        assert plan.get("stages") == [], "Expected empty stages list when None"

    def test_plan_node_handles_missing_targets_in_response(self, base_state):
        """Test handling when LLM response lacks targets."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    # Missing "targets" key
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"
        plan = result.get("plan", {})
        assert plan.get("targets") == [], "Expected empty targets list when missing"

    def test_plan_node_handles_missing_extracted_parameters(self, base_state):
        """Test handling when LLM response lacks extracted_parameters."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    # Missing "extracted_parameters" key
                },
            ):
                result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"
        plan = result.get("plan", {})
        assert plan.get("extracted_parameters") == [], "Expected empty list when missing"

    def test_plan_node_handles_progress_init_failure(self, base_state):
        """Test handling when progress initialization fails."""
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                with patch(
                    "src.agents.planning.initialize_progress_from_plan",
                    side_effect=ValueError("Invalid plan structure"),
                ):
                    result = plan_node(base_state)

        assert result.get("workflow_phase") == "planning"
        assert result.get("last_plan_review_verdict") == "needs_revision", (
            "Expected needs_revision when progress init fails"
        )
        assert result.get("planner_feedback") is not None, (
            "Expected feedback message when progress init fails"
        )
        assert "Progress initialization failed" in result.get("planner_feedback", "")

    def test_plan_node_increments_replan_count_on_progress_failure(self, base_state):
        """Test that replan_count increments when progress init fails."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = 1
        base_state["runtime_config"] = {"max_replans": MAX_REPLANS}

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                with patch(
                    "src.agents.planning.initialize_progress_from_plan",
                    side_effect=ValueError("Invalid plan structure"),
                ):
                    result = plan_node(base_state)

        assert result.get("replan_count") == 2, (
            f"Expected replan_count=2, got {result.get('replan_count')}"
        )

    def test_plan_node_respects_max_replans_on_progress_failure(self, base_state):
        """Test that replan_count doesn't exceed max_replans on progress failure."""
        from src.agents.planning import plan_node

        base_state["replan_count"] = MAX_REPLANS
        base_state["runtime_config"] = {"max_replans": MAX_REPLANS}

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                with patch(
                    "src.agents.planning.initialize_progress_from_plan",
                    side_effect=ValueError("Invalid plan structure"),
                ):
                    result = plan_node(base_state)

        assert result.get("replan_count") == MAX_REPLANS, (
            f"Expected replan_count={MAX_REPLANS}, got {result.get('replan_count')}"
        )

    def test_plan_node_preserves_paper_id_from_state(self, base_state):
        """Test that valid paper_id from state is preserved."""
        from src.agents.planning import plan_node

        base_state["paper_id"] = "preserved_id"

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "llm_provided_id",  # LLM provides different ID
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("paper_id") == "preserved_id", (
            f"Expected paper_id='preserved_id', got {plan.get('paper_id')}"
        )

    def test_plan_node_uses_llm_paper_id_when_state_invalid(self, base_state):
        """Test that LLM paper_id is used when state paper_id is 'unknown'."""
        from src.agents.planning import plan_node

        base_state["paper_id"] = "unknown"

        with patch(
            "src.agents.planning.check_context_or_escalate", return_value=None
        ):
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                return_value={
                    "paper_id": "llm_provided_id",
                    "title": "Test",
                    "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": []}],
                    "targets": [],
                    "extracted_parameters": [],
                },
            ):
                result = plan_node(base_state)

        plan = result.get("plan", {})
        assert plan.get("paper_id") == "llm_provided_id", (
            f"Expected paper_id='llm_provided_id', got {plan.get('paper_id')}"
        )


class TestPlanReviewerEdgeCases:
    """Edge cases for plan_reviewer_node."""

    def test_reviewer_rejects_empty_stages(self, base_state):
        """Test that reviewer rejects plan with empty stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [],  # Empty stages
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision", (
            f"Expected verdict='needs_revision', got {result.get('last_plan_review_verdict')}"
        )
        assert result.get("workflow_phase") == "plan_review"
        assert result.get("planner_feedback") is not None
        assert "no stages" in result.get("planner_feedback", "").lower() or "stage" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_none_stages(self, base_state):
        """Test that reviewer rejects plan with None stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": None,
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "stage" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_none_plan(self, base_state):
        """Test that reviewer handles None plan."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = None

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert result.get("workflow_phase") == "plan_review"

    def test_reviewer_rejects_missing_plan_key(self, base_state):
        """Test that reviewer handles missing plan key."""
        from src.agents.planning import plan_reviewer_node

        if "plan" in base_state:
            del base_state["plan"]

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert result.get("workflow_phase") == "plan_review"

    def test_reviewer_rejects_stages_without_targets(self, base_state):
        """Test that reviewer rejects stages without targets."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    # Missing targets
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "target" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_stages_without_stage_id(self, base_state):
        """Test that reviewer rejects stages without stage_id."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    # Missing stage_id
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["t1"],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "stage_id" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_duplicate_stage_ids(self, base_state):
        """Test that reviewer rejects plans with duplicate stage IDs."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["t1"],
                },
                {
                    "stage_id": "s1",  # Duplicate
                    "stage_type": "SIMULATION",
                    "targets": ["t2"],
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "duplicate" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_circular_dependencies(self, base_state):
        """Test that reviewer rejects plans with circular dependencies."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["t1"],
                    "dependencies": ["s2"],
                },
                {
                    "stage_id": "s2",
                    "stage_type": "SIMULATION",
                    "targets": ["t2"],
                    "dependencies": ["s1"],  # Circular dependency
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_self_dependency(self, base_state):
        """Test that reviewer rejects stages that depend on themselves."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["t1"],
                    "dependencies": ["s1"],  # Self-dependency
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "depend" in result.get("planner_feedback", "").lower() or "itself" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_missing_dependency(self, base_state):
        """Test that reviewer rejects stages with dependencies on non-existent stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "s1",
                    "stage_type": "SIMULATION",
                    "targets": ["t1"],
                    "dependencies": ["nonexistent"],  # Missing stage
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "depend" in result.get("planner_feedback", "").lower() or "missing" in result.get("planner_feedback", "").lower()

    def test_reviewer_rejects_invalid_stage_structure(self, base_state):
        """Test that reviewer rejects non-dict stages."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                "not a dict",  # Invalid structure
                123,  # Also invalid
            ],
            "targets": [],
        }

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "structure" in result.get("planner_feedback", "").lower() or "dict" in result.get("planner_feedback", "").lower()

    def test_reviewer_increments_replan_count_on_rejection(self, base_state):
        """Test that replan_count increments when plan is rejected."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [],  # Invalid - will be rejected
            "targets": [],
        }
        base_state["replan_count"] = 1
        base_state["runtime_config"] = {"max_replans": MAX_REPLANS}

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert result.get("replan_count") == 2, (
            f"Expected replan_count=2, got {result.get('replan_count')}"
        )

    def test_reviewer_respects_max_replans_on_rejection(self, base_state):
        """Test that replan_count doesn't exceed max_replans."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [],  # Invalid
            "targets": [],
        }
        base_state["replan_count"] = MAX_REPLANS
        base_state["runtime_config"] = {"max_replans": MAX_REPLANS}

        result = plan_reviewer_node(base_state)
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert result.get("replan_count") == MAX_REPLANS, (
            f"Expected replan_count={MAX_REPLANS}, got {result.get('replan_count')}"
        )

    def test_reviewer_handles_missing_assumptions(self, base_state, valid_plan):
        """Test that reviewer handles missing assumptions gracefully."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = valid_plan
        if "assumptions" in base_state:
            del base_state["assumptions"]

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={"verdict": "approve", "summary": "OK"},
        ):
            result = plan_reviewer_node(base_state)

        assert result.get("workflow_phase") == "plan_review"
        # Should not crash, should proceed normally

    def test_reviewer_handles_none_assumptions(self, base_state, valid_plan):
        """Test that reviewer handles None assumptions."""
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = valid_plan
        base_state["assumptions"] = None

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={"verdict": "approve", "summary": "OK"},
        ):
            result = plan_reviewer_node(base_state)

        assert result.get("workflow_phase") == "plan_review"


class TestAdaptPromptsEdgeCases:
    """Edge cases for adapt_prompts_node."""

    def test_adapt_prompts_handles_none_paper_text(self, base_state):
        """Test that adapt_prompts handles None paper_text."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_text"] = None

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={"adaptations": []},
        ):
            result = adapt_prompts_node(base_state)

        assert result.get("workflow_phase") == "adapting_prompts"
        assert result.get("prompt_adaptations") == []

    def test_adapt_prompts_handles_empty_paper_text(self, base_state):
        """Test that adapt_prompts handles empty paper_text."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_text"] = ""

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={"adaptations": []},
        ):
            result = adapt_prompts_node(base_state)

        assert result.get("workflow_phase") == "adapting_prompts"
        assert result.get("prompt_adaptations") == []

    def test_adapt_prompts_handles_missing_adaptations_key(self, base_state):
        """Test that adapt_prompts handles missing adaptations in response."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={},  # Missing "adaptations" key
        ):
            result = adapt_prompts_node(base_state)

        assert result.get("prompt_adaptations") == []

    def test_adapt_prompts_handles_none_adaptations(self, base_state):
        """Test that adapt_prompts handles None adaptations."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={"adaptations": None},
        ):
            result = adapt_prompts_node(base_state)

        assert result.get("prompt_adaptations") == []

    def test_adapt_prompts_handles_non_list_adaptations(self, base_state):
        """Test that adapt_prompts handles non-list adaptations."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value={"adaptations": "not a list"},  # Wrong type
        ):
            result = adapt_prompts_node(base_state)

        assert result.get("prompt_adaptations") == []

    def test_adapt_prompts_handles_llm_error(self, base_state):
        """Test that adapt_prompts handles LLM errors gracefully."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = adapt_prompts_node(base_state)

        assert result.get("workflow_phase") == "adapting_prompts"
        assert result.get("prompt_adaptations") == []

    def test_adapt_prompts_handles_context_escalation(self, base_state):
        """Test that adapt_prompts respects context escalation."""
        from src.agents.planning import adapt_prompts_node

        escalation_response = {
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
        }

        # The decorator uses check_context_or_escalate from base module, not planning module
        with patch(
            "src.agents.base.check_context_or_escalate",
            return_value=escalation_response,
        ):
            result = adapt_prompts_node(base_state)

        # Decorator should return escalation response directly when awaiting_user_input is True
        assert result == escalation_response, (
            f"Expected escalation response, got {result}. "
            "The decorator should return escalation directly when awaiting_user_input=True"
        )

