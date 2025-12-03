"""Routing-related graph integration tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from schemas.state import MAX_CODE_REVISIONS, MAX_REPLANS, ReproState
from src import routing
from src.graph import route_after_plan, route_after_select_stage, route_after_supervisor
from tests.integration.graph.state_utils import apply_state_overrides


class TestRoutingIntegration:
    """Tests that routing functions work correctly with real state."""

    @patch("src.routing.save_checkpoint")
    def test_plan_review_approve_routes_to_select_stage(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> select_stage on approve."""
        apply_state_overrides(test_state, last_plan_review_verdict="approve")

        result = routing.route_after_plan_review(test_state)

        assert result == "select_stage"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_review_approve_routes_to_generate_code(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> generate_code on approve."""
        apply_state_overrides(test_state, last_design_review_verdict="approve")

        result = routing.route_after_design_review(test_state)

        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_code_review_approve_routes_to_run_code(self, mock_checkpoint, test_state: ReproState):
        """Test code_review -> run_code on approve."""
        apply_state_overrides(test_state, last_code_review_verdict="approve")

        result = routing.route_after_code_review(test_state)

        assert result == "run_code"

    @patch("src.routing.save_checkpoint")
    def test_execution_pass_routes_to_physics_check(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> physics_check on pass."""
        apply_state_overrides(test_state, execution_verdict="pass")

        result = routing.route_after_execution_check(test_state)

        assert result == "physics_check"

    @patch("src.routing.save_checkpoint")
    def test_physics_pass_routes_to_analyze(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> analyze on pass."""
        apply_state_overrides(test_state, physics_verdict="pass")

        result = routing.route_after_physics_check(test_state)

        assert result == "analyze"

    @patch("src.routing.save_checkpoint")
    def test_comparison_approve_routes_to_supervisor(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> supervisor on approve."""
        apply_state_overrides(test_state, comparison_verdict="approve")

        result = routing.route_after_comparison_check(test_state)

        assert result == "supervisor"


class TestLimitEscalation:
    """Tests that routers correctly escalate when limits are reached."""

    @patch("src.routing.save_checkpoint")
    def test_code_review_escalates_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test code_review escalates to ask_user at revision limit."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )

        result = routing.route_after_code_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_code_review_continues_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test code_review continues normally under limit."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS - 1,
        )

        result = routing.route_after_code_review(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_overrides_default_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that runtime_config can override default limits."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=10,
            runtime_config={"max_code_revisions": 20},
        )

        result = routing.route_after_code_review(test_state)

        assert result == "generate_code"


class TestNoneVerdictHandling:
    """Tests that all routers handle None verdicts gracefully."""

    @pytest.mark.parametrize(
        "router_name,verdict_field",
        [
            ("route_after_plan_review", "last_plan_review_verdict"),
            ("route_after_design_review", "last_design_review_verdict"),
            ("route_after_code_review", "last_code_review_verdict"),
            ("route_after_execution_check", "execution_verdict"),
            ("route_after_physics_check", "physics_verdict"),
            ("route_after_comparison_check", "comparison_verdict"),
        ],
    )
    @patch("src.routing.save_checkpoint")
    def test_router_escalates_on_none(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        test_state: ReproState,
    ):
        """Test that each router escalates to ask_user on None verdict."""
        router = getattr(routing, router_name)
        apply_state_overrides(test_state, **{verdict_field: None})

        result = router(test_state)

        assert result == "ask_user", f"{router_name} should route to ask_user on None"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name


class TestSupervisorRouting:
    """Tests for the complex route_after_supervisor logic."""

    @patch("src.graph.save_checkpoint")
    def test_supervisor_routes_to_select_stage_on_ok(self, mock_checkpoint, test_state: ReproState):
        """Test normal continuation routes to select_stage."""
        apply_state_overrides(test_state, supervisor_verdict="ok_continue")

        assert route_after_supervisor(test_state) == "select_stage"

        mock_checkpoint.assert_called()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "complete" in checkpoint_name

    @patch("src.graph.save_checkpoint")
    def test_supervisor_routes_to_report_if_should_stop(self, mock_checkpoint, test_state: ReproState):
        """Test that should_stop flag forces report generation."""
        apply_state_overrides(test_state, supervisor_verdict="ok_continue", should_stop=True)

        assert route_after_supervisor(test_state) == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_routes_to_material_checkpoint_for_validation_stage(
        self, mock_checkpoint, test_state: ReproState
    ):
        """Test mandatory material checkpoint after MATERIAL_VALIDATION."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
        )

        assert route_after_supervisor(test_state) == "material_checkpoint"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test replan routes to plan if under limit."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS - 1,
        )

        assert route_after_supervisor(test_state) == "plan"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test replan escalates to ask_user if at limit."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS,
        )

        assert route_after_supervisor(test_state) == "ask_user"

        mock_checkpoint.assert_called()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.graph.save_checkpoint")
    def test_supervisor_backtrack(self, mock_checkpoint, test_state: ReproState):
        """Test backtrack verdict routes to handle_backtrack."""
        apply_state_overrides(test_state, supervisor_verdict="backtrack_to_stage")

        assert route_after_supervisor(test_state) == "handle_backtrack"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_all_complete(self, mock_checkpoint, test_state: ReproState):
        """Test all_complete verdict routes to generate_report."""
        apply_state_overrides(test_state, supervisor_verdict="all_complete")

        assert route_after_supervisor(test_state) == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_ask_user(self, mock_checkpoint, test_state: ReproState):
        """Test ask_user verdict routes to ask_user."""
        apply_state_overrides(test_state, supervisor_verdict="ask_user")

        assert route_after_supervisor(test_state) == "ask_user"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_none_verdict(self, mock_checkpoint, test_state: ReproState):
        """Test None verdict escalates to ask_user with error."""
        apply_state_overrides(test_state, supervisor_verdict=None)

        assert route_after_supervisor(test_state) == "ask_user"

        mock_checkpoint.assert_called()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name


class TestSimpleRouting:
    """Tests for simple routing functions."""

    @patch("src.graph.save_checkpoint")
    def test_route_after_plan(self, mock_checkpoint, test_state: ReproState):
        """Test plan always routes to plan_review."""
        assert route_after_plan(test_state) == "plan_review"
        mock_checkpoint.assert_called_once()

    def test_route_after_select_stage_with_stage(self, test_state: ReproState):
        """Test select_stage routes to design when stage is selected."""
        apply_state_overrides(test_state, current_stage_id="stage_1")
        assert route_after_select_stage(test_state) == "design"

    def test_route_after_select_stage_finished(self, test_state: ReproState):
        """Test select_stage routes to report when no stage selected (done)."""
        apply_state_overrides(test_state, current_stage_id=None)
        assert route_after_select_stage(test_state) == "generate_report"

    def test_route_after_ask_user(self):
        """Placeholder test for ask_user routing; verified via edge tests."""
        pass


