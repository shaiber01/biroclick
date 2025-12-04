"""Routing-related graph integration tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from schemas.state import (
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
    ReproState,
)
from src import routing
from src.graph import route_after_plan, route_after_select_stage, route_after_supervisor
from tests.integration.graph.state_utils import apply_state_overrides


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN REVIEW ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanReviewRouting:
    """Tests for route_after_plan_review with all verdict paths and edge cases."""

    @patch("src.routing.save_checkpoint")
    def test_plan_review_approve_routes_to_select_stage(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> select_stage on approve."""
        apply_state_overrides(test_state, last_plan_review_verdict="approve")

        result = routing.route_after_plan_review(test_state)

        assert result == "select_stage"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_plan_review_needs_revision_routes_to_plan(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> plan on needs_revision when under limit."""
        apply_state_overrides(
            test_state,
            last_plan_review_verdict="needs_revision",
            replan_count=0,
        )

        result = routing.route_after_plan_review(test_state)

        assert result == "planning"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_plan_review_needs_revision_escalates_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> ask_user on needs_revision when at limit."""
        apply_state_overrides(
            test_state,
            last_plan_review_verdict="needs_revision",
            replan_count=MAX_REPLANS,
        )

        result = routing.route_after_plan_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_plan_review_needs_revision_continues_just_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> plan on needs_revision when just under limit."""
        apply_state_overrides(
            test_state,
            last_plan_review_verdict="needs_revision",
            replan_count=MAX_REPLANS - 1,
        )

        result = routing.route_after_plan_review(test_state)

        assert result == "planning"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_plan_review_none_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> ask_user on None verdict."""
        apply_state_overrides(test_state, last_plan_review_verdict=None)

        result = routing.route_after_plan_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_plan_review_unknown_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test plan_review -> ask_user on unknown verdict."""
        apply_state_overrides(test_state, last_plan_review_verdict="invalid_verdict")

        result = routing.route_after_plan_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_plan_review_runtime_config_overrides_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that runtime_config can override default replan limit."""
        apply_state_overrides(
            test_state,
            last_plan_review_verdict="needs_revision",
            replan_count=MAX_REPLANS + 5,
            runtime_config={"max_replans": MAX_REPLANS + 10},
        )

        result = routing.route_after_plan_review(test_state)

        assert result == "planning"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN REVIEW ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignReviewRouting:
    """Tests for route_after_design_review with all verdict paths and edge cases."""

    @patch("src.routing.save_checkpoint")
    def test_design_review_approve_routes_to_generate_code(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> generate_code on approve."""
        apply_state_overrides(test_state, last_design_review_verdict="approve")

        result = routing.route_after_design_review(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_review_needs_revision_routes_to_design(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> design on needs_revision when under limit."""
        apply_state_overrides(
            test_state,
            last_design_review_verdict="needs_revision",
            design_revision_count=0,
        )

        result = routing.route_after_design_review(test_state)

        assert result == "design"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_review_needs_revision_escalates_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> ask_user on needs_revision when at limit."""
        apply_state_overrides(
            test_state,
            last_design_review_verdict="needs_revision",
            design_revision_count=MAX_DESIGN_REVISIONS,
        )

        result = routing.route_after_design_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_design_review_continues_just_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> design on needs_revision when just under limit."""
        apply_state_overrides(
            test_state,
            last_design_review_verdict="needs_revision",
            design_revision_count=MAX_DESIGN_REVISIONS - 1,
        )

        result = routing.route_after_design_review(test_state)

        assert result == "design"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_review_none_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> ask_user on None verdict."""
        apply_state_overrides(test_state, last_design_review_verdict=None)

        result = routing.route_after_design_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_design_review_unknown_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test design_review -> ask_user on unknown verdict."""
        apply_state_overrides(test_state, last_design_review_verdict="unknown_verdict")

        result = routing.route_after_design_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_design_review_runtime_config_overrides_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that runtime_config can override default design revision limit."""
        apply_state_overrides(
            test_state,
            last_design_review_verdict="needs_revision",
            design_revision_count=MAX_DESIGN_REVISIONS + 5,
            runtime_config={"max_design_revisions": MAX_DESIGN_REVISIONS + 10},
        )

        result = routing.route_after_design_review(test_state)

        assert result == "design"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEW ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewRouting:
    """Tests for route_after_code_review with all verdict paths and edge cases."""

    @patch("src.routing.save_checkpoint")
    def test_code_review_approve_routes_to_run_code(self, mock_checkpoint, test_state: ReproState):
        """Test code_review -> run_code on approve."""
        apply_state_overrides(test_state, last_code_review_verdict="approve")

        result = routing.route_after_code_review(test_state)

        assert result == "run_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_code_review_needs_revision_routes_to_generate_code(self, mock_checkpoint, test_state: ReproState):
        """Test code_review -> generate_code on needs_revision when under limit."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=0,
        )

        result = routing.route_after_code_review(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

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
    def test_code_review_none_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test code_review -> ask_user on None verdict."""
        apply_state_overrides(test_state, last_code_review_verdict=None)

        result = routing.route_after_code_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_code_review_unknown_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test code_review -> ask_user on unknown verdict."""
        apply_state_overrides(test_state, last_code_review_verdict="unexpected_value")

        result = routing.route_after_code_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

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
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_code_review_escalates_at_runtime_config_limit(self, mock_checkpoint, test_state: ReproState):
        """Test code_review escalates at custom runtime config limit."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=15,
            runtime_config={"max_code_revisions": 15},
        )

        result = routing.route_after_code_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_code_review_approves_regardless_of_count(self, mock_checkpoint, test_state: ReproState):
        """Test approve verdict routes to run_code regardless of revision count."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="approve",
            code_revision_count=999,  # High count shouldn't matter for approve
        )

        result = routing.route_after_code_review(test_state)

        assert result == "run_code"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION CHECK ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionCheckRouting:
    """Tests for route_after_execution_check with all verdict paths."""

    @patch("src.routing.save_checkpoint")
    def test_execution_pass_routes_to_physics_check(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> physics_check on pass."""
        apply_state_overrides(test_state, execution_verdict="pass")

        result = routing.route_after_execution_check(test_state)

        assert result == "physics_check"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_execution_warning_routes_to_physics_check(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> physics_check on warning (pass-through verdict)."""
        apply_state_overrides(test_state, execution_verdict="warning")

        result = routing.route_after_execution_check(test_state)

        assert result == "physics_check"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_execution_fail_routes_to_generate_code(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> generate_code on fail when under limit."""
        apply_state_overrides(
            test_state,
            execution_verdict="fail",
            execution_failure_count=0,
        )

        result = routing.route_after_execution_check(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_execution_fail_escalates_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> ask_user on fail when at limit."""
        apply_state_overrides(
            test_state,
            execution_verdict="fail",
            execution_failure_count=MAX_EXECUTION_FAILURES,
        )

        result = routing.route_after_execution_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_execution_fail_continues_just_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> generate_code on fail when just under limit."""
        apply_state_overrides(
            test_state,
            execution_verdict="fail",
            execution_failure_count=MAX_EXECUTION_FAILURES - 1,
        )

        result = routing.route_after_execution_check(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_execution_none_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> ask_user on None verdict."""
        apply_state_overrides(test_state, execution_verdict=None)

        result = routing.route_after_execution_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_execution_unknown_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test execution_check -> ask_user on unknown verdict."""
        apply_state_overrides(test_state, execution_verdict="invalid")

        result = routing.route_after_execution_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_execution_pass_ignores_count_limit(self, mock_checkpoint, test_state: ReproState):
        """Test pass verdict routes to physics_check regardless of failure count."""
        apply_state_overrides(
            test_state,
            execution_verdict="pass",
            execution_failure_count=999,
        )

        result = routing.route_after_execution_check(test_state)

        assert result == "physics_check"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_execution_warning_ignores_count_limit(self, mock_checkpoint, test_state: ReproState):
        """Test warning verdict routes to physics_check regardless of failure count."""
        apply_state_overrides(
            test_state,
            execution_verdict="warning",
            execution_failure_count=999,
        )

        result = routing.route_after_execution_check(test_state)

        assert result == "physics_check"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_execution_runtime_config_overrides_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that runtime_config can override default execution failure limit."""
        apply_state_overrides(
            test_state,
            execution_verdict="fail",
            execution_failure_count=MAX_EXECUTION_FAILURES + 5,
            runtime_config={"max_execution_failures": MAX_EXECUTION_FAILURES + 10},
        )

        result = routing.route_after_execution_check(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS CHECK ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhysicsCheckRouting:
    """Tests for route_after_physics_check with all verdict paths."""

    @patch("src.routing.save_checkpoint")
    def test_physics_pass_routes_to_analyze(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> analyze on pass."""
        apply_state_overrides(test_state, physics_verdict="pass")

        result = routing.route_after_physics_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_warning_routes_to_analyze(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> analyze on warning (pass-through verdict)."""
        apply_state_overrides(test_state, physics_verdict="warning")

        result = routing.route_after_physics_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_fail_routes_to_generate_code(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> generate_code on fail when under limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="fail",
            physics_failure_count=0,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_fail_escalates_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> ask_user on fail when at limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="fail",
            physics_failure_count=MAX_PHYSICS_FAILURES,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_physics_fail_continues_just_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> generate_code on fail when just under limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="fail",
            physics_failure_count=MAX_PHYSICS_FAILURES - 1,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_design_flaw_routes_to_design(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> design on design_flaw when under limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="design_flaw",
            design_revision_count=0,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "design"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_design_flaw_escalates_at_design_limit(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> ask_user on design_flaw when at design revision limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="design_flaw",
            design_revision_count=MAX_DESIGN_REVISIONS,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_physics_none_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> ask_user on None verdict."""
        apply_state_overrides(test_state, physics_verdict=None)

        result = routing.route_after_physics_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_physics_unknown_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test physics_check -> ask_user on unknown verdict."""
        apply_state_overrides(test_state, physics_verdict="invalid_verdict")

        result = routing.route_after_physics_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_physics_pass_ignores_count_limit(self, mock_checkpoint, test_state: ReproState):
        """Test pass verdict routes to analyze regardless of failure count."""
        apply_state_overrides(
            test_state,
            physics_verdict="pass",
            physics_failure_count=999,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_warning_ignores_count_limit(self, mock_checkpoint, test_state: ReproState):
        """Test warning verdict routes to analyze regardless of failure count."""
        apply_state_overrides(
            test_state,
            physics_verdict="warning",
            physics_failure_count=999,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_physics_runtime_config_overrides_failure_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that runtime_config can override default physics failure limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="fail",
            physics_failure_count=MAX_PHYSICS_FAILURES + 5,
            runtime_config={"max_physics_failures": MAX_PHYSICS_FAILURES + 10},
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHECK ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestComparisonCheckRouting:
    """Tests for route_after_comparison_check with all verdict paths."""

    @patch("src.routing.save_checkpoint")
    def test_comparison_approve_routes_to_supervisor(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> supervisor on approve."""
        apply_state_overrides(test_state, comparison_verdict="approve")

        result = routing.route_after_comparison_check(test_state)

        assert result == "supervisor"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_comparison_needs_revision_routes_to_analyze(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> analyze on needs_revision when under limit."""
        apply_state_overrides(
            test_state,
            comparison_verdict="needs_revision",
            analysis_revision_count=0,
        )

        result = routing.route_after_comparison_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_comparison_needs_revision_routes_to_ask_user_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> ask_user on needs_revision when at limit (consistent with others)."""
        # NOTE: Now consistent with other routers - comparison routes to ask_user at limit
        apply_state_overrides(
            test_state,
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS,
        )

        result = routing.route_after_comparison_check(test_state)

        # Now consistent with other routers - routes to ask_user at limit
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_comparison_needs_revision_continues_just_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> analyze on needs_revision when just under limit."""
        apply_state_overrides(
            test_state,
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS - 1,
        )

        result = routing.route_after_comparison_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_comparison_none_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> ask_user on None verdict."""
        apply_state_overrides(test_state, comparison_verdict=None)

        result = routing.route_after_comparison_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_comparison_unknown_verdict_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test comparison_check -> ask_user on unknown verdict."""
        apply_state_overrides(test_state, comparison_verdict="invalid_verdict")

        result = routing.route_after_comparison_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_comparison_runtime_config_overrides_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that runtime_config can override default analysis revision limit."""
        apply_state_overrides(
            test_state,
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS + 5,
            runtime_config={"max_analysis_revisions": MAX_ANALYSIS_REVISIONS + 10},
        )

        result = routing.route_after_comparison_check(test_state)

        assert result == "analyze"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupervisorRouting:
    """Tests for the complex route_after_supervisor logic."""

    @patch("src.graph.save_checkpoint")
    def test_supervisor_ok_continue_routes_to_select_stage(self, mock_checkpoint, test_state: ReproState):
        """Test normal continuation routes to select_stage."""
        apply_state_overrides(test_state, supervisor_verdict="ok_continue")

        result = route_after_supervisor(test_state)

        assert result == "select_stage"
        mock_checkpoint.assert_called()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "complete" in checkpoint_name

    @patch("src.graph.save_checkpoint")
    def test_supervisor_change_priority_routes_to_select_stage(self, mock_checkpoint, test_state: ReproState):
        """Test change_priority verdict routes to select_stage."""
        apply_state_overrides(test_state, supervisor_verdict="change_priority")

        result = route_after_supervisor(test_state)

        assert result == "select_stage"
        mock_checkpoint.assert_called()

    @patch("src.graph.save_checkpoint")
    def test_supervisor_routes_to_report_if_should_stop(self, mock_checkpoint, test_state: ReproState):
        """Test that should_stop flag forces report generation."""
        apply_state_overrides(test_state, supervisor_verdict="ok_continue", should_stop=True)

        result = route_after_supervisor(test_state)

        assert result == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_change_priority_with_should_stop(self, mock_checkpoint, test_state: ReproState):
        """Test change_priority also respects should_stop flag."""
        apply_state_overrides(test_state, supervisor_verdict="change_priority", should_stop=True)

        result = route_after_supervisor(test_state)

        assert result == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_routes_to_material_checkpoint_for_validation_stage(
        self, mock_checkpoint, test_state: ReproState
    ):
        """Test mandatory material checkpoint after MATERIAL_VALIDATION stage."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
        )

        result = route_after_supervisor(test_state)

        assert result == "material_checkpoint"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_skips_material_checkpoint_if_already_done(self, mock_checkpoint, test_state: ReproState):
        """Test that material checkpoint is not triggered again if already confirmed."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={"material_checkpoint": {"confirmed": True}},
        )

        result = route_after_supervisor(test_state)

        # Should skip material checkpoint and go to select_stage
        assert result == "select_stage"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_under_limit(self, mock_checkpoint, test_state: ReproState):
        """Test replan routes to plan if under limit."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS - 1,
        )

        result = route_after_supervisor(test_state)

        assert result == "planning"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_at_limit(self, mock_checkpoint, test_state: ReproState):
        """Test replan escalates to ask_user if at limit."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS,
        )

        result = route_after_supervisor(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_above_limit(self, mock_checkpoint, test_state: ReproState):
        """Test replan escalates to ask_user if above limit."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS + 5,
        )

        result = route_after_supervisor(test_state)

        assert result == "ask_user"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_at_zero_count(self, mock_checkpoint, test_state: ReproState):
        """Test replan routes to plan when count is zero."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=0,
        )

        result = route_after_supervisor(test_state)

        assert result == "planning"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_backtrack(self, mock_checkpoint, test_state: ReproState):
        """Test backtrack verdict routes to handle_backtrack."""
        apply_state_overrides(test_state, supervisor_verdict="backtrack_to_stage")

        result = route_after_supervisor(test_state)

        assert result == "handle_backtrack"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_all_complete(self, mock_checkpoint, test_state: ReproState):
        """Test all_complete verdict routes to generate_report."""
        apply_state_overrides(test_state, supervisor_verdict="all_complete")

        result = route_after_supervisor(test_state)

        assert result == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_ask_user(self, mock_checkpoint, test_state: ReproState):
        """Test ask_user verdict routes to ask_user."""
        apply_state_overrides(test_state, supervisor_verdict="ask_user")

        result = route_after_supervisor(test_state)

        assert result == "ask_user"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_none_verdict(self, mock_checkpoint, test_state: ReproState):
        """Test None verdict escalates to ask_user with error."""
        apply_state_overrides(test_state, supervisor_verdict=None)

        result = route_after_supervisor(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.graph.save_checkpoint")
    def test_supervisor_unknown_verdict_fallback(self, mock_checkpoint, test_state: ReproState):
        """Test unknown verdict falls back to ask_user."""
        apply_state_overrides(test_state, supervisor_verdict="invalid_verdict_xyz")

        result = route_after_supervisor(test_state)

        assert result == "ask_user"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_material_checkpoint_does_not_apply_to_other_stages(
        self, mock_checkpoint, test_state: ReproState
    ):
        """Test material checkpoint is NOT triggered for non-MATERIAL_VALIDATION stages."""
        for stage_type in ["SINGLE_STRUCTURE", "ARRAY_SYSTEM", "PARAMETER_SWEEP", "COMPLEX_PHYSICS"]:
            apply_state_overrides(
                test_state,
                supervisor_verdict="ok_continue",
                current_stage_type=stage_type,
            )

            result = route_after_supervisor(test_state)

            assert result == "select_stage", f"Stage type {stage_type} should route to select_stage"


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimpleRouting:
    """Tests for simple routing functions."""

    @patch("src.graph.save_checkpoint")
    def test_route_after_plan(self, mock_checkpoint, test_state: ReproState):
        """Test plan always routes to plan_review and saves checkpoint."""
        result = route_after_plan(test_state)

        assert result == "plan_review"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "after_plan" in checkpoint_name

    def test_route_after_select_stage_with_stage(self, test_state: ReproState):
        """Test select_stage routes to design when stage is selected."""
        apply_state_overrides(test_state, current_stage_id="stage_1")

        result = route_after_select_stage(test_state)

        assert result == "design"

    def test_route_after_select_stage_finished(self, test_state: ReproState):
        """Test select_stage routes to report when no stage selected (done)."""
        apply_state_overrides(test_state, current_stage_id=None)

        result = route_after_select_stage(test_state)

        assert result == "generate_report"

    def test_route_after_select_stage_with_empty_string(self, test_state: ReproState):
        """Test select_stage routes to report when stage_id is empty string."""
        apply_state_overrides(test_state, current_stage_id="")

        result = route_after_select_stage(test_state)

        # Empty string is falsy, should route to generate_report
        assert result == "generate_report"

    def test_route_after_select_stage_with_various_stage_ids(self, test_state: ReproState):
        """Test select_stage handles various stage_id formats."""
        test_cases = [
            "stage_1",
            "stage_0",
            "material_validation",
            "stage_abc_123",
            "s",  # Single character
        ]

        for stage_id in test_cases:
            apply_state_overrides(test_state, current_stage_id=stage_id)

            result = route_after_select_stage(test_state)

            assert result == "design", f"stage_id '{stage_id}' should route to design"


# ═══════════════════════════════════════════════════════════════════════════════
# NONE VERDICT HANDLING TESTS (PARAMETRIZED)
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# UNKNOWN VERDICT HANDLING TESTS (PARAMETRIZED)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnknownVerdictHandling:
    """Tests that all routers handle unknown verdicts gracefully."""

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
    def test_router_escalates_on_unknown_verdict(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        test_state: ReproState,
    ):
        """Test that each router escalates to ask_user on unknown verdict."""
        router = getattr(routing, router_name)
        apply_state_overrides(test_state, **{verdict_field: "completely_invalid_verdict_value"})

        result = router(test_state)

        assert result == "ask_user", f"{router_name} should route to ask_user on unknown verdict"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name


# ═══════════════════════════════════════════════════════════════════════════════
# LIMIT ESCALATION TESTS (PARAMETRIZED)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLimitEscalation:
    """Tests that routers correctly escalate when limits are reached."""

    @pytest.mark.parametrize(
        "router_name,verdict_field,verdict_value,count_field,max_count,expected_under_limit,expected_at_limit",
        [
            (
                "route_after_plan_review",
                "last_plan_review_verdict",
                "needs_revision",
                "replan_count",
                MAX_REPLANS,
                "planning",
                "ask_user",
            ),
            (
                "route_after_design_review",
                "last_design_review_verdict",
                "needs_revision",
                "design_revision_count",
                MAX_DESIGN_REVISIONS,
                "design",
                "ask_user",
            ),
            (
                "route_after_code_review",
                "last_code_review_verdict",
                "needs_revision",
                "code_revision_count",
                MAX_CODE_REVISIONS,
                "generate_code",
                "ask_user",
            ),
            (
                "route_after_execution_check",
                "execution_verdict",
                "fail",
                "execution_failure_count",
                MAX_EXECUTION_FAILURES,
                "generate_code",
                "ask_user",
            ),
            (
                "route_after_physics_check",
                "physics_verdict",
                "fail",
                "physics_failure_count",
                MAX_PHYSICS_FAILURES,
                "generate_code",
                "ask_user",
            ),
            # comparison_check now routes to ask_user at limit (consistent with others)
            (
                "route_after_comparison_check",
                "comparison_verdict",
                "needs_revision",
                "analysis_revision_count",
                MAX_ANALYSIS_REVISIONS,
                "analyze",
                "ask_user",  # Now consistent with other routers
            ),
        ],
    )
    @patch("src.routing.save_checkpoint")
    def test_limit_behavior(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        verdict_value: str,
        count_field: str,
        max_count: int,
        expected_under_limit: str,
        expected_at_limit: str,
        test_state: ReproState,
    ):
        """Test that each router handles count limits correctly."""
        router = getattr(routing, router_name)

        # Test under limit
        apply_state_overrides(
            test_state,
            **{verdict_field: verdict_value, count_field: max_count - 1},
        )
        result = router(test_state)
        assert (
            result == expected_under_limit
        ), f"{router_name} should route to {expected_under_limit} when under limit"

        # Reset mock for next test
        mock_checkpoint.reset_mock()

        # Test at limit
        apply_state_overrides(
            test_state,
            **{verdict_field: verdict_value, count_field: max_count},
        )
        result = router(test_state)
        assert result == expected_at_limit, f"{router_name} should route to {expected_at_limit} when at limit"
        mock_checkpoint.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT NAMING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointNaming:
    """Tests that checkpoints are named correctly for different scenarios."""

    @pytest.mark.parametrize(
        "router_name,verdict_field,expected_prefix",
        [
            ("route_after_plan_review", "last_plan_review_verdict", "plan_review"),
            ("route_after_design_review", "last_design_review_verdict", "design_review"),
            ("route_after_code_review", "last_code_review_verdict", "code_review"),
            ("route_after_execution_check", "execution_verdict", "execution"),
            ("route_after_physics_check", "physics_verdict", "physics"),
            ("route_after_comparison_check", "comparison_verdict", "comparison"),
        ],
    )
    @patch("src.routing.save_checkpoint")
    def test_error_checkpoint_naming(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        expected_prefix: str,
        test_state: ReproState,
    ):
        """Test that error checkpoints include the correct prefix."""
        router = getattr(routing, router_name)
        apply_state_overrides(test_state, **{verdict_field: None})

        router(test_state)

        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert expected_prefix in checkpoint_name, f"Checkpoint should contain prefix '{expected_prefix}'"
        assert "error" in checkpoint_name

    @pytest.mark.parametrize(
        "router_name,verdict_field,expected_prefix",
        [
            ("route_after_plan_review", "last_plan_review_verdict", "plan_review"),
            ("route_after_design_review", "last_design_review_verdict", "design_review"),
            ("route_after_code_review", "last_code_review_verdict", "code_review"),
            ("route_after_execution_check", "execution_verdict", "execution"),
            ("route_after_physics_check", "physics_verdict", "physics"),
            ("route_after_comparison_check", "comparison_verdict", "comparison"),
        ],
    )
    @patch("src.routing.save_checkpoint")
    def test_fallback_checkpoint_naming(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        expected_prefix: str,
        test_state: ReproState,
    ):
        """Test that fallback checkpoints include the correct prefix."""
        router = getattr(routing, router_name)
        apply_state_overrides(test_state, **{verdict_field: "unknown_verdict_xyz"})

        router(test_state)

        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert expected_prefix in checkpoint_name, f"Checkpoint should contain prefix '{expected_prefix}'"
        assert "fallback" in checkpoint_name


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES AND BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions in routing."""

    @patch("src.routing.save_checkpoint")
    def test_missing_runtime_config(self, mock_checkpoint, test_state: ReproState):
        """Test routing works when runtime_config is None."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS - 1,
            runtime_config=None,
        )

        result = routing.route_after_code_review(test_state)

        # Should use default limit and route to generate_code
        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_empty_runtime_config(self, mock_checkpoint, test_state: ReproState):
        """Test routing works when runtime_config is empty dict."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS - 1,
            runtime_config={},
        )

        result = routing.route_after_code_review(test_state)

        # Should use default limit and route to generate_code
        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_count_field_missing_defaults_to_zero(self, mock_checkpoint, test_state: ReproState):
        """Test routing works when count field is missing (defaults to 0)."""
        # Remove code_revision_count from state
        apply_state_overrides(test_state, last_code_review_verdict="needs_revision")
        test_state.pop("code_revision_count", None)

        result = routing.route_after_code_review(test_state)

        # Should treat as count=0 and route to generate_code
        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_count_field_none_defaults_to_zero(self, mock_checkpoint, test_state: ReproState):
        """Test routing works when count field is explicitly None."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=None,
        )

        result = routing.route_after_code_review(test_state)

        # Should treat as count=0 and route to generate_code
        assert result == "generate_code"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_replan_count_missing(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor handles missing replan_count."""
        apply_state_overrides(test_state, supervisor_verdict="replan_needed")
        test_state.pop("replan_count", None)

        result = route_after_supervisor(test_state)

        # Should treat as count=0 and route to plan
        assert result == "planning"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_user_responses_missing(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor handles missing user_responses for material checkpoint check."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
        )
        test_state.pop("user_responses", None)

        result = route_after_supervisor(test_state)

        # Should route to material_checkpoint since no previous checkpoint response exists
        assert result == "material_checkpoint"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_user_responses_empty(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor handles empty user_responses dict."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={},
        )

        result = route_after_supervisor(test_state)

        # Should route to material_checkpoint
        assert result == "material_checkpoint"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_current_stage_type_missing(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor handles missing current_stage_type."""
        apply_state_overrides(test_state, supervisor_verdict="ok_continue")
        test_state.pop("current_stage_type", None)

        result = route_after_supervisor(test_state)

        # Should route to select_stage (no material checkpoint needed)
        assert result == "select_stage"

    @patch("src.graph.save_checkpoint")
    def test_supervisor_current_stage_type_empty(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor handles empty current_stage_type."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="",
        )

        result = route_after_supervisor(test_state)

        # Should route to select_stage
        assert result == "select_stage"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS - VERIFY ROUTING CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoutingConsistency:
    """Integration tests to verify routing is consistent across the codebase."""

    def test_all_router_functions_exist(self):
        """Verify all expected routing functions are defined."""
        expected_routers = [
            "route_after_plan_review",
            "route_after_design_review",
            "route_after_code_review",
            "route_after_execution_check",
            "route_after_physics_check",
            "route_after_comparison_check",
        ]

        for router_name in expected_routers:
            assert hasattr(routing, router_name), f"Missing router: {router_name}"
            router = getattr(routing, router_name)
            assert callable(router), f"Router {router_name} is not callable"

    def test_all_routers_accept_repro_state(self, test_state: ReproState):
        """Verify all routers can accept a ReproState."""
        routers = [
            routing.route_after_plan_review,
            routing.route_after_design_review,
            routing.route_after_code_review,
            routing.route_after_execution_check,
            routing.route_after_physics_check,
            routing.route_after_comparison_check,
        ]

        # Set all verdict fields to None to trigger ask_user
        with patch("src.routing.save_checkpoint"):
            for router in routers:
                # Should not raise - all routers should handle the state
                result = router(test_state)
                # All should return ask_user for None verdicts
                assert result == "ask_user"

    def test_all_routers_return_valid_route_types(self, test_state: ReproState):
        """Verify all routers return valid route type strings."""
        valid_routes = {
            "planning",
            "plan_review",
            "select_stage",
            "design",
            "design_review",
            "generate_code",
            "code_review",
            "run_code",
            "execution_check",
            "physics_check",
            "analyze",
            "comparison_check",
            "supervisor",
            "ask_user",
            "generate_report",
            "handle_backtrack",
            "material_checkpoint",
        }

        routers_and_verdicts = [
            (routing.route_after_plan_review, "last_plan_review_verdict", ["approve", "needs_revision", None]),
            (routing.route_after_design_review, "last_design_review_verdict", ["approve", "needs_revision", None]),
            (routing.route_after_code_review, "last_code_review_verdict", ["approve", "needs_revision", None]),
            (
                routing.route_after_execution_check,
                "execution_verdict",
                ["pass", "warning", "fail", None],
            ),
            (
                routing.route_after_physics_check,
                "physics_verdict",
                ["pass", "warning", "fail", "design_flaw", None],
            ),
            (routing.route_after_comparison_check, "comparison_verdict", ["approve", "needs_revision", None]),
        ]

        with patch("src.routing.save_checkpoint"):
            for router, verdict_field, verdicts in routers_and_verdicts:
                for verdict in verdicts:
                    apply_state_overrides(test_state, **{verdict_field: verdict})
                    result = router(test_state)
                    assert result in valid_routes, (
                        f"Router returned invalid route '{result}' for verdict '{verdict}'"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# INVALID VERDICT TYPE HANDLING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestInvalidVerdictTypes:
    """Tests that routers handle non-string verdict types correctly."""

    @pytest.mark.parametrize(
        "invalid_verdict",
        [
            123,  # Integer
            12.5,  # Float
            True,  # Boolean
            False,  # Boolean False
            ["approve"],  # List
            {"verdict": "approve"},  # Dict
            ("approve",),  # Tuple
        ],
    )
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
    def test_router_handles_non_string_verdict(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        invalid_verdict,
        test_state: ReproState,
    ):
        """Test that routers escalate on non-string verdict types."""
        router = getattr(routing, router_name)
        apply_state_overrides(test_state, **{verdict_field: invalid_verdict})

        result = router(test_state)

        # Non-string verdicts should escalate to ask_user
        assert result == "ask_user", (
            f"{router_name} should route to ask_user for non-string verdict type "
            f"{type(invalid_verdict).__name__}"
        )
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE AND BOUNDARY COUNT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNegativeAndBoundaryCounts:
    """Tests for negative count values and boundary conditions."""

    @pytest.mark.parametrize(
        "router_name,verdict_field,verdict_value,count_field,expected_route",
        [
            ("route_after_code_review", "last_code_review_verdict", "needs_revision", "code_revision_count", "generate_code"),
            ("route_after_design_review", "last_design_review_verdict", "needs_revision", "design_revision_count", "design"),
            ("route_after_plan_review", "last_plan_review_verdict", "needs_revision", "replan_count", "planning"),
            ("route_after_execution_check", "execution_verdict", "fail", "execution_failure_count", "generate_code"),
            ("route_after_physics_check", "physics_verdict", "fail", "physics_failure_count", "generate_code"),
            ("route_after_comparison_check", "comparison_verdict", "needs_revision", "analysis_revision_count", "analyze"),
        ],
    )
    @patch("src.routing.save_checkpoint")
    def test_negative_count_allows_routing(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        verdict_value: str,
        count_field: str,
        expected_route: str,
        test_state: ReproState,
    ):
        """Test that negative counts don't cause escalation (treated as under limit)."""
        router = getattr(routing, router_name)
        apply_state_overrides(
            test_state,
            **{verdict_field: verdict_value, count_field: -5},
        )

        result = router(test_state)

        # Negative counts should be below limit, allowing normal routing
        assert result == expected_route
        mock_checkpoint.assert_not_called()

    @pytest.mark.parametrize(
        "router_name,verdict_field,verdict_value,count_field,max_count",
        [
            ("route_after_code_review", "last_code_review_verdict", "needs_revision", "code_revision_count", MAX_CODE_REVISIONS),
            ("route_after_design_review", "last_design_review_verdict", "needs_revision", "design_revision_count", MAX_DESIGN_REVISIONS),
            ("route_after_execution_check", "execution_verdict", "fail", "execution_failure_count", MAX_EXECUTION_FAILURES),
        ],
    )
    @patch("src.routing.save_checkpoint")
    def test_count_well_above_limit_still_escalates(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        verdict_value: str,
        count_field: str,
        max_count: int,
        test_state: ReproState,
    ):
        """Test that counts well above limit still trigger escalation."""
        router = getattr(routing, router_name)
        apply_state_overrides(
            test_state,
            **{verdict_field: verdict_value, count_field: max_count * 100},
        )

        result = router(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME CONFIG EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestRuntimeConfigEdgeCases:
    """Tests for edge cases in runtime configuration."""

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_zero_limit_always_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test that zero limit in runtime_config causes immediate escalation."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=0,
            runtime_config={"max_code_revisions": 0},
        )

        result = routing.route_after_code_review(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_negative_limit_always_escalates(self, mock_checkpoint, test_state: ReproState):
        """Test that negative limit in runtime_config causes immediate escalation."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=0,
            runtime_config={"max_code_revisions": -5},
        )

        result = routing.route_after_code_review(test_state)

        # count (0) >= limit (-5), so escalates
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_very_large_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that very large limit allows many revisions."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=999999,
            runtime_config={"max_code_revisions": 1000000},
        )

        result = routing.route_after_code_review(test_state)

        assert result == "generate_code"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_wrong_key_uses_default(self, mock_checkpoint, test_state: ReproState):
        """Test that wrong key in runtime_config causes default limit to be used."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
            runtime_config={"wrong_key": 100},  # Wrong key, should use default
        )

        result = routing.route_after_code_review(test_state)

        # Should use default MAX_CODE_REVISIONS limit
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.graph.save_checkpoint")
    def test_supervisor_runtime_config_overrides_replan_limit(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor respects runtime_config for replan limit."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS + 5,
            runtime_config={"max_replans": MAX_REPLANS + 10},
        )

        result = route_after_supervisor(test_state)

        assert result == "planning"  # Under the custom limit

    @patch("src.graph.save_checkpoint")
    def test_supervisor_runtime_config_zero_replan_limit(self, mock_checkpoint, test_state: ReproState):
        """Test supervisor with zero replan limit always escalates."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            replan_count=0,
            runtime_config={"max_replans": 0},
        )

        result = route_after_supervisor(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called()


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT STATE VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointStateVerification:
    """Tests that verify checkpoint receives correct state."""

    @patch("src.routing.save_checkpoint")
    def test_checkpoint_receives_state_on_error(self, mock_checkpoint, test_state: ReproState):
        """Test that save_checkpoint receives the state dict on error escalation."""
        apply_state_overrides(test_state, last_code_review_verdict=None)

        routing.route_after_code_review(test_state)

        mock_checkpoint.assert_called_once()
        # First argument should be the state
        state_arg = mock_checkpoint.call_args[0][0]
        assert state_arg is test_state
        # Second argument should be checkpoint name
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert isinstance(checkpoint_name, str)
        assert len(checkpoint_name) > 0

    @patch("src.routing.save_checkpoint")
    def test_checkpoint_receives_state_on_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that save_checkpoint receives correct state on limit escalation."""
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )

        routing.route_after_code_review(test_state)

        mock_checkpoint.assert_called_once()
        state_arg = mock_checkpoint.call_args[0][0]
        assert state_arg is test_state
        assert state_arg.get("code_revision_count") == MAX_CODE_REVISIONS

    @patch("src.graph.save_checkpoint")
    def test_route_after_plan_checkpoint_receives_state(self, mock_checkpoint, test_state: ReproState):
        """Test that route_after_plan passes state to checkpoint."""
        route_after_plan(test_state)

        mock_checkpoint.assert_called_once()
        state_arg = mock_checkpoint.call_args[0][0]
        assert state_arg is test_state


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MUTATION SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateMutationSafety:
    """Tests that routing functions don't unexpectedly mutate state."""

    @patch("src.routing.save_checkpoint")
    def test_router_does_not_mutate_verdict(self, mock_checkpoint, test_state: ReproState):
        """Test that routing doesn't modify the verdict field."""
        import copy

        original_verdict = "approve"
        apply_state_overrides(test_state, last_code_review_verdict=original_verdict)
        state_before = copy.deepcopy(test_state)

        routing.route_after_code_review(test_state)

        assert test_state.get("last_code_review_verdict") == original_verdict
        # The verdict should remain unchanged
        assert test_state.get("last_code_review_verdict") == state_before.get("last_code_review_verdict")

    @patch("src.routing.save_checkpoint")
    def test_router_does_not_mutate_count(self, mock_checkpoint, test_state: ReproState):
        """Test that routing doesn't modify the count field."""
        import copy

        original_count = 2
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=original_count,
        )
        state_before = copy.deepcopy(test_state)

        routing.route_after_code_review(test_state)

        assert test_state.get("code_revision_count") == original_count
        assert test_state.get("code_revision_count") == state_before.get("code_revision_count")

    @patch("src.graph.save_checkpoint")
    def test_supervisor_does_not_mutate_state(self, mock_checkpoint, test_state: ReproState):
        """Test that supervisor routing doesn't unexpectedly mutate state."""
        import copy

        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            replan_count=1,
        )
        state_before = copy.deepcopy(test_state)

        route_after_supervisor(test_state)

        # Key fields should remain unchanged
        assert test_state.get("supervisor_verdict") == state_before.get("supervisor_verdict")
        assert test_state.get("replan_count") == state_before.get("replan_count")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoggingBehavior:
    """Tests that verify proper logging occurs."""

    @patch("src.routing.save_checkpoint")
    def test_error_logged_on_none_verdict(self, mock_checkpoint, test_state: ReproState, caplog):
        """Test that error is logged when verdict is None."""
        import logging
        apply_state_overrides(test_state, last_code_review_verdict=None)

        with caplog.at_level(logging.ERROR, logger="src.routing"):
            routing.route_after_code_review(test_state)

        assert any("None" in record.message for record in caplog.records)
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    @patch("src.routing.save_checkpoint")
    def test_error_logged_on_invalid_type(self, mock_checkpoint, test_state: ReproState, caplog):
        """Test that error is logged when verdict has invalid type."""
        import logging
        apply_state_overrides(test_state, last_code_review_verdict=123)  # Integer

        with caplog.at_level(logging.ERROR, logger="src.routing"):
            routing.route_after_code_review(test_state)

        assert any("invalid type" in record.message.lower() for record in caplog.records)

    @patch("src.routing.save_checkpoint")
    def test_warning_logged_on_unknown_verdict(self, mock_checkpoint, test_state: ReproState, caplog):
        """Test that warning is logged when verdict is unknown."""
        import logging
        apply_state_overrides(test_state, last_code_review_verdict="invalid_unknown_value")

        with caplog.at_level(logging.WARNING, logger="src.routing"):
            routing.route_after_code_review(test_state)

        assert any("not a recognized verdict" in record.message for record in caplog.records)

    @patch("src.routing.save_checkpoint")
    def test_warning_logged_on_limit_reached(self, mock_checkpoint, test_state: ReproState, caplog):
        """Test that warning is logged when limit is reached."""
        import logging
        apply_state_overrides(
            test_state,
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )

        with caplog.at_level(logging.WARNING, logger="src.routing"):
            routing.route_after_code_review(test_state)

        assert any("escalating" in record.message.lower() for record in caplog.records)

    @patch("src.graph.save_checkpoint")
    def test_supervisor_logs_warning_on_none_verdict(self, mock_checkpoint, test_state: ReproState, caplog):
        """Test that supervisor logs warning on None verdict."""
        import logging
        apply_state_overrides(test_state, supervisor_verdict=None)

        with caplog.at_level(logging.WARNING):
            route_after_supervisor(test_state)

        assert any("None" in record.message for record in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR COMBINED CONDITIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupervisorCombinedConditions:
    """Tests for supervisor with multiple state conditions."""

    @patch("src.graph.save_checkpoint")
    def test_material_validation_with_should_stop_goes_to_report(self, mock_checkpoint, test_state: ReproState):
        """Test that should_stop takes precedence even for MATERIAL_VALIDATION."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
            should_stop=True,
        )

        result = route_after_supervisor(test_state)

        # should_stop should take precedence
        assert result == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_material_checkpoint_with_confirmed_false(self, mock_checkpoint, test_state: ReproState):
        """Test material checkpoint when user response exists but confirmed is False."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={"material_checkpoint": {"confirmed": False}},
        )

        result = route_after_supervisor(test_state)

        # The presence of material_checkpoint key should skip the checkpoint
        # regardless of the confirmed value (the check is just for key presence)
        assert result == "select_stage"

    @patch("src.graph.save_checkpoint")
    def test_material_checkpoint_with_other_responses(self, mock_checkpoint, test_state: ReproState):
        """Test material checkpoint when user_responses has other keys but not material_checkpoint."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="ok_continue",
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={"some_other_response": {"data": "value"}},
        )

        result = route_after_supervisor(test_state)

        # Should still route to material_checkpoint since material_checkpoint key is missing
        assert result == "material_checkpoint"

    @patch("src.graph.save_checkpoint")
    def test_replan_needed_with_should_stop_flag(self, mock_checkpoint, test_state: ReproState):
        """Test that replan_needed verdict is processed before should_stop check."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="replan_needed",
            should_stop=True,
            replan_count=0,
        )

        result = route_after_supervisor(test_state)

        # replan_needed should route to plan, not be affected by should_stop
        # (should_stop is only checked for ok_continue and change_priority)
        assert result == "planning"

    @patch("src.graph.save_checkpoint")
    def test_all_complete_overrides_material_validation(self, mock_checkpoint, test_state: ReproState):
        """Test that all_complete goes to report regardless of stage type."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="all_complete",
            current_stage_type="MATERIAL_VALIDATION",
        )

        result = route_after_supervisor(test_state)

        assert result == "generate_report"

    @patch("src.graph.save_checkpoint")
    def test_backtrack_with_material_validation_stage(self, mock_checkpoint, test_state: ReproState):
        """Test that backtrack routes to handle_backtrack regardless of stage."""
        apply_state_overrides(
            test_state,
            supervisor_verdict="backtrack_to_stage",
            current_stage_type="MATERIAL_VALIDATION",
        )

        result = route_after_supervisor(test_state)

        assert result == "handle_backtrack"


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS CHECK DESIGN_FLAW TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhysicsCheckDesignFlaw:
    """Comprehensive tests for design_flaw verdict in physics check."""

    @patch("src.routing.save_checkpoint")
    def test_design_flaw_just_under_design_limit(self, mock_checkpoint, test_state: ReproState):
        """Test design_flaw continues just under the design revision limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="design_flaw",
            design_revision_count=MAX_DESIGN_REVISIONS - 1,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "design"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_flaw_with_runtime_config_override(self, mock_checkpoint, test_state: ReproState):
        """Test design_flaw respects runtime_config design revision limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="design_flaw",
            design_revision_count=MAX_DESIGN_REVISIONS + 5,
            runtime_config={"max_design_revisions": MAX_DESIGN_REVISIONS + 10},
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "design"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_flaw_uses_design_limit_not_physics_limit(self, mock_checkpoint, test_state: ReproState):
        """Test that design_flaw uses design revision limit, not physics failure limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="design_flaw",
            design_revision_count=0,
            physics_failure_count=MAX_PHYSICS_FAILURES + 100,  # This should be ignored
        )

        result = routing.route_after_physics_check(test_state)

        # Should route to design based on design_revision_count, not physics_failure_count
        assert result == "design"
        mock_checkpoint.assert_not_called()

    @patch("src.routing.save_checkpoint")
    def test_design_flaw_escalates_at_exact_limit(self, mock_checkpoint, test_state: ReproState):
        """Test design_flaw escalates exactly at the limit."""
        apply_state_overrides(
            test_state,
            physics_verdict="design_flaw",
            design_revision_count=MAX_DESIGN_REVISIONS,
        )

        result = routing.route_after_physics_check(test_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHECK SPECIAL CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestComparisonCheckSpecialCases:
    """Tests for comparison_check special routing behavior."""

    @patch("src.routing.save_checkpoint")
    def test_comparison_limit_routes_to_ask_user(self, mock_checkpoint, test_state: ReproState):
        """Verify comparison_check routes to ask_user at limit.

        This is now consistent with other routers - comparison_check
        uses route_on_limit='ask_user' like code_review_limit, etc.
        """
        apply_state_overrides(
            test_state,
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS + 10,  # Well over limit
        )

        result = routing.route_after_comparison_check(test_state)

        # Now consistent with other routers - routes to ask_user
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_comparison_approve_ignores_count(self, mock_checkpoint, test_state: ReproState):
        """Test that approve verdict routes to supervisor regardless of count."""
        apply_state_overrides(
            test_state,
            comparison_verdict="approve",
            analysis_revision_count=9999,
        )

        result = routing.route_after_comparison_check(test_state)

        assert result == "supervisor"
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION AND PHYSICS PASS-THROUGH TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassThroughVerdicts:
    """Tests for pass-through verdicts that bypass count limits."""

    @pytest.mark.parametrize("verdict", ["pass", "warning"])
    @patch("src.routing.save_checkpoint")
    def test_execution_pass_through_with_various_counts(
        self, mock_checkpoint, verdict: str, test_state: ReproState
    ):
        """Test execution pass-through verdicts work with any failure count."""
        for count in [0, 1, MAX_EXECUTION_FAILURES, MAX_EXECUTION_FAILURES * 100]:
            apply_state_overrides(
                test_state,
                execution_verdict=verdict,
                execution_failure_count=count,
            )

            result = routing.route_after_execution_check(test_state)

            assert result == "physics_check", f"Failed for count={count}"
            mock_checkpoint.assert_not_called()

    @pytest.mark.parametrize("verdict", ["pass", "warning"])
    @patch("src.routing.save_checkpoint")
    def test_physics_pass_through_with_various_counts(
        self, mock_checkpoint, verdict: str, test_state: ReproState
    ):
        """Test physics pass-through verdicts work with any failure count."""
        for count in [0, 1, MAX_PHYSICS_FAILURES, MAX_PHYSICS_FAILURES * 100]:
            apply_state_overrides(
                test_state,
                physics_verdict=verdict,
                physics_failure_count=count,
            )

            result = routing.route_after_physics_check(test_state)

            assert result == "analyze", f"Failed for count={count}"
            mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY STRING VERDICT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyStringVerdicts:
    """Tests for empty string verdict handling."""

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
    def test_empty_string_verdict_escalates(
        self,
        mock_checkpoint,
        router_name: str,
        verdict_field: str,
        test_state: ReproState,
    ):
        """Test that empty string verdict escalates to ask_user."""
        router = getattr(routing, router_name)
        apply_state_overrides(test_state, **{verdict_field: ""})

        result = router(test_state)

        # Empty string is not a valid verdict, should escalate
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name


# ═══════════════════════════════════════════════════════════════════════════════
# WHITESPACE VERDICT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestWhitespaceVerdicts:
    """Tests for whitespace-only verdict handling."""

    @pytest.mark.parametrize("whitespace_verdict", [" ", "  ", "\t", "\n", "  \t\n  "])
    @patch("src.routing.save_checkpoint")
    def test_whitespace_verdict_escalates(
        self,
        mock_checkpoint,
        whitespace_verdict: str,
        test_state: ReproState,
    ):
        """Test that whitespace-only verdict escalates to ask_user."""
        apply_state_overrides(test_state, last_code_review_verdict=whitespace_verdict)

        result = routing.route_after_code_review(test_state)

        # Whitespace is not a valid verdict, should escalate via fallback
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# CASE SENSITIVITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCaseSensitivity:
    """Tests to verify verdict matching is case-sensitive."""

    @pytest.mark.parametrize(
        "wrong_case_verdict",
        ["Approve", "APPROVE", "ApPrOvE", "NEEDS_REVISION", "Needs_Revision"],
    )
    @patch("src.routing.save_checkpoint")
    def test_verdicts_are_case_sensitive(
        self,
        mock_checkpoint,
        wrong_case_verdict: str,
        test_state: ReproState,
    ):
        """Test that verdict matching is case-sensitive."""
        apply_state_overrides(test_state, last_code_review_verdict=wrong_case_verdict)

        result = routing.route_after_code_review(test_state)

        # Wrong case should not match, should escalate via fallback
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name
