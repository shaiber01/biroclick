"""Integration-style tests for pre-configured routing helpers.

These tests focus on:
1. Integration between routers and state management
2. Boundary conditions for count limits
3. Runtime config overrides
4. Checkpoint naming consistency
5. State isolation (routing doesn't corrupt state)
6. Realistic routing scenarios
"""

import copy
import logging
import pytest
from typing import get_args
from unittest.mock import patch

from schemas.state import (
    MAX_ANALYSIS_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_REPLANS,
)
from src.routing import (
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
    RouteType,
)


class TestRouterIntegration:
    """
    Tests that pre-configured routers match expected behavior and configuration.
    Uses parameterized tests for better coverage and maintainability.
    """

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, count_val, expected_route",
        [
            # Code Review Router
            (route_after_code_review, "last_code_review_verdict", "approve", None, 0, "run_code"),
            (
                route_after_code_review,
                "last_code_review_verdict",
                "needs_revision",
                "code_revision_count",
                0,
                "generate_code",
            ),
            (
                route_after_code_review,
                "last_code_review_verdict",
                "needs_revision",
                "code_revision_count",
                MAX_CODE_REVISIONS,
                "ask_user",
            ),
            # Execution Check Router
            (route_after_execution_check, "execution_verdict", "pass", None, 0, "physics_check"),
            (route_after_execution_check, "execution_verdict", "warning", None, 0, "physics_check"),
            (
                route_after_execution_check,
                "execution_verdict",
                "fail",
                "execution_failure_count",
                0,
                "generate_code",
            ),
            (
                route_after_execution_check,
                "execution_verdict",
                "fail",
                "execution_failure_count",
                MAX_EXECUTION_FAILURES,
                "ask_user",
            ),
            # Physics Check Router
            (route_after_physics_check, "physics_verdict", "pass", None, 0, "analyze"),
            (route_after_physics_check, "physics_verdict", "warning", None, 0, "analyze"),
            (
                route_after_physics_check,
                "physics_verdict",
                "fail",
                "physics_failure_count",
                0,
                "generate_code",
            ),
            (
                route_after_physics_check,
                "physics_verdict",
                "fail",
                "physics_failure_count",
                MAX_PHYSICS_FAILURES,
                "ask_user",
            ),
            (
                route_after_physics_check,
                "physics_verdict",
                "design_flaw",
                "design_revision_count",
                0,
                "design",
            ),
            (
                route_after_physics_check,
                "physics_verdict",
                "design_flaw",
                "design_revision_count",
                MAX_DESIGN_REVISIONS,
                "ask_user",
            ),
            # Design Review Router
            (route_after_design_review, "last_design_review_verdict", "approve", None, 0, "generate_code"),
            (
                route_after_design_review,
                "last_design_review_verdict",
                "needs_revision",
                "design_revision_count",
                0,
                "design",
            ),
            (
                route_after_design_review,
                "last_design_review_verdict",
                "needs_revision",
                "design_revision_count",
                MAX_DESIGN_REVISIONS,
                "ask_user",
            ),
            # Plan Review Router
            (route_after_plan_review, "last_plan_review_verdict", "approve", None, 0, "select_stage"),
            (
                route_after_plan_review,
                "last_plan_review_verdict",
                "needs_revision",
                "replan_count",
                0,
                "planning",
            ),
            (
                route_after_plan_review,
                "last_plan_review_verdict",
                "needs_revision",
                "replan_count",
                MAX_REPLANS,
                "ask_user",
            ),
            # Comparison Check Router
            (route_after_comparison_check, "comparison_verdict", "approve", None, 0, "supervisor"),
            (
                route_after_comparison_check,
                "comparison_verdict",
                "needs_revision",
                "analysis_revision_count",
                0,
                "analyze",
            ),
            (
                route_after_comparison_check,
                "comparison_verdict",
                "needs_revision",
                "analysis_revision_count",
                MAX_ANALYSIS_REVISIONS + 1,
                "ask_user",
            ),  # NOTE: Now routes to ask_user on limit (consistent with other routers)
        ],
    )
    def test_router_logic(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        count_val,
        expected_route,
        base_state,
        mock_save_checkpoint,
    ):
        """
        Parameterized test for all pre-configured routers.
        Verifies verdict -> route mapping and count limit handling.
        """
        _ = mock_save_checkpoint  # Ensure checkpoint writes remain mocked

        # Setup state
        base_state[verdict_field] = verdict_val
        if count_field:
            base_state[count_field] = count_val

        # Execute
        result = router(base_state)

        # Assert
        assert result == expected_route

    def test_all_routers_return_valid_route_types(self, base_state, mock_save_checkpoint):
        """Verify that all possible outputs of routers are valid RouteType strings."""
        # Get valid routes from Literal
        valid_routes = get_args(RouteType)

        # List of all routers to test
        routers = [
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        ]

        # Test with "unknown" verdict (fallback case)
        for router in routers:
            base_state["last_plan_review_verdict"] = "unknown"  # Garbage to trigger fallback
            base_state["last_design_review_verdict"] = "unknown"
            base_state["last_code_review_verdict"] = "unknown"
            base_state["execution_verdict"] = "unknown"
            base_state["physics_verdict"] = "unknown"
            base_state["comparison_verdict"] = "unknown"

            result = router(base_state)
            assert result in valid_routes, f"Router {router} returned invalid route: {result}"


class TestBoundaryConditions:
    """Tests for boundary conditions at count limits.
    
    These tests verify behavior at exact boundaries:
    - count = max - 1 (just under limit - should proceed)
    - count = max (at limit - should escalate)
    - count = max + 1 (over limit - should escalate)
    """

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, max_count",
        [
            (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", MAX_REPLANS),
            (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", MAX_DESIGN_REVISIONS),
            (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", MAX_CODE_REVISIONS),
            (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", MAX_EXECUTION_FAILURES),
            (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", MAX_PHYSICS_FAILURES),
        ],
    )
    def test_count_just_under_limit_proceeds(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        max_count,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that count = max - 1 allows the router to proceed normally."""
        base_state[verdict_field] = verdict_val
        base_state[count_field] = max_count - 1

        result = router(base_state)

        # Should NOT escalate to ask_user
        assert result != "ask_user", (
            f"Count {max_count - 1} (one under limit {max_count}) should NOT escalate"
        )
        mock_save_checkpoint.assert_not_called()

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, max_count",
        [
            (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", MAX_REPLANS),
            (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", MAX_DESIGN_REVISIONS),
            (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", MAX_CODE_REVISIONS),
            (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", MAX_EXECUTION_FAILURES),
            (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", MAX_PHYSICS_FAILURES),
        ],
    )
    def test_count_at_limit_escalates(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        max_count,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that count = max triggers escalation."""
        base_state[verdict_field] = verdict_val
        base_state[count_field] = max_count

        result = router(base_state)

        # Should escalate to ask_user
        assert result == "ask_user", (
            f"Count {max_count} (at limit) should escalate to ask_user, got {result}"
        )
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name, (
            f"Checkpoint name should contain 'limit', got '{checkpoint_name}'"
        )

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, max_count",
        [
            (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", MAX_REPLANS),
            (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", MAX_DESIGN_REVISIONS),
            (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", MAX_CODE_REVISIONS),
            (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", MAX_EXECUTION_FAILURES),
            (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", MAX_PHYSICS_FAILURES),
        ],
    )
    def test_count_over_limit_escalates(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        max_count,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that count > max triggers escalation."""
        base_state[verdict_field] = verdict_val
        base_state[count_field] = max_count + 1

        result = router(base_state)

        # Should escalate to ask_user
        assert result == "ask_user", (
            f"Count {max_count + 1} (over limit) should escalate to ask_user, got {result}"
        )
        mock_save_checkpoint.assert_called_once()

    def test_comparison_check_boundary_routes_to_ask_user(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test comparison_check routes to ask_user at limit (consistent with others).
        
        This is now consistent with other routers - comparison_check has route_on_limit="ask_user".
        """
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS

        result = route_after_comparison_check(base_state)

        # Now consistent with other routers - routes to ask_user
        assert result == "ask_user", (
            f"Comparison check should route to 'ask_user' at limit, got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    def test_physics_check_design_flaw_boundary(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test physics_check design_flaw verdict respects design_revision_count limit."""
        base_state["physics_verdict"] = "design_flaw"
        
        # Under limit - should route to design
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS - 1
        result = route_after_physics_check(base_state)
        assert result == "design"
        mock_save_checkpoint.assert_not_called()

        # At limit - should escalate
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        result = route_after_physics_check(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()


class TestRuntimeConfigOverrides:
    """Tests that runtime_config properly overrides default limits."""

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, config_key, default_max, expected_route_under_limit",
        [
            (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", "max_replans", MAX_REPLANS, "planning"),
            (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS, "design"),
            (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", "max_code_revisions", MAX_CODE_REVISIONS, "generate_code"),
            (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", "max_execution_failures", MAX_EXECUTION_FAILURES, "generate_code"),
            (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", "max_physics_failures", MAX_PHYSICS_FAILURES, "generate_code"),
            (route_after_comparison_check, "comparison_verdict", "needs_revision", "analysis_revision_count", "max_analysis_revisions", MAX_ANALYSIS_REVISIONS, "analyze"),
        ],
    )
    def test_runtime_config_increases_limit(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        config_key,
        default_max,
        expected_route_under_limit,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that runtime_config can increase the limit above default."""
        # Set count to default_max (would normally trigger escalation)
        base_state[verdict_field] = verdict_val
        base_state[count_field] = default_max
        
        # Increase limit via runtime_config
        base_state["runtime_config"] = {config_key: default_max + 10}

        result = router(base_state)

        # Should proceed normally because config limit is higher
        assert result == expected_route_under_limit, (
            f"With runtime_config {config_key}={default_max + 10}, "
            f"count={default_max} should proceed, got '{result}'"
        )
        mock_save_checkpoint.assert_not_called()

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, config_key, default_max",
        [
            (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", "max_replans", MAX_REPLANS),
            (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS),
            (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", "max_code_revisions", MAX_CODE_REVISIONS),
            (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", "max_execution_failures", MAX_EXECUTION_FAILURES),
            (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", "max_physics_failures", MAX_PHYSICS_FAILURES),
        ],
    )
    def test_runtime_config_decreases_limit(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        config_key,
        default_max,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that runtime_config can decrease the limit below default."""
        # Set count to 1 (would normally be under default limit)
        base_state[verdict_field] = verdict_val
        base_state[count_field] = 1
        
        # Decrease limit via runtime_config to 1
        base_state["runtime_config"] = {config_key: 1}

        result = router(base_state)

        # Should escalate because config limit is 1 and count is 1
        assert result == "ask_user", (
            f"With runtime_config {config_key}=1, count=1 should escalate, got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    def test_runtime_config_zero_limit_escalates_immediately(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that runtime_config limit of 0 causes immediate escalation."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        base_state["runtime_config"] = {"max_code_revisions": 0}

        result = route_after_code_review(base_state)

        # 0 >= 0 should escalate
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_runtime_config_negative_limit_escalates(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that negative runtime_config limit causes escalation (count >= negative)."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        base_state["runtime_config"] = {"max_code_revisions": -1}

        result = route_after_code_review(base_state)

        # 0 >= -1 is true, should escalate
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()


class TestCheckpointNaming:
    """Tests that checkpoint names follow consistent naming conventions."""

    @pytest.mark.parametrize(
        "router, verdict_field, checkpoint_prefix",
        [
            (route_after_plan_review, "last_plan_review_verdict", "plan_review"),
            (route_after_design_review, "last_design_review_verdict", "design_review"),
            (route_after_code_review, "last_code_review_verdict", "code_review"),
            (route_after_execution_check, "execution_verdict", "execution"),
            (route_after_physics_check, "physics_verdict", "physics"),
            (route_after_comparison_check, "comparison_verdict", "comparison"),
        ],
    )
    def test_none_verdict_checkpoint_name(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
    ):
        """Test checkpoint name format for None verdict: before_ask_user_{prefix}_error."""
        base_state[verdict_field] = None

        router(base_state)

        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        expected_name = f"before_ask_user_{checkpoint_prefix}_error"
        assert checkpoint_name == expected_name, (
            f"Expected checkpoint name '{expected_name}', got '{checkpoint_name}'"
        )

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val, count_field, max_count, checkpoint_prefix",
        [
            (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", MAX_REPLANS, "plan_review"),
            (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", MAX_DESIGN_REVISIONS, "design_review"),
            (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", MAX_CODE_REVISIONS, "code_review"),
            (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", MAX_EXECUTION_FAILURES, "execution"),
            (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", MAX_PHYSICS_FAILURES, "physics"),
            (route_after_comparison_check, "comparison_verdict", "needs_revision", "analysis_revision_count", MAX_ANALYSIS_REVISIONS, "comparison"),
        ],
    )
    def test_limit_reached_checkpoint_name(
        self,
        router,
        verdict_field,
        verdict_val,
        count_field,
        max_count,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
    ):
        """Test checkpoint name format when limit reached: before_ask_user_{prefix}_limit."""
        base_state[verdict_field] = verdict_val
        base_state[count_field] = max_count

        router(base_state)

        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        expected_name = f"before_ask_user_{checkpoint_prefix}_limit"
        assert checkpoint_name == expected_name, (
            f"Expected checkpoint name '{expected_name}', got '{checkpoint_name}'"
        )

    @pytest.mark.parametrize(
        "router, verdict_field, checkpoint_prefix",
        [
            (route_after_plan_review, "last_plan_review_verdict", "plan_review"),
            (route_after_design_review, "last_design_review_verdict", "design_review"),
            (route_after_code_review, "last_code_review_verdict", "code_review"),
            (route_after_execution_check, "execution_verdict", "execution"),
            (route_after_physics_check, "physics_verdict", "physics"),
            (route_after_comparison_check, "comparison_verdict", "comparison"),
        ],
    )
    def test_unknown_verdict_checkpoint_name(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
    ):
        """Test checkpoint name format for unknown verdict: before_ask_user_{prefix}_fallback."""
        base_state[verdict_field] = "unknown_garbage_verdict"

        router(base_state)

        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        expected_name = f"before_ask_user_{checkpoint_prefix}_fallback"
        assert checkpoint_name == expected_name, (
            f"Expected checkpoint name '{expected_name}', got '{checkpoint_name}'"
        )


class TestStateIsolation:
    """Tests that routing decisions don't corrupt or modify state inappropriately."""

    @pytest.mark.parametrize(
        "router, verdict_field, verdict_val",
        [
            (route_after_plan_review, "last_plan_review_verdict", "approve"),
            (route_after_design_review, "last_design_review_verdict", "approve"),
            (route_after_code_review, "last_code_review_verdict", "approve"),
            (route_after_execution_check, "execution_verdict", "pass"),
            (route_after_physics_check, "physics_verdict", "pass"),
            (route_after_comparison_check, "comparison_verdict", "approve"),
        ],
    )
    def test_router_does_not_modify_state_on_success(
        self,
        router,
        verdict_field,
        verdict_val,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that successful routing doesn't modify any state fields."""
        base_state[verdict_field] = verdict_val
        
        # Deep copy state before routing
        state_snapshot = copy.deepcopy(dict(base_state))

        router(base_state)

        # Verify no fields were modified
        for key in state_snapshot:
            assert base_state.get(key) == state_snapshot[key], (
                f"State field '{key}' was modified during routing"
            )

    @pytest.mark.parametrize(
        "router, verdict_field",
        [
            (route_after_plan_review, "last_plan_review_verdict"),
            (route_after_design_review, "last_design_review_verdict"),
            (route_after_code_review, "last_code_review_verdict"),
            (route_after_execution_check, "execution_verdict"),
            (route_after_physics_check, "physics_verdict"),
            (route_after_comparison_check, "comparison_verdict"),
        ],
    )
    def test_router_does_not_modify_state_on_escalation(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that escalation routing doesn't modify state (except checkpoint call).
        
        Routers should be pure functions that only return the next node name.
        State modifications should be done by nodes, not routers.
        The ask_user_node has a safety net to set trigger/questions if missing.
        """
        base_state[verdict_field] = None  # Triggers escalation
        
        # Deep copy state before routing
        state_snapshot = copy.deepcopy(dict(base_state))

        router(base_state)

        # Verify no fields were modified (routers should be pure)
        for key in state_snapshot:
            assert base_state.get(key) == state_snapshot[key], (
                f"State field '{key}' was modified during escalation routing"
            )

    def test_routers_are_independent(self, base_state, mock_save_checkpoint):
        """Test that each router only examines its own verdict field."""
        # Set all verdicts to "approve" or equivalent pass
        base_state["last_plan_review_verdict"] = "approve"
        base_state["last_design_review_verdict"] = "approve"
        base_state["last_code_review_verdict"] = "approve"
        base_state["execution_verdict"] = "pass"
        base_state["physics_verdict"] = "pass"
        base_state["comparison_verdict"] = "approve"

        # Now set one to None - only that router should see it
        base_state["last_code_review_verdict"] = None

        # Other routers should work normally
        assert route_after_plan_review(base_state) == "select_stage"
        mock_save_checkpoint.reset_mock()
        
        assert route_after_design_review(base_state) == "generate_code"
        mock_save_checkpoint.reset_mock()
        
        # This one should escalate
        assert route_after_code_review(base_state) == "ask_user"
        mock_save_checkpoint.assert_called_once()
        mock_save_checkpoint.reset_mock()
        
        # These should still work
        assert route_after_execution_check(base_state) == "physics_check"
        assert route_after_physics_check(base_state) == "analyze"
        assert route_after_comparison_check(base_state) == "supervisor"


class TestAllValidVerdicts:
    """Tests that each router handles all its valid verdict values correctly."""

    def test_plan_review_all_verdicts(self, base_state, mock_save_checkpoint):
        """Test plan_review router handles all valid verdicts."""
        # Test 'approve'
        base_state["last_plan_review_verdict"] = "approve"
        assert route_after_plan_review(base_state) == "select_stage"
        
        # Test 'needs_revision' under limit
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 0
        assert route_after_plan_review(base_state) == "planning"
        
        mock_save_checkpoint.assert_not_called()

    def test_design_review_all_verdicts(self, base_state, mock_save_checkpoint):
        """Test design_review router handles all valid verdicts."""
        # Test 'approve'
        base_state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(base_state) == "generate_code"
        
        # Test 'needs_revision' under limit
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 0
        assert route_after_design_review(base_state) == "design"
        
        mock_save_checkpoint.assert_not_called()

    def test_code_review_all_verdicts(self, base_state, mock_save_checkpoint):
        """Test code_review router handles all valid verdicts."""
        # Test 'approve'
        base_state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(base_state) == "run_code"
        
        # Test 'needs_revision' under limit
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        assert route_after_code_review(base_state) == "generate_code"
        
        mock_save_checkpoint.assert_not_called()

    def test_execution_check_all_verdicts(self, base_state, mock_save_checkpoint):
        """Test execution_check router handles all valid verdicts."""
        # Test 'pass'
        base_state["execution_verdict"] = "pass"
        assert route_after_execution_check(base_state) == "physics_check"
        
        # Test 'warning'
        base_state["execution_verdict"] = "warning"
        assert route_after_execution_check(base_state) == "physics_check"
        
        # Test 'fail' under limit
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0
        assert route_after_execution_check(base_state) == "generate_code"
        
        mock_save_checkpoint.assert_not_called()

    def test_physics_check_all_verdicts(self, base_state, mock_save_checkpoint):
        """Test physics_check router handles all valid verdicts."""
        # Test 'pass'
        base_state["physics_verdict"] = "pass"
        assert route_after_physics_check(base_state) == "analyze"
        
        # Test 'warning'
        base_state["physics_verdict"] = "warning"
        assert route_after_physics_check(base_state) == "analyze"
        
        # Test 'fail' under limit
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = 0
        assert route_after_physics_check(base_state) == "generate_code"
        
        # Test 'design_flaw' under limit
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0
        assert route_after_physics_check(base_state) == "design"
        
        mock_save_checkpoint.assert_not_called()

    def test_comparison_check_all_verdicts(self, base_state, mock_save_checkpoint):
        """Test comparison_check router handles all valid verdicts."""
        # Test 'approve'
        base_state["comparison_verdict"] = "approve"
        assert route_after_comparison_check(base_state) == "supervisor"
        
        # Test 'needs_revision' under limit
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 0
        assert route_after_comparison_check(base_state) == "analyze"
        
        mock_save_checkpoint.assert_not_called()


class TestPassThroughVerdicts:
    """Tests that pass-through verdicts (pass, warning) skip count checking."""

    def test_execution_check_pass_ignores_failure_count(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that 'pass' verdict ignores execution_failure_count."""
        base_state["execution_verdict"] = "pass"
        base_state["execution_failure_count"] = 1000  # Way over limit
        
        result = route_after_execution_check(base_state)
        
        assert result == "physics_check"
        mock_save_checkpoint.assert_not_called()

    def test_execution_check_warning_ignores_failure_count(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that 'warning' verdict ignores execution_failure_count."""
        base_state["execution_verdict"] = "warning"
        base_state["execution_failure_count"] = 1000  # Way over limit
        
        result = route_after_execution_check(base_state)
        
        assert result == "physics_check"
        mock_save_checkpoint.assert_not_called()

    def test_physics_check_pass_ignores_failure_count(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that 'pass' verdict ignores physics_failure_count."""
        base_state["physics_verdict"] = "pass"
        base_state["physics_failure_count"] = 1000  # Way over limit
        
        result = route_after_physics_check(base_state)
        
        assert result == "analyze"
        mock_save_checkpoint.assert_not_called()

    def test_physics_check_warning_ignores_failure_count(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that 'warning' verdict ignores physics_failure_count."""
        base_state["physics_verdict"] = "warning"
        base_state["physics_failure_count"] = 1000  # Way over limit
        
        result = route_after_physics_check(base_state)
        
        assert result == "analyze"
        mock_save_checkpoint.assert_not_called()


class TestRealisticRoutingScenarios:
    """Integration tests simulating realistic routing scenarios."""

    def test_successful_code_flow_no_escalation(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test a successful flow: code approved -> execution pass -> physics pass."""
        # Step 1: Code review approved
        base_state["last_code_review_verdict"] = "approve"
        route1 = route_after_code_review(base_state)
        assert route1 == "run_code"
        
        # Step 2: Execution passes
        base_state["execution_verdict"] = "pass"
        route2 = route_after_execution_check(base_state)
        assert route2 == "physics_check"
        
        # Step 3: Physics passes
        base_state["physics_verdict"] = "pass"
        route3 = route_after_physics_check(base_state)
        assert route3 == "analyze"
        
        # Step 4: Comparison approved
        base_state["comparison_verdict"] = "approve"
        route4 = route_after_comparison_check(base_state)
        assert route4 == "supervisor"
        
        # No checkpoints should have been saved
        mock_save_checkpoint.assert_not_called()

    def test_code_revision_cycle_until_limit(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test code revision cycle that eventually hits the limit."""
        base_state["last_code_review_verdict"] = "needs_revision"
        
        # Simulate multiple revision cycles
        for count in range(MAX_CODE_REVISIONS):
            base_state["code_revision_count"] = count
            mock_save_checkpoint.reset_mock()
            
            result = route_after_code_review(base_state)
            assert result == "generate_code", (
                f"Should route to generate_code at count {count}"
            )
            mock_save_checkpoint.assert_not_called()
        
        # At limit, should escalate
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        result = route_after_code_review(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_physics_design_flaw_escalation_path(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test design_flaw verdict routing back to design."""
        base_state["physics_verdict"] = "design_flaw"
        
        # First design_flaw should route to design
        base_state["design_revision_count"] = 0
        result = route_after_physics_check(base_state)
        assert result == "design"
        
        # Multiple design flaws should eventually escalate
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        result = route_after_physics_check(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_execution_failure_with_recovery(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test execution failure -> code regeneration -> eventual success."""
        # First execution fails
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0
        route1 = route_after_execution_check(base_state)
        assert route1 == "generate_code"
        
        # After code regen, execution passes
        base_state["execution_verdict"] = "pass"
        base_state["execution_failure_count"] = 1  # Count was incremented
        route2 = route_after_execution_check(base_state)
        assert route2 == "physics_check"
        
        # No escalation occurred
        mock_save_checkpoint.assert_not_called()

    def test_full_workflow_with_revisions(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test a complete workflow with some revision cycles."""
        # Plan approved
        base_state["last_plan_review_verdict"] = "approve"
        assert route_after_plan_review(base_state) == "select_stage"
        
        # Design needs one revision
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 0
        assert route_after_design_review(base_state) == "design"
        
        # Design approved after revision
        base_state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(base_state) == "generate_code"
        
        # Code approved
        base_state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(base_state) == "run_code"
        
        # Execution with warning (still proceeds)
        base_state["execution_verdict"] = "warning"
        assert route_after_execution_check(base_state) == "physics_check"
        
        # Physics passes
        base_state["physics_verdict"] = "pass"
        assert route_after_physics_check(base_state) == "analyze"
        
        # Analysis needs revision once
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 0
        assert route_after_comparison_check(base_state) == "analyze"
        
        # Analysis approved
        base_state["comparison_verdict"] = "approve"
        assert route_after_comparison_check(base_state) == "supervisor"
        
        # No escalation checkpoints
        mock_save_checkpoint.assert_not_called()


class TestLoggingBehavior:
    """Tests that routers log appropriate messages."""

    def test_limit_reached_logs_warning(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that reaching a limit logs a WARNING level message."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        with caplog.at_level(logging.WARNING):
            route_after_code_review(base_state)
        
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 1
        assert "code_revision_count" in caplog.text
        assert str(MAX_CODE_REVISIONS) in caplog.text

    def test_none_verdict_logs_error(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that None verdict logs an ERROR level message."""
        base_state["last_code_review_verdict"] = None
        
        with caplog.at_level(logging.ERROR):
            route_after_code_review(base_state)
        
        assert len([r for r in caplog.records if r.levelno == logging.ERROR]) >= 1
        assert "None" in caplog.text

    def test_unknown_verdict_logs_warning(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that unknown verdict logs a WARNING level message."""
        base_state["last_code_review_verdict"] = "unknown_garbage"
        
        with caplog.at_level(logging.WARNING):
            route_after_code_review(base_state)
        
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 1
        assert "unknown_garbage" in caplog.text
        assert "not a recognized verdict" in caplog.text


class TestReturnTypeConsistency:
    """Tests that routers always return the correct type."""

    @pytest.mark.parametrize(
        "router, verdict_field",
        [
            (route_after_plan_review, "last_plan_review_verdict"),
            (route_after_design_review, "last_design_review_verdict"),
            (route_after_code_review, "last_code_review_verdict"),
            (route_after_execution_check, "execution_verdict"),
            (route_after_physics_check, "physics_verdict"),
            (route_after_comparison_check, "comparison_verdict"),
        ],
    )
    def test_router_returns_string(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that router always returns a string type."""
        test_cases = [
            ("approve", None),
            ("pass", None),
            ("needs_revision", None),
            (None, None),
            ("unknown", None),
        ]
        
        for verdict, _ in test_cases:
            base_state[verdict_field] = verdict
            mock_save_checkpoint.reset_mock()
            
            result = router(base_state)
            
            assert isinstance(result, str), (
                f"Router should return str, got {type(result).__name__} for verdict '{verdict}'"
            )

    @pytest.mark.parametrize(
        "router",
        [
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        ],
    )
    def test_router_returns_valid_route_type(
        self,
        router,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that router returns a valid RouteType value."""
        valid_routes = get_args(RouteType)
        
        # Set all verdicts to various values
        verdicts = ["approve", "needs_revision", "pass", "warning", "fail", None, "garbage"]
        
        for verdict in verdicts:
            base_state["last_plan_review_verdict"] = verdict
            base_state["last_design_review_verdict"] = verdict
            base_state["last_code_review_verdict"] = verdict
            base_state["execution_verdict"] = verdict
            base_state["physics_verdict"] = verdict
            base_state["comparison_verdict"] = verdict
            mock_save_checkpoint.reset_mock()
            
            result = router(base_state)
            
            assert result in valid_routes, (
                f"Router returned '{result}' which is not a valid RouteType"
            )


class TestSpecialCases:
    """Tests for special edge cases and corner cases."""

    def test_comparison_check_ask_user_on_limit(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Verify comparison_check uses ask_user when limit reached (consistent with others)."""
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS
        
        result = route_after_comparison_check(base_state)
        
        # Now consistent with other routers
        assert result == "ask_user", (
            f"comparison_check should route to 'ask_user' at limit. "
            f"Got '{result}'"
        )

    def test_physics_design_flaw_uses_design_revision_count(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that design_flaw verdict uses design_revision_count, not physics_failure_count."""
        base_state["physics_verdict"] = "design_flaw"
        base_state["physics_failure_count"] = 0  # Low
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS  # At limit
        
        result = route_after_physics_check(base_state)
        
        # Should escalate based on design_revision_count, not physics_failure_count
        assert result == "ask_user"

    def test_multiple_counts_independent(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that different count fields are independent."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0  # Under limit
        
        # Set other counts to high values - should not affect code_review
        base_state["design_revision_count"] = 100
        base_state["replan_count"] = 100
        base_state["execution_failure_count"] = 100
        base_state["physics_failure_count"] = 100
        base_state["analysis_revision_count"] = 100
        
        result = route_after_code_review(base_state)
        
        # Should proceed because code_revision_count is 0
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()

    def test_router_with_empty_state(self, mock_save_checkpoint):
        """Test router behavior with minimal/empty state."""
        # Create a minimal state (verdict field missing)
        minimal_state = {}
        
        result = route_after_code_review(minimal_state)
        
        # Missing verdict is treated as None -> ask_user
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_count_field_can_be_none(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None count field is treated as 0."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = None  # Explicitly None
        
        result = route_after_code_review(base_state)
        
        # None should be treated as 0 (under limit)
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()
