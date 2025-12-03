"""Integration-style tests for pre-configured routing helpers."""

import pytest
from typing import get_args

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
                "plan",
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
                "supervisor",
            ),  # NOTE: Routes to supervisor on limit
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

    def test_all_routers_return_valid_route_types(self, base_state):
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

