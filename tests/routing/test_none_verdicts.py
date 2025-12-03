"""Regression tests that ensure None verdicts escalate properly."""

import pytest

from src.routing import (
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
)


class TestNoneVerdictHandling:
    """Tests that all routers handle None verdicts correctly."""

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
    def test_router_handles_none(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that each router handles None verdict by escalating to ask_user."""
        base_state[verdict_field] = None

        result = router(base_state)

        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args[0][1] == f"before_ask_user_{checkpoint_prefix}_error"

