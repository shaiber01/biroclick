"""Unit tests for the routing verdict router factory."""

import copy
import logging

import pytest

from src.routing import (
    create_verdict_router,
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
)
from schemas.state import (
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
)


class TestVerdictRouterFactory:
    """Tests for the create_verdict_router factory function."""

    def test_factory_creates_callable_router(self):
        """Test that factory returns a callable function."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        assert callable(router)

    def test_factory_returns_function_not_other_callable(self):
        """Test that factory returns specifically a function type."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        # Verify it's a function, not just any callable (e.g., not a class)
        import types
        assert isinstance(router, types.FunctionType)

    def test_router_handles_none_verdict_gracefully(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that None verdict escalates to ask_user with checkpoint and error log."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        # Verdict is None (not set)
        base_state["test_verdict"] = None

        with caplog.at_level(logging.ERROR):
            result = router(base_state)

        assert result == "ask_user"

        # Verify checkpoint
        mock_save_checkpoint.assert_called_once()
        call_args = mock_save_checkpoint.call_args
        # Strictly verify identity of state object passed
        assert call_args[0][0] is base_state
        assert call_args[0][1] == "before_ask_user_test_error"

        # Verify logging
        assert "test_verdict is None" in caplog.text
        assert "Escalating to user" in caplog.text

    @pytest.mark.parametrize("invalid_verdict,expected_type_name", [
        (123, "int"),
        (12.5, "float"),
        (True, "bool"),
        (False, "bool"),
        (["approve"], "list"),
        ({"verdict": "approve"}, "dict"),
        (("approve",), "tuple"),
    ])
    def test_router_handles_non_string_verdict_types(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
        invalid_verdict,
        expected_type_name,
    ):
        """Test that non-string verdicts are rejected and escalated to ask_user."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = invalid_verdict

        with caplog.at_level(logging.ERROR):
            result = router(base_state)

        assert result == "ask_user", f"Expected ask_user for {type(invalid_verdict).__name__} verdict"

        # Verify checkpoint was saved
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args[0][1] == "before_ask_user_test_error"

        # Verify error log mentions invalid type
        assert f"invalid type {expected_type_name}" in caplog.text
        assert "expected str" in caplog.text

    def test_router_returns_correct_route_for_verdict(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that router returns correct route for each verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "approve": {"route": "success_node"},
                "reject": {"route": "retry_node"},
            },
            checkpoint_prefix="test",
        )

        # Test approve verdict
        base_state["test_verdict"] = "approve"
        assert router(base_state) == "success_node"
        mock_save_checkpoint.assert_not_called()

        # Test reject verdict
        base_state["test_verdict"] = "reject"
        assert router(base_state) == "retry_node"
        mock_save_checkpoint.assert_not_called()

    def test_router_respects_count_limits(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that router escalates when count limit is reached."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"

        # Under limit - should route to retry_node
        base_state["revision_count"] = 2
        assert router(base_state) == "retry_node"
        mock_save_checkpoint.assert_not_called()

        # At limit - should escalate to ask_user
        base_state["revision_count"] = 3

        with caplog.at_level(logging.WARNING):
            result = router(base_state)

        assert result == "ask_user"

        # Verify logging
        assert "revision_count=3 >= max_revisions=3" in caplog.text
        assert "escalating to ask_user" in caplog.text

        # Verify checkpoint
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args[0][1] == "before_ask_user_test_limit"

    def test_router_escalates_when_count_over_limit(
        self,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that router escalates when count is well over the limit."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 100  # Way over limit

        with caplog.at_level(logging.WARNING):
            result = router(base_state)

        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        # Verify log shows actual count and limit
        assert "revision_count=100" in caplog.text
        assert "max_revisions=3" in caplog.text

    def test_router_allows_count_just_under_limit(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that count just under limit proceeds normally."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 2  # One under limit

        result = router(base_state)

        assert result == "retry_node"
        mock_save_checkpoint.assert_not_called()

    def test_router_handles_count_field_missing_from_state(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that missing count field is treated as 0 (under limit)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        # Explicitly ensure count field doesn't exist
        if "revision_count" in base_state:
            del base_state["revision_count"]

        result = router(base_state)

        # Missing count should be treated as 0, which is under limit
        assert result == "retry_node"
        mock_save_checkpoint.assert_not_called()

    def test_router_uses_runtime_config_for_limits(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that router uses runtime_config limits when available."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 5

        # Set higher limit in runtime_config
        base_state["runtime_config"] = {"max_revisions": 10}

        # Should route to retry_node because config limit is 10
        assert router(base_state) == "retry_node"
        mock_save_checkpoint.assert_not_called()

    def test_router_handles_missing_runtime_config_gracefully(self, base_state):
        """Test that router handles missing runtime_config using default_max."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 5,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 4

        # Delete runtime_config if present
        if "runtime_config" in base_state:
            del base_state["runtime_config"]

        # Should use default_max=5, so 4 is OK
        assert router(base_state) == "retry_node"

    def test_router_fails_if_runtime_config_is_none(self, base_state):
        """
        Test behavior when runtime_config is explicitly None.
        This exposes a potential bug if the code assumes .get() on None works.
        """
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 5,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 4
        base_state["runtime_config"] = None  # Explicitly None

        try:
            result = router(base_state)
            assert result == "retry_node"
        except AttributeError:
            pytest.fail("Router crashed when runtime_config was None")
        except Exception as exc:
            pytest.fail(f"Router crashed with unexpected error: {exc}")

    def test_router_handles_none_count_gracefully(self, base_state, mock_save_checkpoint):
        """Test behavior when count field is None (should treat as 0)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = None  # Explicitly None

        # Should treat None as 0 and route to retry_node
        try:
            assert router(base_state) == "retry_node"
        except TypeError:
            pytest.fail("Router crashed when revision_count was None")

    def test_router_handles_unknown_verdict(self, base_state, mock_save_checkpoint, caplog):
        """Test that unknown verdict escalates to ask_user with warning."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "unknown_verdict"

        with caplog.at_level(logging.WARNING):
            result = router(base_state)

        assert result == "ask_user"

        # Verify logging
        assert "unknown_verdict" in caplog.text
        assert "not a recognized verdict" in caplog.text

        # Verify checkpoint
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args[0][1] == "before_ask_user_test_fallback"

    def test_pass_through_verdicts_skip_count_check(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that pass_through_verdicts skip count checking."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "pass": {
                    "route": "next_node",
                    "count_limit": {
                        "count_field": "fail_count",
                        "max_count_key": "max_fails",
                        "default_max": 1,
                    },
                },
            },
            checkpoint_prefix="test",
            pass_through_verdicts=["pass"],
        )

        base_state["test_verdict"] = "pass"
        base_state["fail_count"] = 100  # Way over limit

        # Should still route to next_node because "pass" is pass-through
        assert router(base_state) == "next_node"
        mock_save_checkpoint.assert_not_called()

    def test_custom_route_on_limit(self, base_state, mock_save_checkpoint):
        """Test that custom route_on_limit is respected."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 1,
                        "route_on_limit": "supervisor",  # Custom route
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 1  # At limit

        result = router(base_state)

        assert result == "supervisor"  # Not ask_user

        # Checkpoint should still be saved with "limit" in name
        mock_save_checkpoint.assert_called_once()
        assert "limit" in mock_save_checkpoint.call_args[0][1]

    def test_router_handles_empty_routes(self, base_state, mock_save_checkpoint):
        """Test router behavior with empty routes config."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={},
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "approve"

        # Should fall back to unknown verdict handling
        assert router(base_state) == "ask_user"
        mock_save_checkpoint.assert_called_once()
        assert "fallback" in mock_save_checkpoint.call_args[0][1]

    def test_router_handles_missing_route_key(self, base_state):
        """Test that missing 'route' key in config defaults to 'ask_user'."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {}},  # Empty config, missing "route"
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "approve"

        # Should default to ask_user
        assert router(base_state) == "ask_user"

    def test_router_cross_talk_isolation(self, base_state):
        """
        Test that router ignores other verdict fields.
        Ensures that if we set 'other_verdict' to something that would trigger routing,
        our router ignores it and only looks at 'test_verdict'.
        """
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "success_node"}},
            checkpoint_prefix="test",
        )

        # Set unrelated verdict
        base_state["other_verdict"] = "approve"
        base_state["test_verdict"] = "reject"  # Not in routes

        # Should fallback to ask_user because "reject" is not in routes
        # If it read "other_verdict", it might have returned "success_node"
        assert router(base_state) == "ask_user"

    def test_router_does_not_modify_state(self, base_state, mock_save_checkpoint):
        """Test that router doesn't modify the state object (except through save_checkpoint)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "approve": {"route": "success_node"},
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "approve"
        base_state["revision_count"] = 1
        base_state["runtime_config"] = {"max_revisions": 5}

        # Deep copy the state before routing
        state_snapshot = copy.deepcopy(dict(base_state))

        router(base_state)

        # Compare relevant fields - router should not modify state
        for key in state_snapshot:
            assert base_state.get(key) == state_snapshot[key], f"State field '{key}' was modified"

    def test_router_handles_negative_count(self, base_state, mock_save_checkpoint):
        """Test that negative count values are handled correctly (should be under limit)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = -1  # Negative count

        result = router(base_state)

        # Negative is less than limit, should route normally
        assert result == "retry_node"
        mock_save_checkpoint.assert_not_called()

    def test_router_handles_zero_count(self, base_state, mock_save_checkpoint):
        """Test that zero count is handled correctly."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 0

        result = router(base_state)

        assert result == "retry_node"
        mock_save_checkpoint.assert_not_called()

    def test_router_handles_zero_max_limit(self, base_state, mock_save_checkpoint):
        """Test that zero max limit escalates immediately even with zero count."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 0,  # No retries allowed
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 0

        result = router(base_state)

        # 0 >= 0 should trigger escalation
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_router_with_multiple_verdicts_and_different_limits(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test complex routing with multiple verdicts having different count limits."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "pass": {"route": "success_node"},
                "needs_code_fix": {
                    "route": "code_node",
                    "count_limit": {
                        "count_field": "code_fix_count",
                        "max_count_key": "max_code_fixes",
                        "default_max": 5,
                    },
                },
                "needs_design_fix": {
                    "route": "design_node",
                    "count_limit": {
                        "count_field": "design_fix_count",
                        "max_count_key": "max_design_fixes",
                        "default_max": 2,
                        "route_on_limit": "supervisor",
                    },
                },
            },
            checkpoint_prefix="test",
        )

        # Test pass verdict (no count limit)
        base_state["test_verdict"] = "pass"
        assert router(base_state) == "success_node"

        # Test code fix under limit
        base_state["test_verdict"] = "needs_code_fix"
        base_state["code_fix_count"] = 3
        assert router(base_state) == "code_node"

        # Test code fix at limit
        base_state["code_fix_count"] = 5
        assert router(base_state) == "ask_user"

        # Test design fix under limit
        mock_save_checkpoint.reset_mock()
        base_state["test_verdict"] = "needs_design_fix"
        base_state["design_fix_count"] = 1
        assert router(base_state) == "design_node"

        # Test design fix at limit (custom route_on_limit)
        base_state["design_fix_count"] = 2
        result = router(base_state)
        assert result == "supervisor", "Custom route_on_limit should be used"
        mock_save_checkpoint.assert_called()

    def test_pass_through_verdicts_with_none_list(self, base_state, mock_save_checkpoint):
        """Test that pass_through_verdicts=None is handled like empty list."""
        # Create router with explicit None for pass_through_verdicts
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "pass": {
                    "route": "next_node",
                    "count_limit": {
                        "count_field": "fail_count",
                        "max_count_key": "max_fails",
                        "default_max": 1,
                    },
                },
            },
            checkpoint_prefix="test",
            pass_through_verdicts=None,  # Explicit None
        )

        base_state["test_verdict"] = "pass"
        base_state["fail_count"] = 100  # Over limit

        # Without pass-through, should escalate
        assert router(base_state) == "ask_user"
        mock_save_checkpoint.assert_called_once()


class TestPreConfiguredRouters:
    """Tests for the pre-configured router instances exported by routing.py."""

    def test_route_after_plan_review_approve(self, base_state, mock_save_checkpoint):
        """Test plan_review router routes to select_stage on approve."""
        base_state["last_plan_review_verdict"] = "approve"
        result = route_after_plan_review(base_state)
        assert result == "select_stage"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_plan_review_needs_revision_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test plan_review router routes to plan when under replan limit."""
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 0
        result = route_after_plan_review(base_state)
        assert result == "planning"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_plan_review_needs_revision_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test plan_review router escalates when replan limit reached."""
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = MAX_REPLANS
        result = route_after_plan_review(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        assert "limit" in mock_save_checkpoint.call_args[0][1]

    def test_route_after_design_review_approve(self, base_state, mock_save_checkpoint):
        """Test design_review router routes to generate_code on approve."""
        base_state["last_design_review_verdict"] = "approve"
        result = route_after_design_review(base_state)
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_design_review_needs_revision_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test design_review router routes to design when under limit."""
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 0
        result = route_after_design_review(base_state)
        assert result == "design"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_design_review_needs_revision_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test design_review router escalates when limit reached."""
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        result = route_after_design_review(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_route_after_code_review_approve(self, base_state, mock_save_checkpoint):
        """Test code_review router routes to run_code on approve."""
        base_state["last_code_review_verdict"] = "approve"
        result = route_after_code_review(base_state)
        assert result == "run_code"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_code_review_needs_revision_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test code_review router routes to generate_code when under limit."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        result = route_after_code_review(base_state)
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_code_review_needs_revision_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test code_review router escalates when limit reached."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        result = route_after_code_review(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_route_after_execution_check_pass(self, base_state, mock_save_checkpoint):
        """Test execution_check router routes to physics_check on pass."""
        base_state["execution_verdict"] = "pass"
        base_state["execution_failure_count"] = 100  # Should be ignored (pass-through)
        result = route_after_execution_check(base_state)
        assert result == "physics_check"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_execution_check_warning(self, base_state, mock_save_checkpoint):
        """Test execution_check router routes to physics_check on warning."""
        base_state["execution_verdict"] = "warning"
        base_state["execution_failure_count"] = 100  # Should be ignored (pass-through)
        result = route_after_execution_check(base_state)
        assert result == "physics_check"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_execution_check_fail_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test execution_check router routes to generate_code on fail when under limit."""
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0
        result = route_after_execution_check(base_state)
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_execution_check_fail_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test execution_check router escalates on fail when at limit."""
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        result = route_after_execution_check(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_route_after_physics_check_pass(self, base_state, mock_save_checkpoint):
        """Test physics_check router routes to analyze on pass."""
        base_state["physics_verdict"] = "pass"
        result = route_after_physics_check(base_state)
        assert result == "analyze"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_physics_check_warning(self, base_state, mock_save_checkpoint):
        """Test physics_check router routes to analyze on warning."""
        base_state["physics_verdict"] = "warning"
        result = route_after_physics_check(base_state)
        assert result == "analyze"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_physics_check_fail_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test physics_check router routes to generate_code on fail when under limit."""
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = 0
        result = route_after_physics_check(base_state)
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_physics_check_fail_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test physics_check router escalates on fail when at limit."""
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        result = route_after_physics_check(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_route_after_physics_check_design_flaw_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test physics_check router routes to design on design_flaw when under limit."""
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0
        result = route_after_physics_check(base_state)
        assert result == "design"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_physics_check_design_flaw_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test physics_check router escalates on design_flaw when at limit."""
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        result = route_after_physics_check(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    def test_route_after_comparison_check_approve(self, base_state, mock_save_checkpoint):
        """Test comparison_check router routes to supervisor on approve."""
        base_state["comparison_verdict"] = "approve"
        result = route_after_comparison_check(base_state)
        assert result == "supervisor"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_comparison_check_needs_revision_under_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test comparison_check router routes to analyze when under limit."""
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 0
        result = route_after_comparison_check(base_state)
        assert result == "analyze"
        mock_save_checkpoint.assert_not_called()

    def test_route_after_comparison_check_needs_revision_at_limit(
        self, base_state, mock_save_checkpoint
    ):
        """Test comparison_check router routes to supervisor (not ask_user) when at limit."""
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS
        result = route_after_comparison_check(base_state)
        # Comparison check has route_on_limit="supervisor" (not ask_user)
        assert result == "supervisor"
        mock_save_checkpoint.assert_called_once()

    def test_preconfigured_routers_handle_none_verdict(
        self, base_state, mock_save_checkpoint, caplog
    ):
        """Test that all pre-configured routers handle None verdict correctly."""
        routers_and_fields = [
            (route_after_plan_review, "last_plan_review_verdict"),
            (route_after_design_review, "last_design_review_verdict"),
            (route_after_code_review, "last_code_review_verdict"),
            (route_after_execution_check, "execution_verdict"),
            (route_after_physics_check, "physics_verdict"),
            (route_after_comparison_check, "comparison_verdict"),
        ]

        for router, verdict_field in routers_and_fields:
            mock_save_checkpoint.reset_mock()
            caplog.clear()
            base_state[verdict_field] = None

            with caplog.at_level(logging.ERROR):
                result = router(base_state)

            assert result == "ask_user", f"{router.__name__} should return ask_user for None"
            mock_save_checkpoint.assert_called_once()
            assert "is None" in caplog.text

    def test_preconfigured_routers_respect_runtime_config(
        self, base_state, mock_save_checkpoint
    ):
        """Test that pre-configured routers use runtime_config for limits."""
        # Set a very high limit in runtime_config
        base_state["runtime_config"] = {"max_code_revisions": 100}
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 50  # Would exceed default but not runtime config

        result = route_after_code_review(base_state)

        # Should route normally because runtime_config limit is 100
        assert result == "generate_code"
        mock_save_checkpoint.assert_not_called()


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    def test_verdict_field_not_in_state_at_all(self, base_state, mock_save_checkpoint, caplog):
        """Test behavior when verdict field doesn't exist in state (should be None)."""
        router = create_verdict_router(
            verdict_field="nonexistent_verdict_field",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        # Don't set the verdict field at all
        with caplog.at_level(logging.ERROR):
            result = router(base_state)

        assert result == "ask_user"
        assert "is None" in caplog.text
        mock_save_checkpoint.assert_called_once()

    def test_empty_string_verdict(self, base_state, mock_save_checkpoint, caplog):
        """Test that empty string verdict is treated as unknown verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = ""

        with caplog.at_level(logging.WARNING):
            result = router(base_state)

        # Empty string is a string, but not a known verdict
        assert result == "ask_user"
        assert "not a recognized verdict" in caplog.text

    def test_whitespace_verdict(self, base_state, mock_save_checkpoint, caplog):
        """Test that whitespace-only verdict is treated as unknown verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "   "  # Only whitespace

        with caplog.at_level(logging.WARNING):
            result = router(base_state)

        # Whitespace is a string, but not a known verdict (not normalized)
        assert result == "ask_user"
        assert "not a recognized verdict" in caplog.text

    def test_case_sensitive_verdict_matching(self, base_state, mock_save_checkpoint):
        """Test that verdict matching is case-sensitive."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        # Test various case variations
        for verdict in ["Approve", "APPROVE", "ApPrOvE"]:
            base_state["test_verdict"] = verdict
            result = router(base_state)
            assert result == "ask_user", f"'{verdict}' should not match 'approve'"

        # Correct case should work
        base_state["test_verdict"] = "approve"
        result = router(base_state)
        assert result == "next_node"

    def test_verdict_with_leading_trailing_whitespace(
        self, base_state, mock_save_checkpoint
    ):
        """Test that verdict with whitespace doesn't match trimmed version."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        # Leading/trailing whitespace should NOT match
        for verdict in [" approve", "approve ", " approve "]:
            base_state["test_verdict"] = verdict
            result = router(base_state)
            assert result == "ask_user", f"'{verdict}' should not match 'approve'"

    def test_checkpoint_name_with_special_characters_in_prefix(
        self, base_state, mock_save_checkpoint
    ):
        """Test that checkpoint prefix with special chars is preserved."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test_with-special.chars",
        )

        base_state["test_verdict"] = None

        router(base_state)

        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert "test_with-special.chars" in checkpoint_name

    def test_route_with_very_long_name(self, base_state, mock_save_checkpoint):
        """Test that very long route names work correctly."""
        long_route_name = "a" * 1000
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": long_route_name}},
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "approve"
        result = router(base_state)

        assert result == long_route_name

    def test_route_config_with_extra_fields(self, base_state, mock_save_checkpoint):
        """Test that extra fields in route config don't break routing."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "approve": {
                    "route": "next_node",
                    "extra_field": "ignored",
                    "another_field": 123,
                }
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "approve"
        result = router(base_state)

        assert result == "next_node"

    def test_count_limit_with_missing_optional_fields(
        self, base_state, mock_save_checkpoint
    ):
        """Test count_limit with only required fields specified."""
        # Minimal count_limit config - only required fields
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        # Rely on defaults for everything
                    },
                },
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "needs_revision"

        # Should not crash, defaults should apply
        result = router(base_state)
        # Default count_field is "", which won't exist, so count=0
        # Default default_max is 3, so 0 < 3, should route normally
        assert result == "retry_node"

    def test_route_is_deterministic(self, base_state, mock_save_checkpoint):
        """Test that routing is deterministic (same input = same output)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "approve": {"route": "next_node"},
                "reject": {"route": "error_node"},
            },
            checkpoint_prefix="test",
        )

        base_state["test_verdict"] = "approve"

        # Call multiple times and verify same result
        results = [router(base_state) for _ in range(10)]
        assert all(r == "next_node" for r in results)

        base_state["test_verdict"] = "reject"
        results = [router(base_state) for _ in range(10)]
        assert all(r == "error_node" for r in results)

