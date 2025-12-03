"""Unit tests for the routing verdict router factory."""

import logging

import pytest

from src.routing import create_verdict_router


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

