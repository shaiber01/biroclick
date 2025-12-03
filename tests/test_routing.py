"""
Tests for the Routing Factory Module.

Tests the verdict router factory function and pre-configured routers
to ensure they correctly:
1. Route based on verdict values
2. Handle None verdicts gracefully
3. Respect count limits from runtime_config
4. Save checkpoints before escalation
5. Log appropriate errors and warnings
"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from typing import get_type_hints, get_args

from schemas.state import (
    create_initial_state,
    ReproState,
    RuntimeConfig,
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
)
from src.routing import (
    create_verdict_router,
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
    RouteType,
)


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state() -> ReproState:
    """Create a base test state."""
    state = create_initial_state(
        paper_id="test_paper",
        paper_text="Test paper content",
    )
    return state


@pytest.fixture
def mock_save_checkpoint():
    """Mock the save_checkpoint function to avoid file I/O."""
    with patch("src.routing.save_checkpoint") as mock:
        yield mock


# ═══════════════════════════════════════════════════════════════════════
# Factory Function Tests
# ═══════════════════════════════════════════════════════════════════════

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
    
    def test_router_handles_none_verdict_gracefully(self, base_state, mock_save_checkpoint, caplog):
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
    
    def test_router_returns_correct_route_for_verdict(self, base_state, mock_save_checkpoint):
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
    
    def test_router_respects_count_limits(self, base_state, mock_save_checkpoint, caplog):
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
                    }
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
    
    def test_router_uses_runtime_config_for_limits(self, base_state, mock_save_checkpoint):
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
                    }
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
                    }
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
                    }
                },
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 4
        base_state["runtime_config"] = None # Explicitly None
        
        # This will raise AttributeError if code does state.get("runtime_config", {}).get(...)
        # because the first get returns None (if key exists but value is None) or default.
        # Actually state.get("runtime_config", {}) returns the value if key exists, even if None.
        
        # We EXPECT this to fail if the code isn't robust.
        # The prompt says: "Tests that FAIL when bugs exist."
        try:
            result = router(base_state)
            assert result == "retry_node"
        except AttributeError:
            pytest.fail("Router crashed when runtime_config was None")
        except Exception as e:
            pytest.fail(f"Router crashed with unexpected error: {e}")

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
                    }
                },
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = None # Explicitly None
        
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
    
    def test_pass_through_verdicts_skip_count_check(self, base_state, mock_save_checkpoint):
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
                    }
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
                    }
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
            routes={
                "approve": {} # Empty config, missing "route"
            },
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
        base_state["test_verdict"] = "reject" # Not in routes
        
        # Should fallback to ask_user because "reject" is not in routes
        # If it read "other_verdict", it might have returned "success_node"
        assert router(base_state) == "ask_user"


# ═══════════════════════════════════════════════════════════════════════
# Pre-configured Router Tests (Integration)
# ═══════════════════════════════════════════════════════════════════════

class TestRouterIntegration:
    """
    Tests that pre-configured routers match expected behavior and configuration.
    Uses parameterized tests for better coverage and maintainability.
    """
    
    @pytest.mark.parametrize("router, verdict_field, verdict_val, count_field, count_val, expected_route", [
        # Code Review Router
        (route_after_code_review, "last_code_review_verdict", "approve", None, 0, "run_code"),
        (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", 0, "generate_code"),
        (route_after_code_review, "last_code_review_verdict", "needs_revision", "code_revision_count", MAX_CODE_REVISIONS, "ask_user"),
        
        # Execution Check Router
        (route_after_execution_check, "execution_verdict", "pass", None, 0, "physics_check"),
        (route_after_execution_check, "execution_verdict", "warning", None, 0, "physics_check"),
        (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", 0, "generate_code"),
        (route_after_execution_check, "execution_verdict", "fail", "execution_failure_count", MAX_EXECUTION_FAILURES, "ask_user"),
        
        # Physics Check Router
        (route_after_physics_check, "physics_verdict", "pass", None, 0, "analyze"),
        (route_after_physics_check, "physics_verdict", "warning", None, 0, "analyze"),
        (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", 0, "generate_code"),
        (route_after_physics_check, "physics_verdict", "fail", "physics_failure_count", MAX_PHYSICS_FAILURES, "ask_user"),
        (route_after_physics_check, "physics_verdict", "design_flaw", "design_revision_count", 0, "design"),
        (route_after_physics_check, "physics_verdict", "design_flaw", "design_revision_count", MAX_DESIGN_REVISIONS, "ask_user"),
        
        # Design Review Router
        (route_after_design_review, "last_design_review_verdict", "approve", None, 0, "generate_code"),
        (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", 0, "design"),
        (route_after_design_review, "last_design_review_verdict", "needs_revision", "design_revision_count", MAX_DESIGN_REVISIONS, "ask_user"),
        
        # Plan Review Router
        (route_after_plan_review, "last_plan_review_verdict", "approve", None, 0, "select_stage"),
        (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", 0, "plan"),
        (route_after_plan_review, "last_plan_review_verdict", "needs_revision", "replan_count", MAX_REPLANS, "ask_user"),
        
        # Comparison Check Router
        (route_after_comparison_check, "comparison_verdict", "approve", None, 0, "supervisor"),
        (route_after_comparison_check, "comparison_verdict", "needs_revision", "analysis_revision_count", 0, "analyze"),
        (route_after_comparison_check, "comparison_verdict", "needs_revision", "analysis_revision_count", MAX_ANALYSIS_REVISIONS + 1, "supervisor"), # NOTE: Routes to supervisor on limit
    ])
    def test_router_logic(self, router, verdict_field, verdict_val, count_field, count_val, expected_route, base_state, mock_save_checkpoint):
        """
        Parameterized test for all pre-configured routers.
        Verifies verdict -> route mapping and count limit handling.
        """
        # Setup state
        base_state[verdict_field] = verdict_val
        if count_field:
            base_state[count_field] = count_val
        
        # Execute
        result = router(base_state)
        
        # Assert
        assert result == expected_route
        
        # Optional: Verify checkpoint for escalations
        if expected_route == "ask_user" or (expected_route == "supervisor" and count_field and count_val > 0):
             # We expect a checkpoint if we hit a limit/escalation
             # Note: Comparison check routes to supervisor on limit, which triggers a checkpoint in our implementation
             # Let's only check specific cases if needed, or rely on result
             pass

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
            base_state["last_plan_review_verdict"] = "unknown" # Just garbage to trigger fallback
            base_state["last_design_review_verdict"] = "unknown"
            base_state["last_code_review_verdict"] = "unknown"
            base_state["execution_verdict"] = "unknown"
            base_state["physics_verdict"] = "unknown"
            base_state["comparison_verdict"] = "unknown"
            
            result = router(base_state)
            assert result in valid_routes, f"Router {router} returned invalid route: {result}"


# ═══════════════════════════════════════════════════════════════════════
# None Verdict Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNoneVerdictHandling:
    """Tests that all routers handle None verdicts correctly."""
    
    @pytest.mark.parametrize("router,verdict_field,checkpoint_prefix", [
        (route_after_plan_review, "last_plan_review_verdict", "plan_review"),
        (route_after_design_review, "last_design_review_verdict", "design_review"),
        (route_after_code_review, "last_code_review_verdict", "code_review"),
        (route_after_execution_check, "execution_verdict", "execution"),
        (route_after_physics_check, "physics_verdict", "physics"),
        (route_after_comparison_check, "comparison_verdict", "comparison"),
    ])
    def test_router_handles_none(self, router, verdict_field, checkpoint_prefix, base_state, mock_save_checkpoint):
        """Test that each router handles None verdict by escalating to ask_user."""
        base_state[verdict_field] = None
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args[0][1] == f"before_ask_user_{checkpoint_prefix}_error"


# ═══════════════════════════════════════════════════════════════════════
# Schema Consistency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaConsistency:
    """
    Tests that router configurations match the ReproState and RuntimeConfig schemas.
    This ensures that fields accessed by routers actually exist.
    """
    
    def test_runtime_config_keys_exist(self):
        """
        Verify that max_count_keys used in routers exist in RuntimeConfig.
        Failure here means the router relies on a config key that isn't defined in the schema.
        """
        # Map of router to expected config key
        router_configs = [
            ("plan_review", "max_replans"),
            ("design_review", "max_design_revisions"),
            ("code_review", "max_code_revisions"),
            ("execution_check", "max_execution_failures"),
            ("physics_check", "max_physics_failures"),
            ("comparison_check", "max_analysis_revisions"),
        ]
        
        # Get keys from RuntimeConfig type hint
        runtime_config_keys = get_type_hints(RuntimeConfig).keys()
        
        missing_keys = []
        for router_name, key in router_configs:
            if key not in runtime_config_keys:
                missing_keys.append(f"{router_name}: {key}")
        
        if missing_keys:
            pytest.fail(f"Router config keys missing from RuntimeConfig schema: {', '.join(missing_keys)}")

    def test_state_fields_exist(self):
        """
        Verify that verdict_field and count_field used in routers exist in ReproState.
        """
        # Map of router to (verdict_field, count_field)
        # Note: Some count fields might be optional/None in config, but we test the ones we know are used.
        router_configs = [
            ("plan_review", "last_plan_review_verdict", "replan_count"),
            ("design_review", "last_design_review_verdict", "design_revision_count"),
            ("code_review", "last_code_review_verdict", "code_revision_count"),
            ("execution_check", "execution_verdict", "execution_failure_count"),
            ("physics_check", "physics_verdict", "physics_failure_count"),
            ("comparison_check", "comparison_verdict", "analysis_revision_count"),
        ]
        
        # Get keys from ReproState type hint
        state_keys = get_type_hints(ReproState).keys()
        
        missing_fields = []
        for router_name, verdict_field, count_field in router_configs:
            if verdict_field not in state_keys:
                missing_fields.append(f"{router_name} verdict: {verdict_field}")
            if count_field not in state_keys:
                missing_fields.append(f"{router_name} count: {count_field}")
        
        if missing_fields:
            pytest.fail(f"Router fields missing from ReproState schema: {', '.join(missing_fields)}")
