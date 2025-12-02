"""
Tests for the Routing Factory Module.

Tests the verdict router factory function and pre-configured routers
to ensure they correctly:
1. Route based on verdict values
2. Handle None verdicts gracefully
3. Respect count limits from runtime_config
4. Save checkpoints before escalation
"""

import pytest
from unittest.mock import patch, MagicMock

from schemas.state import (
    create_initial_state,
    ReproState,
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
)
from src.routing import (
    create_verdict_router,
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
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
    
    def test_router_handles_none_verdict_gracefully(self, base_state, mock_save_checkpoint):
        """Test that None verdict escalates to ask_user with checkpoint."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        # Verdict is None (not set)
        base_state["test_verdict"] = None
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        # Check checkpoint name contains "error"
        call_args = mock_save_checkpoint.call_args
        assert "error" in call_args[0][1]
    
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
        
        # Test reject verdict
        base_state["test_verdict"] = "reject"
        assert router(base_state) == "retry_node"
    
    def test_router_respects_count_limits(self, base_state, mock_save_checkpoint):
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
        
        # At limit - should escalate to ask_user
        base_state["revision_count"] = 3
        result = router(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.assert_called()
    
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
    
    def test_router_saves_checkpoint_on_escalation(self, base_state, mock_save_checkpoint):
        """Test that router saves checkpoint before escalating."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 1,
                    }
                },
            },
            checkpoint_prefix="myprefix",
        )
        
        base_state["test_verdict"] = "needs_revision"
        base_state["revision_count"] = 1  # At limit
        
        router(base_state)
        
        mock_save_checkpoint.assert_called_once()
        call_args = mock_save_checkpoint.call_args[0]
        assert call_args[0] == base_state  # First arg is state
        assert "myprefix" in call_args[1]  # Checkpoint name contains prefix
        assert "limit" in call_args[1]  # Checkpoint name indicates limit reached
    
    def test_router_handles_unknown_verdict(self, base_state, mock_save_checkpoint):
        """Test that unknown verdict escalates to ask_user."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "unknown_verdict"
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        call_args = mock_save_checkpoint.call_args[0]
        assert "fallback" in call_args[1]
    
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


# ═══════════════════════════════════════════════════════════════════════
# Pre-configured Router Tests (Integration)
# ═══════════════════════════════════════════════════════════════════════

class TestRouterIntegration:
    """Tests that pre-configured routers match expected behavior."""
    
    def test_code_review_router_approve(self, base_state, mock_save_checkpoint):
        """Test code review router routes to run_code on approve."""
        base_state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(base_state) == "run_code"
    
    def test_code_review_router_needs_revision(self, base_state, mock_save_checkpoint):
        """Test code review router routes to generate_code on needs_revision."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        assert route_after_code_review(base_state) == "generate_code"
    
    def test_code_review_router_revision_limit(self, base_state, mock_save_checkpoint):
        """Test code review router escalates on revision limit."""
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        assert route_after_code_review(base_state) == "ask_user"
    
    def test_execution_check_router_pass(self, base_state, mock_save_checkpoint):
        """Test execution check router routes to physics_check on pass."""
        base_state["execution_verdict"] = "pass"
        assert route_after_execution_check(base_state) == "physics_check"
    
    def test_execution_check_router_warning(self, base_state, mock_save_checkpoint):
        """Test execution check router routes to physics_check on warning."""
        base_state["execution_verdict"] = "warning"
        assert route_after_execution_check(base_state) == "physics_check"
    
    def test_execution_check_router_fail(self, base_state, mock_save_checkpoint):
        """Test execution check router routes to generate_code on fail."""
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0
        assert route_after_execution_check(base_state) == "generate_code"
    
    def test_execution_check_router_failure_limit(self, base_state, mock_save_checkpoint):
        """Test execution check router escalates on failure limit."""
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        assert route_after_execution_check(base_state) == "ask_user"
    
    def test_physics_check_router_design_flaw(self, base_state, mock_save_checkpoint):
        """Test physics check router routes to design on design_flaw verdict."""
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0
        assert route_after_physics_check(base_state) == "design"
    
    def test_physics_check_router_design_flaw_limit(self, base_state, mock_save_checkpoint):
        """Test physics check router escalates design_flaw at design limit."""
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        assert route_after_physics_check(base_state) == "ask_user"
    
    def test_design_review_router_approve(self, base_state, mock_save_checkpoint):
        """Test design review router routes to generate_code on approve."""
        base_state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(base_state) == "generate_code"
    
    def test_plan_review_router_approve(self, base_state, mock_save_checkpoint):
        """Test plan review router routes to select_stage on approve."""
        base_state["last_plan_review_verdict"] = "approve"
        assert route_after_plan_review(base_state) == "select_stage"
    
    def test_comparison_check_router_limit_routes_to_supervisor(self, base_state, mock_save_checkpoint):
        """Test comparison check router routes to supervisor (not ask_user) on limit."""
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 10  # Over any reasonable limit
        # Comparison router has custom route_on_limit: "supervisor"
        assert route_after_comparison_check(base_state) == "supervisor"


# ═══════════════════════════════════════════════════════════════════════
# None Verdict Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNoneVerdictHandling:
    """Tests that all routers handle None verdicts correctly."""
    
    @pytest.mark.parametrize("router,verdict_field", [
        (route_after_plan_review, "last_plan_review_verdict"),
        (route_after_design_review, "last_design_review_verdict"),
        (route_after_code_review, "last_code_review_verdict"),
        (route_after_execution_check, "execution_verdict"),
        (route_after_physics_check, "physics_verdict"),
        (route_after_comparison_check, "comparison_verdict"),
    ])
    def test_router_handles_none(self, router, verdict_field, base_state, mock_save_checkpoint):
        """Test that each router handles None verdict by escalating to ask_user."""
        base_state[verdict_field] = None
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()



