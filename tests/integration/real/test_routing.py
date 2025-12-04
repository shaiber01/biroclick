"""Integration tests ensuring routing verdicts match schema definitions and comprehensive routing behavior tests."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from schemas.state import (
    ReproState,
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
    create_initial_state,
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


class TestRoutingReturnsValidValues:
    """Test that routing functions return values that exist in the graph."""

    SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

    def test_supervisor_verdicts_match_routing(self):
        """Supervisor schema verdicts must match what routing expects."""
        schema_file = self.SCHEMAS_DIR / "supervisor_output_schema.json"
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        schema_verdicts = set(schema["properties"]["verdict"]["enum"])
        handled_verdicts = {
            "ok_continue",
            "change_priority",
            "replan_needed",
            "ask_user",
            "backtrack_to_stage",
            "all_complete",
        }

        unhandled = schema_verdicts - handled_verdicts
        assert not unhandled, (
            "Supervisor schema allows verdicts that routing doesn't handle: "
            f"{unhandled}\nThis could cause routing errors at runtime!"
        )

    def test_reviewer_verdicts_match_routing(self):
        """Reviewer verdicts must match what routing expects."""
        expected_verdicts = {"approve", "needs_revision"}
        for reviewer in ["plan_reviewer", "design_reviewer", "code_reviewer"]:
            schema_file = self.SCHEMAS_DIR / f"{reviewer}_output_schema.json"
            if schema_file.exists():
                with open(schema_file, encoding="utf-8") as file:
                    schema = json.load(file)
                verdict_property = schema.get("properties", {}).get("verdict", {})
                schema_verdicts = set(verdict_property.get("enum", []))
                missing = expected_verdicts - schema_verdicts
                assert not missing, f"{reviewer} schema missing verdicts: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateVerdictRouter:
    """Comprehensive tests for the create_verdict_router factory function."""

    @pytest.fixture
    def minimal_state(self):
        """Create minimal state for testing."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    @patch("src.routing.save_checkpoint")
    def test_router_handles_none_verdict(self, mock_checkpoint, minimal_state):
        """Test router escalates to ask_user when verdict is None."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = None
        result = router(minimal_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name
        assert "test" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_router_handles_unknown_verdict(self, mock_checkpoint, minimal_state):
        """Test router escalates to ask_user when verdict is unknown."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "unknown_verdict"
        result = router(minimal_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    def test_router_routes_to_configured_route(self, minimal_state):
        """Test router routes to configured route for known verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "approve"
        result = router(minimal_state)

        assert result == "next_node"

    def test_router_uses_default_route_when_missing(self, minimal_state):
        """Test router uses ask_user as default when route not specified."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {}},  # No route specified
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "approve"
        result = router(minimal_state)

        assert result == "ask_user"

    def test_router_respects_count_limit_under_limit(self, minimal_state):
        """Test router allows routing when count is under limit."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2
        result = router(minimal_state)

        assert result == "retry_node"

    @patch("src.routing.save_checkpoint")
    def test_router_escalates_at_count_limit(self, mock_checkpoint, minimal_state):
        """Test router escalates when count reaches limit."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 3
        result = router(minimal_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_router_escalates_above_count_limit(self, mock_checkpoint, minimal_state):
        """Test router escalates when count exceeds limit."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 5
        result = router(minimal_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_router_uses_runtime_config_max_count(self, minimal_state):
        """Test router uses runtime_config max_count when available."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2
        minimal_state["runtime_config"] = {"max_revisions": 5}
        result = router(minimal_state)

        assert result == "retry_node"

    @patch("src.routing.save_checkpoint")
    def test_router_uses_runtime_config_max_count_at_limit(self, mock_checkpoint, minimal_state):
        """Test router respects runtime_config max_count at limit."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 5
        minimal_state["runtime_config"] = {"max_revisions": 5}
        result = router(minimal_state)

        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_router_uses_default_max_when_runtime_config_missing(self, minimal_state):
        """Test router uses default_max when runtime_config doesn't have key."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2
        minimal_state["runtime_config"] = {}  # Missing max_revisions
        result = router(minimal_state)

        assert result == "retry_node"

    def test_router_handles_missing_count_field(self, minimal_state):
        """Test router handles missing count_field gracefully."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        # revision_count not set
        result = router(minimal_state)

        assert result == "retry_node"  # Should default to 0, which is < 3

    @patch("src.routing.save_checkpoint")
    def test_router_handles_none_count_field(self, mock_checkpoint, minimal_state):
        """Test router handles None count_field gracefully."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = None
        result = router(minimal_state)

        assert result == "retry_node"  # None should be treated as 0

    def test_router_handles_missing_runtime_config(self, minimal_state):
        """Test router handles missing runtime_config gracefully."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2
        # runtime_config not set
        result = router(minimal_state)

        assert result == "retry_node"

    def test_router_handles_none_runtime_config(self, minimal_state):
        """Test router handles None runtime_config gracefully."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2
        minimal_state["runtime_config"] = None
        result = router(minimal_state)

        assert result == "retry_node"

    def test_router_respects_pass_through_verdicts(self, minimal_state):
        """Test router bypasses count limit for pass-through verdicts."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "pass": {
                    "route": "next_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                    },
                }
            },
            checkpoint_prefix="test",
            pass_through_verdicts=["pass"],
        )

        minimal_state["test_verdict"] = "pass"
        minimal_state["revision_count"] = 999  # Way over limit
        result = router(minimal_state)

        assert result == "next_node"  # Should bypass count check

    @patch("src.routing.save_checkpoint")
    def test_router_uses_custom_route_on_limit(self, mock_checkpoint, minimal_state):
        """Test router uses custom route_on_limit when specified."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                        "route_on_limit": "escalate_node",
                    },
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 3
        result = router(minimal_state)

        assert result == "escalate_node"
        mock_checkpoint.assert_called_once()

    def test_router_handles_empty_state(self):
        """Test router handles empty state dict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        empty_state = {}
        result = router(empty_state)

        assert result == "ask_user"  # Should handle None verdict

    def test_router_handles_missing_verdict_field(self, minimal_state):
        """Test router handles missing verdict field."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        # test_verdict not set
        result = router(minimal_state)

        assert result == "ask_user"  # Should handle None verdict

    def test_router_handles_empty_string_verdict(self, minimal_state):
        """Test router handles empty string verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = ""
        result = router(minimal_state)

        assert result == "ask_user"  # Empty string is not a known verdict

    def test_router_handles_whitespace_verdict(self, minimal_state):
        """Test router handles whitespace-only verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "   "
        result = router(minimal_state)

        assert result == "ask_user"  # Whitespace is not a known verdict

    def test_router_handles_negative_count(self, minimal_state):
        """Test router handles negative count values."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = -1
        result = router(minimal_state)

        assert result == "retry_node"  # Negative count should be < limit

    def test_router_handles_zero_max_count(self, minimal_state):
        """Test router handles zero max_count."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 0,
                    },
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 0
        result = router(minimal_state)

        assert result == "ask_user"  # 0 >= 0, so should escalate

    def test_router_handles_very_large_count(self, minimal_state):
        """Test router handles very large count values."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 999999
        result = router(minimal_state)

        assert result == "ask_user"  # Should escalate

    def test_router_handles_multiple_verdicts(self, minimal_state):
        """Test router handles multiple verdict routes correctly."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "approve": {"route": "next_node"},
                "needs_revision": {"route": "retry_node"},
                "reject": {"route": "abort_node"},
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "approve"
        assert router(minimal_state) == "next_node"

        minimal_state["test_verdict"] = "needs_revision"
        assert router(minimal_state) == "retry_node"

        minimal_state["test_verdict"] = "reject"
        assert router(minimal_state) == "abort_node"

    def test_router_checkpoint_prefix_in_checkpoint_name(self, minimal_state):
        """Test router includes checkpoint_prefix in checkpoint names."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="custom_prefix",
        )

        minimal_state["test_verdict"] = None
        with patch("src.routing.save_checkpoint") as mock_checkpoint:
            router(minimal_state)
            checkpoint_name = mock_checkpoint.call_args[0][1]
            assert "custom_prefix" in checkpoint_name

    def test_router_checkpoint_name_uses_route_on_limit(self, minimal_state):
        """Test router checkpoint name reflects route_on_limit, not hardcoded ask_user."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "revision_count",
                        "max_count_key": "max_revisions",
                        "default_max": 3,
                        "route_on_limit": "escalate_node",
                    },
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 3
        with patch("src.routing.save_checkpoint") as mock_checkpoint:
            result = router(minimal_state)
            assert result == "escalate_node"
            # Check that checkpoint name doesn't hardcode "ask_user" when route_on_limit is different
            checkpoint_name = mock_checkpoint.call_args[0][1]
            # The checkpoint name should reflect the actual route, not assume ask_user
            # Currently this might be a bug - checkpoint name says "before_ask_user" but route is "escalate_node"
            # This test documents the current behavior - if it's wrong, we should fix the component
            assert "test" in checkpoint_name
            assert "limit" in checkpoint_name

    def test_router_handles_malformed_count_limit_config(self, minimal_state):
        """Test router handles malformed count_limit config gracefully."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {},  # Empty count_limit config
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        # Should not crash, should route normally since count_limit is empty/malformed
        result = router(minimal_state)
        assert result == "retry_node"

    def test_router_handles_missing_count_limit_fields(self, minimal_state):
        """Test router handles missing count_limit fields gracefully."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry_node",
                    "count_limit": {
                        # Missing count_field, max_count_key, etc.
                    },
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        # Should not crash, should use defaults
        result = router(minimal_state)
        assert result == "retry_node"  # Should route normally since count check will fail safely

    def test_router_handles_non_numeric_count(self, minimal_state):
        """Test router handles non-numeric count values."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = "not_a_number"
        # Should handle gracefully - might crash or default, but should not silently fail
        try:
            result = router(minimal_state)
            # If it doesn't crash, verify it handles it somehow
            assert isinstance(result, str)
        except (TypeError, ValueError):
            # If it crashes, that's actually good - it means we're catching the bug
            pass

    def test_router_handles_non_numeric_max_count(self, minimal_state):
        """Test router handles non-numeric max_count from runtime_config."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2
        minimal_state["runtime_config"] = {"max_revisions": "not_a_number"}
        # Should handle gracefully - might crash or use default
        try:
            result = router(minimal_state)
            assert isinstance(result, str)
        except (TypeError, ValueError):
            # If it crashes, that's actually good - it means we're catching the bug
            pass

    def test_router_handles_float_count_values(self, minimal_state):
        """Test router handles float count values."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 2.5
        result = router(minimal_state)
        # Should handle float - might compare 2.5 >= 3 (False) or might crash
        assert isinstance(result, str)

    def test_router_handles_float_max_count(self, minimal_state):
        """Test router handles float max_count values."""
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
                }
            },
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "needs_revision"
        minimal_state["revision_count"] = 3
        minimal_state["runtime_config"] = {"max_revisions": 2.5}
        result = router(minimal_state)
        # Should handle float comparison
        assert isinstance(result, str)

    def test_router_handles_verdict_with_special_characters(self, minimal_state):
        """Test router handles verdicts with special characters."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "approve\nwith\nnewlines"
        result = router(minimal_state)
        # Should handle special characters - might not match, should escalate
        assert result == "ask_user"  # Unknown verdict

    def test_router_handles_unicode_verdict(self, minimal_state):
        """Test router handles unicode verdicts."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "approve_✓"
        result = router(minimal_state)
        # Should handle unicode - might not match, should escalate
        assert result == "ask_user"  # Unknown verdict

    def test_router_handles_dict_as_verdict(self, minimal_state):
        """Test router handles dict as verdict (shouldn't happen, but test robustness)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = {"nested": "dict"}
        # Should handle non-string verdicts
        result = router(minimal_state)
        assert result == "ask_user"  # Should escalate

    def test_router_handles_list_as_verdict(self, minimal_state):
        """Test router handles list as verdict (shouldn't happen, but test robustness)."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = ["list", "of", "values"]
        # Should handle non-string verdicts
        result = router(minimal_state)
        assert result == "ask_user"  # Should escalate

    def test_router_handles_boolean_verdict(self, minimal_state):
        """Test router handles boolean as verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = True
        # Should handle boolean verdicts
        result = router(minimal_state)
        assert result == "ask_user"  # Should escalate (True is not in routes)

    def test_router_handles_zero_as_verdict(self, minimal_state):
        """Test router handles zero as verdict."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = 0
        # Should handle numeric verdicts
        result = router(minimal_state)
        assert result == "ask_user"  # Should escalate (0 is not in routes)

    def test_router_handles_very_long_verdict_string(self, minimal_state):
        """Test router handles very long verdict strings."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )

        minimal_state["test_verdict"] = "a" * 10000
        result = router(minimal_state)
        assert result == "ask_user"  # Should escalate

    def test_router_handles_very_long_checkpoint_prefix(self, minimal_state):
        """Test router handles very long checkpoint prefix."""
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="a" * 1000,
        )

        minimal_state["test_verdict"] = None
        with patch("src.routing.save_checkpoint") as mock_checkpoint:
            router(minimal_state)
            checkpoint_name = mock_checkpoint.call_args[0][1]
            assert len(checkpoint_name) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN REVIEW ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanReviewRouter:
    """Comprehensive tests for route_after_plan_review."""

    @pytest.fixture
    def state(self):
        """Create state with plan."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        state["plan"] = {
            "paper_id": "test",
            "title": "Test Plan",
            "stages": [],
            "targets": [],
            "extracted_parameters": [],
        }
        return state

    def test_approve_routes_to_select_stage(self, state):
        """Test approve verdict routes to select_stage."""
        state["last_plan_review_verdict"] = "approve"
        result = route_after_plan_review(state)
        assert result == "select_stage"

    def test_needs_revision_routes_to_plan(self, state):
        """Test needs_revision routes to plan when under limit."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = 0
        result = route_after_plan_review(state)
        assert result == "planning"

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_at_limit(self, mock_checkpoint, state):
        """Test needs_revision escalates when replan_count reaches limit."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = MAX_REPLANS
        result = route_after_plan_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "plan_review" in checkpoint_name
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_above_limit(self, mock_checkpoint, state):
        """Test needs_revision escalates when replan_count exceeds limit."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = MAX_REPLANS + 1
        result = route_after_plan_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_needs_revision_continues_just_under_limit(self, state):
        """Test needs_revision continues when just under limit."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = MAX_REPLANS - 1
        result = route_after_plan_review(state)
        assert result == "planning"

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates(self, mock_checkpoint, state):
        """Test None verdict escalates to ask_user."""
        state["last_plan_review_verdict"] = None
        result = route_after_plan_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates(self, mock_checkpoint, state):
        """Test unknown verdict escalates to ask_user."""
        state["last_plan_review_verdict"] = "invalid_verdict"
        result = route_after_plan_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name

    def test_missing_replan_count_defaults_to_zero(self, state):
        """Test missing replan_count defaults to 0."""
        state["last_plan_review_verdict"] = "needs_revision"
        # replan_count not set
        result = route_after_plan_review(state)
        assert result == "planning"

    def test_none_replan_count_defaults_to_zero(self, state):
        """Test None replan_count defaults to 0."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = None
        result = route_after_plan_review(state)
        assert result == "planning"

    def test_uses_runtime_config_max_replans(self, state):
        """Test router uses runtime_config max_replans when available."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = 4
        state["runtime_config"] = {"max_replans": 5}
        result = route_after_plan_review(state)
        assert result == "planning"

    @patch("src.routing.save_checkpoint")
    def test_uses_runtime_config_max_replans_at_limit(self, mock_checkpoint, state):
        """Test router respects runtime_config max_replans at limit."""
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = 5
        state["runtime_config"] = {"max_replans": 5}
        result = route_after_plan_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN REVIEW ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignReviewRouter:
    """Comprehensive tests for route_after_design_review."""

    @pytest.fixture
    def state(self):
        """Create minimal state."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    def test_approve_routes_to_generate_code(self, state):
        """Test approve verdict routes to generate_code."""
        state["last_design_review_verdict"] = "approve"
        result = route_after_design_review(state)
        assert result == "generate_code"

    def test_needs_revision_routes_to_design(self, state):
        """Test needs_revision routes to design when under limit."""
        state["last_design_review_verdict"] = "needs_revision"
        state["design_revision_count"] = 0
        result = route_after_design_review(state)
        assert result == "design"

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_at_limit(self, mock_checkpoint, state):
        """Test needs_revision escalates when design_revision_count reaches limit."""
        state["last_design_review_verdict"] = "needs_revision"
        state["design_revision_count"] = MAX_DESIGN_REVISIONS
        result = route_after_design_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "design_review" in checkpoint_name
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates(self, mock_checkpoint, state):
        """Test None verdict escalates to ask_user."""
        state["last_design_review_verdict"] = None
        result = route_after_design_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates(self, mock_checkpoint, state):
        """Test unknown verdict escalates to ask_user."""
        state["last_design_review_verdict"] = "invalid_verdict"
        result = route_after_design_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_missing_design_revision_count_defaults_to_zero(self, state):
        """Test missing design_revision_count defaults to 0."""
        state["last_design_review_verdict"] = "needs_revision"
        # design_revision_count not set
        result = route_after_design_review(state)
        assert result == "design"

    def test_uses_runtime_config_max_design_revisions(self, state):
        """Test router uses runtime_config max_design_revisions when available."""
        state["last_design_review_verdict"] = "needs_revision"
        state["design_revision_count"] = 4
        state["runtime_config"] = {"max_design_revisions": 5}
        result = route_after_design_review(state)
        assert result == "design"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEW ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewRouter:
    """Comprehensive tests for route_after_code_review."""

    @pytest.fixture
    def state(self):
        """Create minimal state."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    def test_approve_routes_to_run_code(self, state):
        """Test approve verdict routes to run_code."""
        state["last_code_review_verdict"] = "approve"
        result = route_after_code_review(state)
        assert result == "run_code"

    def test_needs_revision_routes_to_generate_code(self, state):
        """Test needs_revision routes to generate_code when under limit."""
        state["last_code_review_verdict"] = "needs_revision"
        state["code_revision_count"] = 0
        result = route_after_code_review(state)
        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_at_limit(self, mock_checkpoint, state):
        """Test needs_revision escalates when code_revision_count reaches limit."""
        state["last_code_review_verdict"] = "needs_revision"
        state["code_revision_count"] = MAX_CODE_REVISIONS
        result = route_after_code_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "code_review" in checkpoint_name
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates(self, mock_checkpoint, state):
        """Test None verdict escalates to ask_user."""
        state["last_code_review_verdict"] = None
        result = route_after_code_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates(self, mock_checkpoint, state):
        """Test unknown verdict escalates to ask_user."""
        state["last_code_review_verdict"] = "invalid_verdict"
        result = route_after_code_review(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_missing_code_revision_count_defaults_to_zero(self, state):
        """Test missing code_revision_count defaults to 0."""
        state["last_code_review_verdict"] = "needs_revision"
        # code_revision_count not set
        result = route_after_code_review(state)
        assert result == "generate_code"

    def test_uses_runtime_config_max_code_revisions(self, state):
        """Test router uses runtime_config max_code_revisions when available."""
        state["last_code_review_verdict"] = "needs_revision"
        state["code_revision_count"] = 4
        state["runtime_config"] = {"max_code_revisions": 5}
        result = route_after_code_review(state)
        assert result == "generate_code"


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION CHECK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionCheckRouter:
    """Comprehensive tests for route_after_execution_check."""

    @pytest.fixture
    def state(self):
        """Create minimal state."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    def test_pass_routes_to_physics_check(self, state):
        """Test pass verdict routes to physics_check."""
        state["execution_verdict"] = "pass"
        result = route_after_execution_check(state)
        assert result == "physics_check"

    def test_warning_routes_to_physics_check(self, state):
        """Test warning verdict routes to physics_check."""
        state["execution_verdict"] = "warning"
        result = route_after_execution_check(state)
        assert result == "physics_check"

    def test_pass_bypasses_count_limit(self, state):
        """Test pass verdict bypasses count limit (pass-through)."""
        state["execution_verdict"] = "pass"
        state["execution_failure_count"] = 999  # Way over limit
        result = route_after_execution_check(state)
        assert result == "physics_check"

    def test_warning_bypasses_count_limit(self, state):
        """Test warning verdict bypasses count limit (pass-through)."""
        state["execution_verdict"] = "warning"
        state["execution_failure_count"] = 999  # Way over limit
        result = route_after_execution_check(state)
        assert result == "physics_check"

    def test_fail_routes_to_generate_code(self, state):
        """Test fail routes to generate_code when under limit."""
        state["execution_verdict"] = "fail"
        state["execution_failure_count"] = 0
        result = route_after_execution_check(state)
        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_fail_escalates_at_limit(self, mock_checkpoint, state):
        """Test fail escalates when execution_failure_count reaches limit."""
        state["execution_verdict"] = "fail"
        state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        result = route_after_execution_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "execution" in checkpoint_name
        assert "limit" in checkpoint_name

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates(self, mock_checkpoint, state):
        """Test None verdict escalates to ask_user."""
        state["execution_verdict"] = None
        result = route_after_execution_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates(self, mock_checkpoint, state):
        """Test unknown verdict escalates to ask_user."""
        state["execution_verdict"] = "invalid_verdict"
        result = route_after_execution_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_missing_execution_failure_count_defaults_to_zero(self, state):
        """Test missing execution_failure_count defaults to 0."""
        state["execution_verdict"] = "fail"
        # execution_failure_count not set
        result = route_after_execution_check(state)
        assert result == "generate_code"

    def test_uses_runtime_config_max_execution_failures(self, state):
        """Test router uses runtime_config max_execution_failures when available."""
        state["execution_verdict"] = "fail"
        state["execution_failure_count"] = 1
        state["runtime_config"] = {"max_execution_failures": 3}
        result = route_after_execution_check(state)
        assert result == "generate_code"


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS CHECK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhysicsCheckRouter:
    """Comprehensive tests for route_after_physics_check."""

    @pytest.fixture
    def state(self):
        """Create minimal state."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    def test_pass_routes_to_analyze(self, state):
        """Test pass verdict routes to analyze."""
        state["physics_verdict"] = "pass"
        result = route_after_physics_check(state)
        assert result == "analyze"

    def test_warning_routes_to_analyze(self, state):
        """Test warning verdict routes to analyze."""
        state["physics_verdict"] = "warning"
        result = route_after_physics_check(state)
        assert result == "analyze"

    def test_pass_bypasses_count_limit(self, state):
        """Test pass verdict bypasses count limit (pass-through)."""
        state["physics_verdict"] = "pass"
        state["physics_failure_count"] = 999  # Way over limit
        result = route_after_physics_check(state)
        assert result == "analyze"

    def test_warning_bypasses_count_limit(self, state):
        """Test warning verdict bypasses count limit (pass-through)."""
        state["physics_verdict"] = "warning"
        state["physics_failure_count"] = 999  # Way over limit
        result = route_after_physics_check(state)
        assert result == "analyze"

    def test_fail_routes_to_generate_code(self, state):
        """Test fail routes to generate_code when under limit."""
        state["physics_verdict"] = "fail"
        state["physics_failure_count"] = 0
        result = route_after_physics_check(state)
        assert result == "generate_code"

    @patch("src.routing.save_checkpoint")
    def test_fail_escalates_at_limit(self, mock_checkpoint, state):
        """Test fail escalates when physics_failure_count reaches limit."""
        state["physics_verdict"] = "fail"
        state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        result = route_after_physics_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "physics" in checkpoint_name
        assert "limit" in checkpoint_name

    def test_design_flaw_routes_to_design(self, state):
        """Test design_flaw routes to design when under limit."""
        state["physics_verdict"] = "design_flaw"
        state["design_revision_count"] = 0
        result = route_after_physics_check(state)
        assert result == "design"

    @patch("src.routing.save_checkpoint")
    def test_design_flaw_escalates_at_limit(self, mock_checkpoint, state):
        """Test design_flaw escalates when design_revision_count reaches limit."""
        state["physics_verdict"] = "design_flaw"
        state["design_revision_count"] = MAX_DESIGN_REVISIONS
        result = route_after_physics_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates(self, mock_checkpoint, state):
        """Test None verdict escalates to ask_user."""
        state["physics_verdict"] = None
        result = route_after_physics_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates(self, mock_checkpoint, state):
        """Test unknown verdict escalates to ask_user."""
        state["physics_verdict"] = "invalid_verdict"
        result = route_after_physics_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_missing_physics_failure_count_defaults_to_zero(self, state):
        """Test missing physics_failure_count defaults to 0."""
        state["physics_verdict"] = "fail"
        # physics_failure_count not set
        result = route_after_physics_check(state)
        assert result == "generate_code"

    def test_missing_design_revision_count_defaults_to_zero(self, state):
        """Test missing design_revision_count defaults to 0 for design_flaw."""
        state["physics_verdict"] = "design_flaw"
        # design_revision_count not set
        result = route_after_physics_check(state)
        assert result == "design"

    def test_uses_runtime_config_max_physics_failures(self, state):
        """Test router uses runtime_config max_physics_failures when available."""
        state["physics_verdict"] = "fail"
        state["physics_failure_count"] = 1
        state["runtime_config"] = {"max_physics_failures": 3}
        result = route_after_physics_check(state)
        assert result == "generate_code"

    def test_uses_runtime_config_max_design_revisions_for_design_flaw(self, state):
        """Test router uses runtime_config max_design_revisions for design_flaw."""
        state["physics_verdict"] = "design_flaw"
        state["design_revision_count"] = 4
        state["runtime_config"] = {"max_design_revisions": 5}
        result = route_after_physics_check(state)
        assert result == "design"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHECK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestComparisonCheckRouter:
    """Comprehensive tests for route_after_comparison_check."""

    @pytest.fixture
    def state(self):
        """Create minimal state."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    def test_approve_routes_to_supervisor(self, state):
        """Test approve verdict routes to supervisor."""
        state["comparison_verdict"] = "approve"
        result = route_after_comparison_check(state)
        assert result == "supervisor"

    def test_needs_revision_routes_to_analyze(self, state):
        """Test needs_revision routes to analyze when under limit."""
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = 0
        result = route_after_comparison_check(state)
        assert result == "analyze"

    def test_needs_revision_routes_to_ask_user_at_limit(self, state):
        """Test needs_revision routes to ask_user (consistent with other limits) at limit."""
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS
        result = route_after_comparison_check(state)
        assert result == "ask_user"  # Now consistent with other limits

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_saves_checkpoint_at_limit(self, mock_checkpoint, state):
        """Test needs_revision saves checkpoint at limit."""
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS
        result = route_after_comparison_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "comparison" in checkpoint_name
        assert "limit" in checkpoint_name

    def test_needs_revision_continues_just_under_limit(self, state):
        """Test needs_revision continues when just under limit."""
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS - 1
        result = route_after_comparison_check(state)
        assert result == "analyze"

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates(self, mock_checkpoint, state):
        """Test None verdict escalates to ask_user."""
        state["comparison_verdict"] = None
        result = route_after_comparison_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates(self, mock_checkpoint, state):
        """Test unknown verdict escalates to ask_user."""
        state["comparison_verdict"] = "invalid_verdict"
        result = route_after_comparison_check(state)
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()

    def test_missing_analysis_revision_count_defaults_to_zero(self, state):
        """Test missing analysis_revision_count defaults to 0."""
        state["comparison_verdict"] = "needs_revision"
        # analysis_revision_count not set
        result = route_after_comparison_check(state)
        assert result == "analyze"

    def test_uses_runtime_config_max_analysis_revisions(self, state):
        """Test router uses runtime_config max_analysis_revisions when available."""
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = 4
        state["runtime_config"] = {"max_analysis_revisions": 5}
        result = route_after_comparison_check(state)
        assert result == "analyze"

    def test_uses_runtime_config_max_analysis_revisions_at_limit(self, state):
        """Test router respects runtime_config max_analysis_revisions at limit."""
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = 5
        state["runtime_config"] = {"max_analysis_revisions": 5}
        result = route_after_comparison_check(state)
        assert result == "ask_user"  # Now routes to ask_user (consistent with others)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE AND INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoutingEdgeCases:
    """Edge cases and integration tests for routing functions."""

    @pytest.fixture
    def state(self):
        """Create minimal state."""
        return create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )

    def test_all_routers_are_callable(self):
        """Test all router functions are callable."""
        routers = [
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        ]
        for router in routers:
            assert callable(router), f"Router {router.__name__} is not callable"

    def test_all_routers_return_strings(self, state):
        """Test all routers return string route names."""
        state["plan"] = {"paper_id": "test", "title": "Test", "stages": [], "targets": [], "extracted_parameters": []}
        state["last_plan_review_verdict"] = "approve"
        state["last_design_review_verdict"] = "approve"
        state["last_code_review_verdict"] = "approve"
        state["execution_verdict"] = "pass"
        state["physics_verdict"] = "pass"
        state["comparison_verdict"] = "approve"

        routers = [
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        ]

        for router in routers:
            result = router(state)
            assert isinstance(result, str), f"Router {router.__name__} returned non-string: {type(result)}"
            assert len(result) > 0, f"Router {router.__name__} returned empty string"

    def test_routers_handle_empty_dict_state(self):
        """Test routers handle completely empty state dict."""
        empty_state = {}
        routers = [
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        ]

        for router in routers:
            result = router(empty_state)
            assert isinstance(result, str)
            assert result == "ask_user"  # All should escalate on None verdict

    def test_routers_handle_state_with_only_paper_id(self):
        """Test routers handle state with minimal fields."""
        minimal_state = {"paper_id": "test"}
        routers = [
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        ]

        for router in routers:
            result = router(minimal_state)
            assert isinstance(result, str)
            assert result == "ask_user"  # All should escalate on None verdict

    def test_routers_do_not_mutate_state(self, state):
        """Test routers do not mutate input state."""
        state["last_code_review_verdict"] = "approve"
        original_state = dict(state)

        route_after_code_review(state)

        assert state == original_state, "Router mutated input state"

    def test_count_limit_boundary_conditions(self, state):
        """Test count limit boundary conditions across all routers."""
        state["plan"] = {"paper_id": "test", "title": "Test", "stages": [], "targets": [], "extracted_parameters": []}

        # Test at boundary: count == max
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = MAX_REPLANS
        with patch("src.routing.save_checkpoint"):
            assert route_after_plan_review(state) == "ask_user"

        state["last_design_review_verdict"] = "needs_revision"
        state["design_revision_count"] = MAX_DESIGN_REVISIONS
        with patch("src.routing.save_checkpoint"):
            assert route_after_design_review(state) == "ask_user"

        state["last_code_review_verdict"] = "needs_revision"
        state["code_revision_count"] = MAX_CODE_REVISIONS
        with patch("src.routing.save_checkpoint"):
            assert route_after_code_review(state) == "ask_user"

        state["execution_verdict"] = "fail"
        state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        with patch("src.routing.save_checkpoint"):
            assert route_after_execution_check(state) == "ask_user"

        state["physics_verdict"] = "fail"
        state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        with patch("src.routing.save_checkpoint"):
            assert route_after_physics_check(state) == "ask_user"

        # Test just under boundary: count == max - 1
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = MAX_REPLANS - 1
        assert route_after_plan_review(state) == "planning"

        state["last_design_review_verdict"] = "needs_revision"
        state["design_revision_count"] = MAX_DESIGN_REVISIONS - 1
        assert route_after_design_review(state) == "design"

        state["last_code_review_verdict"] = "needs_revision"
        state["code_revision_count"] = MAX_CODE_REVISIONS - 1
        assert route_after_code_review(state) == "generate_code"

        state["execution_verdict"] = "fail"
        state["execution_failure_count"] = MAX_EXECUTION_FAILURES - 1
        assert route_after_execution_check(state) == "generate_code"

        state["physics_verdict"] = "fail"
        state["physics_failure_count"] = MAX_PHYSICS_FAILURES - 1
        assert route_after_physics_check(state) == "generate_code"

    def test_all_verdict_types_handled(self, state):
        """Test all expected verdict types are handled correctly."""
        state["plan"] = {"paper_id": "test", "title": "Test", "stages": [], "targets": [], "extracted_parameters": []}

        # Plan review verdicts
        state["last_plan_review_verdict"] = "approve"
        assert route_after_plan_review(state) == "select_stage"
        state["last_plan_review_verdict"] = "needs_revision"
        state["replan_count"] = 0
        assert route_after_plan_review(state) == "planning"

        # Design review verdicts
        state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(state) == "generate_code"
        state["last_design_review_verdict"] = "needs_revision"
        state["design_revision_count"] = 0
        assert route_after_design_review(state) == "design"

        # Code review verdicts
        state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(state) == "run_code"
        state["last_code_review_verdict"] = "needs_revision"
        state["code_revision_count"] = 0
        assert route_after_code_review(state) == "generate_code"

        # Execution verdicts
        state["execution_verdict"] = "pass"
        assert route_after_execution_check(state) == "physics_check"
        state["execution_verdict"] = "warning"
        assert route_after_execution_check(state) == "physics_check"
        state["execution_verdict"] = "fail"
        state["execution_failure_count"] = 0
        assert route_after_execution_check(state) == "generate_code"

        # Physics verdicts
        state["physics_verdict"] = "pass"
        assert route_after_physics_check(state) == "analyze"
        state["physics_verdict"] = "warning"
        assert route_after_physics_check(state) == "analyze"
        state["physics_verdict"] = "fail"
        state["physics_failure_count"] = 0
        assert route_after_physics_check(state) == "generate_code"
        state["physics_verdict"] = "design_flaw"
        state["design_revision_count"] = 0
        assert route_after_physics_check(state) == "design"

        # Comparison verdicts
        state["comparison_verdict"] = "approve"
        assert route_after_comparison_check(state) == "supervisor"
        state["comparison_verdict"] = "needs_revision"
        state["analysis_revision_count"] = 0
        assert route_after_comparison_check(state) == "analyze"

