"""Regression tests that ensure None verdicts escalate properly.

These tests specifically verify that None verdict handling:
1. Returns 'ask_user' for all routers
2. Saves checkpoints with correct naming
3. Logs appropriate error messages
4. Takes precedence over other routing logic (count limits, etc.)
5. Handles edge cases correctly (missing fields, isolation)
"""

import logging
import pytest

from src.routing import (
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
    def test_router_handles_none_verdict_by_escalating(
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

        # Must return ask_user for None verdict
        assert result == "ask_user", (
            f"Router for {verdict_field} should return 'ask_user' for None verdict, "
            f"got '{result}'"
        )
        
        # Must save checkpoint exactly once
        mock_save_checkpoint.assert_called_once()
        
        # Verify exact checkpoint name format
        expected_checkpoint_name = f"before_ask_user_{checkpoint_prefix}_error"
        actual_checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert actual_checkpoint_name == expected_checkpoint_name, (
            f"Expected checkpoint name '{expected_checkpoint_name}', "
            f"got '{actual_checkpoint_name}'"
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
    def test_router_logs_error_for_none_verdict(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that None verdict logs an ERROR with informative message."""
        base_state[verdict_field] = None

        with caplog.at_level(logging.ERROR):
            router(base_state)

        # Verify error was logged
        assert len(caplog.records) >= 1, "Expected at least one log record"
        
        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) >= 1, "Expected at least one ERROR level log"
        
        # Verify log message contains essential information
        log_text = caplog.text.lower()
        assert verdict_field.lower() in log_text, (
            f"Error log should mention the verdict field '{verdict_field}'"
        )
        assert "none" in log_text, "Error log should mention 'None'"
        assert "escalat" in log_text, (
            "Error log should mention escalation (e.g., 'Escalating to user')"
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
    def test_router_passes_correct_state_to_checkpoint(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that checkpoint receives the exact same state object."""
        base_state[verdict_field] = None

        router(base_state)

        # Verify checkpoint was called with the same state object (identity check)
        mock_save_checkpoint.assert_called_once()
        checkpoint_state = mock_save_checkpoint.call_args[0][0]
        assert checkpoint_state is base_state, (
            "Checkpoint should receive the exact same state object, not a copy"
        )


class TestNoneVerdictPrecedence:
    """Tests that None verdict handling takes precedence over other routing logic."""

    @pytest.mark.parametrize(
        "router, verdict_field, count_field, max_count",
        [
            (route_after_plan_review, "last_plan_review_verdict", "replan_count", MAX_REPLANS),
            (route_after_design_review, "last_design_review_verdict", "design_revision_count", MAX_DESIGN_REVISIONS),
            (route_after_code_review, "last_code_review_verdict", "code_revision_count", MAX_CODE_REVISIONS),
            (route_after_execution_check, "execution_verdict", "execution_failure_count", MAX_EXECUTION_FAILURES),
            (route_after_physics_check, "physics_verdict", "physics_failure_count", MAX_PHYSICS_FAILURES),
            (route_after_comparison_check, "comparison_verdict", "analysis_revision_count", MAX_ANALYSIS_REVISIONS),
        ],
    )
    def test_none_verdict_checked_before_count_limits(
        self,
        router,
        verdict_field,
        count_field,
        max_count,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None verdict check happens BEFORE count limit checks.
        
        This ensures that even if the count limit would trigger escalation,
        the None verdict is detected and reported correctly.
        """
        base_state[verdict_field] = None
        base_state[count_field] = max_count + 100  # Way over limit

        result = router(base_state)

        # Should still get ask_user from None handling, not from count limit
        assert result == "ask_user"
        
        # Checkpoint name should indicate "error" (None verdict), not "limit"
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name, (
            f"Checkpoint name should contain 'error' for None verdict, "
            f"got '{checkpoint_name}'. This suggests None check didn't happen first."
        )
        assert "limit" not in checkpoint_name, (
            f"Checkpoint name should NOT contain 'limit' when verdict is None, "
            f"got '{checkpoint_name}'"
        )


class TestMissingVerdictField:
    """Tests for when verdict field is completely missing from state (not just None)."""

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
    def test_router_handles_missing_verdict_field(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test router behavior when verdict field doesn't exist in state at all.
        
        This is different from explicit None - the key is completely absent.
        Should be treated the same as None (escalate to ask_user).
        """
        # Explicitly remove the verdict field if it exists
        if verdict_field in base_state:
            del base_state[verdict_field]
        
        # Also ensure it's not there via dict access
        assert verdict_field not in base_state, f"Field {verdict_field} should not exist in state"

        with caplog.at_level(logging.ERROR):
            result = router(base_state)

        # Missing field should be treated as None
        assert result == "ask_user", (
            f"Missing verdict field should escalate to 'ask_user', got '{result}'"
        )
        
        # Verify checkpoint was saved with error suffix
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert checkpoint_name == f"before_ask_user_{checkpoint_prefix}_error", (
            f"Expected checkpoint name 'before_ask_user_{checkpoint_prefix}_error', "
            f"got '{checkpoint_name}'"
        )


class TestNoneVerdictIsolation:
    """Tests that None verdict in one field doesn't affect routing for other fields."""

    def test_plan_review_none_doesnt_affect_other_verdicts(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None in plan_review doesn't prevent other routers from working."""
        # Set plan_review to None
        base_state["last_plan_review_verdict"] = None
        
        # But set other verdicts to valid values
        base_state["last_design_review_verdict"] = "approve"
        base_state["last_code_review_verdict"] = "approve"
        base_state["execution_verdict"] = "pass"
        base_state["physics_verdict"] = "pass"
        base_state["comparison_verdict"] = "approve"

        # Plan review should return ask_user
        result = route_after_plan_review(base_state)
        assert result == "ask_user"
        mock_save_checkpoint.reset_mock()

        # Other routers should work normally
        assert route_after_design_review(base_state) == "generate_code"
        assert route_after_code_review(base_state) == "run_code"
        assert route_after_execution_check(base_state) == "physics_check"
        assert route_after_physics_check(base_state) == "analyze"
        assert route_after_comparison_check(base_state) == "supervisor"

    def test_each_router_only_checks_its_own_verdict(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that each router only looks at its designated verdict field."""
        # Set all verdicts to valid values
        base_state["last_plan_review_verdict"] = "approve"
        base_state["last_design_review_verdict"] = "approve"
        base_state["last_code_review_verdict"] = "approve"
        base_state["execution_verdict"] = "pass"
        base_state["physics_verdict"] = "pass"
        base_state["comparison_verdict"] = "approve"

        # Test each router with only ITS verdict as None
        routers_and_fields = [
            (route_after_plan_review, "last_plan_review_verdict", "select_stage"),
            (route_after_design_review, "last_design_review_verdict", "generate_code"),
            (route_after_code_review, "last_code_review_verdict", "run_code"),
            (route_after_execution_check, "execution_verdict", "physics_check"),
            (route_after_physics_check, "physics_verdict", "analyze"),
            (route_after_comparison_check, "comparison_verdict", "supervisor"),
        ]

        for router, verdict_field, expected_route_when_valid in routers_and_fields:
            # Reset to valid state
            base_state["last_plan_review_verdict"] = "approve"
            base_state["last_design_review_verdict"] = "approve"
            base_state["last_code_review_verdict"] = "approve"
            base_state["execution_verdict"] = "pass"
            base_state["physics_verdict"] = "pass"
            base_state["comparison_verdict"] = "approve"
            mock_save_checkpoint.reset_mock()

            # First verify router works with valid verdict
            result = router(base_state)
            assert result == expected_route_when_valid, (
                f"Router for {verdict_field} should return '{expected_route_when_valid}' "
                f"with valid verdict, got '{result}'"
            )
            mock_save_checkpoint.assert_not_called()

            # Now set only this verdict to None
            base_state[verdict_field] = None
            result = router(base_state)
            assert result == "ask_user", (
                f"Router for {verdict_field} should return 'ask_user' when its "
                f"verdict is None, got '{result}'"
            )
            mock_save_checkpoint.assert_called_once()


class TestInvalidVerdictTypes:
    """Tests that non-string verdict types are handled correctly (similar to None)."""

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
    @pytest.mark.parametrize(
        "invalid_verdict,type_name",
        [
            (123, "int"),
            (45.67, "float"),
            (True, "bool"),
            (False, "bool"),
            (["approve"], "list"),
            ({"verdict": "approve"}, "dict"),
            (("approve",), "tuple"),
            (set(["approve"]), "set"),
        ],
    )
    def test_router_handles_non_string_verdict(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        invalid_verdict,
        type_name,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that non-string verdicts are rejected and escalated to ask_user."""
        base_state[verdict_field] = invalid_verdict

        with caplog.at_level(logging.ERROR):
            result = router(base_state)

        # Should escalate to ask_user
        assert result == "ask_user", (
            f"Non-string verdict ({type_name}) should escalate to 'ask_user', "
            f"got '{result}'"
        )
        
        # Verify checkpoint was saved
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        
        # Checkpoint should have "error" suffix (same as None)
        assert "error" in checkpoint_name, (
            f"Checkpoint for invalid type should contain 'error', "
            f"got '{checkpoint_name}'"
        )
        
        # Verify error log mentions the type
        assert type_name in caplog.text.lower() or "invalid type" in caplog.text.lower(), (
            f"Error log should mention the invalid type '{type_name}'"
        )


class TestUnknownVerdictVsNone:
    """Tests distinguishing between None verdict and unknown string verdict."""

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
    def test_none_vs_unknown_string_have_different_checkpoints(
        self,
        router,
        verdict_field,
        checkpoint_prefix,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None and unknown string verdicts have different checkpoint names.
        
        None verdict should have "_error" suffix.
        Unknown string verdict should have "_fallback" suffix.
        This distinction is important for debugging.
        """
        # Test None verdict
        base_state[verdict_field] = None
        router(base_state)
        
        none_checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert none_checkpoint_name == f"before_ask_user_{checkpoint_prefix}_error"
        
        mock_save_checkpoint.reset_mock()
        
        # Test unknown string verdict
        base_state[verdict_field] = "unknown_garbage_verdict"
        router(base_state)
        
        unknown_checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert unknown_checkpoint_name == f"before_ask_user_{checkpoint_prefix}_fallback", (
            f"Unknown string verdict should use '_fallback' suffix, "
            f"got '{unknown_checkpoint_name}'"
        )
        
        # Verify they're different
        assert none_checkpoint_name != unknown_checkpoint_name, (
            "None and unknown string should have different checkpoint names"
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
    def test_none_logs_error_unknown_logs_warning(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
        caplog,
    ):
        """Test that None logs ERROR while unknown string logs WARNING.
        
        This distinction helps identify the severity of the issue:
        - None = likely a bug or failed node
        - Unknown string = misconfigured verdict value
        """
        # Test None verdict - should log ERROR
        base_state[verdict_field] = None
        with caplog.at_level(logging.DEBUG):
            router(base_state)
        
        none_errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(none_errors) >= 1, "None verdict should log at ERROR level"
        
        caplog.clear()
        mock_save_checkpoint.reset_mock()
        
        # Test unknown string verdict - should log WARNING
        base_state[verdict_field] = "unknown_garbage_verdict"
        with caplog.at_level(logging.DEBUG):
            router(base_state)
        
        unknown_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(unknown_warnings) >= 1, "Unknown string verdict should log at WARNING level"


class TestNoneVerdictWithRuntimeConfig:
    """Tests None verdict handling interacts correctly with runtime_config."""

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
    def test_none_verdict_ignores_runtime_config_limits(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None verdict handling doesn't depend on runtime_config.
        
        Even if runtime_config specifies very high limits, None should
        still escalate immediately.
        """
        base_state[verdict_field] = None
        base_state["runtime_config"] = {
            "max_replans": 1000,
            "max_design_revisions": 1000,
            "max_code_revisions": 1000,
            "max_execution_failures": 1000,
            "max_physics_failures": 1000,
            "max_analysis_revisions": 1000,
        }

        result = router(base_state)

        assert result == "ask_user", (
            "None verdict should escalate regardless of runtime_config limits"
        )
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name

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
    def test_none_verdict_with_missing_runtime_config(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None verdict handling works when runtime_config is missing."""
        base_state[verdict_field] = None
        if "runtime_config" in base_state:
            del base_state["runtime_config"]

        result = router(base_state)

        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

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
    def test_none_verdict_with_none_runtime_config(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that None verdict handling works when runtime_config is explicitly None."""
        base_state[verdict_field] = None
        base_state["runtime_config"] = None

        result = router(base_state)

        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()


class TestSpecificRouterNoneHandling:
    """Tests for specific router behaviors with None verdict."""

    def test_execution_check_none_with_pass_through_context(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test execution_check router handles None despite having pass_through_verdicts.
        
        The execution_check router has pass_through_verdicts=["pass", "warning"].
        Verify that None doesn't somehow get treated as pass-through.
        """
        base_state["execution_verdict"] = None
        # Set failure count very high - shouldn't matter for None
        base_state["execution_failure_count"] = 1000

        result = route_after_execution_check(base_state)

        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert checkpoint_name == "before_ask_user_execution_error"

    def test_physics_check_none_with_pass_through_context(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test physics_check router handles None despite having pass_through_verdicts.
        
        The physics_check router has pass_through_verdicts=["pass", "warning"].
        Verify that None doesn't somehow get treated as pass-through.
        """
        base_state["physics_verdict"] = None
        # Set failure count very high - shouldn't matter for None
        base_state["physics_failure_count"] = 1000

        result = route_after_physics_check(base_state)

        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert checkpoint_name == "before_ask_user_physics_error"

    def test_comparison_check_none_uses_ask_user_not_supervisor(
        self,
        base_state,
        mock_save_checkpoint,
    ):
        """Test comparison_check router uses ask_user for None, not supervisor.
        
        The comparison_check router has route_on_limit="supervisor" for count limits.
        But None should still route to ask_user, not supervisor.
        """
        base_state["comparison_verdict"] = None
        # Set revision count very high - shouldn't matter for None
        base_state["analysis_revision_count"] = 1000

        result = route_after_comparison_check(base_state)

        assert result == "ask_user", (
            "None verdict should route to 'ask_user', not the custom "
            "'route_on_limit' value of 'supervisor'"
        )
        mock_save_checkpoint.assert_called_once()
        checkpoint_name = mock_save_checkpoint.call_args[0][1]
        assert checkpoint_name == "before_ask_user_comparison_error"
        assert "limit" not in checkpoint_name


class TestNoneVerdictReturnValueStability:
    """Tests that None verdict handling returns consistent results."""

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
    def test_none_verdict_is_deterministic(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that calling router multiple times with None gives same result."""
        base_state[verdict_field] = None

        # Call multiple times and verify consistency
        results = []
        for _ in range(5):
            mock_save_checkpoint.reset_mock()
            result = router(base_state)
            results.append(result)
            mock_save_checkpoint.assert_called_once()

        # All results should be ask_user
        assert all(r == "ask_user" for r in results), (
            f"Router should consistently return 'ask_user' for None, "
            f"got results: {results}"
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
    def test_none_verdict_return_type_is_string(
        self,
        router,
        verdict_field,
        base_state,
        mock_save_checkpoint,
    ):
        """Test that router returns a string type, not something else."""
        base_state[verdict_field] = None

        result = router(base_state)

        assert isinstance(result, str), (
            f"Router return type should be str, got {type(result).__name__}"
        )
        assert result == "ask_user"
