"""Routing decisions following reviewer/verifier verdicts.

This module provides comprehensive tests for the routing factory and
pre-configured routers that control the workflow state machine transitions.

Key areas tested:
1. Happy-path routing decisions (approve/pass verdicts)
2. Revision routing decisions (needs_revision/fail verdicts)
3. Count limit enforcement and escalation
4. Invalid verdict handling (None, non-string types, unknown values)
5. Runtime config overrides for limits
6. Pass-through verdicts (bypassing count checks)
7. Factory function behavior
"""

import pytest
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN REVIEW ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterPlanReview:
    """Tests for route_after_plan_review routing function."""

    def test_approve_routes_to_select_stage(self, base_state, valid_plan):
        """Approved plan should route to select_stage to begin execution."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "approve"
        base_state["replan_count"] = 0
        
        result = route_after_plan_review(base_state)
        
        assert result == "select_stage", f"Expected 'select_stage' but got '{result}'"

    def test_approve_routes_to_select_stage_regardless_of_count(
        self, base_state, valid_plan
    ):
        """Approved plan should route to select_stage even with high replan_count."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "approve"
        base_state["replan_count"] = 100  # High count should not affect approve
        
        result = route_after_plan_review(base_state)
        
        assert result == "select_stage", (
            f"Approve verdict should route to select_stage regardless of count, got '{result}'"
        )

    def test_needs_revision_routes_to_plan_under_limit(
        self, base_state, valid_plan
    ):
        """needs_revision should route to plan when under replan limit."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 0
        
        result = route_after_plan_review(base_state)
        
        assert result == "planning", f"Expected 'planning' but got '{result}'"

    def test_needs_revision_routes_to_plan_at_limit_minus_one(
        self, base_state, valid_plan
    ):
        """needs_revision should still route to planning at count = limit - 1."""
        from src.routing import route_after_plan_review
        from schemas.state import MAX_REPLANS

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = MAX_REPLANS - 1
        
        result = route_after_plan_review(base_state)
        
        assert result == "planning", (
            f"At count={MAX_REPLANS-1} (limit-1), should route to 'planning', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_at_limit(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """needs_revision should escalate to ask_user when at replan limit."""
        from src.routing import route_after_plan_review
        from schemas.state import MAX_REPLANS

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = MAX_REPLANS
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_REPLANS} (at limit), should escalate to 'ask_user', got '{result}'"
        )
        # Verify checkpoint is saved before escalation
        mock_save_checkpoint.assert_called_once()
        call_args = mock_save_checkpoint.call_args
        assert "plan_review" in call_args[0][1], (
            f"Checkpoint name should contain 'plan_review', got '{call_args[0][1]}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_above_limit(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """needs_revision should escalate to ask_user when above replan limit."""
        from src.routing import route_after_plan_review
        from schemas.state import MAX_REPLANS

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = MAX_REPLANS + 5
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_REPLANS+5} (above limit), should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """None verdict should escalate to ask_user with checkpoint."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = None
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", f"None verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_integer_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """Integer verdict (invalid type) should escalate to ask_user."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = 42
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", f"Integer verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_list_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """List verdict (invalid type) should escalate to ask_user."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = ["approve"]
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", f"List verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_dict_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """Dict verdict (invalid type) should escalate to ask_user."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = {"verdict": "approve"}
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", f"Dict verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """Unknown string verdict should escalate to ask_user (fallback)."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "invalid_verdict_value"
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", (
            f"Unknown verdict 'invalid_verdict_value' should escalate to 'ask_user', got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_empty_string_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """Empty string verdict should escalate to ask_user (not a valid verdict)."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = ""
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", (
            f"Empty string verdict should escalate to 'ask_user', got '{result}'"
        )

    def test_runtime_config_overrides_default_limit(self, base_state, valid_plan):
        """runtime_config should be able to override the default replan limit."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 5
        # Override to allow 10 replans
        base_state["runtime_config"] = {"max_replans": 10}
        
        result = route_after_plan_review(base_state)
        
        assert result == "planning", (
            f"With runtime_config max_replans=10 and count=5, should route to 'planning', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_lower_limit_escalates(
        self, mock_save_checkpoint, base_state, valid_plan
    ):
        """runtime_config with lower limit should escalate earlier."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 1
        # Override to only allow 1 replan
        base_state["runtime_config"] = {"max_replans": 1}
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", (
            f"With runtime_config max_replans=1 and count=1, should escalate to 'ask_user', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN REVIEW ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterDesignReview:
    """Tests for route_after_design_review routing function."""

    def test_approve_routes_to_generate_code(self, base_state):
        """Approved design should route to code generation."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "approve"
        
        result = route_after_design_review(base_state)
        
        assert result == "generate_code", f"Expected 'generate_code' but got '{result}'"

    def test_approve_routes_to_generate_code_regardless_of_count(self, base_state):
        """Approved design should route to generate_code even with high revision count."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "approve"
        base_state["design_revision_count"] = 100
        
        result = route_after_design_review(base_state)
        
        assert result == "generate_code", (
            f"Approve verdict should route to generate_code regardless of count, got '{result}'"
        )

    def test_needs_revision_routes_to_design_under_limit(self, base_state):
        """needs_revision should route to design when under limit."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 0
        
        result = route_after_design_review(base_state)
        
        assert result == "design", f"Expected 'design' but got '{result}'"

    def test_needs_revision_routes_to_design_at_limit_minus_one(self, base_state):
        """needs_revision should still route to design at count = limit - 1."""
        from src.routing import route_after_design_review
        from schemas.state import MAX_DESIGN_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS - 1
        
        result = route_after_design_review(base_state)
        
        assert result == "design", (
            f"At count={MAX_DESIGN_REVISIONS-1} (limit-1), should route to 'design', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_at_limit(
        self, mock_save_checkpoint, base_state
    ):
        """needs_revision should escalate to ask_user when at design revision limit."""
        from src.routing import route_after_design_review
        from schemas.state import MAX_DESIGN_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        
        result = route_after_design_review(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_DESIGN_REVISIONS} (at limit), should escalate to 'ask_user', got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates_to_ask_user(self, mock_save_checkpoint, base_state):
        """None verdict should escalate to ask_user."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = None
        
        result = route_after_design_review(base_state)
        
        assert result == "ask_user", f"None verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_invalid_type_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Non-string verdict type should escalate to ask_user."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = {"verdict": "approve"}
        
        result = route_after_design_review(base_state)
        
        assert result == "ask_user", (
            f"Dict verdict should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Unknown string verdict should escalate to ask_user (fallback)."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "pass"  # Wrong verdict type
        
        result = route_after_design_review(base_state)
        
        assert result == "ask_user", (
            f"Unknown verdict 'pass' should escalate to 'ask_user', got '{result}'"
        )

    def test_runtime_config_overrides_default_limit(self, base_state):
        """runtime_config should override the default design revision limit."""
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 5
        base_state["runtime_config"] = {"max_design_revisions": 10}
        
        result = route_after_design_review(base_state)
        
        assert result == "design", (
            f"With runtime_config max_design_revisions=10 and count=5, should route to 'design', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEW ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterCodeReview:
    """Tests for route_after_code_review routing function."""

    def test_approve_routes_to_run_code(self, base_state):
        """Approved code should route to run_code for execution."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "approve"
        
        result = route_after_code_review(base_state)
        
        assert result == "run_code", f"Expected 'run_code' but got '{result}'"

    def test_approve_routes_to_run_code_regardless_of_count(self, base_state):
        """Approved code should route to run_code even with high revision count."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "approve"
        base_state["code_revision_count"] = 100
        
        result = route_after_code_review(base_state)
        
        assert result == "run_code", (
            f"Approve verdict should route to run_code regardless of count, got '{result}'"
        )

    def test_needs_revision_routes_to_generate_code_under_limit(self, base_state):
        """needs_revision should route to generate_code when under limit."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        
        result = route_after_code_review(base_state)
        
        assert result == "generate_code", f"Expected 'generate_code' but got '{result}'"

    def test_needs_revision_routes_to_generate_code_at_limit_minus_one(self, base_state):
        """needs_revision should still route to generate_code at count = limit - 1."""
        from src.routing import route_after_code_review
        from schemas.state import MAX_CODE_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS - 1
        
        result = route_after_code_review(base_state)
        
        assert result == "generate_code", (
            f"At count={MAX_CODE_REVISIONS-1} (limit-1), should route to 'generate_code', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_escalates_at_limit(self, mock_save_checkpoint, base_state):
        """needs_revision should escalate to ask_user when at code revision limit."""
        from src.routing import route_after_code_review
        from schemas.state import MAX_CODE_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_CODE_REVISIONS} (at limit), should escalate to 'ask_user', got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates_to_ask_user(self, mock_save_checkpoint, base_state):
        """None verdict should escalate to ask_user."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = None
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", f"None verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_invalid_type_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Non-string verdict type should escalate to ask_user."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = 123
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"Integer verdict should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Unknown string verdict should escalate to ask_user (fallback)."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "reject"  # Not a valid verdict
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"Unknown verdict 'reject' should escalate to 'ask_user', got '{result}'"
        )

    def test_runtime_config_overrides_default_limit(self, base_state):
        """runtime_config should override the default code revision limit."""
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 5
        base_state["runtime_config"] = {"max_code_revisions": 10}
        
        result = route_after_code_review(base_state)
        
        assert result == "generate_code", (
            f"With runtime_config max_code_revisions=10 and count=5, should route to 'generate_code', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION CHECK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterExecutionCheck:
    """Tests for route_after_execution_check routing function.
    
    Key behaviors:
    - 'pass' and 'warning' are pass-through verdicts (no count check)
    - 'fail' triggers count limit check for execution_failure_count
    """

    def test_pass_routes_to_physics_check(self, base_state):
        """Pass verdict should route to physics_check."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "pass"
        
        result = route_after_execution_check(base_state)
        
        assert result == "physics_check", f"Expected 'physics_check' but got '{result}'"

    def test_warning_routes_to_physics_check(self, base_state):
        """Warning verdict should route to physics_check (pass-through)."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "warning"
        
        result = route_after_execution_check(base_state)
        
        assert result == "physics_check", f"Expected 'physics_check' but got '{result}'"

    def test_pass_is_pass_through_ignores_high_failure_count(self, base_state):
        """Pass verdict should route to physics_check regardless of failure count.
        
        This tests the pass_through_verdicts functionality - pass/warning should
        not trigger count limit checks even if count is high.
        """
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "pass"
        base_state["execution_failure_count"] = 100
        
        result = route_after_execution_check(base_state)
        
        assert result == "physics_check", (
            f"Pass verdict should route to physics_check regardless of count, got '{result}'"
        )

    def test_warning_is_pass_through_ignores_high_failure_count(self, base_state):
        """Warning verdict should route to physics_check regardless of failure count.
        
        This tests the pass_through_verdicts functionality.
        """
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "warning"
        base_state["execution_failure_count"] = 100
        
        result = route_after_execution_check(base_state)
        
        assert result == "physics_check", (
            f"Warning verdict should route to physics_check regardless of count, got '{result}'"
        )

    def test_fail_routes_to_generate_code_under_limit(self, base_state):
        """Fail verdict should route to generate_code when under limit."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0
        
        result = route_after_execution_check(base_state)
        
        assert result == "generate_code", f"Expected 'generate_code' but got '{result}'"

    def test_fail_routes_to_generate_code_at_limit_minus_one(self, base_state):
        """Fail verdict should still route to generate_code at count = limit - 1."""
        from src.routing import route_after_execution_check
        from schemas.state import MAX_EXECUTION_FAILURES

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = MAX_EXECUTION_FAILURES - 1
        
        result = route_after_execution_check(base_state)
        
        assert result == "generate_code", (
            f"At count={MAX_EXECUTION_FAILURES-1} (limit-1), should route to 'generate_code', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_fail_escalates_at_limit(self, mock_save_checkpoint, base_state):
        """Fail verdict should escalate to ask_user when at execution failure limit."""
        from src.routing import route_after_execution_check
        from schemas.state import MAX_EXECUTION_FAILURES

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        
        result = route_after_execution_check(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_EXECUTION_FAILURES} (at limit), should escalate to 'ask_user', got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_fail_escalates_above_limit(self, mock_save_checkpoint, base_state):
        """Fail verdict should escalate to ask_user when above execution failure limit."""
        from src.routing import route_after_execution_check
        from schemas.state import MAX_EXECUTION_FAILURES

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = MAX_EXECUTION_FAILURES + 5
        
        result = route_after_execution_check(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_EXECUTION_FAILURES+5} (above limit), should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates_to_ask_user(self, mock_save_checkpoint, base_state):
        """None verdict should escalate to ask_user."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = None
        
        result = route_after_execution_check(base_state)
        
        assert result == "ask_user", f"None verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_invalid_type_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Non-string verdict type should escalate to ask_user."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = True
        
        result = route_after_execution_check(base_state)
        
        assert result == "ask_user", (
            f"Boolean verdict should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Unknown string verdict should escalate to ask_user (fallback)."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "approve"  # Wrong verdict type
        
        result = route_after_execution_check(base_state)
        
        assert result == "ask_user", (
            f"Unknown verdict 'approve' should escalate to 'ask_user', got '{result}'"
        )

    def test_runtime_config_overrides_default_limit(self, base_state):
        """runtime_config should override the default execution failure limit."""
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 5
        base_state["runtime_config"] = {"max_execution_failures": 10}
        
        result = route_after_execution_check(base_state)
        
        assert result == "generate_code", (
            f"With runtime_config max_execution_failures=10 and count=5, should route to 'generate_code', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS CHECK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterPhysicsCheck:
    """Tests for route_after_physics_check routing function.
    
    Key behaviors:
    - 'pass' and 'warning' are pass-through verdicts (no count check)
    - 'fail' triggers count limit check for physics_failure_count
    - 'design_flaw' triggers count limit check for design_revision_count
    """

    def test_pass_routes_to_analyze(self, base_state):
        """Pass verdict should route to analyze."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "pass"
        
        result = route_after_physics_check(base_state)
        
        assert result == "analyze", f"Expected 'analyze' but got '{result}'"

    def test_warning_routes_to_analyze(self, base_state):
        """Warning verdict should route to analyze (pass-through)."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "warning"
        
        result = route_after_physics_check(base_state)
        
        assert result == "analyze", f"Expected 'analyze' but got '{result}'"

    def test_pass_is_pass_through_ignores_high_failure_count(self, base_state):
        """Pass verdict should route to analyze regardless of failure count."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "pass"
        base_state["physics_failure_count"] = 100
        
        result = route_after_physics_check(base_state)
        
        assert result == "analyze", (
            f"Pass verdict should route to analyze regardless of count, got '{result}'"
        )

    def test_warning_is_pass_through_ignores_high_failure_count(self, base_state):
        """Warning verdict should route to analyze regardless of failure count."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "warning"
        base_state["physics_failure_count"] = 100
        
        result = route_after_physics_check(base_state)
        
        assert result == "analyze", (
            f"Warning verdict should route to analyze regardless of count, got '{result}'"
        )

    def test_fail_routes_to_generate_code_under_limit(self, base_state):
        """Fail verdict should route to generate_code when under limit."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = 0
        
        result = route_after_physics_check(base_state)
        
        assert result == "generate_code", f"Expected 'generate_code' but got '{result}'"

    def test_fail_routes_to_generate_code_at_limit_minus_one(self, base_state):
        """Fail verdict should still route to generate_code at count = limit - 1."""
        from src.routing import route_after_physics_check
        from schemas.state import MAX_PHYSICS_FAILURES

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = MAX_PHYSICS_FAILURES - 1
        
        result = route_after_physics_check(base_state)
        
        assert result == "generate_code", (
            f"At count={MAX_PHYSICS_FAILURES-1} (limit-1), should route to 'generate_code', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_fail_escalates_at_limit(self, mock_save_checkpoint, base_state):
        """Fail verdict should escalate to ask_user when at physics failure limit."""
        from src.routing import route_after_physics_check
        from schemas.state import MAX_PHYSICS_FAILURES

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        
        result = route_after_physics_check(base_state)
        
        assert result == "ask_user", (
            f"At count={MAX_PHYSICS_FAILURES} (at limit), should escalate to 'ask_user', got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    def test_design_flaw_routes_to_design_under_limit(self, base_state):
        """design_flaw verdict should route to design when under limit."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0
        
        result = route_after_physics_check(base_state)
        
        assert result == "design", f"Expected 'design' but got '{result}'"

    def test_design_flaw_routes_to_design_at_limit_minus_one(self, base_state):
        """design_flaw verdict should still route to design at count = limit - 1."""
        from src.routing import route_after_physics_check
        from schemas.state import MAX_DESIGN_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS - 1
        
        result = route_after_physics_check(base_state)
        
        assert result == "design", (
            f"At count={MAX_DESIGN_REVISIONS-1} (limit-1), should route to 'design', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_design_flaw_escalates_at_limit(self, mock_save_checkpoint, base_state):
        """design_flaw verdict should escalate to ask_user when at design revision limit."""
        from src.routing import route_after_physics_check
        from schemas.state import MAX_DESIGN_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        
        result = route_after_physics_check(base_state)
        
        assert result == "ask_user", (
            f"At design_revision_count={MAX_DESIGN_REVISIONS} (at limit), should escalate to 'ask_user', got '{result}'"
        )
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates_to_ask_user(self, mock_save_checkpoint, base_state):
        """None verdict should escalate to ask_user."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = None
        
        result = route_after_physics_check(base_state)
        
        assert result == "ask_user", f"None verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_invalid_type_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Non-string verdict type should escalate to ask_user."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = ["pass", "warning"]
        
        result = route_after_physics_check(base_state)
        
        assert result == "ask_user", (
            f"List verdict should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Unknown string verdict should escalate to ask_user (fallback)."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "approve"  # Wrong verdict type
        
        result = route_after_physics_check(base_state)
        
        assert result == "ask_user", (
            f"Unknown verdict 'approve' should escalate to 'ask_user', got '{result}'"
        )

    def test_runtime_config_overrides_physics_limit(self, base_state):
        """runtime_config should override the default physics failure limit."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = 5
        base_state["runtime_config"] = {"max_physics_failures": 10}
        
        result = route_after_physics_check(base_state)
        
        assert result == "generate_code", (
            f"With runtime_config max_physics_failures=10 and count=5, should route to 'generate_code', got '{result}'"
        )

    def test_runtime_config_overrides_design_limit_for_design_flaw(self, base_state):
        """runtime_config should override the design revision limit for design_flaw verdict."""
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 5
        base_state["runtime_config"] = {"max_design_revisions": 10}
        
        result = route_after_physics_check(base_state)
        
        assert result == "design", (
            f"With runtime_config max_design_revisions=10 and count=5, should route to 'design', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHECK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterComparisonCheck:
    """Tests for route_after_comparison_check routing function.
    
    Key behaviors:
    - 'approve' routes to supervisor
    - 'needs_revision' routes to analyze (with count limit check)
    - When limit is reached, routes to 'supervisor' (NOT ask_user - this is different!)
    """

    def test_approve_routes_to_supervisor(self, base_state):
        """Approve verdict should route to supervisor."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "approve"
        
        result = route_after_comparison_check(base_state)
        
        assert result == "supervisor", f"Expected 'supervisor' but got '{result}'"

    def test_approve_routes_to_supervisor_regardless_of_count(self, base_state):
        """Approve verdict should route to supervisor even with high revision count."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "approve"
        base_state["analysis_revision_count"] = 100
        
        result = route_after_comparison_check(base_state)
        
        assert result == "supervisor", (
            f"Approve verdict should route to supervisor regardless of count, got '{result}'"
        )

    def test_needs_revision_routes_to_analyze_under_limit(self, base_state):
        """needs_revision should route to analyze when under limit."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 0
        
        result = route_after_comparison_check(base_state)
        
        assert result == "analyze", f"Expected 'analyze' but got '{result}'"

    def test_needs_revision_routes_to_analyze_at_limit_minus_one(self, base_state):
        """needs_revision should still route to analyze at count = limit - 1."""
        from src.routing import route_after_comparison_check
        from schemas.state import MAX_ANALYSIS_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS - 1
        
        result = route_after_comparison_check(base_state)
        
        assert result == "analyze", (
            f"At count={MAX_ANALYSIS_REVISIONS-1} (limit-1), should route to 'analyze', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_routes_to_supervisor_at_limit(
        self, mock_save_checkpoint, base_state
    ):
        """needs_revision should route to supervisor (NOT ask_user) when at analysis limit.
        
        This is a key difference from other routers - comparison_check escalates to
        supervisor instead of ask_user to proceed with a flag indicating analysis limit reached.
        """
        from src.routing import route_after_comparison_check
        from schemas.state import MAX_ANALYSIS_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS
        
        result = route_after_comparison_check(base_state)
        
        assert result == "supervisor", (
            f"At count={MAX_ANALYSIS_REVISIONS} (at limit), should route to 'supervisor' (not ask_user), got '{result}'"
        )
        # Verify checkpoint is saved before proceeding to supervisor
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_needs_revision_routes_to_supervisor_above_limit(
        self, mock_save_checkpoint, base_state
    ):
        """needs_revision should route to supervisor when above analysis limit."""
        from src.routing import route_after_comparison_check
        from schemas.state import MAX_ANALYSIS_REVISIONS

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS + 5
        
        result = route_after_comparison_check(base_state)
        
        assert result == "supervisor", (
            f"At count={MAX_ANALYSIS_REVISIONS+5} (above limit), should route to 'supervisor', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_none_verdict_escalates_to_ask_user(self, mock_save_checkpoint, base_state):
        """None verdict should escalate to ask_user."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = None
        
        result = route_after_comparison_check(base_state)
        
        assert result == "ask_user", f"None verdict should escalate to 'ask_user', got '{result}'"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_invalid_type_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Non-string verdict type should escalate to ask_user."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = {"result": "approve"}
        
        result = route_after_comparison_check(base_state)
        
        assert result == "ask_user", (
            f"Dict verdict should escalate to 'ask_user', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_unknown_verdict_escalates_to_ask_user(
        self, mock_save_checkpoint, base_state
    ):
        """Unknown string verdict should escalate to ask_user (fallback)."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "pass"  # Wrong verdict type
        
        result = route_after_comparison_check(base_state)
        
        assert result == "ask_user", (
            f"Unknown verdict 'pass' should escalate to 'ask_user', got '{result}'"
        )

    def test_runtime_config_overrides_default_limit(self, base_state):
        """runtime_config should override the default analysis revision limit."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 5
        base_state["runtime_config"] = {"max_analysis_revisions": 10}
        
        result = route_after_comparison_check(base_state)
        
        assert result == "analyze", (
            f"With runtime_config max_analysis_revisions=10 and count=5, should route to 'analyze', got '{result}'"
        )

    @patch("src.routing.save_checkpoint")
    def test_runtime_config_lower_limit_routes_to_supervisor(
        self, mock_save_checkpoint, base_state
    ):
        """runtime_config with lower limit should route to supervisor earlier."""
        from src.routing import route_after_comparison_check

        base_state["current_stage_id"] = "stage_0"
        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = 1
        base_state["runtime_config"] = {"max_analysis_revisions": 1}
        
        result = route_after_comparison_check(base_state)
        
        assert result == "supervisor", (
            f"With runtime_config max_analysis_revisions=1 and count=1, should route to 'supervisor', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateVerdictRouter:
    """Tests for the create_verdict_router factory function."""

    @patch("src.routing.save_checkpoint")
    def test_factory_creates_working_router(self, mock_save_checkpoint, base_state):
        """Factory should create a functioning router from configuration."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "approve": {"route": "next_node"},
                "reject": {"route": "previous_node"},
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "approve"
        assert router(base_state) == "next_node"
        
        base_state["test_verdict"] = "reject"
        assert router(base_state) == "previous_node"

    @patch("src.routing.save_checkpoint")
    def test_factory_handles_none_verdict(self, mock_save_checkpoint, base_state):
        """Factory-created router should handle None verdict by escalating."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = None
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()
        # Verify checkpoint name contains the prefix
        call_args = mock_save_checkpoint.call_args
        assert "test" in call_args[0][1]

    @patch("src.routing.save_checkpoint")
    def test_factory_handles_invalid_type_verdict(
        self, mock_save_checkpoint, base_state
    ):
        """Factory-created router should handle non-string verdict types."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = 12345
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_factory_handles_unknown_verdict(self, mock_save_checkpoint, base_state):
        """Factory-created router should handle unknown verdict values."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "unknown_verdict"
        
        result = router(base_state)
        
        assert result == "ask_user"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_factory_count_limit_enforcement(self, mock_save_checkpoint, base_state):
        """Factory should correctly enforce count limits."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "retry": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "retry_count",
                        "max_count_key": "max_retries",
                        "default_max": 3,
                    }
                }
            },
            checkpoint_prefix="test",
        )
        
        # Under limit - should route normally
        base_state["test_verdict"] = "retry"
        base_state["retry_count"] = 2
        assert router(base_state) == "retry_node"
        
        # At limit - should escalate
        base_state["retry_count"] = 3
        assert router(base_state) == "ask_user"
        mock_save_checkpoint.assert_called_once()

    @patch("src.routing.save_checkpoint")
    def test_factory_count_limit_with_custom_escalation_route(
        self, mock_save_checkpoint, base_state
    ):
        """Factory should support custom route_on_limit."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "retry": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "retry_count",
                        "max_count_key": "max_retries",
                        "default_max": 3,
                        "route_on_limit": "custom_handler",
                    }
                }
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "retry"
        base_state["retry_count"] = 3
        
        result = router(base_state)
        
        assert result == "custom_handler", (
            f"Should route to custom_handler when limit reached, got '{result}'"
        )

    def test_factory_pass_through_verdicts_bypass_count_check(self, base_state):
        """Factory should support pass_through_verdicts that bypass count checks."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "success": {
                    "route": "success_node",
                    "count_limit": {
                        "count_field": "attempt_count",
                        "max_count_key": "max_attempts",
                        "default_max": 3,
                    }
                },
                "partial": {
                    "route": "success_node",
                    "count_limit": {
                        "count_field": "attempt_count",
                        "max_count_key": "max_attempts",
                        "default_max": 3,
                    }
                },
            },
            checkpoint_prefix="test",
            pass_through_verdicts=["success", "partial"],
        )
        
        # High count should not affect pass-through verdicts
        base_state["test_verdict"] = "success"
        base_state["attempt_count"] = 100
        
        result = router(base_state)
        
        assert result == "success_node", (
            f"Pass-through verdict should bypass count check, got '{result}'"
        )

    def test_factory_runtime_config_overrides_default_limit(self, base_state):
        """Factory should respect runtime_config overrides for limits."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "retry": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "retry_count",
                        "max_count_key": "max_retries",
                        "default_max": 3,
                    }
                }
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "retry"
        base_state["retry_count"] = 5
        base_state["runtime_config"] = {"max_retries": 10}
        
        result = router(base_state)
        
        assert result == "retry_node", (
            f"With runtime_config override, should route normally, got '{result}'"
        )

    def test_factory_missing_verdict_field_returns_ask_user(self, base_state):
        """Factory-created router should handle missing verdict field."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="nonexistent_field",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        # Don't set the field - it should return None from state.get()
        with patch("src.routing.save_checkpoint"):
            result = router(base_state)
        
        assert result == "ask_user", (
            f"Missing verdict field should return 'ask_user', got '{result}'"
        )

    def test_factory_missing_count_field_uses_zero(self, base_state):
        """Factory should default to 0 if count field is missing."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "retry": {
                    "route": "retry_node",
                    "count_limit": {
                        "count_field": "nonexistent_count",
                        "max_count_key": "max_retries",
                        "default_max": 3,
                    }
                }
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "retry"
        # Don't set the count field
        
        result = router(base_state)
        
        assert result == "retry_node", (
            f"Missing count field should default to 0 and route normally, got '{result}'"
        )

    def test_factory_route_defaults_to_ask_user_if_not_specified(self, base_state):
        """Factory should default to ask_user if route not specified in config."""
        from src.routing import create_verdict_router

        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "mystery": {}  # No route specified
            },
            checkpoint_prefix="test",
        )
        
        base_state["test_verdict"] = "mystery"
        
        result = router(base_state)
        
        assert result == "ask_user", (
            f"Missing route config should default to 'ask_user', got '{result}'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoutingEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_routers_handle_none_runtime_config(self, base_state, valid_plan):
        """All routers should handle None runtime_config gracefully."""
        from src.routing import (
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        )

        # Explicitly set runtime_config to None
        base_state["runtime_config"] = None
        base_state["plan"] = valid_plan
        
        # Each router should use defaults when runtime_config is None
        base_state["last_plan_review_verdict"] = "approve"
        assert route_after_plan_review(base_state) == "select_stage"
        
        base_state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(base_state) == "generate_code"
        
        base_state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(base_state) == "run_code"
        
        base_state["execution_verdict"] = "pass"
        assert route_after_execution_check(base_state) == "physics_check"
        
        base_state["physics_verdict"] = "pass"
        assert route_after_physics_check(base_state) == "analyze"
        
        base_state["comparison_verdict"] = "approve"
        assert route_after_comparison_check(base_state) == "supervisor"

    def test_all_routers_handle_empty_runtime_config(self, base_state, valid_plan):
        """All routers should handle empty runtime_config gracefully."""
        from src.routing import (
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
            route_after_comparison_check,
        )

        # Explicitly set empty runtime_config
        base_state["runtime_config"] = {}
        base_state["plan"] = valid_plan
        
        # Each router should use defaults when runtime_config is empty
        base_state["last_plan_review_verdict"] = "approve"
        assert route_after_plan_review(base_state) == "select_stage"
        
        base_state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(base_state) == "generate_code"
        
        base_state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(base_state) == "run_code"
        
        base_state["execution_verdict"] = "pass"
        assert route_after_execution_check(base_state) == "physics_check"
        
        base_state["physics_verdict"] = "pass"
        assert route_after_physics_check(base_state) == "analyze"
        
        base_state["comparison_verdict"] = "approve"
        assert route_after_comparison_check(base_state) == "supervisor"

    @patch("src.routing.save_checkpoint")
    def test_count_at_zero_with_zero_limit_escalates(
        self, mock_save_checkpoint, base_state
    ):
        """When max limit is set to 0, even count=0 should escalate."""
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        base_state["runtime_config"] = {"max_code_revisions": 0}
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"With max_code_revisions=0 and count=0, should escalate to 'ask_user', got '{result}'"
        )

    def test_negative_count_treated_as_under_limit(self, base_state):
        """Negative counts (if they somehow occur) should be treated as under limit."""
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = -5  # Shouldn't happen, but let's be safe
        
        result = route_after_code_review(base_state)
        
        assert result == "generate_code", (
            f"Negative count should be treated as under limit, got '{result}'"
        )

    def test_very_large_count_escalates(self, base_state):
        """Very large counts should definitely escalate."""
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 999999
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"Very large count should escalate to 'ask_user', got '{result}'"
        )

    def test_verdict_with_extra_whitespace_not_recognized(self, base_state):
        """Verdict with extra whitespace should not be recognized."""
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "  approve  "  # With whitespace
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"Verdict with whitespace '  approve  ' should not match 'approve', got '{result}'"
        )

    def test_verdict_case_sensitive(self, base_state):
        """Verdicts should be case-sensitive."""
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "APPROVE"  # Wrong case
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_code_review(base_state)
        
        assert result == "ask_user", (
            f"Verdict 'APPROVE' should not match 'approve' (case-sensitive), got '{result}'"
        )

    def test_checkpoint_prefix_appears_in_checkpoint_name(self, base_state):
        """Checkpoint names should include the configured prefix."""
        from src.routing import create_verdict_router

        with patch("src.routing.save_checkpoint") as mock_checkpoint:
            router = create_verdict_router(
                verdict_field="test_verdict",
                routes={"approve": {"route": "next"}},
                checkpoint_prefix="my_custom_prefix",
            )
            
            base_state["test_verdict"] = None
            router(base_state)
            
            mock_checkpoint.assert_called_once()
            checkpoint_name = mock_checkpoint.call_args[0][1]
            assert "my_custom_prefix" in checkpoint_name, (
                f"Checkpoint name should contain prefix 'my_custom_prefix', got '{checkpoint_name}'"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDARY TESTS FOR DEFAULT LIMITS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultLimitBoundaries:
    """Tests to verify exact boundary behavior at default limits.
    
    These tests ensure the routing logic correctly handles the exact
    boundary conditions (limit-1, limit, limit+1) for all routers.
    """

    @pytest.mark.parametrize("count,expected_route", [
        (0, "plan"),
        (1, "ask_user"),  # MAX_REPLANS=2, so at 1 it should still route to plan
        (2, "ask_user"),
    ])
    def test_plan_review_boundary(
        self, base_state, valid_plan, count, expected_route
    ):
        """Test plan_review routing at various count boundaries."""
        from src.routing import route_after_plan_review
        from schemas.state import MAX_REPLANS

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = count
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_plan_review(base_state)
        
        # Adjust expected based on actual limit
        if count < MAX_REPLANS:
            assert result == "planning", f"At count={count}, expected 'planning', got '{result}'"
        else:
            assert result == "ask_user", f"At count={count}, expected 'ask_user', got '{result}'"

    @pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
    def test_code_review_boundary_parametrized(self, base_state, count):
        """Test code_review routing at various count values."""
        from src.routing import route_after_code_review
        from schemas.state import MAX_CODE_REVISIONS

        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = count
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_code_review(base_state)
        
        if count < MAX_CODE_REVISIONS:
            assert result == "generate_code", (
                f"At count={count} (< limit {MAX_CODE_REVISIONS}), expected 'generate_code', got '{result}'"
            )
        else:
            assert result == "ask_user", (
                f"At count={count} (>= limit {MAX_CODE_REVISIONS}), expected 'ask_user', got '{result}'"
            )

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_execution_check_boundary_parametrized(self, base_state, count):
        """Test execution_check routing at various count values."""
        from src.routing import route_after_execution_check
        from schemas.state import MAX_EXECUTION_FAILURES

        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = count
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(base_state)
        
        if count < MAX_EXECUTION_FAILURES:
            assert result == "generate_code", (
                f"At count={count} (< limit {MAX_EXECUTION_FAILURES}), expected 'generate_code', got '{result}'"
            )
        else:
            assert result == "ask_user", (
                f"At count={count} (>= limit {MAX_EXECUTION_FAILURES}), expected 'ask_user', got '{result}'"
            )

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_physics_check_fail_boundary_parametrized(self, base_state, count):
        """Test physics_check fail verdict routing at various count values."""
        from src.routing import route_after_physics_check
        from schemas.state import MAX_PHYSICS_FAILURES

        base_state["physics_verdict"] = "fail"
        base_state["physics_failure_count"] = count
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(base_state)
        
        if count < MAX_PHYSICS_FAILURES:
            assert result == "generate_code", (
                f"At count={count} (< limit {MAX_PHYSICS_FAILURES}), expected 'generate_code', got '{result}'"
            )
        else:
            assert result == "ask_user", (
                f"At count={count} (>= limit {MAX_PHYSICS_FAILURES}), expected 'ask_user', got '{result}'"
            )

    @pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
    def test_physics_check_design_flaw_boundary_parametrized(self, base_state, count):
        """Test physics_check design_flaw verdict routing at various count values."""
        from src.routing import route_after_physics_check
        from schemas.state import MAX_DESIGN_REVISIONS

        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = count
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(base_state)
        
        if count < MAX_DESIGN_REVISIONS:
            assert result == "design", (
                f"At count={count} (< limit {MAX_DESIGN_REVISIONS}), expected 'design', got '{result}'"
            )
        else:
            assert result == "ask_user", (
                f"At count={count} (>= limit {MAX_DESIGN_REVISIONS}), expected 'ask_user', got '{result}'"
            )

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_comparison_check_boundary_parametrized(self, base_state, count):
        """Test comparison_check routing at various count values.
        
        Note: comparison_check escalates to 'supervisor', not 'ask_user'.
        """
        from src.routing import route_after_comparison_check
        from schemas.state import MAX_ANALYSIS_REVISIONS

        base_state["comparison_verdict"] = "needs_revision"
        base_state["analysis_revision_count"] = count
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_comparison_check(base_state)
        
        if count < MAX_ANALYSIS_REVISIONS:
            assert result == "analyze", (
                f"At count={count} (< limit {MAX_ANALYSIS_REVISIONS}), expected 'analyze', got '{result}'"
            )
        else:
            assert result == "supervisor", (
                f"At count={count} (>= limit {MAX_ANALYSIS_REVISIONS}), expected 'supervisor' (not ask_user), got '{result}'"
            )
