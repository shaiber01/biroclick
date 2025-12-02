"""Unit tests for src/agents/helpers/context.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.helpers.context import (
    check_context_or_escalate,
    validate_user_responses,
    validate_state_or_warn,
)


class TestCheckContextOrEscalate:
    """Tests for check_context_or_escalate function."""

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_returns_none_when_context_ok(self, mock_check):
        """Should return None when context is OK."""
        mock_check.return_value = {"ok": True}
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert result is None
        mock_check.assert_called_once_with(state, "test_node", auto_recover=True)

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_returns_state_updates_on_auto_recovery(self, mock_check):
        """Should return state updates when auto-recovery applied."""
        mock_check.return_value = {
            "ok": True,
            "state_updates": {"some_field": "recovered_value"}
        }
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert result == {"some_field": "recovered_value"}

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_returns_escalation_when_context_critical(self, mock_check):
        """Should return escalation when context overflow is critical."""
        mock_check.return_value = {
            "ok": False,
            "escalate": True,
            "user_question": "Context overflow. Please advise."
        }
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "context_overflow"
        assert result["last_node_before_ask_user"] == "test_node"
        assert "Context overflow" in result["pending_user_questions"][0]

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_fallback_escalation_for_unknown_state(self, mock_check):
        """Should fallback to escalation if response is ambiguous."""
        mock_check.return_value = {"ok": False, "escalate": False}
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "unknown_node")
        
        assert result["awaiting_user_input"] is True
        assert "unknown_node" in result["pending_user_questions"][0]


class TestValidateUserResponses:
    """Tests for validate_user_responses function."""

    def test_empty_responses_returns_error(self):
        """Should return error when no responses provided."""
        errors = validate_user_responses("material_checkpoint", {}, ["Q1"])
        assert "No responses provided" in errors

    def test_material_checkpoint_valid_approve(self):
        """Should accept APPROVE for material_checkpoint."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "APPROVE"},
            ["q"]
        )
        assert errors == []

    def test_material_checkpoint_valid_yes(self):
        """Should accept YES for material_checkpoint."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "yes please"},
            ["q"]
        )
        assert errors == []

    def test_material_checkpoint_invalid(self):
        """Should reject invalid response for material_checkpoint."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "maybe later"},
            ["q"]
        )
        assert len(errors) == 1
        assert "APPROVE" in errors[0]

    def test_code_review_limit_valid_skip(self):
        """Should accept SKIP for code_review_limit."""
        errors = validate_user_responses(
            "code_review_limit",
            {"q": "let's SKIP this"},
            ["q"]
        )
        assert errors == []

    def test_code_review_limit_valid_hint(self):
        """Should accept PROVIDE_HINT for code_review_limit."""
        errors = validate_user_responses(
            "code_review_limit",
            {"q": "PROVIDE_HINT"},
            ["q"]
        )
        assert errors == []

    def test_design_review_limit_invalid(self):
        """Should reject invalid response for design_review_limit."""
        errors = validate_user_responses(
            "design_review_limit",
            {"q": "continue please"},
            ["q"]
        )
        assert len(errors) == 1

    def test_execution_failure_limit_valid_retry(self):
        """Should accept RETRY for execution_failure_limit."""
        errors = validate_user_responses(
            "execution_failure_limit",
            {"q": "RETRY with more memory"},
            ["q"]
        )
        assert errors == []

    def test_physics_failure_limit_valid_accept(self):
        """Should accept ACCEPT for physics_failure_limit."""
        errors = validate_user_responses(
            "physics_failure_limit",
            {"q": "ACCEPT PARTIAL results"},
            ["q"]
        )
        assert errors == []

    def test_backtrack_approval_valid_approve(self):
        """Should accept APPROVE for backtrack_approval."""
        errors = validate_user_responses(
            "backtrack_approval",
            {"q": "APPROVE the backtrack"},
            ["q"]
        )
        assert errors == []

    def test_backtrack_approval_valid_yes(self):
        """Should accept YES for backtrack_approval."""
        errors = validate_user_responses(
            "backtrack_approval",
            {"q": "YES"},
            ["q"]
        )
        assert errors == []

    def test_replan_limit_valid_force(self):
        """Should accept FORCE for replan_limit."""
        errors = validate_user_responses(
            "replan_limit",
            {"q": "FORCE ACCEPT the plan"},
            ["q"]
        )
        assert errors == []

    def test_unknown_trigger_accepts_nonempty(self):
        """Should accept any non-empty response for unknown triggers."""
        errors = validate_user_responses(
            "some_unknown_trigger",
            {"q": "any response"},
            ["q"]
        )
        assert errors == []

    def test_context_overflow_accepts_any(self):
        """Should accept any response for context_overflow."""
        errors = validate_user_responses(
            "context_overflow",
            {"q": "anything"},
            ["q"]
        )
        assert errors == []

    def test_case_insensitive(self):
        """Should be case insensitive."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "approve"},
            ["q"]
        )
        assert errors == []


class TestValidateStateOrWarn:
    """Tests for validate_state_or_warn function."""

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_returns_empty_list_when_valid(self, mock_validate):
        """Should return empty list when state is valid."""
        mock_validate.return_value = []
        state = {"paper_text": "test"}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == []
        mock_validate.assert_called_once_with(state, "test_node")

    @patch("src.agents.helpers.context.validate_state_for_node")
    @patch("src.agents.helpers.context.logging")
    def test_logs_warnings_for_issues(self, mock_logging, mock_validate):
        """Should log warnings for validation issues."""
        mock_validate.return_value = ["Missing field X", "Invalid value Y"]
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        state = {"paper_text": "test"}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == ["Missing field X", "Invalid value Y"]
        assert mock_logger.warning.call_count == 2

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_returns_issues_list(self, mock_validate):
        """Should return list of issues."""
        mock_validate.return_value = ["Issue 1", "Issue 2"]
        state = {}
        
        result = validate_state_or_warn(state, "node")
        
        assert len(result) == 2
        assert "Issue 1" in result



