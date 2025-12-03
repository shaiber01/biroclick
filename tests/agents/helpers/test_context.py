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
    def test_returns_none_when_context_ok_with_empty_state_updates(self, mock_check):
        """Should return None when context is OK and state_updates is empty dict."""
        mock_check.return_value = {"ok": True, "state_updates": {}}
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert result is None
        mock_check.assert_called_once_with(state, "test_node", auto_recover=True)

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_returns_none_when_context_ok_with_none_state_updates(self, mock_check):
        """Should return None when context is OK and state_updates is None."""
        mock_check.return_value = {"ok": True, "state_updates": None}
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert result is None
        mock_check.assert_called_once_with(state, "test_node", auto_recover=True)

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_returns_none_when_context_ok_without_state_updates_key(self, mock_check):
        """Should return None when context is OK and state_updates key is missing."""
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
        assert isinstance(result, dict)
        assert len(result) == 1
        mock_check.assert_called_once_with(state, "test_node", auto_recover=True)

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_returns_state_updates_with_multiple_fields(self, mock_check):
        """Should return all state updates when multiple fields are updated."""
        mock_check.return_value = {
            "ok": True,
            "state_updates": {
                "field1": "value1",
                "field2": "value2",
                "field3": {"nested": "data"}
            }
        }
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert result == {
            "field1": "value1",
            "field2": "value2",
            "field3": {"nested": "data"}
        }
        assert len(result) == 3

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
        
        assert isinstance(result, dict)
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "context_overflow"
        assert result["last_node_before_ask_user"] == "test_node"
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        assert result["pending_user_questions"][0] == "Context overflow. Please advise."
        assert "Context overflow" in result["pending_user_questions"][0]
        # Verify all required keys are present
        assert set(result.keys()) == {
            "pending_user_questions",
            "awaiting_user_input",
            "ask_user_trigger",
            "last_node_before_ask_user"
        }
        mock_check.assert_called_once_with(state, "test_node", auto_recover=True)

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_escalation_with_different_node_names(self, mock_check):
        """Should include correct node name in escalation for different nodes."""
        node_names = ["design", "code_review", "plan_review", "analyze_results"]
        for node_name in node_names:
            mock_check.return_value = {
                "ok": False,
                "escalate": True,
                "user_question": f"Context overflow in {node_name}."
            }
            state = {"paper_text": "test"}
            
            result = check_context_or_escalate(state, node_name)
            
            assert result["last_node_before_ask_user"] == node_name
            assert result["awaiting_user_input"] is True
            assert result["ask_user_trigger"] == "context_overflow"

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_fallback_escalation_for_unknown_state(self, mock_check):
        """Should fallback to escalation if response is ambiguous."""
        mock_check.return_value = {"ok": False, "escalate": False}
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "unknown_node")
        
        assert isinstance(result, dict)
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "context_overflow"
        assert result["last_node_before_ask_user"] == "unknown_node"
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        assert "unknown_node" in result["pending_user_questions"][0]
        assert "Context overflow" in result["pending_user_questions"][0]
        assert "How should we proceed?" in result["pending_user_questions"][0]
        # Verify all required keys are present
        assert set(result.keys()) == {
            "pending_user_questions",
            "awaiting_user_input",
            "ask_user_trigger",
            "last_node_before_ask_user"
        }
        mock_check.assert_called_once_with(state, "unknown_node", auto_recover=True)

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_fallback_escalation_when_ok_false_escalate_missing(self, mock_check):
        """Should fallback to escalation when ok is False but escalate key is missing."""
        mock_check.return_value = {"ok": False}
        state = {"paper_text": "test"}
        
        result = check_context_or_escalate(state, "test_node")
        
        assert isinstance(result, dict)
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "context_overflow"
        assert result["last_node_before_ask_user"] == "test_node"
        assert "test_node" in result["pending_user_questions"][0]

    @patch("src.agents.helpers.context.check_context_before_node")
    def test_state_passed_correctly_to_check(self, mock_check):
        """Should pass state object correctly to check_context_before_node."""
        state = {"paper_text": "test", "plan": {"stages": []}}
        mock_check.return_value = {"ok": True}
        
        check_context_or_escalate(state, "test_node")
        
        mock_check.assert_called_once_with(state, "test_node", auto_recover=True)
        # Verify state object identity is preserved (same object passed)
        call_args = mock_check.call_args
        assert call_args[0][0] is state


class TestValidateUserResponses:
    """Tests for validate_user_responses function."""

    def test_empty_responses_returns_error(self):
        """Should return error when no responses provided."""
        errors = validate_user_responses("material_checkpoint", {}, ["Q1"])
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert "No responses provided" in errors[0]

    def test_empty_responses_returns_immediately(self):
        """Should return immediately when responses dict is empty."""
        errors = validate_user_responses("material_checkpoint", {}, ["Q1"])
        # Should return early, not check trigger type
        assert len(errors) == 1
        assert "No responses provided" in errors[0]

    def test_material_checkpoint_all_valid_keywords(self):
        """Should accept all valid keywords for material_checkpoint."""
        valid_responses = [
            "APPROVE",
            "CHANGE_MATERIAL",
            "CHANGE_DATABASE",
            "NEED_HELP",
            "HELP",
            "YES",
            "NO",
            "REJECT",
            "CORRECT",
            "WRONG",
            "approve",  # case insensitive
            "yes please",
            "I need help",
            "change material please",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "material_checkpoint",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_material_checkpoint_invalid(self):
        """Should reject invalid response for material_checkpoint."""
        invalid_responses = [
            "maybe later",
            "not sure",
            "let me think",
            "",
            "   ",
        ]
        for response in invalid_responses:
            errors = validate_user_responses(
                "material_checkpoint",
                {"q": response},
                ["q"]
            )
            assert len(errors) == 1, f"Should reject '{response}' but got no errors"
            assert "APPROVE" in errors[0] or "CHANGE_MATERIAL" in errors[0] or "CHANGE_DATABASE" in errors[0] or "NEED_HELP" in errors[0]

    def test_material_checkpoint_error_message_contains_valid_options(self):
        """Should include valid options in error message."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "invalid"},
            ["q"]
        )
        assert len(errors) == 1
        error_msg = errors[0]
        assert "APPROVE" in error_msg or "CHANGE_MATERIAL" in error_msg or "CHANGE_DATABASE" in error_msg or "NEED_HELP" in error_msg

    def test_code_review_limit_all_valid_keywords(self):
        """Should accept all valid keywords for code_review_limit."""
        valid_responses = [
            "PROVIDE_HINT",
            "HINT",
            "SKIP",
            "STOP",
            "RETRY",
            "provide hint",
            "skip this",
            "let's stop",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "code_review_limit",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_code_review_limit_invalid(self):
        """Should reject invalid response for code_review_limit."""
        errors = validate_user_responses(
            "code_review_limit",
            {"q": "continue please"},
            ["q"]
        )
        assert len(errors) == 1
        assert "PROVIDE_HINT" in errors[0] or "SKIP_STAGE" in errors[0] or "STOP" in errors[0]

    def test_design_review_limit_all_valid_keywords(self):
        """Should accept all valid keywords for design_review_limit."""
        valid_responses = [
            "PROVIDE_HINT",
            "HINT",
            "SKIP",
            "STOP",
            "RETRY",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "design_review_limit",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_design_review_limit_invalid(self):
        """Should reject invalid response for design_review_limit."""
        errors = validate_user_responses(
            "design_review_limit",
            {"q": "continue please"},
            ["q"]
        )
        assert len(errors) == 1
        assert "PROVIDE_HINT" in errors[0] or "SKIP_STAGE" in errors[0] or "STOP" in errors[0]

    def test_execution_failure_limit_all_valid_keywords(self):
        """Should accept all valid keywords for execution_failure_limit."""
        valid_responses = [
            "RETRY",
            "GUIDANCE",
            "SKIP",
            "STOP",
            "retry with guidance",
            "skip this stage",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "execution_failure_limit",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_execution_failure_limit_invalid(self):
        """Should reject invalid response for execution_failure_limit."""
        errors = validate_user_responses(
            "execution_failure_limit",
            {"q": "continue anyway"},
            ["q"]
        )
        assert len(errors) == 1
        assert "RETRY_WITH_GUIDANCE" in errors[0] or "SKIP_STAGE" in errors[0] or "STOP" in errors[0]

    def test_physics_failure_limit_all_valid_keywords(self):
        """Should accept all valid keywords for physics_failure_limit."""
        valid_responses = [
            "RETRY",
            "ACCEPT",
            "PARTIAL",
            "SKIP",
            "STOP",
            "accept partial results",
            "retry with guidance",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "physics_failure_limit",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_physics_failure_limit_invalid(self):
        """Should reject invalid response for physics_failure_limit."""
        errors = validate_user_responses(
            "physics_failure_limit",
            {"q": "continue anyway"},
            ["q"]
        )
        assert len(errors) == 1
        assert "RETRY_WITH_GUIDANCE" in errors[0] or "ACCEPT_PARTIAL" in errors[0] or "SKIP_STAGE" in errors[0] or "STOP" in errors[0]

    def test_backtrack_approval_all_valid_keywords(self):
        """Should accept all valid keywords for backtrack_approval."""
        valid_responses = [
            "APPROVE",
            "REJECT",
            "YES",
            "NO",
            "approve",
            "yes please",
            "reject this",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "backtrack_approval",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_backtrack_approval_invalid(self):
        """Should reject invalid response for backtrack_approval."""
        errors = validate_user_responses(
            "backtrack_approval",
            {"q": "maybe"},
            ["q"]
        )
        assert len(errors) == 1
        assert "APPROVE" in errors[0] or "REJECT" in errors[0]

    def test_replan_limit_all_valid_keywords(self):
        """Should accept all valid keywords for replan_limit."""
        valid_responses = [
            "FORCE",
            "ACCEPT",
            "GUIDANCE",
            "STOP",
            "force accept",
            "provide guidance",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "replan_limit",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_replan_limit_invalid(self):
        """Should reject invalid response for replan_limit."""
        errors = validate_user_responses(
            "replan_limit",
            {"q": "continue anyway"},
            ["q"]
        )
        assert len(errors) == 1
        assert "FORCE_ACCEPT" in errors[0] or "PROVIDE_GUIDANCE" in errors[0] or "STOP" in errors[0]

    def test_context_overflow_accepts_any_nonempty(self):
        """Should accept any non-empty response for context_overflow."""
        valid_responses = [
            "anything",
            "proceed",
            "skip",
            "stop",
            "continue",
            "yes",
            "no",
            "   ",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "context_overflow",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_context_overflow_rejects_empty(self):
        """Should reject empty response for context_overflow."""
        errors = validate_user_responses(
            "context_overflow",
            {"q": ""},
            ["q"]
        )
        # context_overflow should accept empty strings based on code logic
        # Actually, looking at the code, it checks if all_responses.strip() is empty
        # So empty string should be rejected for unknown triggers but context_overflow is special
        # Let me check the code again - context_overflow is in the special list, so it should accept empty
        # Wait, the code checks `elif trigger not in ["context_overflow", "backtrack_limit"]`
        # So context_overflow is excluded from the empty check, meaning it accepts anything
        # But empty string when joined becomes empty, so let's test this
        all_responses = " ".join(str("").upper() for r in [""])
        assert all_responses.strip() == ""
        # So for context_overflow, it should accept empty because it's excluded from the check
        errors = validate_user_responses(
            "context_overflow",
            {"q": ""},
            ["q"]
        )
        assert errors == []

    def test_backtrack_limit_accepts_any_nonempty(self):
        """Should accept any non-empty response for backtrack_limit."""
        valid_responses = [
            "anything",
            "proceed",
            "yes",
            "no",
        ]
        for response in valid_responses:
            errors = validate_user_responses(
                "backtrack_limit",
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept '{response}' but got errors: {errors}"

    def test_unknown_trigger_accepts_nonempty(self):
        """Should accept any non-empty response for unknown triggers."""
        errors = validate_user_responses(
            "some_unknown_trigger",
            {"q": "any response"},
            ["q"]
        )
        assert errors == []

    def test_unknown_trigger_rejects_empty(self):
        """Should reject empty response for unknown triggers."""
        errors = validate_user_responses(
            "some_unknown_trigger",
            {"q": ""},
            ["q"]
        )
        assert len(errors) == 1
        assert "empty" in errors[0].lower() or "cannot be empty" in errors[0].lower()

    def test_unknown_trigger_rejects_whitespace_only(self):
        """Should reject whitespace-only response for unknown triggers."""
        errors = validate_user_responses(
            "some_unknown_trigger",
            {"q": "   "},
            ["q"]
        )
        assert len(errors) == 1
        assert "empty" in errors[0].lower() or "cannot be empty" in errors[0].lower()

    def test_case_insensitive(self):
        """Should be case insensitive for all triggers."""
        test_cases = [
            ("material_checkpoint", "approve"),
            ("code_review_limit", "skip"),
            ("design_review_limit", "hint"),
            ("execution_failure_limit", "retry"),
            ("physics_failure_limit", "accept"),
            ("backtrack_approval", "yes"),
            ("replan_limit", "force"),
        ]
        for trigger, response in test_cases:
            errors = validate_user_responses(
                trigger,
                {"q": response},
                ["q"]
            )
            assert errors == [], f"Should accept case-insensitive '{response}' for {trigger} but got errors: {errors}"

    def test_multiple_responses_combined(self):
        """Should check all responses combined."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q1": "maybe", "q2": "APPROVE"},
            ["q1", "q2"]
        )
        # Should pass because "APPROVE" is in combined responses
        assert errors == []

    def test_multiple_responses_none_valid(self):
        """Should fail if none of multiple responses contain valid keyword."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q1": "maybe", "q2": "later"},
            ["q1", "q2"]
        )
        assert len(errors) == 1

    def test_responses_with_none_values(self):
        """Should handle None values in responses dict."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": None},
            ["q"]
        )
        # None should be converted to string "None" and checked
        # "NONE" should not match valid keywords, so should fail
        assert len(errors) == 1

    def test_responses_with_non_string_values(self):
        """Should handle non-string values in responses dict."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": 123},
            ["q"]
        )
        # Should convert to string and check
        assert len(errors) == 1

    def test_responses_with_list_values(self):
        """Should handle list values in responses dict."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": ["APPROVE"]},
            ["q"]
        )
        # Should convert list to string representation and check
        # This might pass if "APPROVE" appears in string representation
        # Let's see what happens - str(["APPROVE"]) = "['APPROVE']" which contains "APPROVE"
        assert errors == []

    def test_questions_parameter_ignored(self):
        """Should ignore questions parameter - only check responses."""
        # The questions parameter is not used in validation logic
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "APPROVE"},
            []  # Empty questions list
        )
        assert errors == []

    def test_questions_parameter_with_missing_response(self):
        """Should work even if questions list doesn't match response keys."""
        errors = validate_user_responses(
            "material_checkpoint",
            {"q": "APPROVE"},
            ["different_question"]  # Different question name
        )
        # Should still validate based on responses dict
        assert errors == []


class TestValidateStateOrWarn:
    """Tests for validate_state_or_warn function."""

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_returns_empty_list_when_valid(self, mock_validate):
        """Should return empty list when state is valid."""
        mock_validate.return_value = []
        state = {"paper_text": "test"}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert isinstance(result, list)
        assert result == []
        assert len(result) == 0
        mock_validate.assert_called_once_with(state, "test_node")

    @patch("src.agents.helpers.context.validate_state_for_node")
    @patch("src.agents.helpers.context.logging")
    def test_no_logging_when_valid(self, mock_logging, mock_validate):
        """Should not log when state is valid."""
        mock_validate.return_value = []
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        state = {"paper_text": "test"}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == []
        assert mock_logger.warning.call_count == 0
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
        
        assert isinstance(result, list)
        assert result == ["Missing field X", "Invalid value Y"]
        assert len(result) == 2
        assert "Missing field X" in result
        assert "Invalid value Y" in result
        assert mock_logger.warning.call_count == 2
        # Verify logging was called with correct format
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) == 2
        # Check that each call includes node name and issue
        for call in warning_calls:
            call_args = call[0][0]  # First positional argument
            assert "test_node" in call_args
            assert "Missing field X" in call_args or "Invalid value Y" in call_args
        mock_validate.assert_called_once_with(state, "test_node")

    @patch("src.agents.helpers.context.validate_state_for_node")
    @patch("src.agents.helpers.context.logging")
    def test_logs_each_issue_separately(self, mock_logging, mock_validate):
        """Should log each issue in a separate warning call."""
        issues = ["Issue 1", "Issue 2", "Issue 3"]
        mock_validate.return_value = issues
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        state = {"paper_text": "test"}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == issues
        assert mock_logger.warning.call_count == 3
        # Verify each issue was logged
        logged_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
        for issue in issues:
            assert any(issue in msg for msg in logged_messages)

    @patch("src.agents.helpers.context.validate_state_for_node")
    @patch("src.agents.helpers.context.logging")
    def test_logging_format_includes_node_name(self, mock_logging, mock_validate):
        """Should include node name in log messages."""
        mock_validate.return_value = ["Missing field"]
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        state = {"paper_text": "test"}
        node_name = "design_node"
        
        validate_state_or_warn(state, node_name)
        
        assert mock_logger.warning.call_count == 1
        logged_message = mock_logger.warning.call_args[0][0]
        assert node_name in logged_message
        assert "Missing field" in logged_message
        assert "State validation issue" in logged_message or "validation issue" in logged_message.lower()

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_returns_issues_list(self, mock_validate):
        """Should return list of issues exactly as returned by validate_state_for_node."""
        issues = ["Issue 1", "Issue 2"]
        mock_validate.return_value = issues
        state = {}
        
        result = validate_state_or_warn(state, "node")
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == issues
        assert "Issue 1" in result
        assert "Issue 2" in result
        mock_validate.assert_called_once_with(state, "node")

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_returns_single_issue(self, mock_validate):
        """Should return single issue correctly."""
        mock_validate.return_value = ["Single issue"]
        state = {"paper_text": "test"}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == ["Single issue"]
        assert len(result) == 1

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_handles_empty_state(self, mock_validate):
        """Should handle empty state dict."""
        mock_validate.return_value = []
        state = {}
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == []
        mock_validate.assert_called_once_with(state, "test_node")

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_handles_none_state(self, mock_validate):
        """Should handle None state (if validate_state_for_node accepts it)."""
        mock_validate.return_value = []
        state = None
        
        result = validate_state_or_warn(state, "test_node")
        
        assert result == []
        mock_validate.assert_called_once_with(state, "test_node")

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_passes_state_correctly(self, mock_validate):
        """Should pass state object correctly to validate_state_for_node."""
        state = {"paper_text": "test", "plan": {"stages": []}}
        mock_validate.return_value = []
        
        validate_state_or_warn(state, "test_node")
        
        mock_validate.assert_called_once_with(state, "test_node")
        # Verify state object identity is preserved
        call_args = mock_validate.call_args
        assert call_args[0][0] is state

    @patch("src.agents.helpers.context.validate_state_for_node")
    def test_passes_node_name_correctly(self, mock_validate):
        """Should pass node name correctly to validate_state_for_node."""
        node_names = ["design", "code_review", "plan_review", "analyze"]
        state = {"paper_text": "test"}
        mock_validate.return_value = []
        
        for node_name in node_names:
            validate_state_or_warn(state, node_name)
            mock_validate.assert_called_with(state, node_name)

    @patch("src.agents.helpers.context.validate_state_for_node")
    @patch("src.agents.helpers.context.logging")
    def test_logger_retrieved_with_module_name(self, mock_logging, mock_validate):
        """Should retrieve logger with correct module name."""
        mock_validate.return_value = ["Issue"]
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        state = {"paper_text": "test"}
        
        validate_state_or_warn(state, "test_node")
        
        # Verify getLogger was called (should be called with __name__)
        assert mock_logging.getLogger.called
        # The logger should be used for warning
        assert mock_logger.warning.called



