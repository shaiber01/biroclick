"""Unit tests for src/agents/base.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.base import (
    with_context_check,
    bounded_increment,
    parse_user_response,
    check_keywords,
    increment_counter_with_max,
    create_llm_error_auto_approve,
    create_llm_error_escalation,
    create_llm_error_fallback,
)


class TestWithContextCheck:
    """Tests for with_context_check decorator."""

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_when_awaiting_input(self, mock_check):
        """Should return escalation when context check requires user input."""
        mock_check.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        @with_context_check("test_node")
        def test_node(state):
            return {"result": "success"}
        
        result = test_node({"some": "state"})
        
        assert result["awaiting_user_input"] is True
        assert result["pending_user_questions"] == ["Context overflow"]

    @patch("src.agents.base.check_context_or_escalate")
    def test_updates_state_before_calling_func(self, mock_check):
        """Should update state with context changes BEFORE calling function."""
        mock_check.return_value = {"metrics": {"tokens": 100}}
        
        @with_context_check("test_node")
        def test_node(state):
            # Verify state has been updated inside function
            if state.get("metrics") != {"tokens": 100}:
                raise ValueError("State was not updated before function call")
            return {"result": "success"}
        
        # This should NOT raise ValueError
        result = test_node({"original": "data"})
        assert result["result"] == "success"

    @patch("src.agents.base.check_context_or_escalate")
    def test_continues_when_no_escalation(self, mock_check):
        """Should continue to wrapped function when no escalation needed."""
        mock_check.return_value = None
        
        @with_context_check("test_node")
        def test_node(state):
            return {"workflow_phase": "testing", "data": state.get("data")}
        
        result = test_node({"data": "test_value"})
        
        assert result["workflow_phase"] == "testing"
        assert result["data"] == "test_value"

    @patch("src.agents.base.check_context_or_escalate")
    def test_calls_check_with_correct_node_name(self, mock_check):
        """Should call check_context_or_escalate with correct node name."""
        mock_check.return_value = None
        
        @with_context_check("my_custom_node")
        def my_node(state):
            return {}
        
        my_node({"state": "data"})
        
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        assert call_args[0][1] == "my_custom_node"

    @patch("src.agents.base.check_context_or_escalate")
    def test_preserves_function_metadata(self, mock_check):
        """Should preserve function name and docstring."""
        mock_check.return_value = None
        
        @with_context_check("test")
        def my_documented_function(state):
            """This is my docstring."""
            return {}
        
        assert my_documented_function.__name__ == "my_documented_function"
        assert "docstring" in my_documented_function.__doc__


class TestBoundedIncrement:
    """Tests for bounded_increment function."""

    def test_increments_below_max(self):
        """Should increment when below max."""
        assert bounded_increment(0, 5) == 1
        assert bounded_increment(2, 5) == 3
        assert bounded_increment(4, 5) == 5

    def test_stops_at_max(self):
        """Should not exceed max value."""
        assert bounded_increment(5, 5) == 5
        assert bounded_increment(10, 5) == 5

    def test_handles_zero_max(self):
        """Should handle zero max value."""
        assert bounded_increment(0, 0) == 0


class TestParseUserResponse:
    """Tests for parse_user_response function."""

    def test_returns_empty_for_empty_dict(self):
        """Should return empty string for empty responses."""
        assert parse_user_response({}) == ""

    def test_returns_last_response_uppercased(self):
        """Should return last response uppercased."""
        responses = {
            "Q1": "first answer",
            "Q2": "second answer",
            "Q3": "approve",
        }
        assert parse_user_response(responses) == "APPROVE"

    def test_handles_non_string_values(self):
        """Should convert non-string values to string."""
        responses = {"Q1": 123}
        assert parse_user_response(responses) == "123"

    def test_handles_single_response(self):
        """Should handle single response correctly."""
        responses = {"Question?": "yes"}
        assert parse_user_response(responses) == "YES"


class TestCheckKeywords:
    """Tests for check_keywords function."""

    def test_finds_present_keyword(self):
        """Should return True when keyword is present."""
        assert check_keywords("APPROVE THE PLAN", ["APPROVE", "REJECT"]) is True
        assert check_keywords("I REJECT THIS", ["APPROVE", "REJECT"]) is True

    def test_returns_false_when_no_match(self):
        """Should return False when no keyword matches."""
        assert check_keywords("MAYBE", ["APPROVE", "REJECT"]) is False

    def test_handles_empty_keywords(self):
        """Should return False for empty keywords list."""
        assert check_keywords("APPROVE", []) is False

    def test_handles_empty_response(self):
        """Should return False for empty response."""
        assert check_keywords("", ["APPROVE"]) is False

    def test_partial_match(self):
        """Should match partial words."""
        assert check_keywords("APPROVED BY USER", ["APPROVE"]) is True


class TestIncrementCounterWithMax:
    """Tests for increment_counter_with_max function."""

    def test_increments_when_below_max(self):
        """Should increment counter when below max."""
        state = {
            "code_revision_count": 1,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 2
        assert was_incremented is True

    def test_stops_at_max(self):
        """Should not increment when at max."""
        state = {
            "code_revision_count": 5,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 5
        assert was_incremented is False

    def test_uses_default_max_when_not_in_config(self):
        """Should use default max when not in runtime_config."""
        state = {
            "code_revision_count": 1,
            "runtime_config": {},  # No max_code_revisions
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3  # default_max=3
        )
        
        assert new_count == 2
        assert was_incremented is True

    def test_handles_missing_counter(self):
        """Should default counter to 0 if missing."""
        state = {"runtime_config": {}}  # No counter in state
        
        new_count, was_incremented = increment_counter_with_max(
            state, "missing_counter", "max_missing", 5
        )
        
        assert new_count == 1
        assert was_incremented is True

    def test_handles_missing_runtime_config(self):
        """Should handle missing runtime_config."""
        state = {"code_revision_count": 0}  # No runtime_config
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 1
        assert was_incremented is True


class TestCreateLlmErrorAutoApprove:
    """Tests for create_llm_error_auto_approve function."""

    def test_creates_approve_verdict(self):
        """Should create response with approve verdict."""
        error = Exception("API timeout")
        
        result = create_llm_error_auto_approve("code_reviewer", error)
        
        assert result["verdict"] == "approve"
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "minor"
        assert "API timeout" in result["issues"][0]["description"]
        # Verify summary format
        assert result["summary"] == "Code Reviewer auto-approved due to LLM unavailability"

    def test_creates_pass_verdict_for_validators(self):
        """Should create response with pass verdict for validators."""
        error = Exception("Connection error")
        
        result = create_llm_error_auto_approve("execution_validator", error, default_verdict="pass")
        
        assert result["verdict"] == "pass"
        assert result["summary"] == "Execution Validator auto-passed due to LLM unavailability"

    def test_truncates_long_error_message(self):
        """Should truncate long error messages."""
        long_error = Exception("A" * 500)
        
        result = create_llm_error_auto_approve("test_agent", long_error, error_truncate_len=50)
        
        # The description contains "LLM review unavailable: " + error_msg
        description = result["issues"][0]["description"]
        assert len(description) < 100 # "LLM review unavailable: " is ~24 chars + 50 = 74
        assert "..." not in description # Python slice doesn't add ellipses automatically, unless implemented manually
        # Wait, the implementation is just str(error)[:len].
        # So if str(error) is "A"*500, slicing to 50 gives "A"*50.
        # The test should verify it matches exactly the sliced version.
        
        expected_msg = ("A" * 500)[:50]
        assert expected_msg in description

    def test_includes_summary(self):
        """Should include summary message."""
        error = Exception("Test error")
        
        result = create_llm_error_auto_approve("code_reviewer", error)
        
        assert "Code Reviewer" in result["summary"]
        assert "auto-approve" in result["summary"].lower()


class TestCreateLlmErrorEscalation:
    """Tests for create_llm_error_escalation function."""

    def test_creates_escalation_response(self):
        """Should create user escalation response."""
        error = Exception("API key invalid")
        
        result = create_llm_error_escalation("code_generator", "code_generation", error)
        
        assert result["workflow_phase"] == "code_generation"
        assert result["ask_user_trigger"] == "llm_error"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) == 1

    def test_includes_error_in_question(self):
        """Should include error message in question."""
        error = Exception("API key invalid")
        
        result = create_llm_error_escalation("planner", "planning", error)
        
        assert "API key invalid" in result["pending_user_questions"][0]

    def test_truncates_long_error_in_question(self):
        """Should truncate long error messages in question."""
        long_error = Exception("B" * 1000)
        
        result = create_llm_error_escalation("test", "test", long_error, error_truncate_len=100)
        
        assert len(result["pending_user_questions"][0]) < 200

    def test_formats_agent_name(self):
        """Should format agent name in question."""
        error = Exception("Error")
        
        result = create_llm_error_escalation("code_generator", "code_generation", error)
        
        assert "Code Generator" in result["pending_user_questions"][0]
        # Check full message structure
        assert "Code Generator failed: Error. Please check API and try again." in result["pending_user_questions"][0]


class TestCreateLlmErrorFallback:
    """Tests for create_llm_error_fallback function."""

    def test_creates_fallback_handler(self):
        """Should create a callable fallback handler."""
        handler = create_llm_error_fallback("supervisor", "ok_continue")
        
        assert callable(handler)

    def test_handler_returns_verdict_and_feedback(self):
        """Should return dict with verdict and feedback."""
        handler = create_llm_error_fallback("supervisor", "ok_continue")
        error = Exception("Test error")
        
        result = handler(error)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Test error" in result["supervisor_feedback"]

    def test_uses_custom_feedback_format(self):
        """Should use custom feedback format if provided."""
        handler = create_llm_error_fallback(
            "supervisor", "ok_continue", 
            feedback_msg="Custom message: {error}"
        )
        error = Exception("Connection lost")
        
        result = handler(error)
        
        assert result["supervisor_feedback"] == "Custom message: Connection lost"

    def test_truncates_error_in_feedback(self):
        """Should truncate long errors in feedback."""
        handler = create_llm_error_fallback("test", "default", error_truncate_len=20)
        error = Exception("C" * 100)
        
        result = handler(error)
        
        assert len(result["test_feedback"]) < 50

