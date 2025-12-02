"""Unit tests for src/agents/base.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.base import (
    with_context_check,
    bounded_increment,
    parse_user_response,
    check_keywords,
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
    def test_merges_state_updates_and_continues(self, mock_check):
        """Should merge context updates into state and continue."""
        mock_check.return_value = {"metrics": {"tokens": 100}}
        
        @with_context_check("test_node")
        def test_node(state):
            # Verify state has been updated
            assert state.get("metrics") == {"tokens": 100}
            return {"result": "success"}
        
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

