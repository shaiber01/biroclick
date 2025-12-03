"""Tests for shared counter/keyword utilities in src.agents.base."""

import pytest

from src.agents.base import (
    bounded_increment,
    check_keywords,
    increment_counter_with_max,
    parse_user_response,
)

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

    def test_handles_negative_current(self):
        """Should handle negative current values correctly."""
        assert bounded_increment(-1, 5) == 0
        assert bounded_increment(-5, 5) == -4

    def test_behavior_when_current_exceeds_max(self):
        """
        Verify behavior when current is already above max.
        Current implementation: min(current + 1, max)
        If current=10, max=5 -> min(11, 5) = 5.
        This clamps the value down to max immediately.
        """
        assert bounded_increment(10, 5) == 5

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

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        # This test currently FAILS if implementation doesn't strip
        responses = {"Q1": "  yes  "}
        assert parse_user_response(responses) == "YES"

    def test_handles_none_responses(self):
        """Should handle None values gracefully."""
        # If dictionary values can be None (e.g. failed to capture), it should handle it
        responses = {"Q1": None}
        assert parse_user_response(responses) == "NONE"

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

    def test_partial_match_handling(self):
        """Should NOT match false positives where keyword is part of another word."""
        # Example: 'DISAPPROVE' should NOT match 'APPROVE' if we want strict keyword matching.
        # But the current implementation uses `in` operator which is substring search.
        # This test asserts correct behavior for semantic keywords.
        # If 'APPROVE' is a command, 'DISAPPROVE' should not trigger it.
        assert check_keywords("DISAPPROVE", ["APPROVE"]) is False
        assert check_keywords("SOMEOTHERWORD", ["WORD"]) is False

    def test_case_insensitivity(self):
        """Should handle mixed case input robustly."""
        # The docstring says input 'should be uppercased', but robust code should handle it.
        assert check_keywords("approve", ["APPROVE"]) is True

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

    def test_behavior_when_exceeding_max(self):
        """Should not increment and should preserve value if already above max."""
        state = {
            "code_revision_count": 10,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 10
        assert was_incremented is False

