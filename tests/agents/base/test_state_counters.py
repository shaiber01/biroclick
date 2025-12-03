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
        # Verify exact boundary: max - 1 should increment to max
        assert bounded_increment(4, 5) == 5
        assert bounded_increment(0, 1) == 1

    def test_stops_at_max(self):
        """Should not exceed max value."""
        assert bounded_increment(5, 5) == 5
        assert bounded_increment(10, 5) == 5
        assert bounded_increment(100, 5) == 5
        # Verify exact max boundary
        assert bounded_increment(5, 5) == 5
        assert bounded_increment(1, 1) == 1

    def test_handles_zero_max(self):
        """Should handle zero max value."""
        assert bounded_increment(0, 0) == 0
        assert bounded_increment(-1, 0) == 0
        assert bounded_increment(5, 0) == 0

    def test_handles_negative_current(self):
        """Should handle negative current values correctly."""
        assert bounded_increment(-1, 5) == 0
        assert bounded_increment(-5, 5) == -4
        assert bounded_increment(-10, 5) == -9
        # Negative current with negative max: min(current + 1, max)
        assert bounded_increment(-5, -2) == -4  # min(-4, -2) = -4

    def test_handles_negative_max(self):
        """Should handle negative max values correctly."""
        # If max is negative: min(current + 1, max)
        assert bounded_increment(0, -5) == -5  # min(1, -5) = -5
        assert bounded_increment(-1, -5) == -5  # min(0, -5) = -5
        assert bounded_increment(-6, -5) == -5  # min(-5, -5) = -5
        assert bounded_increment(-10, -5) == -9  # min(-9, -5) = -9 (current+1 is less than max)

    def test_behavior_when_current_exceeds_max(self):
        """
        Verify behavior when current is already above max.
        Current implementation: min(current + 1, max)
        If current=10, max=5 -> min(11, 5) = 5.
        This clamps the value down to max immediately.
        """
        assert bounded_increment(10, 5) == 5
        assert bounded_increment(100, 5) == 5
        assert bounded_increment(6, 5) == 5

    def test_type_errors_reveal_bugs(self):
        """Should fail with type errors for invalid inputs - these reveal bugs."""
        # These tests should FAIL if implementation doesn't handle types correctly
        with pytest.raises((TypeError, AttributeError)):
            bounded_increment(None, 5)
        
        with pytest.raises((TypeError, AttributeError)):
            bounded_increment(5, None)
        
        with pytest.raises((TypeError, AttributeError)):
            bounded_increment("5", 5)
        
        with pytest.raises((TypeError, AttributeError)):
            bounded_increment(5, "5")

    def test_large_numbers(self):
        """Should handle large numbers correctly."""
        large_num = 10**10
        assert bounded_increment(large_num - 1, large_num) == large_num
        assert bounded_increment(large_num, large_num) == large_num
        assert bounded_increment(large_num + 1, large_num) == large_num

class TestParseUserResponse:
    """Tests for parse_user_response function."""

    def test_returns_empty_for_empty_dict(self):
        """Should return empty string for empty responses."""
        result = parse_user_response({})
        assert result == ""
        assert isinstance(result, str)
        assert len(result) == 0

    def test_returns_last_response_uppercased(self):
        """Should return last response uppercased."""
        responses = {
            "Q1": "first answer",
            "Q2": "second answer",
            "Q3": "approve",
        }
        result = parse_user_response(responses)
        assert result == "APPROVE"
        assert result.isupper()
        # Verify it's actually the LAST value, not first
        responses2 = {
            "Q3": "approve",
            "Q1": "first answer",
            "Q2": "second answer",
        }
        result2 = parse_user_response(responses2)
        assert result2 == "SECOND ANSWER"  # Last insertion order

    def test_handles_non_string_values(self):
        """Should convert non-string values to string."""
        assert parse_user_response({"Q1": 123}) == "123"
        assert parse_user_response({"Q1": 0}) == "0"
        assert parse_user_response({"Q1": -42}) == "-42"
        assert parse_user_response({"Q1": 3.14}) == "3.14"
        # Boolean values are uppercased
        assert parse_user_response({"Q1": True}) == "TRUE"
        assert parse_user_response({"Q1": False}) == "FALSE"

    def test_handles_single_response(self):
        """Should handle single response correctly."""
        responses = {"Question?": "yes"}
        result = parse_user_response(responses)
        assert result == "YES"
        assert result.isupper()

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        # This test currently FAILS if implementation doesn't strip
        assert parse_user_response({"Q1": "  yes  "}) == "YES"
        assert parse_user_response({"Q1": "\t\napprove\t\n"}) == "APPROVE"
        assert parse_user_response({"Q1": "   "}) == ""  # Whitespace-only becomes empty
        assert parse_user_response({"Q1": "  test  test  "}) == "TEST  TEST"  # Only strips edges

    def test_handles_none_responses(self):
        """Should handle None values gracefully."""
        # If dictionary values can be None (e.g. failed to capture), it should handle it
        result = parse_user_response({"Q1": None})
        assert result == "NONE"  # str(None) == "None", then uppercased
        assert isinstance(result, str)

    def test_handles_empty_strings(self):
        """Should handle empty string values."""
        assert parse_user_response({"Q1": ""}) == ""
        assert parse_user_response({"Q1": "   "}) == ""  # Whitespace-only
        # Empty string as last value
        assert parse_user_response({"Q1": "yes", "Q2": ""}) == ""

    def test_handles_complex_objects(self):
        """Should convert complex objects to string representation."""
        # These should not crash, but may produce unexpected strings
        result = parse_user_response({"Q1": [1, 2, 3]})
        assert isinstance(result, str)
        assert result == "[1, 2, 3]"
        
        result = parse_user_response({"Q1": {"nested": "dict"}})
        assert isinstance(result, str)
        # Exact format depends on dict ordering, but should be uppercased
        assert result.isupper() or result == ""

    def test_preserves_order_dependency(self):
        """Should return last value based on insertion order (Python 3.7+ dict order)."""
        # In Python 3.7+, dicts preserve insertion order
        responses1 = {"first": "a", "second": "b", "third": "c"}
        assert parse_user_response(responses1) == "C"
        
        responses2 = {"third": "c", "first": "a", "second": "b"}
        assert parse_user_response(responses2) == "B"  # Last insertion

    def test_type_errors_reveal_bugs(self):
        """Should fail with type errors for invalid inputs - these reveal bugs."""
        # Note: Current implementation handles None gracefully (None is falsy, returns "")
        # This might be intentional behavior, but we test it explicitly
        assert parse_user_response(None) == ""  # None is falsy, so returns ""
        
        # These should fail with type errors
        with pytest.raises((TypeError, AttributeError)):
            parse_user_response("not a dict")
        
        with pytest.raises((TypeError, AttributeError)):
            parse_user_response([])

class TestCheckKeywords:
    """Tests for check_keywords function."""

    def test_finds_present_keyword(self):
        """Should return True when keyword is present."""
        assert check_keywords("APPROVE THE PLAN", ["APPROVE", "REJECT"]) is True
        assert check_keywords("I REJECT THIS", ["APPROVE", "REJECT"]) is True
        # Keyword at start
        assert check_keywords("APPROVE", ["APPROVE"]) is True
        # Keyword at end
        assert check_keywords("THE PLAN IS APPROVE", ["APPROVE"]) is True
        # Keyword in middle
        assert check_keywords("I APPROVE THIS", ["APPROVE"]) is True

    def test_returns_false_when_no_match(self):
        """Should return False when no keyword matches."""
        assert check_keywords("MAYBE", ["APPROVE", "REJECT"]) is False
        assert check_keywords("SOMETHING ELSE", ["APPROVE"]) is False
        assert check_keywords("NOTHING", ["APPROVE", "REJECT"]) is False

    def test_handles_empty_keywords(self):
        """Should return False for empty keywords list."""
        assert check_keywords("APPROVE", []) is False
        assert check_keywords("ANYTHING", []) is False

    def test_handles_empty_response(self):
        """Should return False for empty response."""
        assert check_keywords("", ["APPROVE"]) is False
        assert check_keywords("   ", ["APPROVE"]) is False  # Whitespace-only

    def test_partial_match_handling(self):
        """Should NOT match false positives where keyword is part of another word."""
        # Example: 'DISAPPROVE' should NOT match 'APPROVE' if we want strict keyword matching.
        # The implementation uses word boundaries, so this should work correctly.
        assert check_keywords("DISAPPROVE", ["APPROVE"]) is False
        assert check_keywords("SOMEOTHERWORD", ["WORD"]) is False
        assert check_keywords("UNAPPROVED", ["APPROVE"]) is False
        assert check_keywords("APPROVING", ["APPROVE"]) is False  # Different word form

    def test_case_insensitivity(self):
        """Should handle mixed case input robustly."""
        # The docstring says input 'should be uppercased', but robust code should handle it.
        assert check_keywords("approve", ["APPROVE"]) is True
        assert check_keywords("Approve", ["APPROVE"]) is True
        assert check_keywords("APPROVE", ["approve"]) is True
        assert check_keywords("ApPrOvE", ["APPROVE"]) is True

    def test_keyword_with_punctuation(self):
        """Should match keywords even when surrounded by punctuation."""
        assert check_keywords("APPROVE!", ["APPROVE"]) is True
        assert check_keywords("APPROVE.", ["APPROVE"]) is True
        assert check_keywords("(APPROVE)", ["APPROVE"]) is True
        assert check_keywords("APPROVE, please", ["APPROVE"]) is True

    def test_multiple_keywords(self):
        """Should return True if any keyword matches."""
        assert check_keywords("APPROVE", ["APPROVE", "REJECT"]) is True
        assert check_keywords("REJECT", ["APPROVE", "REJECT"]) is True
        assert check_keywords("APPROVE OR REJECT", ["APPROVE", "REJECT"]) is True

    def test_empty_keyword_in_list(self):
        """Should skip empty keywords in the list."""
        # Empty strings should be skipped, so should return False
        assert check_keywords("APPROVE", ["", "APPROVE"]) is True
        assert check_keywords("APPROVE", ["", ""]) is False
        assert check_keywords("ANYTHING", ["", "APPROVE"]) is False

    def test_regex_special_characters_in_keyword(self):
        """Should escape regex special characters in keywords."""
        # These should not cause regex errors and should match literally
        assert check_keywords("TEST.123", ["TEST.123"]) is True
        assert check_keywords("TEST*123", ["TEST*123"]) is True
        assert check_keywords("TEST+123", ["TEST+123"]) is True
        assert check_keywords("TEST(123)", ["TEST(123)"]) is True
        assert check_keywords("TEST[123]", ["TEST[123]"]) is True
        assert check_keywords("TEST^123", ["TEST^123"]) is True
        assert check_keywords("TEST$123", ["TEST$123"]) is True

    def test_regex_special_characters_in_response(self):
        """Should handle regex special characters in response."""
        # Response is uppercased and searched as literal text
        # Word boundaries work: TEST is a whole word in "TEST.123" (before the dot)
        assert check_keywords("TEST.123", ["TEST"]) is True  # TEST is a whole word before the dot
        assert check_keywords("TEST*123", ["TEST"]) is True  # TEST is a whole word before the *
        assert check_keywords("TEST+123", ["TEST"]) is True  # TEST is a whole word before the +
        # But not when TEST is part of another word
        assert check_keywords("TESTING.123", ["TEST"]) is False  # TEST is part of TESTING

    def test_type_errors_reveal_bugs(self):
        """Should fail with type errors for invalid inputs - these reveal bugs."""
        # Current implementation handles None response gracefully (returns False)
        assert check_keywords(None, ["APPROVE"]) is False  # None is falsy
        
        # These should fail with type errors
        with pytest.raises((TypeError, AttributeError)):
            check_keywords("APPROVE", None)
        
        # Non-string keyword should be converted to string, then uppercased
        # This should work now that we convert to str(kw).upper()
        assert check_keywords("APPROVE", [123]) is False  # "123" doesn't match "APPROVE"
        assert check_keywords("123", [123]) is True  # "123" matches "123"
        
        # None keyword should be skipped (falsy check skips it)
        assert check_keywords("APPROVE", [None]) is False  # None is skipped, not converted
        assert check_keywords("NONE", [None]) is False  # None is skipped, not converted to "NONE"

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
        assert isinstance(new_count, int)
        assert isinstance(was_incremented, bool)
        
        # Test boundary: max - 1 should increment to max
        state2 = {
            "code_revision_count": 4,
            "runtime_config": {"max_code_revisions": 5},
        }
        new_count2, was_incremented2 = increment_counter_with_max(
            state2, "code_revision_count", "max_code_revisions", 3
        )
        assert new_count2 == 5
        assert was_incremented2 is True

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
        assert isinstance(new_count, int)
        assert isinstance(was_incremented, bool)

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
        
        # Test that default max is actually used
        state2 = {
            "code_revision_count": 3,
            "runtime_config": {},
        }
        new_count2, was_incremented2 = increment_counter_with_max(
            state2, "code_revision_count", "max_code_revisions", 3
        )
        assert new_count2 == 3  # At default max, should not increment
        assert was_incremented2 is False

    def test_handles_missing_counter(self):
        """Should default counter to 0 if missing."""
        state = {"runtime_config": {}}  # No counter in state
        
        new_count, was_incremented = increment_counter_with_max(
            state, "missing_counter", "max_missing", 5
        )
        
        assert new_count == 1
        assert was_incremented is True
        # Verify counter defaults to 0
        assert state.get("missing_counter") is None  # Original state unchanged

    def test_handles_missing_runtime_config(self):
        """Should handle missing runtime_config."""
        state = {"code_revision_count": 0}  # No runtime_config
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 1
        assert was_incremented is True
        
        # Test with None runtime_config explicitly
        state2 = {"code_revision_count": 0, "runtime_config": None}
        new_count2, was_incremented2 = increment_counter_with_max(
            state2, "code_revision_count", "max_code_revisions", 3
        )
        assert new_count2 == 1
        assert was_incremented2 is True

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
        # Verify original state is not modified
        assert state["code_revision_count"] == 10

    def test_handles_negative_counter(self):
        """Should handle negative counter values."""
        state = {
            "code_revision_count": -1,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 0
        assert was_incremented is True

    def test_handles_negative_max(self):
        """Should handle negative max values."""
        state = {
            "code_revision_count": 0,
            "runtime_config": {"max_code_revisions": -5},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        # If max is negative and current is 0, 0 < -5 is False, so should not increment
        assert new_count == 0
        assert was_incremented is False

    def test_handles_zero_max(self):
        """Should handle zero max value."""
        state = {
            "code_revision_count": 0,
            "runtime_config": {"max_code_revisions": 0},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3
        )
        
        assert new_count == 0
        assert was_incremented is False

    def test_handles_zero_default_max(self):
        """Should handle zero default max."""
        state = {
            "code_revision_count": 0,
            "runtime_config": {},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 0
        )
        
        assert new_count == 0
        assert was_incremented is False

    def test_config_overrides_default(self):
        """Should use config value over default when both exist."""
        state = {
            "code_revision_count": 1,
            "runtime_config": {"max_code_revisions": 10},
        }
        
        new_count, was_incremented = increment_counter_with_max(
            state, "code_revision_count", "max_code_revisions", 3  # default ignored
        )
        
        assert new_count == 2
        assert was_incremented is True
        
        # Verify config max is used, not default
        state2 = {
            "code_revision_count": 10,
            "runtime_config": {"max_code_revisions": 10},
        }
        new_count2, was_incremented2 = increment_counter_with_max(
            state2, "code_revision_count", "max_code_revisions", 3
        )
        assert new_count2 == 10
        assert was_incremented2 is False

    def test_type_errors_reveal_bugs(self):
        """Should fail with type errors for invalid inputs - these reveal bugs."""
        # These tests should FAIL if implementation doesn't handle types correctly
        with pytest.raises((TypeError, AttributeError, KeyError)):
            increment_counter_with_max(None, "counter", "max_key", 5)
        
        # Counter value is None - should be treated as 0 (fixed bug)
        state_none = {
            "code_revision_count": None,
            "runtime_config": {"max_code_revisions": 5},
        }
        new_count, was_incremented = increment_counter_with_max(state_none, "code_revision_count", "max_code_revisions", 3)
        assert new_count == 1
        assert was_incremented is True
        
        # Counter value is string - should crash on comparison
        state_str = {
            "code_revision_count": "5",
            "runtime_config": {"max_code_revisions": 5},
        }
        with pytest.raises((TypeError, AttributeError)):
            increment_counter_with_max(state_str, "code_revision_count", "max_code_revisions", 3)
        
        # Max value from config is None - should use default_max (handled gracefully)
        state_max_none = {
            "code_revision_count": 1,
            "runtime_config": {"max_code_revisions": None},
        }
        new_count3, was_incremented3 = increment_counter_with_max(state_max_none, "code_revision_count", "max_code_revisions", 3)
        assert new_count3 == 2
        assert was_incremented3 is True  # Uses default_max=3, so 1 < 3, increments to 2
        
        # Max value from config is string - should crash on comparison
        state_max_str = {
            "code_revision_count": 1,
            "runtime_config": {"max_code_revisions": "5"},
        }
        with pytest.raises((TypeError, AttributeError)):
            increment_counter_with_max(state_max_str, "code_revision_count", "max_code_revisions", 3)
        
        # Runtime_config is not a dict - should crash on .get()
        state_bad_config = {
            "code_revision_count": 1,
            "runtime_config": "not a dict",
        }
        with pytest.raises((TypeError, AttributeError)):
            increment_counter_with_max(state_bad_config, "code_revision_count", "max_code_revisions", 3)

