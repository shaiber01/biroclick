"""Unit tests for the check_paper_length helper."""

import pytest

from src.paper_loader import (
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_VERY_LONG,
    check_paper_length,
)
from src.paper_loader.config import CHARS_PER_TOKEN


class TestCheckPaperLength:
    """Tests for check_paper_length function."""

    def test_normal_length_no_warnings(self):
        """Normal length paper returns no warnings."""
        text = "A" * PAPER_LENGTH_NORMAL
        warnings = check_paper_length(text)
        assert warnings == []

    def test_below_long_threshold_no_warning(self):
        """Text just below long threshold returns no warnings."""
        text = "A" * (PAPER_LENGTH_LONG - 1)
        warnings = check_paper_length(text)
        assert warnings == []

    def test_exactly_long_threshold_no_warning(self):
        """Text exactly at long threshold returns no warnings (boundary is exclusive)."""
        text = "A" * PAPER_LENGTH_LONG
        warnings = check_paper_length(text)
        assert warnings == []

    def test_just_above_long_threshold_warning(self):
        """Text just above long threshold returns long warning."""
        text = "A" * (PAPER_LENGTH_LONG + 1)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0]
        # Verify warning contains char count
        assert str(PAPER_LENGTH_LONG + 1) in warnings[0] or f"{PAPER_LENGTH_LONG + 1:,}" in warnings[0]
        # Verify warning contains token estimate
        expected_tokens = (PAPER_LENGTH_LONG + 1) // CHARS_PER_TOKEN
        assert str(expected_tokens) in warnings[0] or f"{expected_tokens:,}" in warnings[0]

    def test_between_long_and_very_long_warning(self):
        """Text between long and very long thresholds returns long warning."""
        text = "A" * (PAPER_LENGTH_LONG + 50_000)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0]

    def test_exactly_very_long_threshold_warning(self):
        """Text exactly at very long threshold returns long warning (not VERY LONG)."""
        text = "A" * PAPER_LENGTH_VERY_LONG
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0]
        # Verify warning contains char count
        assert str(PAPER_LENGTH_VERY_LONG) in warnings[0] or f"{PAPER_LENGTH_VERY_LONG:,}" in warnings[0]

    def test_just_above_very_long_threshold_warning(self):
        """Text just above very long threshold returns VERY LONG warning."""
        text = "A" * (PAPER_LENGTH_VERY_LONG + 1)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "VERY LONG" in warnings[0]
        # Verify warning contains char count
        assert str(PAPER_LENGTH_VERY_LONG + 1) in warnings[0] or f"{PAPER_LENGTH_VERY_LONG + 1:,}" in warnings[0]
        # Verify warning contains token estimate
        expected_tokens = (PAPER_LENGTH_VERY_LONG + 1) // CHARS_PER_TOKEN
        assert str(expected_tokens) in warnings[0] or f"{expected_tokens:,}" in warnings[0]

    def test_very_large_text_very_long_warning(self):
        """Very large text returns VERY LONG warning."""
        text = "A" * (PAPER_LENGTH_VERY_LONG * 2)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "VERY LONG" in warnings[0]

    def test_long_warning_message_format(self):
        """Long warning message has correct format and content."""
        text = "A" * (PAPER_LENGTH_LONG + 1000)
        warnings = check_paper_length(text, label="TestPaper")
        assert len(warnings) == 1
        warning = warnings[0]
        # Check label appears
        assert "TestPaper" in warning
        # Check char count appears (formatted with commas)
        assert f"{len(text):,}" in warning or str(len(text)) in warning
        # Check token estimate appears
        expected_tokens = len(text) // CHARS_PER_TOKEN
        assert f"{expected_tokens:,}" in warning or str(expected_tokens) in warning
        # Check warning mentions tokens
        assert "token" in warning.lower()
        # Check warning mentions cost reduction suggestion
        assert "trimming" in warning.lower() or "reduce" in warning.lower()

    def test_very_long_warning_message_format(self):
        """VERY LONG warning message has correct format and content."""
        text = "A" * (PAPER_LENGTH_VERY_LONG + 1000)
        warnings = check_paper_length(text, label="TestPaper")
        assert len(warnings) == 1
        warning = warnings[0]
        # Check label appears
        assert "TestPaper" in warning
        # Check char count appears (formatted with commas)
        assert f"{len(text):,}" in warning or str(len(text)) in warning
        # Check token estimate appears
        expected_tokens = len(text) // CHARS_PER_TOKEN
        assert f"{expected_tokens:,}" in warning or str(expected_tokens) in warning
        # Check warning mentions tokens
        assert "token" in warning.lower()
        # Check warning mentions context limits
        assert "context" in warning.lower() or "limit" in warning.lower()
        # Check warning mentions cost increase
        assert "cost" in warning.lower() or "increase" in warning.lower()

    def test_custom_label_in_long_warning(self):
        """Custom label appears in long warning."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="Supplementary")
        assert len(warnings) == 1
        assert "Supplementary" in warnings[0]
        assert warnings[0].startswith("Supplementary")

    def test_custom_label_in_very_long_warning(self):
        """Custom label appears in VERY LONG warning."""
        text = "A" * (PAPER_LENGTH_VERY_LONG + 100)
        warnings = check_paper_length(text, label="Supplementary")
        assert len(warnings) == 1
        assert "Supplementary" in warnings[0]
        assert warnings[0].startswith("Supplementary")

    def test_default_label_is_paper(self):
        """Default label is 'Paper'."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert warnings[0].startswith("Paper")

    def test_default_label_in_very_long_warning(self):
        """Default label 'Paper' appears in VERY LONG warning."""
        text = "A" * (PAPER_LENGTH_VERY_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert warnings[0].startswith("Paper")

    def test_empty_string(self):
        """Empty string returns no warnings."""
        warnings = check_paper_length("")
        assert warnings == []
        assert isinstance(warnings, list)

    def test_single_character(self):
        """Single character returns no warnings."""
        warnings = check_paper_length("A")
        assert warnings == []

    def test_whitespace_only_string(self):
        """Whitespace-only string returns no warnings if below threshold."""
        text = " " * PAPER_LENGTH_NORMAL
        warnings = check_paper_length(text)
        assert warnings == []

    def test_whitespace_only_long_string(self):
        """Whitespace-only long string returns warning."""
        text = " " * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()

    def test_newlines_in_text(self):
        """Text with newlines is counted correctly."""
        text = "A\n" * (PAPER_LENGTH_LONG // 2)
        warnings = check_paper_length(text)
        assert warnings == []
        # Add more to trigger warning
        text = "A\n" * ((PAPER_LENGTH_LONG + 100) // 2)
        warnings = check_paper_length(text)
        assert len(warnings) == 1

    def test_unicode_characters(self):
        """Unicode characters are counted correctly."""
        text = "α" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        # Verify char count includes unicode chars
        assert str(len(text)) in warnings[0] or f"{len(text):,}" in warnings[0]

    def test_mixed_content_text(self):
        """Text with mixed content (letters, numbers, symbols) is handled correctly."""
        text = "A1!@#" * ((PAPER_LENGTH_LONG + 100) // 5)
        warnings = check_paper_length(text)
        assert len(warnings) == 1

    def test_token_estimate_calculation(self):
        """Token estimate is calculated correctly using CHARS_PER_TOKEN."""
        text = "A" * (CHARS_PER_TOKEN * 1000)  # Exactly 1000 tokens
        warnings = check_paper_length(text)
        assert warnings == []  # Should be below threshold
        
        text = "A" * (CHARS_PER_TOKEN * 10000 + PAPER_LENGTH_LONG)  # Large number of tokens
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        # Verify token estimate is approximately correct
        expected_tokens = len(text) // CHARS_PER_TOKEN
        warning = warnings[0]
        # Token count should appear in warning
        assert str(expected_tokens) in warning or f"{expected_tokens:,}" in warning

    def test_none_input_raises_error(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length(None)  # type: ignore[arg-type]

    def test_integer_input_raises_error(self):
        """Integer input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length(12345)  # type: ignore[arg-type]

    def test_list_input_raises_error(self):
        """List input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length(["text"])  # type: ignore[arg-type]

    def test_dict_input_raises_error(self):
        """Dictionary input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length({"text": "value"})  # type: ignore[arg-type]

    def test_bytes_input_raises_error(self):
        """Bytes input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length(b"text")  # type: ignore[arg-type]

    def test_return_type_is_list(self):
        """Function always returns a list."""
        warnings = check_paper_length("")
        assert isinstance(warnings, list)
        
        warnings = check_paper_length("A" * PAPER_LENGTH_NORMAL)
        assert isinstance(warnings, list)
        
        warnings = check_paper_length("A" * (PAPER_LENGTH_LONG + 100))
        assert isinstance(warnings, list)
        assert len(warnings) == 1

    def test_idempotent_calls(self):
        """Multiple calls with same input return same results."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings1 = check_paper_length(text)
        warnings2 = check_paper_length(text)
        assert warnings1 == warnings2
        assert warnings1 is not warnings2  # Should be new list each time

    def test_warning_is_single_string_in_list(self):
        """Warning is a single string in a list, not nested."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert isinstance(warnings[0], str)
        assert not isinstance(warnings[0], list)

    def test_only_one_warning_returned(self):
        """Only one warning is returned even for very long text."""
        text = "A" * (PAPER_LENGTH_VERY_LONG * 10)
        warnings = check_paper_length(text)
        assert len(warnings) == 1  # Should only have VERY LONG warning, not both

    def test_warning_contains_helpful_suggestions(self):
        """Warning messages contain helpful suggestions."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        # Long warning should mention trimming
        assert "trimming" in warnings[0].lower() or "reduce" in warnings[0].lower()
        
        text = "A" * (PAPER_LENGTH_VERY_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        # VERY LONG warning should mention removing sections
        assert "removing" in warnings[0].lower() or "sections" in warnings[0].lower() or "references" in warnings[0].lower()

    def test_char_count_formatting(self):
        """Char count in warning is formatted with commas for readability."""
        text = "A" * 200_000  # Should have comma formatting
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        # Check that formatted number appears (with comma)
        assert "200,000" in warnings[0] or "200000" in warnings[0]

    def test_token_count_formatting(self):
        """Token count in warning is formatted with commas for readability."""
        text = "A" * 200_000  # Should produce ~50,000 tokens
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        expected_tokens = 200_000 // CHARS_PER_TOKEN  # 50,000
        # Check that formatted number appears (with comma)
        assert f"{expected_tokens:,}" in warnings[0] or str(expected_tokens) in warnings[0]

    def test_empty_label(self):
        """Empty label is handled correctly."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="")
        assert len(warnings) == 1
        # Warning should still be generated, just without label prefix
        assert "is long" in warnings[0] or "long" in warnings[0].lower()

    def test_label_with_special_characters(self):
        """Label with special characters is handled correctly."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="Test-Paper_2024")
        assert len(warnings) == 1
        assert "Test-Paper_2024" in warnings[0]

    def test_label_with_unicode(self):
        """Label with unicode characters is handled correctly."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="论文")
        assert len(warnings) == 1
        assert "论文" in warnings[0]

    def test_exactly_normal_threshold(self):
        """Text exactly at normal threshold returns no warnings."""
        text = "A" * PAPER_LENGTH_NORMAL
        warnings = check_paper_length(text)
        assert warnings == []

    def test_just_below_normal_threshold(self):
        """Text just below normal threshold returns no warnings."""
        text = "A" * (PAPER_LENGTH_NORMAL - 1)
        warnings = check_paper_length(text)
        assert warnings == []

    def test_zero_length_string(self):
        """Zero-length string (empty) returns no warnings."""
        warnings = check_paper_length("")
        assert warnings == []
        assert len(warnings) == 0

    def test_warning_message_structure(self):
        """Warning messages have expected structure components."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="Test")
        assert len(warnings) == 1
        warning = warnings[0]
        # Should contain label
        assert "Test" in warning
        # Should contain "is long" or "is VERY LONG"
        assert "is" in warning
        assert ("long" in warning.lower() or "VERY LONG" in warning)
        # Should contain "chars"
        assert "char" in warning.lower()
        # Should contain "token"
        assert "token" in warning.lower()

    def test_very_long_warning_message_structure(self):
        """VERY LONG warning messages have expected structure components."""
        text = "A" * (PAPER_LENGTH_VERY_LONG + 100)
        warnings = check_paper_length(text, label="Test")
        assert len(warnings) == 1
        warning = warnings[0]
        # Should contain label
        assert "Test" in warning
        # Should contain "VERY LONG"
        assert "VERY LONG" in warning
        # Should contain "chars"
        assert "char" in warning.lower()
        # Should contain "token"
        assert "token" in warning.lower()
        # Should mention context limits or costs
        assert "context" in warning.lower() or "cost" in warning.lower() or "limit" in warning.lower()

    def test_warning_does_not_contain_duplicate_info(self):
        """Warning message should not contain duplicate information."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        warning = warnings[0]
        # Count occurrences of key terms - should appear once each
        assert warning.count("chars") == 1
        assert warning.count("tokens") == 1 or warning.count("token") == 1

    def test_token_estimate_matches_actual_calculation(self):
        """Token estimate in warning matches manual calculation."""
        text = "A" * 160_000  # Exactly divisible by CHARS_PER_TOKEN
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        expected_tokens = len(text) // CHARS_PER_TOKEN
        warning = warnings[0]
        # Extract token count from warning (look for number before "token")
        import re
        token_match = re.search(r'~(\d{1,3}(?:,\d{3})*)\s*token', warning)
        if token_match:
            token_str = token_match.group(1).replace(',', '')
            actual_tokens = int(token_str)
            assert actual_tokens == expected_tokens, f"Expected {expected_tokens} tokens, found {actual_tokens} in warning: {warning}"

    def test_char_count_matches_input_length(self):
        """Char count in warning matches actual input length."""
        text = "A" * 160_000
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        warning = warnings[0]
        # Extract char count from warning
        import re
        char_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*char', warning)
        if char_match:
            char_str = char_match.group(1).replace(',', '')
            actual_chars = int(char_str)
            assert actual_chars == len(text), f"Expected {len(text)} chars, found {actual_chars} in warning: {warning}"

    def test_warning_is_not_empty_string(self):
        """Warning string should not be empty."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert len(warnings[0]) > 0
        assert warnings[0].strip() != ""

    def test_multiple_calls_with_different_labels(self):
        """Multiple calls with different labels return different warnings."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings1 = check_paper_length(text, label="Paper1")
        warnings2 = check_paper_length(text, label="Paper2")
        assert len(warnings1) == 1
        assert len(warnings2) == 1
        assert warnings1[0] != warnings2[0]
        assert "Paper1" in warnings1[0]
        assert "Paper2" in warnings2[0]

