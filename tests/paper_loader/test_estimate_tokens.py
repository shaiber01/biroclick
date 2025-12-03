"""Unit tests for the estimate_tokens helper."""

import pytest

from src.paper_loader import CHARS_PER_TOKEN, estimate_tokens


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_basic_estimate(self):
        """Estimates tokens as chars / CHARS_PER_TOKEN."""
        text = "A" * 100
        tokens = estimate_tokens(text)
        assert tokens == 25  # 100 / 4 = 25
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_empty_string_returns_zero(self):
        """Empty string returns 0 tokens."""
        result = estimate_tokens("")
        assert result == 0
        assert isinstance(result, int)

    def test_single_character(self):
        """Single character returns 0 tokens (floor division)."""
        result = estimate_tokens("A")
        assert result == 0  # 1 // 4 = 0
        assert isinstance(result, int)

    def test_exactly_chars_per_token(self):
        """String of exactly CHARS_PER_TOKEN characters returns 1 token."""
        text = "A" * CHARS_PER_TOKEN
        result = estimate_tokens(text)
        assert result == 1
        assert len(text) == CHARS_PER_TOKEN

    def test_chars_per_token_minus_one(self):
        """String of CHARS_PER_TOKEN - 1 characters returns 0 tokens."""
        text = "A" * (CHARS_PER_TOKEN - 1)
        result = estimate_tokens(text)
        assert result == 0  # (CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN = 0
        assert len(text) == CHARS_PER_TOKEN - 1

    def test_chars_per_token_plus_one(self):
        """String of CHARS_PER_TOKEN + 1 characters returns 1 token."""
        text = "A" * (CHARS_PER_TOKEN + 1)
        result = estimate_tokens(text)
        assert result == 1  # (CHARS_PER_TOKEN + 1) // CHARS_PER_TOKEN = 1
        assert len(text) == CHARS_PER_TOKEN + 1

    def test_integer_division_floor(self):
        """Uses integer division (floor)."""
        text = "A" * 5
        tokens = estimate_tokens(text)
        assert tokens == 1  # 5/4 -> 1.25 -> floor to 1
        assert isinstance(tokens, int)

        text = "A" * 7
        tokens = estimate_tokens(text)
        assert tokens == 1  # 7/4 -> 1.75 -> floor to 1

        text = "A" * 8
        tokens = estimate_tokens(text)
        assert tokens == 2  # 8/4 -> 2.0 -> 2

        text = "A" * 9
        tokens = estimate_tokens(text)
        assert tokens == 2  # 9/4 -> 2.25 -> floor to 2

    def test_large_text(self):
        """Handles large text."""
        text = "A" * 100_000
        tokens = estimate_tokens(text)
        expected = 100_000 // CHARS_PER_TOKEN
        assert tokens == expected
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_very_large_text(self):
        """Handles very large text (boundary condition)."""
        text = "A" * 1_000_000
        tokens = estimate_tokens(text)
        expected = 1_000_000 // CHARS_PER_TOKEN
        assert tokens == expected
        assert isinstance(tokens, int)

    def test_unicode_characters(self):
        """Handles unicode characters (counts Python chars, not bytes)."""
        text = "👍" * 10
        assert len(text) == 10
        tokens = estimate_tokens(text)
        expected = 10 // CHARS_PER_TOKEN
        assert tokens == expected
        assert isinstance(tokens, int)

    def test_mixed_unicode_ascii(self):
        """Handles mixed unicode and ASCII characters."""
        text = "A👍B" * 5  # 15 characters total
        assert len(text) == 15
        tokens = estimate_tokens(text)
        expected = 15 // CHARS_PER_TOKEN
        assert tokens == expected
        assert isinstance(tokens, int)

    def test_whitespace_only(self):
        """Handles whitespace-only strings."""
        # Single space
        assert estimate_tokens(" ") == 0  # 1 // 4 = 0
        
        # Multiple spaces
        assert estimate_tokens(" " * 4) == 1  # 4 // 4 = 1
        assert estimate_tokens(" " * 5) == 1  # 5 // 4 = 1
        
        # Tabs
        assert estimate_tokens("\t" * 4) == 1
        assert estimate_tokens("\t" * 5) == 1
        
        # Newlines
        assert estimate_tokens("\n" * 4) == 1
        assert estimate_tokens("\n" * 5) == 1
        
        # Mixed whitespace
        assert estimate_tokens(" \t\n " * 2) == 2  # 8 chars // 4 = 2

    def test_special_characters(self):
        """Handles special characters and punctuation."""
        text = "!@#$%^&*()" * 2  # 20 characters
        tokens = estimate_tokens(text)
        assert tokens == 20 // CHARS_PER_TOKEN
        assert isinstance(tokens, int)

    def test_control_characters(self):
        """Handles control characters."""
        text = "\x00\x01\x02\x03" * 2  # 8 characters
        tokens = estimate_tokens(text)
        assert tokens == 8 // CHARS_PER_TOKEN
        assert isinstance(tokens, int)

    def test_multibyte_unicode(self):
        """Handles multibyte unicode characters."""
        # Chinese characters
        text = "中文" * 5  # 10 characters
        assert len(text) == 10
        tokens = estimate_tokens(text)
        assert tokens == 10 // CHARS_PER_TOKEN
        
        # Emoji sequences (zero-width joiners make these longer)
        text = "👨‍👩‍👧‍👦" * 3  # Actually 21 characters due to zero-width joiners
        char_count = len(text)
        tokens = estimate_tokens(text)
        assert tokens == char_count // CHARS_PER_TOKEN
        assert isinstance(tokens, int)
        
        # Simple emoji (single character)
        text = "👍" * 4  # 4 characters
        assert len(text) == 4
        tokens = estimate_tokens(text)
        assert tokens == 4 // CHARS_PER_TOKEN

    def test_realistic_text(self):
        """Tests with realistic text content."""
        text = "This is a sample paper text with multiple words and sentences."
        char_count = len(text)
        tokens = estimate_tokens(text)
        expected = char_count // CHARS_PER_TOKEN
        assert tokens == expected
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_newlines_and_spaces(self):
        """Tests with text containing newlines and spaces."""
        text = "Line 1\nLine 2\nLine 3" * 10
        char_count = len(text)
        tokens = estimate_tokens(text)
        expected = char_count // CHARS_PER_TOKEN
        assert tokens == expected
        assert isinstance(tokens, int)

    def test_return_type(self):
        """Ensures function always returns an integer."""
        test_cases = [
            "",
            "A",
            "A" * CHARS_PER_TOKEN,
            "A" * (CHARS_PER_TOKEN * 10),
            "👍" * 10,
            "Hello, world!",
        ]
        for text in test_cases:
            result = estimate_tokens(text)
            assert isinstance(result, int), f"Expected int for '{text[:20]}...', got {type(result)}"
            assert result >= 0, f"Expected non-negative for '{text[:20]}...', got {result}"

    def test_none_input_raises_error(self):
        """None input raises TypeError with correct message."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(None)  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "NoneType" in error_msg
        assert error_msg == "Expected string for text, got NoneType"

    def test_int_input_raises_error(self):
        """Integer input raises TypeError with correct message."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(123)  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "int" in error_msg
        assert error_msg == "Expected string for text, got int"

    def test_float_input_raises_error(self):
        """Float input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(123.45)  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "float" in error_msg
        assert error_msg == "Expected string for text, got float"

    def test_bool_input_raises_error(self):
        """Boolean input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(True)  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "bool" in error_msg
        assert error_msg == "Expected string for text, got bool"
        
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(False)  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "bool" in error_msg
        assert error_msg == "Expected string for text, got bool"

    def test_list_input_raises_error(self):
        """List input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(["a", "b", "c"])  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "list" in error_msg
        assert error_msg == "Expected string for text, got list"

    def test_dict_input_raises_error(self):
        """Dictionary input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens({"key": "value"})  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "dict" in error_msg
        assert error_msg == "Expected string for text, got dict"

    def test_bytes_input_raises_error(self):
        """Bytes input raises TypeError (not str)."""
        with pytest.raises(TypeError) as exc_info:
            estimate_tokens(b"hello")  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "Expected string" in error_msg
        assert "bytes" in error_msg
        assert error_msg == "Expected string for text, got bytes"

    def test_none_string_inputs(self):
        """Tests that various non-string types all raise TypeError."""
        invalid_inputs = [
            None,
            0,
            1,
            -1,
            3.14,
            True,
            False,
            [],
            {},
            [1, 2, 3],
            {"a": 1},
            b"bytes",
            object(),
        ]
        for invalid_input in invalid_inputs:
            with pytest.raises(TypeError, match="Expected string"):
                estimate_tokens(invalid_input)  # type: ignore[arg-type]

    def test_consistency(self):
        """Tests that the function is consistent (same input = same output)."""
        text = "Consistent test string" * 10
        result1 = estimate_tokens(text)
        result2 = estimate_tokens(text)
        assert result1 == result2
        assert isinstance(result1, int)
        assert isinstance(result2, int)

    def test_no_side_effects(self):
        """Tests that the function doesn't modify input."""
        text = "Original text" * 5
        original_text = text
        estimate_tokens(text)
        assert text == original_text
        assert text is original_text  # Should not create a copy

    def test_boundary_around_chars_per_token(self):
        """Tests boundary conditions around CHARS_PER_TOKEN."""
        # Test all values from CHARS_PER_TOKEN - 2 to CHARS_PER_TOKEN + 2
        for offset in range(-2, 3):
            length = CHARS_PER_TOKEN + offset
            text = "A" * length
            tokens = estimate_tokens(text)
            expected = length // CHARS_PER_TOKEN
            assert tokens == expected, f"Failed for length {length}: expected {expected}, got {tokens}"
            assert isinstance(tokens, int)

    def test_zero_length_edge_cases(self):
        """Tests edge cases with zero-length or minimal input."""
        assert estimate_tokens("") == 0
        
        # Single character should give 0 tokens
        for char in ["A", " ", "\n", "\t", "👍", "中"]:
            result = estimate_tokens(char)
            assert result == 0, f"Single char '{char}' should give 0 tokens, got {result}"
            assert isinstance(result, int)

