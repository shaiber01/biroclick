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

    def test_empty_string_returns_zero(self):
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0

    def test_integer_division_floor(self):
        """Uses integer division (floor)."""
        text = "A" * 5
        tokens = estimate_tokens(text)
        assert tokens == 1  # 5/4 -> 1.25 -> floor to 1

        text = "A" * 7
        tokens = estimate_tokens(text)
        assert tokens == 1

        text = "A" * 8
        tokens = estimate_tokens(text)
        assert tokens == 2

    def test_large_text(self):
        """Handles large text."""
        text = "A" * 100_000
        tokens = estimate_tokens(text)
        assert tokens == 100_000 // CHARS_PER_TOKEN

    def test_unicode_characters(self):
        """Handles unicode characters (counts Python chars, not bytes)."""
        text = "👍" * 10
        assert len(text) == 10
        tokens = estimate_tokens(text)
        assert tokens == 10 // CHARS_PER_TOKEN

    def test_none_input_raises_error(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            estimate_tokens(None)  # type: ignore[arg-type]

    def test_int_input_raises_error(self):
        """Integer input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            estimate_tokens(123)  # type: ignore[arg-type]

