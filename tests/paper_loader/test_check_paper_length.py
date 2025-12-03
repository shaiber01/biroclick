"""Unit tests for the check_paper_length helper."""

import pytest

from src.paper_loader import (
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_VERY_LONG,
    check_paper_length,
)


class TestCheckPaperLength:
    """Tests for check_paper_length function."""

    def test_normal_length_no_warnings(self):
        """Normal length paper returns no warnings."""
        text = "A" * PAPER_LENGTH_NORMAL
        warnings = check_paper_length(text)
        assert warnings == []

    def test_long_paper_boundary(self):
        """Test boundary conditions for long paper warning."""
        text = "A" * PAPER_LENGTH_LONG
        warnings = check_paper_length(text)
        assert warnings == []

        text = "A" * (PAPER_LENGTH_LONG + 1)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0]

    def test_very_long_paper_boundary(self):
        """Test boundary conditions for very long paper warning."""
        text = "A" * PAPER_LENGTH_VERY_LONG
        warnings = check_paper_length(text)

        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0]

        text = "A" * (PAPER_LENGTH_VERY_LONG + 1)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "VERY LONG" in warnings[0]

    def test_custom_label(self):
        """Custom label appears in warning."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="Supplementary")
        assert "Supplementary" in warnings[0]

    def test_default_label_is_paper(self):
        """Default label is 'Paper'."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert warnings[0].startswith("Paper")

    def test_empty_string(self):
        """Empty string returns no warnings."""
        warnings = check_paper_length("")
        assert warnings == []

    def test_none_input_raises_error(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length(None)  # type: ignore[arg-type]

