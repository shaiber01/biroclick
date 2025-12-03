from __future__ import annotations

from unittest.mock import patch

import pytest

from src.paper_loader import ValidationError, validate_paper_input
from schemas.state import CONTEXT_WINDOW_LIMITS


class TestValidatePaperInput:
    """Tests for validate_paper_input function."""

    def test_valid_input_passes(self, valid_paper_input):
        """Valid paper input passes validation with no warnings."""
        warnings = validate_paper_input(valid_paper_input)
        assert warnings == [], f"Expected no warnings, got: {warnings}"

    def test_missing_required_fields_accumulates_errors(self):
        """Missing multiple required fields raises ValidationError with all missing fields listed."""
        paper_input = {}

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Missing required field: paper_id" in error_msg
        assert "Missing required field: paper_title" in error_msg
        assert "Missing required field: paper_text" in error_msg
        assert "Missing required field: figures" in error_msg

    def test_invalid_types_and_values_accumulates_errors(self, paper_input_factory):
        """Invalid types and values should be accumulated in the error message."""
        paper_input = paper_input_factory()
        paper_input["paper_id"] = 123
        paper_input["paper_text"] = "Short"
        paper_input["figures"] = "not a list"

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "paper_id must be a non-empty string" in error_msg
        assert "paper_text is empty or too short" in error_msg
        assert "figures must be a list" in error_msg

    def test_empty_paper_id_raises(self, valid_paper_input):
        """Empty paper_id raises ValidationError."""
        valid_paper_input["paper_id"] = ""

        with pytest.raises(ValidationError, match="paper_id must be a non-empty string"):
            validate_paper_input(valid_paper_input)

    def test_paper_id_with_spaces_warns(self, valid_paper_input):
        """Paper ID with spaces generates warning."""
        valid_paper_input["paper_id"] = "test paper id"

        warnings = validate_paper_input(valid_paper_input)
        assert any("contains spaces" in w for w in warnings)

    def test_empty_paper_text_raises(self, valid_paper_input):
        """Empty paper_text raises ValidationError."""
        valid_paper_input["paper_text"] = ""

        with pytest.raises(ValidationError, match="paper_text is empty or too short"):
            validate_paper_input(valid_paper_input)

    def test_short_paper_text_raises(self, valid_paper_input):
        """Paper text under 100 chars raises ValidationError."""
        valid_paper_input["paper_text"] = "A" * 50

        with pytest.raises(ValidationError, match="paper_text is empty or too short"):
            validate_paper_input(valid_paper_input)

    def test_paper_text_too_long_raises(self, paper_input_factory):
        """Paper text exceeding MAX_PAPER_CHARS raises ValidationError."""
        paper_input = paper_input_factory()
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        paper_input["paper_text"] = "A" * (max_chars + 1)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert f"Paper exceeds maximum length ({max_chars:,} chars)" in str(excinfo.value)

    def test_figures_not_list_raises(self, paper_input_factory):
        """Non-list figures field raises ValidationError."""
        paper_input = paper_input_factory(figures="not a list")

        with pytest.raises(ValidationError, match="figures must be a list"):
            validate_paper_input(paper_input)

    def test_empty_figures_warns(self, paper_input_factory):
        """Empty figures list generates warning."""
        paper_input = paper_input_factory(figures=[])

        warnings = validate_paper_input(paper_input)
        assert any("No figures provided" in w for w in warnings)

    def test_figure_missing_required_fields_accumulates(self, paper_input_factory):
        """Figure with multiple missing fields should report all of them."""
        paper_input = paper_input_factory(figures=[{}])

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: missing 'id' field" in error_msg
        assert "Figure 0 (unknown): missing 'image_path' field" in error_msg

    def test_figure_nonexistent_image_warns(self, paper_input_factory):
        """Figure with non-existent image path generates warning."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "description": "test",
                    "image_path": "/nonexistent/path/image.png",
                }
            ]
        )

        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)

    def test_figure_unusual_format_warns(self, paper_input_factory):
        """Figure with unusual image format generates warning."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "description": "test",
                    "image_path": "image.bmp",
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=True):
            warnings = validate_paper_input(paper_input)

        assert any("unusual image format" in w for w in warnings)

    def test_digitized_data_nonexistent_warns(self, valid_paper_input):
        """Figure with non-existent digitized data path generates warning."""
        valid_paper_input["figures"][0]["digitized_data_path"] = "/nonexistent/data.csv"

        warnings = validate_paper_input(valid_paper_input)
        assert any("digitized data file not found" in w for w in warnings)

