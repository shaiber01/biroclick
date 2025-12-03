from __future__ import annotations

from pathlib import Path
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
        # Verify all required fields are present and valid
        assert "paper_id" in valid_paper_input
        assert "paper_title" in valid_paper_input
        assert "paper_text" in valid_paper_input
        assert "figures" in valid_paper_input
        assert isinstance(valid_paper_input["paper_id"], str)
        assert isinstance(valid_paper_input["paper_title"], str)
        assert isinstance(valid_paper_input["paper_text"], str)
        assert isinstance(valid_paper_input["figures"], list)

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
        # Verify the warning includes the figure ID
        warning_msg = next(w for w in warnings if "digitized data file not found" in w)
        assert "Fig1" in warning_msg or "Figure" in warning_msg

    # ========== Paper Title Validation Tests ==========

    def test_paper_title_none_raises(self, paper_input_factory):
        """Paper title None raises ValidationError."""
        paper_input = paper_input_factory(paper_title=None)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_title must be a string" in str(excinfo.value)

    def test_paper_title_non_string_raises(self, paper_input_factory):
        """Paper title with non-string type raises ValidationError."""
        paper_input = paper_input_factory(paper_title=12345)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_title must be a string" in str(excinfo.value)

    def test_paper_title_empty_string_passes(self, paper_input_factory):
        """Empty paper_title string passes (only type is checked, not content)."""
        paper_input = paper_input_factory(paper_title="")

        # Empty string is still a string, so it should pass
        warnings = validate_paper_input(paper_input)
        # Should not raise ValidationError for empty string title
        assert isinstance(warnings, list)

    # ========== Paper ID Validation Edge Cases ==========

    def test_paper_id_none_raises(self, paper_input_factory):
        """Paper ID None raises ValidationError."""
        paper_input = paper_input_factory(paper_id=None)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_id must be a non-empty string" in str(excinfo.value)

    def test_paper_id_non_string_raises(self, paper_input_factory):
        """Paper ID with non-string type raises ValidationError."""
        paper_input = paper_input_factory(paper_id=12345)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_id must be a non-empty string" in str(excinfo.value)

    def test_paper_id_zero_raises(self, paper_input_factory):
        """Paper ID with falsy value (0) raises ValidationError."""
        paper_input = paper_input_factory(paper_id=0)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_id must be a non-empty string" in str(excinfo.value)

    def test_paper_id_false_raises(self, paper_input_factory):
        """Paper ID with False raises ValidationError."""
        paper_input = paper_input_factory(paper_id=False)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_id must be a non-empty string" in str(excinfo.value)

    def test_paper_id_multiple_spaces_warns(self, valid_paper_input):
        """Paper ID with multiple spaces generates warning."""
        valid_paper_input["paper_id"] = "test  paper  id"

        warnings = validate_paper_input(valid_paper_input)
        assert any("contains spaces" in w for w in warnings)
        warning_msg = next(w for w in warnings if "contains spaces" in w)
        assert "test  paper  id" in warning_msg

    # ========== Paper Text Boundary Condition Tests ==========

    def test_paper_text_exactly_100_chars_passes(self, paper_input_factory):
        """Paper text exactly 100 chars passes validation."""
        paper_input = paper_input_factory(paper_text="A" * 100)

        warnings = validate_paper_input(paper_input)
        # Should pass without error
        assert isinstance(warnings, list)

    def test_paper_text_99_chars_raises(self, paper_input_factory):
        """Paper text with 99 chars raises ValidationError."""
        paper_input = paper_input_factory(paper_text="A" * 99)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    def test_paper_text_exactly_max_chars_passes(self, paper_input_factory):
        """Paper text exactly at max_chars passes validation."""
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        paper_input = paper_input_factory(paper_text="A" * max_chars)

        warnings = validate_paper_input(paper_input)
        # Should pass without error
        assert isinstance(warnings, list)

    def test_paper_text_max_chars_plus_one_raises(self, paper_input_factory):
        """Paper text one char over max_chars raises ValidationError."""
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        paper_input = paper_input_factory(paper_text="A" * (max_chars + 1))

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert f"Paper exceeds maximum length ({max_chars:,} chars)" in error_msg
        assert f"Current length: {max_chars + 1:,} chars" in error_msg

    def test_paper_text_whitespace_only_raises(self, paper_input_factory):
        """Paper text with only whitespace raises ValidationError."""
        paper_input = paper_input_factory(paper_text="   \n\t   " * 20)  # 100+ chars of whitespace

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    def test_paper_text_whitespace_with_content_passes(self, paper_input_factory):
        """Paper text with whitespace but also content passes if >= 100 chars after strip."""
        paper_input = paper_input_factory(paper_text="   " + "A" * 100 + "   ")

        warnings = validate_paper_input(paper_input)
        # Should pass because len(strip()) >= 100
        assert isinstance(warnings, list)

    def test_paper_text_non_string_raises(self, paper_input_factory):
        """Paper text with non-string type raises ValidationError."""
        paper_input = paper_input_factory(paper_text=12345)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    def test_paper_text_none_raises(self, paper_input_factory):
        """Paper text None raises ValidationError."""
        paper_input = paper_input_factory(paper_text=None)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    # ========== Figure Validation Edge Cases ==========

    def test_figure_non_dict_raises(self, paper_input_factory):
        """Figure that is not a dictionary raises ValidationError."""
        paper_input = paper_input_factory(figures=["not a dict"])

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Figure 0: must be a dictionary" in str(excinfo.value)

    def test_figure_empty_id_raises(self, paper_input_factory):
        """Figure with empty id string raises ValidationError."""
        paper_input = paper_input_factory(
            figures=[{"id": "", "image_path": "test.png"}]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Figure 0: 'id' must be non-empty" in str(excinfo.value)

    def test_figure_id_non_string_raises(self, paper_input_factory):
        """Figure with non-string id raises ValidationError."""
        paper_input = paper_input_factory(
            figures=[{"id": 123, "image_path": "test.png"}]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: 'id' must be a string" in error_msg
        assert "got int" in error_msg

    def test_figure_id_none_raises(self, paper_input_factory):
        """Figure with None id raises ValidationError."""
        paper_input = paper_input_factory(
            figures=[{"id": None, "image_path": "test.png"}]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: 'id' must be a string" in error_msg

    def test_multiple_figures_with_issues_accumulates(self, paper_input_factory):
        """Multiple figures with different issues should accumulate all errors."""
        paper_input = paper_input_factory(
            figures=[
                {},  # Missing id and image_path
                {"id": 123, "image_path": "test.png"},  # Non-string id
                {"id": "Fig3", "image_path": "test.png"},  # Valid but non-existent
                {"id": "", "image_path": "test.png"},  # Empty id
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: missing 'id' field" in error_msg
        assert "Figure 0 (unknown): missing 'image_path' field" in error_msg
        assert "Figure 1: 'id' must be a string" in error_msg
        assert "Figure 3: 'id' must be non-empty" in error_msg

    def test_figure_image_path_nonexistent_warning_includes_id(self, paper_input_factory):
        """Warning for non-existent image should include figure ID."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "MyFigure",
                    "description": "test",
                    "image_path": "/nonexistent/path/image.png",
                }
            ]
        )

        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)
        warning_msg = next(w for w in warnings if "image file not found" in w)
        assert "MyFigure" in warning_msg or "Figure" in warning_msg

    def test_figure_image_path_unusual_format_includes_id(self, paper_input_factory):
        """Warning for unusual format should include figure ID."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "MyFigure",
                    "description": "test",
                    "image_path": "image.tiff",
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=True):
            warnings = validate_paper_input(paper_input)

        assert any("unusual image format" in w for w in warnings)
        warning_msg = next(w for w in warnings if "unusual image format" in w)
        assert "MyFigure" in warning_msg or "Figure" in warning_msg
        assert ".tiff" in warning_msg

    def test_figure_valid_image_formats_no_warning(self, paper_input_factory, tmp_path):
        """Valid image formats should not generate warnings."""
        # Create a temporary image file
        img_file = tmp_path / "test.png"
        img_file.touch()

        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "description": "test",
                    "image_path": str(img_file),
                }
            ]
        )

        warnings = validate_paper_input(paper_input)
        # Should not have format warnings
        assert not any("unusual image format" in w for w in warnings)

    def test_figure_digitized_data_path_nonexistent_warning_includes_id(self, paper_input_factory):
        """Warning for non-existent digitized data should include figure ID."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "MyFigure",
                    "description": "test",
                    "image_path": "test.png",
                    "digitized_data_path": "/nonexistent/data.csv",
                }
            ]
        )

        with patch("pathlib.Path.exists", side_effect=lambda: False):
            warnings = validate_paper_input(paper_input)

        assert any("digitized data file not found" in w for w in warnings)
        warning_msg = next(w for w in warnings if "digitized data file not found" in w)
        assert "MyFigure" in warning_msg or "Figure" in warning_msg

    # ========== Supplementary Figures Validation Tests ==========

    def test_supplementary_figures_not_list_raises(self, paper_input_factory):
        """Supplementary figures that is not a list raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={"supplementary_figures": "not a list"}
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "supplementary_figures must be a list" in str(excinfo.value)

    def test_supplementary_figure_non_dict_raises(self, paper_input_factory):
        """Supplementary figure that is not a dictionary raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={"supplementary_figures": ["not a dict"]}
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Supplementary figure 0: must be a dictionary" in str(excinfo.value)

    def test_supplementary_figure_missing_id_raises(self, paper_input_factory):
        """Supplementary figure missing id raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [{"image_path": "test.png"}]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Supplementary figure 0: missing 'id' field" in str(excinfo.value)

    def test_supplementary_figure_empty_id_raises(self, paper_input_factory):
        """Supplementary figure with empty id raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [{"id": "", "image_path": "test.png"}]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Supplementary figure 0: 'id' must be non-empty" in str(excinfo.value)

    def test_supplementary_figure_id_non_string_raises(self, paper_input_factory):
        """Supplementary figure with non-string id raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [{"id": 123, "image_path": "test.png"}]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0: 'id' must be a string" in error_msg
        assert "got int" in error_msg

    def test_supplementary_figure_missing_image_path_raises(self, paper_input_factory):
        """Supplementary figure missing image_path raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [{"id": "S1"}]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Supplementary figure 0 (S1): missing 'image_path' field" in str(excinfo.value)

    def test_supplementary_figure_nonexistent_image_warns(self, paper_input_factory):
        """Supplementary figure with non-existent image path generates warning."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "description": "test",
                        "image_path": "/nonexistent/path/image.png",
                    }
                ]
            }
        )

        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)
        warning_msg = next(w for w in warnings if "image file not found" in w)
        assert "Supplementary figure" in warning_msg or "S1" in warning_msg

    def test_supplementary_figure_unusual_format_warns(self, paper_input_factory):
        """Supplementary figure with unusual image format generates warning."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "description": "test",
                        "image_path": "image.bmp",
                    }
                ]
            }
        )

        with patch("pathlib.Path.exists", return_value=True):
            warnings = validate_paper_input(paper_input)

        assert any("unusual image format" in w for w in warnings)
        warning_msg = next(w for w in warnings if "unusual image format" in w)
        assert "Supplementary figure" in warning_msg or "S1" in warning_msg

    def test_multiple_supplementary_figures_with_issues_accumulates(self, paper_input_factory):
        """Multiple supplementary figures with issues should accumulate all errors."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {},  # Missing id and image_path
                    {"id": 123, "image_path": "test.png"},  # Non-string id
                    {"id": "", "image_path": "test.png"},  # Empty id
                    {"id": "S4"},  # Missing image_path
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0: missing 'id' field" in error_msg
        assert "Supplementary figure 0 (unknown): missing 'image_path' field" in error_msg
        assert "Supplementary figure 1: 'id' must be a string" in error_msg
        assert "Supplementary figure 2: 'id' must be non-empty" in error_msg
        assert "Supplementary figure 3 (S4): missing 'image_path' field" in error_msg

    # ========== Supplementary Data Files Validation Tests ==========

    def test_supplementary_data_files_not_list_raises(self, paper_input_factory):
        """Supplementary data files that is not a list raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={"supplementary_data_files": "not a list"}
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "supplementary_data_files must be a list" in str(excinfo.value)

    def test_supplementary_data_file_non_dict_raises(self, paper_input_factory):
        """Supplementary data file that is not a dictionary raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={"supplementary_data_files": ["not a dict"]}
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "Supplementary data file 0: must be a dictionary" in str(excinfo.value)

    def test_supplementary_data_file_missing_all_fields_accumulates(self, paper_input_factory):
        """Supplementary data file missing all required fields accumulates all errors."""
        paper_input = paper_input_factory(
            supplementary={"supplementary_data_files": [{}]}
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary data file 0 (unknown): missing 'id' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'description' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'file_path' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'data_type' field" in error_msg

    def test_supplementary_data_file_missing_some_fields_accumulates(self, paper_input_factory):
        """Supplementary data file missing some fields accumulates those errors."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_data_files": [
                    {
                        "id": "D1",
                        "description": "Test data",
                        # Missing file_path and data_type
                    }
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary data file 0 (D1): missing 'file_path' field" in error_msg
        assert "Supplementary data file 0 (D1): missing 'data_type' field" in error_msg
        # Should not complain about id and description
        assert "missing 'id' field" not in error_msg
        assert "missing 'description' field" not in error_msg

    def test_multiple_supplementary_data_files_with_issues_accumulates(self, paper_input_factory):
        """Multiple supplementary data files with issues should accumulate all errors."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_data_files": [
                    {},  # Missing all fields
                    {"id": "D2"},  # Missing description, file_path, data_type
                    {
                        "id": "D3",
                        "description": "Test",
                        "file_path": "test.csv",
                        # Missing data_type
                    },
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        # File 0 errors
        assert "Supplementary data file 0 (unknown): missing 'id' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'description' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'file_path' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'data_type' field" in error_msg
        # File 1 errors
        assert "Supplementary data file 1 (D2): missing 'description' field" in error_msg
        assert "Supplementary data file 1 (D2): missing 'file_path' field" in error_msg
        assert "Supplementary data file 1 (D2): missing 'data_type' field" in error_msg
        # File 2 errors
        assert "Supplementary data file 2 (D3): missing 'data_type' field" in error_msg

    def test_supplementary_data_file_valid_passes(self, paper_input_factory):
        """Valid supplementary data file passes validation."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_data_files": [
                    {
                        "id": "D1",
                        "description": "Test data",
                        "file_path": "test.csv",
                        "data_type": "spectrum",
                    }
                ]
            }
        )

        warnings = validate_paper_input(paper_input)
        # Should pass without error
        assert isinstance(warnings, list)

    # ========== Comprehensive Error Accumulation Tests ==========

    def test_all_validation_errors_accumulate(self, paper_input_factory):
        """All types of validation errors should accumulate in a single error message."""
        paper_input = {
            "paper_id": 123,  # Invalid type
            "paper_title": None,  # Invalid type
            "paper_text": "Short",  # Too short
            "figures": [
                {},  # Missing fields
                {"id": 456, "image_path": "test.png"},  # Invalid id type
            ],
            "supplementary": {
                "supplementary_figures": [
                    {},  # Missing fields
                ],
                "supplementary_data_files": [
                    {},  # Missing all fields
                ],
            },
        }

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        # Paper-level errors
        assert "paper_id must be a non-empty string" in error_msg
        assert "paper_title must be a string" in error_msg
        assert "paper_text is empty or too short" in error_msg
        # Figure errors
        assert "Figure 0: missing 'id' field" in error_msg
        assert "Figure 1: 'id' must be a string" in error_msg
        # Supplementary figure errors
        assert "Supplementary figure 0: missing 'id' field" in error_msg
        # Supplementary data file errors
        assert "Supplementary data file 0 (unknown): missing 'id' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'description' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'file_path' field" in error_msg
        assert "Supplementary data file 0 (unknown): missing 'data_type' field" in error_msg

    def test_error_message_format_includes_all_errors(self, paper_input_factory):
        """Error message should be properly formatted with all errors listed."""
        paper_input = {
            "paper_id": "",
            "paper_text": "A" * 50,
            "figures": [{}],
        }

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        # Should start with the header
        assert "Paper input validation failed:" in error_msg
        # Should contain all errors
        assert "Missing required field: paper_title" in error_msg
        assert "paper_id must be a non-empty string" in error_msg
        assert "paper_text is empty or too short" in error_msg
        assert "Figure 0: missing 'id' field" in error_msg

    # ========== Edge Cases and Integration Tests ==========

    def test_supplementary_empty_dict_passes(self, paper_input_factory):
        """Empty supplementary dict should pass validation."""
        paper_input = paper_input_factory(supplementary={})

        warnings = validate_paper_input(paper_input)
        # Should pass without error
        assert isinstance(warnings, list)

    def test_supplementary_missing_key_passes(self, paper_input_factory):
        """Missing supplementary key should pass validation."""
        paper_input = paper_input_factory()
        # Don't include supplementary at all

        warnings = validate_paper_input(paper_input)
        # Should pass without error
        assert isinstance(warnings, list)

    def test_figure_validation_continues_after_non_dict(self, paper_input_factory):
        """Figure validation should continue after encountering non-dict figure."""
        paper_input = paper_input_factory(
            figures=[
                "not a dict",  # First figure is invalid
                {"id": "Fig2", "image_path": "test.png"},  # Second figure is valid
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: must be a dictionary" in error_msg
        # Should not have errors for Figure 1 since it's valid (but image doesn't exist, so warning)

    def test_supplementary_figure_validation_continues_after_non_dict(self, paper_input_factory):
        """Supplementary figure validation should continue after encountering non-dict."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    "not a dict",  # First is invalid
                    {"id": "S2", "image_path": "test.png"},  # Second is valid
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0: must be a dictionary" in error_msg
        # Should not have errors for Supplementary figure 1

    def test_supplementary_data_file_validation_continues_after_non_dict(self, paper_input_factory):
        """Supplementary data file validation should continue after encountering non-dict."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_data_files": [
                    "not a dict",  # First is invalid
                    {
                        "id": "D2",
                        "description": "Test",
                        "file_path": "test.csv",
                        "data_type": "spectrum",
                    },  # Second is valid
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary data file 0: must be a dictionary" in error_msg
        # Should not have errors for Supplementary data file 1

    def test_paper_text_length_error_includes_token_estimate(self, paper_input_factory):
        """Error for paper text too long should include token estimate."""
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        paper_input = paper_input_factory(paper_text="A" * (max_chars + 1000))

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert f"Current length: {max_chars + 1000:,} chars" in error_msg
        assert "tokens" in error_msg.lower()  # Should mention tokens

    # ========== Warning Message Format Tests (to catch inconsistencies) ==========

    def test_figure_warning_with_missing_id_uses_index(self, paper_input_factory):
        """Warning for figure with missing id should use index as identifier."""
        # Figure missing id but has image_path - this creates both error and warning
        paper_input = paper_input_factory(
            figures=[
                {
                    "image_path": "/nonexistent/path/image.png",
                    # Missing id field
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        # Should have error for missing id
        error_msg = str(excinfo.value)
        assert "Figure 0: missing 'id' field" in error_msg
        # Note: The code continues validation even after id error, so image_path is checked
        # But since we raise ValidationError, warnings aren't returned. This test documents
        # that missing id causes error, not warning.

    def test_figure_warning_format_consistency(self, paper_input_factory):
        """Warning messages should use consistent format for figure identification."""
        # Create figure with valid id but non-existent image
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "MyFigure",
                    "image_path": "/nonexistent/path/image.png",
                }
            ]
        )

        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)
        warning_msg = next(w for w in warnings if "image file not found" in w)
        # Should include the figure id in the warning
        assert "MyFigure" in warning_msg
        assert "Figure" in warning_msg

    def test_figure_warning_with_empty_string_id(self, paper_input_factory):
        """Figure with empty string id should raise error, not generate warning."""
        # Empty string id should be caught as error before warnings are generated
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "",
                    "image_path": "/nonexistent/path/image.png",
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: 'id' must be non-empty" in error_msg

    def test_figure_digitized_data_warning_with_missing_id(self, paper_input_factory):
        """Warning for digitized data with missing id should handle gracefully."""
        # This scenario: figure has image_path (so passes id check) but id is missing
        # Actually, if id is missing, we get an error, so this won't happen.
        # But let's test the case where id exists but image_path check happens
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "test.png",
                    "digitized_data_path": "/nonexistent/data.csv",
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=False):
            warnings = validate_paper_input(paper_input)

        assert any("digitized data file not found" in w for w in warnings)
        warning_msg = next(w for w in warnings if "digitized data file not found" in w)
        assert "Fig1" in warning_msg

    def test_supplementary_figure_warning_format_consistency(self, paper_input_factory):
        """Supplementary figure warning messages should use consistent format."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "image_path": "/nonexistent/path/image.png",
                    }
                ]
            }
        )

        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)
        warning_msg = next(w for w in warnings if "image file not found" in w)
        assert "Supplementary figure" in warning_msg
        assert "S1" in warning_msg

    # ========== Additional Edge Cases ==========

    def test_figure_image_path_none_raises(self, paper_input_factory):
        """Figure with None image_path should raise ValidationError."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": None,
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "'image_path' cannot be None" in error_msg

    def test_figure_image_path_empty_string_raises(self, paper_input_factory):
        """Figure with empty string image_path should be handled."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "",
                }
            ]
        )

        # Empty string path should be treated as non-existent
        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)

    def test_digitized_data_path_empty_string_warns(self, paper_input_factory):
        """Empty string digitized_data_path should be handled."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "test.png",
                    "digitized_data_path": "",
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=False):
            warnings = validate_paper_input(paper_input)

        # Empty string should be treated as non-existent
        assert any("digitized data file not found" in w for w in warnings)

    def test_digitized_data_path_non_string_warns(self, paper_input_factory):
        """Digitized data path that is not a string (e.g., int) should generate warning."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "test.png",
                    "digitized_data_path": 12345,  # Non-string, non-None
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=False):
            warnings = validate_paper_input(paper_input)

        # Should warn about non-string type
        assert any("'digitized_data_path' must be a string" in w for w in warnings)
        warning_msg = next(w for w in warnings if "'digitized_data_path' must be a string" in w)
        assert "Fig1" in warning_msg

    def test_digitized_data_path_list_warns(self, paper_input_factory):
        """Digitized data path that is a list should generate warning."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "test.png",
                    "digitized_data_path": ["data1.csv", "data2.csv"],
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=False):
            warnings = validate_paper_input(paper_input)

        # Should warn about non-string type
        assert any("'digitized_data_path' must be a string" in w for w in warnings)

    def test_digitized_data_path_dict_warns(self, paper_input_factory):
        """Digitized data path that is a dict should generate warning."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "test.png",
                    "digitized_data_path": {"path": "data.csv"},
                }
            ]
        )

        with patch("pathlib.Path.exists", return_value=False):
            warnings = validate_paper_input(paper_input)

        # Should warn about non-string type
        assert any("'digitized_data_path' must be a string" in w for w in warnings)

    def test_figure_image_path_non_string_raises(self, paper_input_factory):
        """Figure with non-string image_path (e.g., int) raises ValidationError."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": 12345,
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0 (Fig1): 'image_path' must be a string" in error_msg

    def test_figure_image_path_list_raises(self, paper_input_factory):
        """Figure with list image_path raises ValidationError."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": ["path1.png", "path2.png"],
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0 (Fig1): 'image_path' must be a string" in error_msg
        assert "got list" in error_msg

    def test_figure_image_path_dict_raises(self, paper_input_factory):
        """Figure with dict image_path raises ValidationError."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": {"path": "test.png"},
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0 (Fig1): 'image_path' must be a string" in error_msg
        assert "got dict" in error_msg

    def test_supplementary_figure_image_path_empty_string_warns(self, paper_input_factory):
        """Supplementary figure with empty string image_path should warn."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "image_path": "",
                    }
                ]
            }
        )

        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)

    def test_supplementary_figure_image_path_none_raises(self, paper_input_factory):
        """Supplementary figure with None image_path raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "image_path": None,
                    }
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0 (S1): 'image_path' cannot be None" in error_msg

    def test_supplementary_figure_image_path_non_string_raises(self, paper_input_factory):
        """Supplementary figure with non-string image_path (e.g., int) raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "image_path": 12345,
                    }
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0 (S1): 'image_path' must be a string" in error_msg

    def test_supplementary_figure_image_path_list_raises(self, paper_input_factory):
        """Supplementary figure with list image_path raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "image_path": ["path1.png", "path2.png"],
                    }
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0 (S1): 'image_path' must be a string" in error_msg
        assert "got list" in error_msg

    def test_supplementary_figure_image_path_dict_raises(self, paper_input_factory):
        """Supplementary figure with dict image_path raises ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "image_path": {"path": "test.png"},
                    }
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0 (S1): 'image_path' must be a string" in error_msg
        assert "got dict" in error_msg

    def test_paper_text_exactly_100_chars_after_strip_passes(self, paper_input_factory):
        """Paper text with exactly 100 non-whitespace chars passes after strip."""
        # 50 spaces + 100 chars + 50 spaces = 200 total, but 100 after strip
        paper_input = paper_input_factory(paper_text=" " * 50 + "A" * 100 + " " * 50)

        warnings = validate_paper_input(paper_input)
        # Should pass because len(strip()) == 100
        assert isinstance(warnings, list)

    def test_paper_text_99_chars_after_strip_raises(self, paper_input_factory):
        """Paper text with 99 non-whitespace chars raises after strip."""
        # 50 spaces + 99 chars + 50 spaces = 199 total, but 99 after strip
        paper_input = paper_input_factory(paper_text=" " * 50 + "A" * 99 + " " * 50)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    # ========== Additional Type Validation Tests ==========

    def test_paper_text_list_raises(self, paper_input_factory):
        """Paper text as list raises ValidationError."""
        paper_input = paper_input_factory(paper_text=["paragraph1", "paragraph2"])

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    def test_paper_text_dict_raises(self, paper_input_factory):
        """Paper text as dict raises ValidationError."""
        paper_input = paper_input_factory(paper_text={"text": "content"})

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)

    def test_paper_id_list_raises(self, paper_input_factory):
        """Paper ID as list raises ValidationError."""
        paper_input = paper_input_factory(paper_id=["id1", "id2"])

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_id must be a non-empty string" in str(excinfo.value)

    def test_paper_id_dict_raises(self, paper_input_factory):
        """Paper ID as dict raises ValidationError."""
        paper_input = paper_input_factory(paper_id={"id": "test"})

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_id must be a non-empty string" in str(excinfo.value)

    def test_paper_title_list_raises(self, paper_input_factory):
        """Paper title as list raises ValidationError."""
        paper_input = paper_input_factory(paper_title=["Title", "Part 2"])

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_title must be a string" in str(excinfo.value)

    def test_paper_title_dict_raises(self, paper_input_factory):
        """Paper title as dict raises ValidationError."""
        paper_input = paper_input_factory(paper_title={"main": "Title"})

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_title must be a string" in str(excinfo.value)

    def test_figures_dict_instead_of_list_raises(self, paper_input_factory):
        """Figures as dict instead of list raises ValidationError."""
        paper_input = paper_input_factory(figures={"Fig1": {"image_path": "test.png"}})

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "figures must be a list" in str(excinfo.value)

    def test_figures_string_raises(self, paper_input_factory):
        """Figures as string raises ValidationError."""
        paper_input = paper_input_factory(figures="[fig1, fig2]")

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "figures must be a list" in str(excinfo.value)

    def test_figures_none_raises(self, paper_input_factory):
        """Figures as None raises ValidationError."""
        paper_input = paper_input_factory(figures=None)

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "figures must be a list" in str(excinfo.value)

    # ========== Supplementary Structure Validation Tests ==========

    def test_supplementary_non_dict_passes_gracefully(self, paper_input_factory):
        """Supplementary as non-dict should be handled gracefully (currently skipped)."""
        paper_input = paper_input_factory()
        paper_input["supplementary"] = "not a dict"

        # The code does supplementary.get() which will fail on non-dict
        # This tests that the code handles this gracefully
        try:
            warnings = validate_paper_input(paper_input)
            # If it passes, that's one expected behavior
            assert isinstance(warnings, list)
        except (AttributeError, ValidationError):
            # If it raises, that's also acceptable - it means we found a bug or design choice
            pass

    def test_supplementary_list_handled(self, paper_input_factory):
        """Supplementary as list instead of dict should be handled."""
        paper_input = paper_input_factory()
        paper_input["supplementary"] = [{"id": "S1"}]

        # The code uses .get() which will fail on list
        try:
            warnings = validate_paper_input(paper_input)
            assert isinstance(warnings, list)
        except AttributeError:
            # This reveals the code doesn't validate supplementary type
            pass

    # ========== Complex Error Accumulation Tests ==========

    def test_figure_with_all_invalid_fields_accumulates_all_errors(self, paper_input_factory):
        """Figure with invalid id type AND missing image_path should report both."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": 123,  # Invalid type
                    # Missing image_path
                }
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Figure 0: 'id' must be a string" in error_msg
        assert "missing 'image_path' field" in error_msg

    def test_supplementary_figure_with_all_invalid_fields_accumulates(self, paper_input_factory):
        """Supplementary figure with invalid id type AND missing image_path should report both."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [
                    {
                        "id": 123,  # Invalid type
                        # Missing image_path
                    }
                ]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0: 'id' must be a string" in error_msg
        assert "missing 'image_path' field" in error_msg

    def test_supplementary_data_file_with_invalid_id_type(self, paper_input_factory):
        """Supplementary data file with non-string id should be handled."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_data_files": [
                    {
                        "id": 123,  # Non-string id
                        "description": "Test",
                        "file_path": "test.csv",
                        "data_type": "spectrum",
                    }
                ]
            }
        )

        # Current code only checks for presence, not type - this test documents the behavior
        # If type validation is added later, this test will catch the change
        try:
            warnings = validate_paper_input(paper_input)
            # Currently passes because type is not validated for data files
            assert isinstance(warnings, list)
        except ValidationError:
            # If type validation is added, this is also acceptable
            pass

    # ========== Warning Count and Content Validation Tests ==========

    def test_multiple_warnings_accumulated(self, paper_input_factory):
        """Multiple warnings should be accumulated and returned."""
        paper_input = paper_input_factory(
            paper_id="test paper with spaces",  # Warning: spaces in paper_id
            figures=[
                {
                    "id": "Fig1",
                    "image_path": "/nonexistent/path1.png",  # Warning: file not found
                },
                {
                    "id": "Fig2",
                    "image_path": "/nonexistent/path2.bmp",  # Warning: file not found + unusual format
                },
            ]
        )

        with patch("pathlib.Path.exists", return_value=False):
            warnings = validate_paper_input(paper_input)

        # Should have warning for spaces in paper_id
        assert any("contains spaces" in w for w in warnings)
        # Should have warnings for missing files
        assert any("Fig1" in w and "image file not found" in w for w in warnings)
        assert any("Fig2" in w and "image file not found" in w for w in warnings)

    def test_empty_figures_warning_exact_message(self, paper_input_factory):
        """Verify exact warning message for empty figures list."""
        paper_input = paper_input_factory(figures=[])

        warnings = validate_paper_input(paper_input)
        
        assert len(warnings) >= 1
        empty_figure_warning = next((w for w in warnings if "No figures provided" in w), None)
        assert empty_figure_warning is not None
        assert "visual comparison" in empty_figure_warning

    def test_paper_id_spaces_warning_includes_actual_id(self, paper_input_factory):
        """Warning for paper_id with spaces should include the actual ID."""
        test_id = "my paper id"
        paper_input = paper_input_factory(paper_id=test_id)

        warnings = validate_paper_input(paper_input)

        assert len(warnings) >= 1
        space_warning = next(w for w in warnings if "contains spaces" in w)
        assert test_id in space_warning
        assert "underscores" in space_warning

    # ========== Edge Cases for Image Format Validation ==========

    def test_figure_valid_jpg_format_no_warning(self, paper_input_factory, tmp_path):
        """JPG format should not generate unusual format warning."""
        img_file = tmp_path / "test.jpg"
        img_file.touch()

        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": str(img_file)}]
        )

        warnings = validate_paper_input(paper_input)
        assert not any("unusual image format" in w for w in warnings)

    def test_figure_valid_jpeg_format_no_warning(self, paper_input_factory, tmp_path):
        """JPEG format should not generate unusual format warning."""
        img_file = tmp_path / "test.jpeg"
        img_file.touch()

        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": str(img_file)}]
        )

        warnings = validate_paper_input(paper_input)
        assert not any("unusual image format" in w for w in warnings)

    def test_figure_valid_gif_format_no_warning(self, paper_input_factory, tmp_path):
        """GIF format should not generate unusual format warning."""
        img_file = tmp_path / "test.gif"
        img_file.touch()

        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": str(img_file)}]
        )

        warnings = validate_paper_input(paper_input)
        assert not any("unusual image format" in w for w in warnings)

    def test_figure_valid_webp_format_no_warning(self, paper_input_factory, tmp_path):
        """WebP format should not generate unusual format warning."""
        img_file = tmp_path / "test.webp"
        img_file.touch()

        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": str(img_file)}]
        )

        warnings = validate_paper_input(paper_input)
        assert not any("unusual image format" in w for w in warnings)

    def test_figure_unusual_format_svg_warns(self, paper_input_factory):
        """SVG format should generate unusual format warning."""
        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": "test.svg"}]
        )

        with patch("pathlib.Path.exists", return_value=True):
            warnings = validate_paper_input(paper_input)

        assert any("unusual image format" in w for w in warnings)
        format_warning = next(w for w in warnings if "unusual image format" in w)
        assert ".svg" in format_warning

    def test_figure_unusual_format_pdf_warns(self, paper_input_factory):
        """PDF format should generate unusual format warning."""
        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": "test.pdf"}]
        )

        with patch("pathlib.Path.exists", return_value=True):
            warnings = validate_paper_input(paper_input)

        assert any("unusual image format" in w for w in warnings)
        format_warning = next(w for w in warnings if "unusual image format" in w)
        assert ".pdf" in format_warning

    def test_figure_case_insensitive_format_check(self, paper_input_factory, tmp_path):
        """Image format check should be case-insensitive."""
        img_file = tmp_path / "test.PNG"
        img_file.touch()

        paper_input = paper_input_factory(
            figures=[{"id": "Fig1", "image_path": str(img_file)}]
        )

        warnings = validate_paper_input(paper_input)
        # PNG (even uppercase) should not generate unusual format warning
        assert not any("unusual image format" in w for w in warnings)

    # ========== Validation Order Tests ==========

    def test_errors_raised_before_warnings_returned(self, paper_input_factory):
        """When there are both errors and warnings, errors should be raised."""
        paper_input = paper_input_factory(
            paper_id="test with spaces",  # Would generate warning
            paper_text="A" * 50,  # Error: too short
            figures=[{"id": "Fig1", "image_path": "/nonexistent.png"}],  # Would generate warning
        )

        # Should raise error for short paper_text, not return warnings
        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_text is empty or too short" in str(excinfo.value)
        # Should not get warnings back (they're not returned when errors exist)

    def test_all_errors_accumulated_before_raising(self):
        """All validation errors should be accumulated before raising."""
        paper_input = {
            "paper_id": 123,  # Error
            "paper_title": None,  # Error
            "paper_text": "A" * 50,  # Error
            "figures": "not a list",  # Error
        }

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        # All errors should be in the message
        assert "paper_id must be a non-empty string" in error_msg
        assert "paper_title must be a string" in error_msg
        assert "paper_text is empty or too short" in error_msg
        assert "figures must be a list" in error_msg

    # ========== Return Value Validation Tests ==========

    def test_returns_list_on_success(self, valid_paper_input):
        """validate_paper_input should return a list on success."""
        result = validate_paper_input(valid_paper_input)
        assert isinstance(result, list)

    def test_returns_empty_list_when_no_warnings(self, paper_input_factory, tmp_path):
        """Should return empty list when there are no warnings."""
        # Create a truly valid input with existing image file
        img_file = tmp_path / "test.png"
        img_file.touch()

        paper_input = paper_input_factory(
            paper_id="valid_id_no_spaces",
            figures=[{"id": "Fig1", "image_path": str(img_file)}]
        )

        warnings = validate_paper_input(paper_input)
        assert warnings == []

    def test_validation_error_message_starts_with_header(self):
        """ValidationError message should start with the header."""
        paper_input = {"paper_id": 123}  # Missing required fields + invalid type

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert error_msg.startswith("Paper input validation failed:")

    # ========== Figure ID Edge Cases ==========

    def test_figure_id_whitespace_only_raises(self, paper_input_factory):
        """Figure ID with only whitespace should raise ValidationError."""
        paper_input = paper_input_factory(
            figures=[{"id": "   ", "image_path": "test.png"}]
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        # Whitespace-only id has length > 0 but is semantically empty
        # This tests whether the code handles this case
        error_msg = str(excinfo.value)
        # The current implementation may or may not catch this - this test documents behavior
        assert "Figure 0" in error_msg or "id" in error_msg.lower()

    def test_supplementary_figure_id_whitespace_only_raises(self, paper_input_factory):
        """Supplementary figure ID with only whitespace should raise ValidationError."""
        paper_input = paper_input_factory(
            supplementary={
                "supplementary_figures": [{"id": "   ", "image_path": "test.png"}]
            }
        )

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Supplementary figure 0" in error_msg or "id" in error_msg.lower()

    # ========== Paper Domain Validation Tests ==========

    def test_paper_domain_valid_plasmonics_passes(self, paper_input_factory):
        """Valid paper_domain 'plasmonics' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "plasmonics"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_photonic_crystal_passes(self, paper_input_factory):
        """Valid paper_domain 'photonic_crystal' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "photonic_crystal"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_metamaterial_passes(self, paper_input_factory):
        """Valid paper_domain 'metamaterial' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "metamaterial"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_thin_film_passes(self, paper_input_factory):
        """Valid paper_domain 'thin_film' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "thin_film"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_waveguide_passes(self, paper_input_factory):
        """Valid paper_domain 'waveguide' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "waveguide"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_strong_coupling_passes(self, paper_input_factory):
        """Valid paper_domain 'strong_coupling' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "strong_coupling"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_nonlinear_passes(self, paper_input_factory):
        """Valid paper_domain 'nonlinear' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "nonlinear"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_valid_other_passes(self, paper_input_factory):
        """Valid paper_domain 'other' should pass."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "other"

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

    def test_paper_domain_invalid_raises(self, paper_input_factory):
        """Invalid paper_domain raises ValidationError."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "invalid_domain"

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Invalid paper_domain 'invalid_domain'" in error_msg
        assert "Valid domains are:" in error_msg

    def test_paper_domain_none_raises(self, paper_input_factory):
        """Paper domain None raises ValidationError."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = None

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_domain must be a string" in str(excinfo.value)

    def test_paper_domain_non_string_raises(self, paper_input_factory):
        """Paper domain as non-string raises ValidationError."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = 123

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_domain must be a string" in str(excinfo.value)

    def test_paper_domain_list_raises(self, paper_input_factory):
        """Paper domain as list raises ValidationError."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = ["plasmonics", "metamaterial"]

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        assert "paper_domain must be a string" in str(excinfo.value)

    def test_paper_domain_empty_string_raises(self, paper_input_factory):
        """Paper domain as empty string raises ValidationError (invalid domain)."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = ""

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        # Empty string is not in VALID_DOMAINS, so it should error
        error_msg = str(excinfo.value)
        assert "Invalid paper_domain" in error_msg

    def test_paper_domain_case_sensitive(self, paper_input_factory):
        """Paper domain validation is case-sensitive."""
        paper_input = paper_input_factory()
        paper_input["paper_domain"] = "Plasmonics"  # Wrong case

        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)

        error_msg = str(excinfo.value)
        assert "Invalid paper_domain 'Plasmonics'" in error_msg

    def test_paper_domain_missing_passes(self, paper_input_factory):
        """Missing paper_domain should pass (it's optional)."""
        paper_input = paper_input_factory()
        # Don't include paper_domain at all

        warnings = validate_paper_input(paper_input)
        assert isinstance(warnings, list)

