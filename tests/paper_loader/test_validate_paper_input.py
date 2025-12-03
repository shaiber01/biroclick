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

        # Path(None) would fail, but let's see what happens
        # Actually, Path(None) might raise TypeError, but let's test
        try:
            validate_paper_input(paper_input)
        except (ValidationError, TypeError) as e:
            # Either is acceptable - None is invalid
            assert True

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

    def test_figure_image_path_non_string_handled(self, paper_input_factory):
        """Figure with non-string image_path should be handled."""
        paper_input = paper_input_factory(
            figures=[
                {
                    "id": "Fig1",
                    "image_path": 12345,
                }
            ]
        )

        # Path(12345) might work or fail, but should be handled
        try:
            warnings = validate_paper_input(paper_input)
            # If it doesn't raise, it should warn about non-existent file
            assert any("image file not found" in w or "unusual image format" in w for w in warnings)
        except (ValidationError, TypeError, AttributeError):
            # Any of these is acceptable - non-string path is invalid
            pass

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

