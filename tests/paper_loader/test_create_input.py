"""Tests for creating paper input payloads."""

import pytest

from src.paper_loader import create_paper_input, ValidationError
from schemas.state import CONTEXT_WINDOW_LIMITS


class TestCreatePaperInput:
    """Tests for create_paper_input function."""

    def test_creates_basic_paper_input(self):
        """Creates basic paper input with required fields."""
        paper_input = create_paper_input(
            paper_id="test_paper",
            paper_title="Test Title",
            paper_text="A" * 150,
            figures=[{"id": "Fig1", "description": "Test", "image_path": "test.png"}],
        )

        # Verify all required fields are present and correct
        assert paper_input["paper_id"] == "test_paper"
        assert paper_input["paper_title"] == "Test Title"
        assert paper_input["paper_text"] == "A" * 150
        assert len(paper_input["paper_text"]) == 150
        assert paper_input["paper_domain"] == "other"
        
        # Verify figures structure
        assert isinstance(paper_input["figures"], list)
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["id"] == "Fig1"
        assert paper_input["figures"][0]["description"] == "Test"
        assert paper_input["figures"][0]["image_path"] == "test.png"
        
        # Verify supplementary section is not created when not needed
        assert "supplementary" not in paper_input
        
        # Verify no extra fields
        expected_keys = {"paper_id", "paper_title", "paper_text", "paper_domain", "figures"}
        assert set(paper_input.keys()) == expected_keys

    def test_default_domain_is_other(self):
        """Default paper_domain is 'other'."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
        )

        assert paper_input["paper_domain"] == "other"
        assert isinstance(paper_input["figures"], list)
        assert len(paper_input["figures"]) == 0

    def test_custom_domain(self):
        """Accepts custom paper_domain."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            paper_domain="plasmonics",
        )

        assert paper_input["paper_domain"] == "plasmonics"
        assert paper_input["paper_domain"] != "other"

    def test_all_valid_domains(self):
        """Accepts all valid domain values."""
        from src.paper_loader import VALID_DOMAINS
        
        for domain in VALID_DOMAINS:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                paper_domain=domain,
            )
            assert paper_input["paper_domain"] == domain

    def test_with_supplementary_text(self):
        """Includes supplementary text."""
        supp_text = "Supplementary methods..."
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text=supp_text,
        )

        assert "supplementary" in paper_input
        assert isinstance(paper_input["supplementary"], dict)
        assert paper_input["supplementary"]["supplementary_text"] == supp_text
        assert len(paper_input["supplementary"]["supplementary_text"]) == len(supp_text)
        assert "supplementary_figures" not in paper_input["supplementary"]
        assert "supplementary_data_files" not in paper_input["supplementary"]

    def test_with_supplementary_figures(self):
        """Includes supplementary figures."""
        supp_figs = [{"id": "S1", "description": "Supp fig", "image_path": "s1.png"}]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_figures=supp_figs,
        )

        assert "supplementary" in paper_input
        assert isinstance(paper_input["supplementary"], dict)
        assert paper_input["supplementary"]["supplementary_figures"] == supp_figs
        assert len(paper_input["supplementary"]["supplementary_figures"]) == 1
        assert paper_input["supplementary"]["supplementary_figures"][0]["id"] == "S1"
        assert "supplementary_text" not in paper_input["supplementary"]
        assert "supplementary_data_files" not in paper_input["supplementary"]

    def test_with_supplementary_data_files(self):
        """Includes supplementary data files."""
        data_files = [
            {
                "id": "S_data",
                "description": "Data",
                "file_path": "data.csv",
                "data_type": "spectrum",
            }
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_data_files=data_files,
        )

        assert "supplementary" in paper_input
        assert isinstance(paper_input["supplementary"], dict)
        assert paper_input["supplementary"]["supplementary_data_files"] == data_files
        assert len(paper_input["supplementary"]["supplementary_data_files"]) == 1
        assert paper_input["supplementary"]["supplementary_data_files"][0]["id"] == "S_data"
        assert paper_input["supplementary"]["supplementary_data_files"][0]["data_type"] == "spectrum"
        assert "supplementary_text" not in paper_input["supplementary"]
        assert "supplementary_figures" not in paper_input["supplementary"]

    def test_with_all_supplementary_types(self):
        """Includes all supplementary types simultaneously."""
        supp_text = "Supplementary methods..."
        supp_figs = [{"id": "S1", "description": "Supp fig", "image_path": "s1.png"}]
        data_files = [
            {
                "id": "S_data",
                "description": "Data",
                "file_path": "data.csv",
                "data_type": "spectrum",
            }
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text=supp_text,
            supplementary_figures=supp_figs,
            supplementary_data_files=data_files,
        )

        assert "supplementary" in paper_input
        assert paper_input["supplementary"]["supplementary_text"] == supp_text
        assert paper_input["supplementary"]["supplementary_figures"] == supp_figs
        assert paper_input["supplementary"]["supplementary_data_files"] == data_files
        assert len(paper_input["supplementary"]) == 3

    def test_multiple_figures(self):
        """Handles multiple figures correctly."""
        figures = [
            {"id": "Fig1", "description": "First", "image_path": "fig1.png"},
            {"id": "Fig2", "description": "Second", "image_path": "fig2.png"},
            {"id": "Fig3", "description": "Third", "image_path": "fig3.png"},
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )

        assert len(paper_input["figures"]) == 3
        assert paper_input["figures"][0]["id"] == "Fig1"
        assert paper_input["figures"][1]["id"] == "Fig2"
        assert paper_input["figures"][2]["id"] == "Fig3"
        assert paper_input["figures"][0]["description"] == "First"
        assert paper_input["figures"][1]["description"] == "Second"
        assert paper_input["figures"][2]["description"] == "Third"

    def test_figure_with_digitized_data(self):
        """Handles figures with digitized data paths."""
        figures = [
            {
                "id": "Fig1",
                "description": "Test",
                "image_path": "test.png",
                "digitized_data_path": "data.csv",
            }
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )

        assert paper_input["figures"][0]["digitized_data_path"] == "data.csv"

    def test_validates_on_creation(self):
        """Validates paper input during creation."""
        with pytest.raises(ValidationError):
            create_paper_input(
                paper_id="",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )

    def test_validates_empty_paper_id(self):
        """Raises ValidationError for empty paper_id."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )
        assert "paper_id" in str(exc_info.value).lower()

    def test_validates_none_paper_id(self):
        """Raises ValidationError for None paper_id."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id=None,
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )
        assert "paper_id" in str(exc_info.value).lower()

    def test_validates_non_string_paper_id(self):
        """Raises ValidationError for non-string paper_id."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id=123,
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )
        assert "paper_id" in str(exc_info.value).lower()

    def test_validates_paper_id_with_spaces(self):
        """Accepts paper_id with spaces but may warn."""
        # This should not raise, but may produce warnings
        paper_input = create_paper_input(
            paper_id="test paper id",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
        )
        assert paper_input["paper_id"] == "test paper id"

    def test_validates_empty_paper_text(self):
        """Raises ValidationError for empty paper_text."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="",
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_validates_short_paper_text(self):
        """Raises ValidationError for paper_text shorter than 100 chars."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 99,  # 99 chars, needs at least 100
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_validates_paper_text_exactly_100_chars(self):
        """Accepts paper_text with exactly 100 chars."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 100,
            figures=[],
        )
        assert len(paper_input["paper_text"]) == 100

    def test_validates_paper_text_whitespace_only(self):
        """Raises ValidationError for whitespace-only paper_text."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text=" " * 150,  # Only whitespace
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_validates_none_paper_text(self):
        """Raises ValidationError for None paper_text."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text=None,
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_validates_non_string_paper_text(self):
        """Raises ValidationError for non-string paper_text."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text=12345,
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_validates_very_long_paper_text(self):
        """Raises ValidationError for paper_text exceeding max length."""
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * (max_chars + 1),
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower() or "exceeds" in str(exc_info.value).lower()

    def test_validates_paper_text_at_max_length(self):
        """Accepts paper_text at maximum allowed length."""
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * max_chars,
            figures=[],
        )
        assert len(paper_input["paper_text"]) == max_chars

    def test_validates_figures_not_list(self):
        """Raises ValidationError if figures is not a list."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures="not a list",
            )
        error_msg = str(exc_info.value).lower()
        # Should either say "figures must be a list" or have figure validation errors
        assert "figures" in error_msg or "must be a list" in error_msg or "figure" in error_msg

    def test_validates_figures_none(self):
        """Raises ValidationError if figures is None."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=None,
            )
        assert "figures" in str(exc_info.value).lower()

    def test_validates_figure_missing_id(self):
        """Raises ValidationError if figure is missing id field."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"description": "Test", "image_path": "test.png"}],
            )
        assert "id" in str(exc_info.value).lower() or "figure" in str(exc_info.value).lower()

    def test_validates_figure_missing_image_path(self):
        """Raises ValidationError if figure is missing image_path field."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "Fig1", "description": "Test"}],
            )
        assert "image_path" in str(exc_info.value).lower() or "figure" in str(exc_info.value).lower()

    def test_validates_figure_not_dict(self):
        """Raises ValidationError if figure is not a dictionary."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=["not a dict"],
            )
        assert "figure" in str(exc_info.value).lower() or "dictionary" in str(exc_info.value).lower()

    def test_validates_multiple_figure_errors(self):
        """Raises ValidationError with multiple figure errors."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[
                    {"id": "Fig1"},  # Missing image_path
                    "not a dict",  # Not a dict
                    {"description": "Test", "image_path": "test.png"},  # Missing id
                ],
            )
        error_msg = str(exc_info.value).lower()
        assert "figure" in error_msg

    def test_empty_figures_list(self):
        """Accepts empty figures list (may warn but should not error)."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
        )
        assert isinstance(paper_input["figures"], list)
        assert len(paper_input["figures"]) == 0

    def test_supplementary_text_empty_string(self):
        """Handles empty supplementary_text (should not create supplementary section)."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text="",
        )
        # Empty string should be falsy, so supplementary section should not be created
        assert "supplementary" not in paper_input

    def test_supplementary_figures_empty_list(self):
        """Handles empty supplementary_figures list (should not create supplementary section)."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_figures=[],
        )
        # Empty list should be falsy, so supplementary section should not be created
        assert "supplementary" not in paper_input

    def test_supplementary_data_files_empty_list(self):
        """Handles empty supplementary_data_files list (should not create supplementary section)."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_data_files=[],
        )
        # Empty list should be falsy, so supplementary section should not be created
        assert "supplementary" not in paper_input

    def test_supplementary_figures_missing_fields(self):
        """Validates supplementary figures have required fields."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_figures=[{"id": "S1"}],  # Missing image_path
            )
        assert "image_path" in str(exc_info.value).lower() or "figure" in str(exc_info.value).lower()

    def test_supplementary_data_files_missing_fields(self):
        """Validates supplementary data files have required fields."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=[{"id": "S1"}],  # Missing required fields
            )
        # Note: data files may have different validation - check what's actually validated
        # This test will reveal if validation is missing for data files

    def test_paper_title_preserved(self):
        """Verifies paper_title is preserved exactly as provided."""
        title = "Complex Title: With Special Characters! @#$%"
        paper_input = create_paper_input(
            paper_id="test",
            paper_title=title,
            paper_text="A" * 150,
            figures=[],
        )
        assert paper_input["paper_title"] == title
        assert len(paper_input["paper_title"]) == len(title)

    def test_paper_title_empty_string(self):
        """Handles empty paper_title (may or may not validate)."""
        # Empty title might be valid or invalid - test will reveal
        try:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="",
                paper_text="A" * 150,
                figures=[],
            )
            assert paper_input["paper_title"] == ""
        except ValidationError:
            # If validation catches this, that's fine
            pass

    def test_paper_title_none(self):
        """Handles None paper_title."""
        with pytest.raises(ValidationError):
            create_paper_input(
                paper_id="test",
                paper_title=None,
                paper_text="A" * 150,
                figures=[],
            )

    def test_figure_ids_preserved(self):
        """Verifies figure IDs are preserved exactly."""
        figures = [
            {"id": "Fig1a", "description": "Test", "image_path": "test.png"},
            {"id": "Fig1b", "description": "Test", "image_path": "test.png"},
            {"id": "Fig_S1", "description": "Test", "image_path": "test.png"},
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )
        assert paper_input["figures"][0]["id"] == "Fig1a"
        assert paper_input["figures"][1]["id"] == "Fig1b"
        assert paper_input["figures"][2]["id"] == "Fig_S1"

    def test_figure_descriptions_preserved(self):
        """Verifies figure descriptions are preserved exactly."""
        figures = [
            {"id": "Fig1", "description": "Complex description with symbols: α, β, γ", "image_path": "test.png"},
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )
        assert paper_input["figures"][0]["description"] == "Complex description with symbols: α, β, γ"

    def test_figure_image_paths_preserved(self):
        """Verifies figure image paths are preserved exactly."""
        figures = [
            {"id": "Fig1", "description": "Test", "image_path": "/absolute/path/to/image.png"},
            {"id": "Fig2", "description": "Test", "image_path": "relative/path/to/image.jpg"},
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )
        assert paper_input["figures"][0]["image_path"] == "/absolute/path/to/image.png"
        assert paper_input["figures"][1]["image_path"] == "relative/path/to/image.jpg"

    def test_supplementary_text_preserved(self):
        """Verifies supplementary text is preserved exactly."""
        supp_text = "Complex supplementary text\nWith newlines\nAnd special chars: α, β, γ"
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text=supp_text,
        )
        assert paper_input["supplementary"]["supplementary_text"] == supp_text

    def test_return_type_is_dict(self):
        """Verifies return value is a dictionary."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
        )
        assert isinstance(paper_input, dict)

    def test_no_side_effects_on_input_figures(self):
        """Verifies input figures list is not modified."""
        original_figures = [{"id": "Fig1", "description": "Test", "image_path": "test.png"}]
        figures_copy = original_figures.copy()
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=original_figures,
        )
        assert original_figures == figures_copy
        # Verify returned list is a different object (not the same reference)
        assert paper_input["figures"] is not original_figures
        # Verify mutating returned list doesn't affect original
        paper_input["figures"].append({"id": "Fig2", "description": "New", "image_path": "new.png"})
        assert len(original_figures) == 1
        assert len(paper_input["figures"]) == 2

    def test_no_side_effects_on_supplementary_figures(self):
        """Verifies input supplementary_figures list is not modified."""
        original_supp_figs = [{"id": "S1", "description": "Test", "image_path": "test.png"}]
        supp_figs_copy = original_supp_figs.copy()
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_figures=original_supp_figs,
        )
        assert original_supp_figs == supp_figs_copy
        # Verify returned list is a different object (not the same reference)
        assert paper_input["supplementary"]["supplementary_figures"] is not original_supp_figs
        # Verify mutating returned list doesn't affect original
        paper_input["supplementary"]["supplementary_figures"].append({"id": "S2", "description": "New", "image_path": "new.png"})
        assert len(original_supp_figs) == 1
        assert len(paper_input["supplementary"]["supplementary_figures"]) == 2

    def test_no_side_effects_on_supplementary_data_files(self):
        """Verifies input supplementary_data_files list is not modified."""
        original_data_files = [
            {"id": "S1", "description": "Test", "file_path": "test.csv", "data_type": "spectrum"}
        ]
        data_files_copy = original_data_files.copy()
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_data_files=original_data_files,
        )
        assert original_data_files == data_files_copy
        # Verify returned list is a different object (not the same reference)
        assert paper_input["supplementary"]["supplementary_data_files"] is not original_data_files
        # Verify mutating returned list doesn't affect original
        paper_input["supplementary"]["supplementary_data_files"].append({"id": "S2", "description": "New", "file_path": "new.csv", "data_type": "spectrum"})
        assert len(original_data_files) == 1
        assert len(paper_input["supplementary"]["supplementary_data_files"]) == 2


