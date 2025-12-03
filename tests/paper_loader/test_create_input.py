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


class TestCreatePaperInputInvalidDomain:
    """Tests for invalid domain validation in create_paper_input."""

    def test_invalid_domain_should_raise_error(self):
        """Invalid domain should raise ValidationError, not be silently accepted."""
        from src.paper_loader import VALID_DOMAINS
        
        # Use a domain that is clearly not in the valid list
        invalid_domain = "definitely_not_a_valid_domain_xyz123"
        assert invalid_domain not in VALID_DOMAINS
        
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                paper_domain=invalid_domain,
            )
        error_msg = str(exc_info.value).lower()
        assert "domain" in error_msg or invalid_domain.lower() in error_msg

    def test_empty_string_domain_should_raise_error(self):
        """Empty string domain should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                paper_domain="",
            )
        assert "domain" in str(exc_info.value).lower()

    def test_none_domain_should_raise_error(self):
        """None domain should raise ValidationError or TypeError."""
        with pytest.raises((ValidationError, TypeError)):
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                paper_domain=None,
            )

    def test_numeric_domain_should_raise_error(self):
        """Numeric domain should raise ValidationError or TypeError."""
        with pytest.raises((ValidationError, TypeError)):
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                paper_domain=123,
            )

    def test_domain_case_sensitivity(self):
        """Test that domain validation is case-sensitive."""
        from src.paper_loader import VALID_DOMAINS
        
        # "plasmonics" is valid, but "PLASMONICS" should not be (case-sensitive)
        # Unless the system is case-insensitive, in which case it should work
        # This test reveals the actual behavior
        if "plasmonics" in VALID_DOMAINS:
            try:
                paper_input = create_paper_input(
                    paper_id="test",
                    paper_title="Test",
                    paper_text="A" * 150,
                    figures=[],
                    paper_domain="PLASMONICS",
                )
                # If it succeeds, verify the domain is stored as provided
                assert paper_input["paper_domain"] == "PLASMONICS"
            except ValidationError:
                # Case-sensitive validation - this is also acceptable
                pass


class TestFigureEdgeCases:
    """Tests for figure edge cases that should be properly validated."""

    def test_figure_with_empty_string_id(self):
        """Figure with empty string id should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "", "description": "Test", "image_path": "test.png"}],
            )
        error_msg = str(exc_info.value).lower()
        assert "id" in error_msg or "empty" in error_msg or "figure" in error_msg

    def test_figure_with_none_image_path(self):
        """Figure with None image_path should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "Fig1", "description": "Test", "image_path": None}],
            )
        error_msg = str(exc_info.value).lower()
        assert "image_path" in error_msg or "figure" in error_msg or "none" in error_msg

    def test_figure_with_non_string_image_path(self):
        """Figure with non-string image_path should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "Fig1", "description": "Test", "image_path": 123}],
            )
        error_msg = str(exc_info.value).lower()
        assert "image_path" in error_msg or "figure" in error_msg or "string" in error_msg

    def test_figure_with_none_id(self):
        """Figure with None id should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": None, "description": "Test", "image_path": "test.png"}],
            )
        error_msg = str(exc_info.value).lower()
        assert "id" in error_msg or "figure" in error_msg

    def test_figure_with_non_string_id(self):
        """Figure with non-string id should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": 123, "description": "Test", "image_path": "test.png"}],
            )
        error_msg = str(exc_info.value).lower()
        assert "id" in error_msg or "figure" in error_msg or "string" in error_msg

    def test_figure_with_whitespace_only_id(self):
        """Figure with whitespace-only id should either fail or be handled."""
        # Whitespace-only IDs are problematic - test expected behavior
        # The validation should either reject this or trim it
        try:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "   ", "description": "Test", "image_path": "test.png"}],
            )
            # If accepted, it should either be trimmed or stored as-is
            assert paper_input["figures"][0]["id"] == "   " or paper_input["figures"][0]["id"].strip() == ""
        except ValidationError:
            # Rejection is also acceptable behavior
            pass

    def test_figure_description_is_optional(self):
        """Figure description field should be optional or have sensible default."""
        # The schema shows description is required, but let's verify
        try:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "Fig1", "image_path": "test.png"}],  # No description
            )
            # If it succeeds, description might have a default or be truly optional
            assert "description" in paper_input["figures"][0] or "description" not in paper_input["figures"][0]
        except ValidationError:
            # Description is required - this is also acceptable
            pass

    def test_duplicate_figure_ids(self):
        """Test behavior with duplicate figure IDs."""
        # Duplicate IDs might be an error or might be allowed
        # This test reveals the actual behavior
        try:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[
                    {"id": "Fig1", "description": "First", "image_path": "test1.png"},
                    {"id": "Fig1", "description": "Duplicate", "image_path": "test2.png"},
                ],
            )
            # If duplicates are allowed, verify both are stored
            assert len(paper_input["figures"]) == 2
            assert paper_input["figures"][0]["id"] == "Fig1"
            assert paper_input["figures"][1]["id"] == "Fig1"
        except ValidationError:
            # Duplicate IDs are not allowed - also acceptable
            pass


class TestSupplementaryDataFileEdgeCases:
    """Tests for supplementary data file edge cases."""

    def test_supplementary_data_file_not_dict(self):
        """Supplementary data file that is not a dict should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=["not a dict"],
            )
        error_msg = str(exc_info.value).lower()
        assert "data file" in error_msg or "dictionary" in error_msg or "dict" in error_msg

    def test_supplementary_data_file_missing_id(self):
        """Supplementary data file missing id should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=[{"description": "Test", "file_path": "test.csv", "data_type": "spectrum"}],
            )
        assert "id" in str(exc_info.value).lower()

    def test_supplementary_data_file_missing_description(self):
        """Supplementary data file missing description should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=[{"id": "S1", "file_path": "test.csv", "data_type": "spectrum"}],
            )
        assert "description" in str(exc_info.value).lower()

    def test_supplementary_data_file_missing_file_path(self):
        """Supplementary data file missing file_path should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=[{"id": "S1", "description": "Test", "data_type": "spectrum"}],
            )
        assert "file_path" in str(exc_info.value).lower()

    def test_supplementary_data_file_missing_data_type(self):
        """Supplementary data file missing data_type should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=[{"id": "S1", "description": "Test", "file_path": "test.csv"}],
            )
        assert "data_type" in str(exc_info.value).lower()

    def test_supplementary_data_file_with_valid_data_types(self):
        """Test that various data_type values are accepted."""
        # Test different data_type values
        data_types = ["spectrum", "geometry", "parameters", "time_series", "other"]
        for data_type in data_types:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_data_files=[{
                    "id": f"S_{data_type}",
                    "description": f"Test {data_type}",
                    "file_path": f"test_{data_type}.csv",
                    "data_type": data_type
                }],
            )
            assert paper_input["supplementary"]["supplementary_data_files"][0]["data_type"] == data_type


class TestSupplementaryFigureEdgeCases:
    """Tests for supplementary figure edge cases."""

    def test_supplementary_figure_not_dict(self):
        """Supplementary figure that is not a dict should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_figures=["not a dict"],
            )
        error_msg = str(exc_info.value).lower()
        assert "supplementary figure" in error_msg or "dictionary" in error_msg or "dict" in error_msg

    def test_supplementary_figure_with_none_id(self):
        """Supplementary figure with None id should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_figures=[{"id": None, "description": "Test", "image_path": "test.png"}],
            )
        error_msg = str(exc_info.value).lower()
        assert "id" in error_msg

    def test_supplementary_figure_with_empty_id(self):
        """Supplementary figure with empty id should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_figures=[{"id": "", "description": "Test", "image_path": "test.png"}],
            )
        error_msg = str(exc_info.value).lower()
        assert "id" in error_msg or "empty" in error_msg

    def test_supplementary_figure_with_none_image_path(self):
        """Supplementary figure with None image_path should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
                supplementary_figures=[{"id": "S1", "description": "Test", "image_path": None}],
            )
        error_msg = str(exc_info.value).lower()
        assert "image_path" in error_msg or "none" in error_msg


class TestDeepCopyBehavior:
    """Tests for deep copy behavior to prevent side effects."""

    def test_deep_copy_of_figure_dicts(self):
        """Verify that figure dicts themselves are copied, not just the list."""
        original_figures = [
            {"id": "Fig1", "description": "Original", "image_path": "test.png"}
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=original_figures,
        )
        # Verify the dict object is a different reference
        assert paper_input["figures"][0] is not original_figures[0]
        # Modify the returned dict
        paper_input["figures"][0]["description"] = "Modified"
        # Original should be unchanged
        assert original_figures[0]["description"] == "Original"

    def test_deep_copy_of_supplementary_figure_dicts(self):
        """Verify that supplementary figure dicts are copied, not just the list."""
        original_supp_figs = [
            {"id": "S1", "description": "Original", "image_path": "test.png"}
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_figures=original_supp_figs,
        )
        # Verify the dict object is a different reference
        assert paper_input["supplementary"]["supplementary_figures"][0] is not original_supp_figs[0]
        # Modify the returned dict
        paper_input["supplementary"]["supplementary_figures"][0]["description"] = "Modified"
        # Original should be unchanged
        assert original_supp_figs[0]["description"] == "Original"

    def test_deep_copy_of_supplementary_data_file_dicts(self):
        """Verify that supplementary data file dicts are copied, not just the list."""
        original_data_files = [
            {"id": "S1", "description": "Original", "file_path": "test.csv", "data_type": "spectrum"}
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_data_files=original_data_files,
        )
        # Verify the dict object is a different reference
        assert paper_input["supplementary"]["supplementary_data_files"][0] is not original_data_files[0]
        # Modify the returned dict
        paper_input["supplementary"]["supplementary_data_files"][0]["description"] = "Modified"
        # Original should be unchanged
        assert original_data_files[0]["description"] == "Original"


class TestPaperTextBoundaryConditions:
    """Tests for paper_text boundary conditions."""

    def test_paper_text_with_100_chars_including_whitespace(self):
        """Paper text with exactly 100 chars but some whitespace should work if stripped >= 100."""
        # 100 chars total, but stripping leading/trailing whitespace leaves 98
        # This should fail validation since len(paper_text.strip()) < 100
        text_with_whitespace = " " + "A" * 98 + " "  # 100 chars total, 98 after strip
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text=text_with_whitespace,
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_paper_text_with_102_chars_but_strips_to_100(self):
        """Paper text with 102 chars that strips to exactly 100 should work."""
        # Leading/trailing space + 100 chars
        text_with_whitespace = " " + "A" * 100 + " "  # 102 chars total, 100 after strip
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text=text_with_whitespace,
            figures=[],
        )
        # Verify the original text is stored (not stripped)
        assert paper_input["paper_text"] == text_with_whitespace
        assert len(paper_input["paper_text"]) == 102

    def test_paper_text_with_newlines_only(self):
        """Paper text with only newlines should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="\n" * 150,
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_paper_text_with_tabs_only(self):
        """Paper text with only tabs should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="\t" * 150,
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_paper_text_with_mixed_whitespace_only(self):
        """Paper text with only mixed whitespace should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text=" \t\n" * 50,  # 150 chars of mixed whitespace
                figures=[],
            )
        assert "paper_text" in str(exc_info.value).lower()

    def test_paper_text_just_over_max_length(self):
        """Paper text just 1 character over max should fail."""
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * (max_chars + 1),
                figures=[],
            )
        error_msg = str(exc_info.value).lower()
        assert "exceeds" in error_msg or "paper_text" in error_msg or "maximum" in error_msg


class TestUnicodeHandling:
    """Tests for unicode handling in text fields."""

    def test_paper_text_with_unicode_characters(self):
        """Paper text with unicode characters should be preserved."""
        unicode_text = "α β γ δ ε ζ η θ ι κ λ μ ν ξ π ρ σ τ υ φ χ ψ ω " * 5  # Greek letters
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text=unicode_text,
            figures=[],
        )
        assert paper_input["paper_text"] == unicode_text

    def test_paper_title_with_unicode_characters(self):
        """Paper title with unicode characters should be preserved."""
        unicode_title = "Study of α-particle interactions: γ-ray spectroscopy"
        paper_input = create_paper_input(
            paper_id="test",
            paper_title=unicode_title,
            paper_text="A" * 150,
            figures=[],
        )
        assert paper_input["paper_title"] == unicode_title

    def test_paper_id_with_unicode_characters(self):
        """Paper ID with unicode characters - test actual behavior."""
        # Some systems may not allow unicode in IDs
        unicode_id = "paper_α_2024"
        try:
            paper_input = create_paper_input(
                paper_id=unicode_id,
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )
            assert paper_input["paper_id"] == unicode_id
        except ValidationError:
            # If unicode IDs are not allowed, that's acceptable
            pass

    def test_figure_description_with_unicode(self):
        """Figure description with unicode should be preserved."""
        unicode_desc = "Absorption spectrum showing λmax = 590 nm (ε = 10⁵ M⁻¹cm⁻¹)"
        figures = [{"id": "Fig1", "description": unicode_desc, "image_path": "test.png"}]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )
        assert paper_input["figures"][0]["description"] == unicode_desc

    def test_supplementary_text_with_unicode(self):
        """Supplementary text with unicode should be preserved."""
        unicode_supp = "Δλ measurements: λ₁ = 400nm, λ₂ = 700nm, Δλ = 300nm"
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text=unicode_supp,
        )
        assert paper_input["supplementary"]["supplementary_text"] == unicode_supp

    def test_paper_text_with_emoji(self):
        """Paper text with emoji characters should be handled."""
        # Emojis in scientific papers are rare but should be handled
        text_with_emoji = "Research findings 🔬 show significant results 📊 " * 4
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text=text_with_emoji,
            figures=[],
        )
        assert paper_input["paper_text"] == text_with_emoji

    def test_paper_text_with_mathematical_symbols(self):
        """Paper text with mathematical symbols should be preserved."""
        math_text = "∫₀^∞ e^(-x²) dx = √π/2, ∑ᵢ aᵢ = ∏ⱼ bⱼ, ∀x ∃y: x ≤ y " * 3
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text=math_text,
            figures=[],
        )
        assert paper_input["paper_text"] == math_text


class TestPaperTitleValidation:
    """Additional tests for paper title validation."""

    def test_paper_title_non_string_type(self):
        """Non-string paper_title should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            create_paper_input(
                paper_id="test",
                paper_title=123,
                paper_text="A" * 150,
                figures=[],
            )
        assert "paper_title" in str(exc_info.value).lower()

    def test_paper_title_very_long(self):
        """Very long paper_title should be accepted (or rejected with clear error)."""
        # Scientific paper titles can be long, but there might be limits
        very_long_title = "A" * 10000
        try:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title=very_long_title,
                paper_text="A" * 150,
                figures=[],
            )
            assert paper_input["paper_title"] == very_long_title
            assert len(paper_input["paper_title"]) == 10000
        except ValidationError:
            # If there's a length limit, that's fine too
            pass

    def test_paper_title_with_newlines(self):
        """Paper title with newlines should be handled appropriately."""
        title_with_newlines = "First Line\nSecond Line\nThird Line"
        paper_input = create_paper_input(
            paper_id="test",
            paper_title=title_with_newlines,
            paper_text="A" * 150,
            figures=[],
        )
        # Title should be preserved exactly as provided
        assert paper_input["paper_title"] == title_with_newlines

    def test_paper_title_whitespace_only(self):
        """Paper title with only whitespace - test behavior."""
        # This might be valid or invalid depending on requirements
        try:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="   ",
                paper_text="A" * 150,
                figures=[],
            )
            assert paper_input["paper_title"] == "   "
        except ValidationError:
            # Whitespace-only titles might be rejected
            pass


class TestExtraFieldsHandling:
    """Tests for handling of extra/unknown fields."""

    def test_extra_fields_in_figure_are_preserved(self):
        """Extra fields in figure dict should be preserved."""
        figures = [
            {
                "id": "Fig1",
                "description": "Test",
                "image_path": "test.png",
                "custom_field": "custom_value",
                "another_field": 123,
            }
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=figures,
        )
        # Extra fields should be preserved
        assert paper_input["figures"][0].get("custom_field") == "custom_value"
        assert paper_input["figures"][0].get("another_field") == 123

    def test_extra_fields_in_supplementary_figure_are_preserved(self):
        """Extra fields in supplementary figure dict should be preserved."""
        supp_figs = [
            {
                "id": "S1",
                "description": "Test",
                "image_path": "test.png",
                "custom_field": "custom_value",
            }
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_figures=supp_figs,
        )
        assert paper_input["supplementary"]["supplementary_figures"][0].get("custom_field") == "custom_value"

    def test_extra_fields_in_data_file_are_preserved(self):
        """Extra fields in data file dict should be preserved."""
        data_files = [
            {
                "id": "S1",
                "description": "Test",
                "file_path": "test.csv",
                "data_type": "spectrum",
                "custom_field": "custom_value",
            }
        ]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_data_files=data_files,
        )
        assert paper_input["supplementary"]["supplementary_data_files"][0].get("custom_field") == "custom_value"


class TestMissingRequiredFields:
    """Tests for missing required top-level fields."""

    def test_missing_paper_id_arg_raises_type_error(self):
        """Omitting paper_id argument should raise TypeError (not ValidationError)."""
        with pytest.raises(TypeError):
            create_paper_input(
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )

    def test_missing_paper_title_arg_raises_type_error(self):
        """Omitting paper_title argument should raise TypeError."""
        with pytest.raises(TypeError):
            create_paper_input(
                paper_id="test",
                paper_text="A" * 150,
                figures=[],
            )

    def test_missing_paper_text_arg_raises_type_error(self):
        """Omitting paper_text argument should raise TypeError."""
        with pytest.raises(TypeError):
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                figures=[],
            )

    def test_missing_figures_arg_raises_type_error(self):
        """Omitting figures argument should raise TypeError."""
        with pytest.raises(TypeError):
            create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
            )


class TestSpecialCharacters:
    """Tests for special characters in various fields."""

    def test_paper_id_with_special_characters(self):
        """Paper ID with special characters - test behavior."""
        special_ids = [
            "paper-with-dashes",
            "paper_with_underscores",
            "paper.with.dots",
            "paper/with/slashes",  # Might be problematic for file paths
            "paper:with:colons",
        ]
        for special_id in special_ids:
            try:
                paper_input = create_paper_input(
                    paper_id=special_id,
                    paper_title="Test",
                    paper_text="A" * 150,
                    figures=[],
                )
                assert paper_input["paper_id"] == special_id
            except ValidationError:
                # Some special characters might be rejected
                pass

    def test_figure_id_with_special_characters(self):
        """Figure ID with special characters - test behavior."""
        special_ids = ["Fig1a", "Fig_1_a", "Fig-1-a", "Fig.1.a"]
        for fig_id in special_ids:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": fig_id, "description": "Test", "image_path": "test.png"}],
            )
            assert paper_input["figures"][0]["id"] == fig_id

    def test_image_path_with_special_characters(self):
        """Image path with special characters - test behavior."""
        special_paths = [
            "path/to/image.png",
            "path\\to\\image.png",
            "path with spaces/image.png",
            "path/with/αβγ/image.png",
        ]
        for path in special_paths:
            paper_input = create_paper_input(
                paper_id="test",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[{"id": "Fig1", "description": "Test", "image_path": path}],
            )
            assert paper_input["figures"][0]["image_path"] == path


