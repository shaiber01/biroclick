"""Tests for creating paper input payloads."""

import pytest

from src.paper_loader import create_paper_input, ValidationError


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

        assert paper_input["paper_id"] == "test_paper"
        assert paper_input["paper_title"] == "Test Title"
        assert paper_input["paper_text"] == "A" * 150
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["id"] == "Fig1"
        assert paper_input["paper_domain"] == "other"
        assert "supplementary" not in paper_input

    def test_default_domain_is_other(self):
        """Default paper_domain is 'other'."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
        )

        assert paper_input["paper_domain"] == "other"

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

    def test_with_supplementary_text(self):
        """Includes supplementary text."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text="Supplementary methods...",
        )

        assert "supplementary" in paper_input
        assert (
            paper_input["supplementary"]["supplementary_text"]
            == "Supplementary methods..."
        )
        assert "supplementary_figures" not in paper_input["supplementary"]

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

        assert paper_input["supplementary"]["supplementary_figures"] == supp_figs

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

        assert paper_input["supplementary"]["supplementary_data_files"] == data_files

    def test_validates_on_creation(self):
        """Validates paper input during creation."""
        with pytest.raises(ValidationError):
            create_paper_input(
                paper_id="",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[],
            )


