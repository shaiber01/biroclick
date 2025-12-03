"""Tests for saving paper input data and accessor helpers."""

import json

from src.paper_loader import (
    save_paper_input_json,
    get_figure_by_id,
    list_figure_ids,
    get_supplementary_text,
    get_supplementary_figures,
    get_supplementary_data_files,
    get_data_file_by_type,
    get_all_figures,
)


class TestSavePaperInputJson:
    """Tests for save_paper_input_json function."""

    def test_saves_to_json(self, tmp_path):
        """Saves paper input to JSON file."""
        paper_input = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 150,
            "figures": [],
        }
        output_path = tmp_path / "output.json"

        save_paper_input_json(paper_input, str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["paper_id"] == "test"

    def test_creates_parent_directories(self, tmp_path):
        """Creates parent directories if needed."""
        paper_input = {
            "paper_id": "test",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        output_path = tmp_path / "deep" / "nested" / "output.json"

        save_paper_input_json(paper_input, str(output_path))

        assert output_path.exists()


class TestAccessorFunctions:
    """Tests for paper input accessor functions."""

    def test_get_figure_by_id_found(self, sample_paper_input):
        """get_figure_by_id returns figure when found."""
        fig = get_figure_by_id(sample_paper_input, "Fig1")

        assert fig is not None
        assert fig["description"] == "First"

    def test_get_figure_by_id_not_found(self, sample_paper_input):
        """get_figure_by_id returns None when not found."""
        fig = get_figure_by_id(sample_paper_input, "NonExistent")

        assert fig is None

    def test_list_figure_ids(self, sample_paper_input):
        """list_figure_ids returns all figure IDs."""
        ids = list_figure_ids(sample_paper_input)

        assert ids == ["Fig1", "Fig2"]

    def test_get_supplementary_text(self, sample_paper_input):
        """get_supplementary_text returns supplementary text."""
        text = get_supplementary_text(sample_paper_input)

        assert text == "Supplementary content"

    def test_get_supplementary_text_none(self):
        """get_supplementary_text returns None when not present."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        text = get_supplementary_text(paper_input)

        assert text is None

    def test_get_supplementary_figures(self, sample_paper_input):
        """get_supplementary_figures returns supplementary figures."""
        figs = get_supplementary_figures(sample_paper_input)

        assert len(figs) == 1
        assert figs[0]["id"] == "S1"

    def test_get_supplementary_figures_empty(self):
        """get_supplementary_figures returns empty list when none."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        figs = get_supplementary_figures(paper_input)

        assert figs == []

    def test_get_supplementary_data_files(self, sample_paper_input):
        """get_supplementary_data_files returns data files."""
        files = get_supplementary_data_files(sample_paper_input)

        assert len(files) == 2

    def test_get_data_file_by_type(self, sample_paper_input):
        """get_data_file_by_type filters by data_type."""
        spectrum_files = get_data_file_by_type(sample_paper_input, "spectrum")

        assert len(spectrum_files) == 1
        assert spectrum_files[0]["id"] == "D1"

    def test_get_data_file_by_type_no_match(self, sample_paper_input):
        """get_data_file_by_type returns empty for no match."""
        files = get_data_file_by_type(sample_paper_input, "nonexistent_type")

        assert files == []

    def test_get_all_figures(self, sample_paper_input):
        """get_all_figures returns main + supplementary figures."""
        all_figs = get_all_figures(sample_paper_input)

        assert len(all_figs) == 3
        ids = [f["id"] for f in all_figs]
        assert "Fig1" in ids
        assert "S1" in ids


