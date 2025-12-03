"""Tests for saving paper input data and accessor helpers."""

import json

import pytest

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
        """Saves paper input to JSON file with all fields preserved."""
        paper_input = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 150,
            "figures": [],
        }
        output_path = tmp_path / "output.json"

        save_paper_input_json(paper_input, str(output_path))

        assert output_path.exists()
        with open(output_path, encoding='utf-8') as f:
            loaded = json.load(f)
        
        # Verify ALL fields are preserved exactly
        assert loaded["paper_id"] == "test"
        assert loaded["paper_title"] == "Test"
        assert loaded["paper_text"] == "A" * 150
        assert loaded["figures"] == []
        assert len(loaded) == 4  # Ensure no extra fields

    def test_saves_complex_paper_input(self, tmp_path, sample_paper_input):
        """Saves complex paper input with supplementary materials correctly."""
        output_path = tmp_path / "complex.json"

        save_paper_input_json(sample_paper_input, str(output_path))

        assert output_path.exists()
        with open(output_path, encoding='utf-8') as f:
            loaded = json.load(f)
        
        # Verify all main fields
        assert loaded["paper_id"] == sample_paper_input["paper_id"]
        assert loaded["paper_title"] == sample_paper_input["paper_title"]
        assert loaded["paper_text"] == sample_paper_input["paper_text"]
        assert len(loaded["figures"]) == 2
        assert loaded["figures"][0]["id"] == "Fig1"
        assert loaded["figures"][0]["description"] == "First"
        assert loaded["figures"][0]["image_path"] == "fig1.png"
        
        # Verify supplementary section
        assert "supplementary" in loaded
        assert loaded["supplementary"]["supplementary_text"] == "Supplementary content"
        assert len(loaded["supplementary"]["supplementary_figures"]) == 1
        assert loaded["supplementary"]["supplementary_figures"][0]["id"] == "S1"
        assert len(loaded["supplementary"]["supplementary_data_files"]) == 2
        assert loaded["supplementary"]["supplementary_data_files"][0]["id"] == "D1"
        assert loaded["supplementary"]["supplementary_data_files"][0]["data_type"] == "spectrum"

    def test_saves_with_special_characters(self, tmp_path):
        """Saves paper input with special characters correctly."""
        paper_input = {
            "paper_id": "test_émojis_🎉",
            "paper_title": "Test with émojis 🎉 and unicode 中文",
            "paper_text": "A" * 150,
            "figures": [],
        }
        output_path = tmp_path / "special.json"

        save_paper_input_json(paper_input, str(output_path))

        with open(output_path, encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert loaded["paper_id"] == "test_émojis_🎉"
        assert loaded["paper_title"] == "Test with émojis 🎉 and unicode 中文"

    def test_saves_with_figures(self, tmp_path):
        """Saves paper input with figures correctly."""
        paper_input = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "Fig1", "description": "First", "image_path": "fig1.png"},
                {"id": "Fig2", "description": "Second", "image_path": "fig2.png", "digitized_data_path": "data.csv"},
            ],
        }
        output_path = tmp_path / "with_figures.json"

        save_paper_input_json(paper_input, str(output_path))

        with open(output_path, encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert len(loaded["figures"]) == 2
        assert loaded["figures"][0]["id"] == "Fig1"
        assert loaded["figures"][0]["description"] == "First"
        assert loaded["figures"][0]["image_path"] == "fig1.png"
        assert "digitized_data_path" not in loaded["figures"][0]
        assert loaded["figures"][1]["id"] == "Fig2"
        assert loaded["figures"][1]["digitized_data_path"] == "data.csv"

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
        assert output_path.parent.exists()
        assert (tmp_path / "deep").exists()
        assert (tmp_path / "deep" / "nested").exists()
        
        # Verify content is also correct
        with open(output_path, encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded["paper_id"] == "test"
        assert loaded["paper_title"] == "T"


class TestAccessorFunctions:
    """Tests for paper input accessor functions."""

    def test_get_figure_by_id_found(self, sample_paper_input):
        """get_figure_by_id returns complete figure when found."""
        fig = get_figure_by_id(sample_paper_input, "Fig1")

        assert fig is not None
        assert fig["id"] == "Fig1"
        assert fig["description"] == "First"
        assert fig["image_path"] == "fig1.png"
        assert len(fig) == 3  # Ensure no extra fields

    def test_get_figure_by_id_found_second(self, sample_paper_input):
        """get_figure_by_id returns correct figure for second ID."""
        fig = get_figure_by_id(sample_paper_input, "Fig2")

        assert fig is not None
        assert fig["id"] == "Fig2"
        assert fig["description"] == "Second"
        assert fig["image_path"] == "fig2.png"

    def test_get_figure_by_id_not_found(self, sample_paper_input):
        """get_figure_by_id returns None when not found."""
        fig = get_figure_by_id(sample_paper_input, "NonExistent")

        assert fig is None

    def test_get_figure_by_id_empty_string(self, sample_paper_input):
        """get_figure_by_id returns None for empty string."""
        fig = get_figure_by_id(sample_paper_input, "")

        assert fig is None

    def test_get_figure_by_id_case_sensitive(self, sample_paper_input):
        """get_figure_by_id is case sensitive."""
        fig = get_figure_by_id(sample_paper_input, "fig1")  # lowercase

        assert fig is None  # Should not find "Fig1" with "fig1"

    def test_get_figure_by_id_missing_figures_key(self):
        """get_figure_by_id handles missing figures key."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
        }
        fig = get_figure_by_id(paper_input, "Fig1")

        assert fig is None

    def test_get_figure_by_id_figures_with_missing_id(self):
        """get_figure_by_id handles figures missing id field."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [
                {"description": "No ID", "image_path": "fig.png"},
                {"id": "Fig1", "description": "Has ID", "image_path": "fig1.png"},
            ],
        }
        # Should not crash, should find Fig1
        fig = get_figure_by_id(paper_input, "Fig1")

        assert fig is not None
        assert fig["id"] == "Fig1"
        # Should not find the figure without ID
        fig_no_id = get_figure_by_id(paper_input, None)
        assert fig_no_id is None

    def test_list_figure_ids(self, sample_paper_input):
        """list_figure_ids returns all figure IDs in order."""
        ids = list_figure_ids(sample_paper_input)

        assert ids == ["Fig1", "Fig2"]
        assert len(ids) == 2

    def test_list_figure_ids_empty(self):
        """list_figure_ids returns empty list when no figures."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        ids = list_figure_ids(paper_input)

        assert ids == []
        assert isinstance(ids, list)

    def test_list_figure_ids_missing_figures_key(self):
        """list_figure_ids handles missing figures key."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
        }
        ids = list_figure_ids(paper_input)

        assert ids == []

    def test_list_figure_ids_with_missing_id_fields(self):
        """list_figure_ids handles figures missing id field."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [
                {"description": "No ID", "image_path": "fig.png"},
                {"id": "Fig1", "description": "Has ID", "image_path": "fig1.png"},
            ],
        }
        ids = list_figure_ids(paper_input)

        # Should generate "unknown_0" for first figure, "Fig1" for second
        assert len(ids) == 2
        assert ids[0] == "unknown_0"
        assert ids[1] == "Fig1"

    def test_get_supplementary_text(self, sample_paper_input):
        """get_supplementary_text returns supplementary text."""
        text = get_supplementary_text(sample_paper_input)

        assert text == "Supplementary content"
        assert isinstance(text, str)
        assert len(text) > 0

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

    def test_get_supplementary_text_missing_supplementary_key(self):
        """get_supplementary_text handles missing supplementary key."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        text = get_supplementary_text(paper_input)

        assert text is None

    def test_get_supplementary_text_empty_supplementary(self):
        """get_supplementary_text returns None when supplementary is empty."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {},
        }
        text = get_supplementary_text(paper_input)

        assert text is None

    def test_get_supplementary_figures(self, sample_paper_input):
        """get_supplementary_figures returns complete supplementary figures."""
        figs = get_supplementary_figures(sample_paper_input)

        assert len(figs) == 1
        assert figs[0]["id"] == "S1"
        assert figs[0]["description"] == "Supp fig"
        assert figs[0]["image_path"] == "s1.png"
        assert isinstance(figs, list)

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
        assert isinstance(figs, list)

    def test_get_supplementary_figures_missing_supplementary_key(self):
        """get_supplementary_figures handles missing supplementary key."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        figs = get_supplementary_figures(paper_input)

        assert figs == []

    def test_get_supplementary_figures_empty_supplementary(self):
        """get_supplementary_figures returns empty list when supplementary is empty."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {},
        }
        figs = get_supplementary_figures(paper_input)

        assert figs == []

    def test_get_supplementary_data_files(self, sample_paper_input):
        """get_supplementary_data_files returns complete data files."""
        files = get_supplementary_data_files(sample_paper_input)

        assert len(files) == 2
        assert files[0]["id"] == "D1"
        assert files[0]["description"] == "Data 1"
        assert files[0]["file_path"] == "d1.csv"
        assert files[0]["data_type"] == "spectrum"
        assert files[1]["id"] == "D2"
        assert files[1]["data_type"] == "geometry"
        assert isinstance(files, list)

    def test_get_supplementary_data_files_empty(self):
        """get_supplementary_data_files returns empty list when none."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        files = get_supplementary_data_files(paper_input)

        assert files == []
        assert isinstance(files, list)

    def test_get_supplementary_data_files_missing_supplementary_key(self):
        """get_supplementary_data_files handles missing supplementary key."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        files = get_supplementary_data_files(paper_input)

        assert files == []

    def test_get_data_file_by_type(self, sample_paper_input):
        """get_data_file_by_type filters by data_type correctly."""
        spectrum_files = get_data_file_by_type(sample_paper_input, "spectrum")

        assert len(spectrum_files) == 1
        assert spectrum_files[0]["id"] == "D1"
        assert spectrum_files[0]["data_type"] == "spectrum"
        assert spectrum_files[0]["description"] == "Data 1"
        assert spectrum_files[0]["file_path"] == "d1.csv"

    def test_get_data_file_by_type_geometry(self, sample_paper_input):
        """get_data_file_by_type filters geometry type correctly."""
        geometry_files = get_data_file_by_type(sample_paper_input, "geometry")

        assert len(geometry_files) == 1
        assert geometry_files[0]["id"] == "D2"
        assert geometry_files[0]["data_type"] == "geometry"

    def test_get_data_file_by_type_no_match(self, sample_paper_input):
        """get_data_file_by_type returns empty for no match."""
        files = get_data_file_by_type(sample_paper_input, "nonexistent_type")

        assert files == []
        assert isinstance(files, list)

    def test_get_data_file_by_type_case_sensitive(self, sample_paper_input):
        """get_data_file_by_type is case sensitive."""
        files = get_data_file_by_type(sample_paper_input, "Spectrum")  # Capital S

        assert files == []  # Should not match "spectrum"

    def test_get_data_file_by_type_empty_string(self, sample_paper_input):
        """get_data_file_by_type filters empty string type."""
        files = get_data_file_by_type(sample_paper_input, "")

        assert files == []

    def test_get_data_file_by_type_missing_data_type_field(self):
        """get_data_file_by_type handles files missing data_type field."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_data_files": [
                    {"id": "D1", "description": "Data", "file_path": "d1.csv"},
                    {"id": "D2", "description": "Data", "file_path": "d2.csv", "data_type": "spectrum"},
                ],
            },
        }
        files = get_data_file_by_type(paper_input, "spectrum")

        # Should only return D2, not D1 (which has no data_type)
        assert len(files) == 1
        assert files[0]["id"] == "D2"

    def test_get_all_figures(self, sample_paper_input):
        """get_all_figures returns main + supplementary figures in correct order."""
        all_figs = get_all_figures(sample_paper_input)

        assert len(all_figs) == 3
        # Should be main figures first, then supplementary
        assert all_figs[0]["id"] == "Fig1"
        assert all_figs[1]["id"] == "Fig2"
        assert all_figs[2]["id"] == "S1"
        
        # Verify all fields are present
        assert all_figs[0]["description"] == "First"
        assert all_figs[0]["image_path"] == "fig1.png"
        assert all_figs[2]["description"] == "Supp fig"
        assert all_figs[2]["image_path"] == "s1.png"

    def test_get_all_figures_empty(self):
        """get_all_figures returns empty list when no figures."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
        }
        all_figs = get_all_figures(paper_input)

        assert all_figs == []
        assert isinstance(all_figs, list)

    def test_get_all_figures_only_main(self):
        """get_all_figures returns only main figures when no supplementary."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "Fig1", "description": "First", "image_path": "fig1.png"},
            ],
        }
        all_figs = get_all_figures(paper_input)

        assert len(all_figs) == 1
        assert all_figs[0]["id"] == "Fig1"

    def test_get_all_figures_only_supplementary(self):
        """get_all_figures returns only supplementary figures when no main."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"id": "S1", "description": "Supp", "image_path": "s1.png"},
                ],
            },
        }
        all_figs = get_all_figures(paper_input)

        assert len(all_figs) == 1
        assert all_figs[0]["id"] == "S1"

    def test_get_all_figures_missing_figures_key(self):
        """get_all_figures handles missing figures key."""
        paper_input = {
            "paper_id": "t",
            "paper_title": "T",
            "paper_text": "A" * 150,
            "supplementary": {
                "supplementary_figures": [
                    {"id": "S1", "description": "Supp", "image_path": "s1.png"},
                ],
            },
        }
        all_figs = get_all_figures(paper_input)

        assert len(all_figs) == 1
        assert all_figs[0]["id"] == "S1"


