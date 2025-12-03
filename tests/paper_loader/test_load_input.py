"""Tests for loading paper input data from JSON."""

import json
from unittest.mock import patch

import pytest

from src.paper_loader import load_paper_input


class TestLoadPaperInput:
    """Tests for load_paper_input function."""

    def test_loads_valid_json(self, paper_loader_fixtures_dir):
        """Loads valid paper input from JSON file."""
        json_path = paper_loader_fixtures_dir / "valid_paper_input.json"
        paper_input = load_paper_input(str(json_path))

        assert paper_input["paper_id"] == "test_valid_paper"
        assert "figures" in paper_input
        assert isinstance(paper_input["figures"], list)

    def test_loads_comprehensive_json(self, tmp_path):
        """Loads JSON with all optional fields including supplementary."""
        data = {
            "paper_id": "full_paper",
            "paper_title": "Full Title",
            "paper_text": "A" * 150,
            "paper_domain": "materials",
            "figures": [{"id": "F1", "description": "D1", "image_path": "p1.png"}],
            "supplementary": {
                "supplementary_text": "Supp text",
                "supplementary_figures": [
                    {"id": "S1", "description": "SD1", "image_path": "s1.png"}
                ],
                "supplementary_data_files": [
                    {
                        "id": "D1",
                        "description": "DD1",
                        "file_path": "d1.csv",
                        "data_type": "spectrum",
                    }
                ],
            },
        }
        json_file = tmp_path / "full.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result == data
        assert result["supplementary"]["supplementary_text"] == "Supp text"
        assert len(result["supplementary"]["supplementary_figures"]) == 1
        assert len(result["supplementary"]["supplementary_data_files"]) == 1

    def test_file_not_found_raises(self):
        """Raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Paper input file not found"):
            load_paper_input("/nonexistent/path.json")

    def test_invalid_json_raises_decode_error(self, tmp_path):
        """Raises JSONDecodeError for malformed JSON."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_paper_input(str(bad_json))

    @patch("src.paper_loader.loaders.validate_paper_input")
    def test_validates_loaded_json(self, mock_validate, tmp_path):
        """Ensures validate_paper_input is called on loaded data."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "Text",
            "figures": [],
        }
        json_file = tmp_path / "test.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        load_paper_input(str(json_file))

        mock_validate.assert_called_once_with(data)


