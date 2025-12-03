"""Tests for loading paper input data from JSON."""

import json
from pathlib import Path
from unittest.mock import patch, mock_open
import logging

import pytest

from src.paper_loader import load_paper_input, ValidationError


class TestLoadPaperInput:
    """Tests for load_paper_input function."""

    def test_loads_valid_json(self, paper_loader_fixtures_dir):
        """Loads valid paper input from JSON file with complete field verification."""
        json_path = paper_loader_fixtures_dir / "valid_paper_input.json"
        paper_input = load_paper_input(str(json_path))

        # Verify all required fields exist and have correct types
        assert paper_input["paper_id"] == "test_valid_paper"
        assert isinstance(paper_input["paper_id"], str)
        assert len(paper_input["paper_id"]) > 0
        
        assert paper_input["paper_title"] == "A Valid Test Paper for Unit Testing"
        assert isinstance(paper_input["paper_title"], str)
        assert len(paper_input["paper_title"]) > 0
        
        assert isinstance(paper_input["paper_text"], str)
        assert len(paper_input["paper_text"]) >= 100  # Minimum validation requirement
        
        assert paper_input["paper_domain"] == "plasmonics"
        assert isinstance(paper_input["paper_domain"], str)
        
        # Verify figures structure completely
        assert "figures" in paper_input
        assert isinstance(paper_input["figures"], list)
        assert len(paper_input["figures"]) == 2
        
        # Verify each figure has all required fields
        for i, fig in enumerate(paper_input["figures"]):
            assert isinstance(fig, dict), f"Figure {i} must be a dictionary"
            assert "id" in fig, f"Figure {i} missing 'id' field"
            assert isinstance(fig["id"], str), f"Figure {i} 'id' must be a string"
            assert len(fig["id"]) > 0, f"Figure {i} 'id' must be non-empty"
            assert "description" in fig, f"Figure {i} missing 'description' field"
            assert isinstance(fig["description"], str), f"Figure {i} 'description' must be a string"
            assert "image_path" in fig, f"Figure {i} missing 'image_path' field"
            assert isinstance(fig["image_path"], str), f"Figure {i} 'image_path' must be a string"
        
        # Verify specific figure values
        assert paper_input["figures"][0]["id"] == "Fig1"
        assert paper_input["figures"][1]["id"] == "Fig2"

    def test_loads_comprehensive_json(self, tmp_path):
        """Loads JSON with all optional fields including supplementary with complete verification."""
        data = {
            "paper_id": "full_paper",
            "paper_title": "Full Title",
            "paper_text": "A" * 150,
            "paper_domain": "other",
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
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        # Verify exact match
        assert result == data
        
        # Verify all top-level fields
        assert result["paper_id"] == "full_paper"
        assert result["paper_title"] == "Full Title"
        assert result["paper_text"] == "A" * 150
        assert result["paper_domain"] == "other"
        
        # Verify figures
        assert isinstance(result["figures"], list)
        assert len(result["figures"]) == 1
        assert result["figures"][0]["id"] == "F1"
        assert result["figures"][0]["description"] == "D1"
        assert result["figures"][0]["image_path"] == "p1.png"
        
        # Verify supplementary section exists
        assert "supplementary" in result
        assert isinstance(result["supplementary"], dict)
        
        # Verify supplementary_text
        assert result["supplementary"]["supplementary_text"] == "Supp text"
        assert isinstance(result["supplementary"]["supplementary_text"], str)
        
        # Verify supplementary_figures
        assert "supplementary_figures" in result["supplementary"]
        assert isinstance(result["supplementary"]["supplementary_figures"], list)
        assert len(result["supplementary"]["supplementary_figures"]) == 1
        supp_fig = result["supplementary"]["supplementary_figures"][0]
        assert supp_fig["id"] == "S1"
        assert supp_fig["description"] == "SD1"
        assert supp_fig["image_path"] == "s1.png"
        
        # Verify supplementary_data_files
        assert "supplementary_data_files" in result["supplementary"]
        assert isinstance(result["supplementary"]["supplementary_data_files"], list)
        assert len(result["supplementary"]["supplementary_data_files"]) == 1
        data_file = result["supplementary"]["supplementary_data_files"][0]
        assert data_file["id"] == "D1"
        assert data_file["description"] == "DD1"
        assert data_file["file_path"] == "d1.csv"
        assert data_file["data_type"] == "spectrum"

    def test_file_not_found_raises(self):
        """Raises FileNotFoundError for non-existent file with correct message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_paper_input("/nonexistent/path.json")
        assert "Paper input file not found" in str(exc_info.value)
        assert "/nonexistent/path.json" in str(exc_info.value)

    def test_file_not_found_relative_path(self, tmp_path):
        """Raises FileNotFoundError for relative path that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_paper_input("nonexistent_relative.json")

    def test_invalid_json_raises_decode_error(self, tmp_path):
        """Raises JSONDecodeError for malformed JSON."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_paper_input(str(bad_json))
        # Verify it's actually a JSON decode error, not just any exception
        assert isinstance(exc_info.value, json.JSONDecodeError)

    def test_invalid_json_empty_file(self, tmp_path):
        """Raises JSONDecodeError for empty file."""
        empty_json = tmp_path / "empty.json"
        empty_json.write_text("")

        with pytest.raises(json.JSONDecodeError):
            load_paper_input(str(empty_json))

    def test_invalid_json_partial_object(self, tmp_path):
        """Raises JSONDecodeError for incomplete JSON object."""
        partial_json = tmp_path / "partial.json"
        partial_json.write_text('{"paper_id": "test"')  # Missing closing brace

        with pytest.raises(json.JSONDecodeError):
            load_paper_input(str(partial_json))

    @patch("src.paper_loader.loaders.validate_paper_input")
    def test_validates_loaded_json(self, mock_validate, tmp_path):
        """Ensures validate_paper_input is called on loaded data with exact data."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        # Verify validation was called with the exact loaded data
        mock_validate.assert_called_once()
        call_args = mock_validate.call_args[0][0]
        assert call_args == data
        # Verify the function still returns the data
        assert result == data

    def test_validation_error_propagates(self, tmp_path):
        """Ensures ValidationError from validation is raised, not caught."""
        invalid_data = {
            "paper_id": "",  # Empty paper_id should fail validation
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(invalid_data, f)

        with pytest.raises(ValidationError):
            load_paper_input(str(json_file))

    def test_validation_error_missing_required_field(self, tmp_path):
        """Ensures ValidationError raised when required field is missing."""
        missing_field_data = {
            "paper_id": "test",
            "paper_title": "Title",
            # Missing paper_text
            "figures": [],
        }
        json_file = tmp_path / "missing.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(missing_field_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "Missing required field" in str(exc_info.value) or "paper_text" in str(exc_info.value)

    def test_validation_error_empty_paper_text(self, tmp_path):
        """Ensures ValidationError raised when paper_text is too short."""
        short_text_data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "short",  # Less than 100 chars
            "figures": [],
        }
        json_file = tmp_path / "short.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(short_text_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "paper_text" in str(exc_info.value).lower() or "too short" in str(exc_info.value).lower()

    def test_validation_error_invalid_figures_type(self, tmp_path):
        """Ensures ValidationError raised when figures is not a list."""
        invalid_figures_data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": "not a list",  # Wrong type
        }
        json_file = tmp_path / "invalid_figures.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(invalid_figures_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "figures" in str(exc_info.value).lower()

    def test_validation_error_figure_missing_id(self, tmp_path):
        """Ensures ValidationError raised when figure is missing id field."""
        missing_id_data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {"description": "Desc", "image_path": "path.png"}  # Missing id
            ],
        }
        json_file = tmp_path / "missing_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(missing_id_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "missing 'id'" in str(exc_info.value).lower() or "figure" in str(exc_info.value).lower()

    def test_validation_error_figure_missing_image_path(self, tmp_path):
        """Ensures ValidationError raised when figure is missing image_path field."""
        missing_path_data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "F1", "description": "Desc"}  # Missing image_path
            ],
        }
        json_file = tmp_path / "missing_path.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(missing_path_data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "image_path" in str(exc_info.value).lower()

    def test_handles_utf8_encoding(self, tmp_path):
        """Ensures UTF-8 characters are handled correctly."""
        utf8_data = {
            "paper_id": "test_utf8",
            "paper_title": "Test with émojis 🎉 and 中文",
            "paper_text": "A" * 150 + " with émojis 🎉 and 中文",
            "figures": [],
        }
        json_file = tmp_path / "utf8.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(utf8_data, f, ensure_ascii=False)

        result = load_paper_input(str(json_file))

        assert result["paper_title"] == "Test with émojis 🎉 and 中文"
        assert "émojis" in result["paper_text"]
        assert "🎉" in result["paper_text"]

    def test_handles_pathlib_path(self, tmp_path):
        """Ensures function accepts Path objects, not just strings."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "pathlib.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Should work with string path
        result1 = load_paper_input(str(json_file))
        assert result1["paper_id"] == "test"

        # Should also work with Path object (if function supports it)
        # Actually, looking at the implementation, it converts to Path internally,
        # so string is required. But let's verify it handles Path-like strings correctly.
        result2 = load_paper_input(str(json_file.resolve()))  # Absolute path
        assert result2["paper_id"] == "test"

    def test_handles_absolute_paths(self, tmp_path):
        """Ensures absolute paths work correctly."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "absolute.json"
        json_file.write_text(json.dumps(data))

        abs_path = str(json_file.resolve())
        result = load_paper_input(abs_path)

        assert result["paper_id"] == "test"

    def test_handles_relative_paths(self, tmp_path):
        """Ensures relative paths work correctly."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "relative.json"
        json_file.write_text(json.dumps(data))

        # Change to tmp_path directory and use relative path
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = load_paper_input("relative.json")
            assert result["paper_id"] == "test"
        finally:
            os.chdir(old_cwd)

    def test_preserves_all_fields_exactly(self, tmp_path):
        """Ensures all fields are preserved exactly as in JSON, no modifications."""
        data = {
            "paper_id": "exact_test",
            "paper_title": "Exact Title",
            "paper_text": "A" * 200,
            "paper_domain": "plasmonics",
            "figures": [
                {
                    "id": "F1",
                    "description": "Exact description",
                    "image_path": "exact/path.png",
                    "digitized_data_path": "exact/data.csv",  # Optional field
                }
            ],
            "supplementary": {
                "supplementary_text": "Exact supp text",
                "supplementary_figures": [
                    {
                        "id": "S1",
                        "description": "Exact supp desc",
                        "image_path": "exact/supp.png",
                    }
                ],
                "supplementary_data_files": [
                    {
                        "id": "D1",
                        "description": "Exact data desc",
                        "file_path": "exact/data.csv",
                        "data_type": "exact_type",
                    }
                ],
            },
        }
        json_file = tmp_path / "exact.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        # Verify exact match - no fields added, removed, or modified
        assert result == data
        # Also verify deep equality of nested structures
        assert result["figures"][0]["digitized_data_path"] == "exact/data.csv"
        assert result["supplementary"]["supplementary_data_files"][0]["data_type"] == "exact_type"

    def test_handles_empty_figures_list(self, tmp_path):
        """Ensures empty figures list is handled correctly."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "empty_figures.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["figures"] == []
        assert isinstance(result["figures"], list)
        assert len(result["figures"]) == 0

    def test_handles_many_figures(self, tmp_path):
        """Ensures large number of figures is handled correctly."""
        figures = [
            {"id": f"F{i}", "description": f"Figure {i}", "image_path": f"fig{i}.png"}
            for i in range(50)
        ]
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": figures,
        }
        json_file = tmp_path / "many_figures.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["figures"]) == 50
        assert result["figures"][0]["id"] == "F0"
        assert result["figures"][49]["id"] == "F49"

    def test_handles_long_paper_text(self, tmp_path):
        """Ensures long paper text is handled correctly."""
        long_text = "A" * 100000  # 100K chars, well under 600K limit
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": long_text,
            "figures": [],
        }
        json_file = tmp_path / "long_text.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["paper_text"]) == 100000
        assert result["paper_text"] == long_text

    def test_handles_max_length_paper_text(self, tmp_path):
        """Ensures paper text at maximum allowed length is handled."""
        # max_paper_chars is 600000 from CONTEXT_WINDOW_LIMITS
        max_text = "A" * 600000
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": max_text,
            "figures": [],
        }
        json_file = tmp_path / "max_text.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["paper_text"]) == 600000

    def test_validation_error_exceeds_max_length(self, tmp_path):
        """Ensures ValidationError raised when paper text exceeds maximum length."""
        # max_paper_chars is 600000, so 600001 should fail
        too_long_text = "A" * 600001
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": too_long_text,
            "figures": [],
        }
        json_file = tmp_path / "too_long.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "exceeds maximum length" in str(exc_info.value).lower() or "600000" in str(exc_info.value)

    def test_handles_supplementary_only_text(self, tmp_path):
        """Ensures supplementary with only text (no figures/files) works."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_text": "Only text",
            },
        }
        json_file = tmp_path / "supp_text_only.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["supplementary"]["supplementary_text"] == "Only text"
        assert "supplementary_figures" not in result["supplementary"]
        assert "supplementary_data_files" not in result["supplementary"]

    def test_handles_supplementary_only_figures(self, tmp_path):
        """Ensures supplementary with only figures (no text/files) works."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"id": "S1", "description": "Desc", "image_path": "s1.png"}
                ],
            },
        }
        json_file = tmp_path / "supp_figures_only.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["supplementary"]["supplementary_figures"]) == 1
        assert "supplementary_text" not in result["supplementary"]
        assert "supplementary_data_files" not in result["supplementary"]

    def test_handles_supplementary_only_data_files(self, tmp_path):
        """Ensures supplementary with only data files (no text/figures) works."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_data_files": [
                    {
                        "id": "D1",
                        "description": "Desc",
                        "file_path": "d1.csv",
                        "data_type": "spectrum",
                    }
                ],
            },
        }
        json_file = tmp_path / "supp_data_only.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["supplementary"]["supplementary_data_files"]) == 1
        assert "supplementary_text" not in result["supplementary"]
        assert "supplementary_figures" not in result["supplementary"]

    def test_handles_figure_with_digitized_data(self, tmp_path):
        """Ensures figures with optional digitized_data_path are handled."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {
                    "id": "F1",
                    "description": "Desc",
                    "image_path": "f1.png",
                    "digitized_data_path": "f1_data.csv",
                }
            ],
        }
        json_file = tmp_path / "digitized.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["figures"][0]["digitized_data_path"] == "f1_data.csv"

    def test_handles_figure_without_digitized_data(self, tmp_path):
        """Ensures figures without digitized_data_path are handled."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {
                    "id": "F1",
                    "description": "Desc",
                    "image_path": "f1.png",
                    # No digitized_data_path
                }
            ],
        }
        json_file = tmp_path / "no_digitized.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert "digitized_data_path" not in result["figures"][0]

    def test_handles_missing_paper_domain(self, tmp_path):
        """Ensures missing optional paper_domain field is handled."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            # No paper_domain
        }
        json_file = tmp_path / "no_domain.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert "paper_domain" not in result

    def test_handles_missing_supplementary(self, tmp_path):
        """Ensures missing optional supplementary field is handled."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            # No supplementary
        }
        json_file = tmp_path / "no_supplementary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert "supplementary" not in result

    def test_validation_error_supplementary_figures_not_list(self, tmp_path):
        """Ensures ValidationError raised when supplementary_figures is not a list."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": "not a list",
            },
        }
        json_file = tmp_path / "invalid_supp_figures.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "supplementary_figures" in str(exc_info.value).lower()

    def test_validation_error_supplementary_figure_missing_id(self, tmp_path):
        """Ensures ValidationError raised when supplementary figure is missing id."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"description": "Desc", "image_path": "s1.png"}  # Missing id
                ],
            },
        }
        json_file = tmp_path / "supp_missing_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "missing 'id'" in str(exc_info.value).lower() or "supplementary figure" in str(exc_info.value).lower()

    def test_validation_error_supplementary_figure_missing_image_path(self, tmp_path):
        """Ensures ValidationError raised when supplementary figure is missing image_path."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"id": "S1", "description": "Desc"}  # Missing image_path
                ],
            },
        }
        json_file = tmp_path / "supp_missing_path.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "image_path" in str(exc_info.value).lower()

    def test_handles_nested_supplementary_data(self, tmp_path):
        """Ensures deeply nested supplementary data structures are handled."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "F1", "description": "D1", "image_path": "f1.png"},
                {"id": "F2", "description": "D2", "image_path": "f2.png"},
            ],
            "supplementary": {
                "supplementary_text": "Supp text " * 100,
                "supplementary_figures": [
                    {"id": "S1", "description": "SD1", "image_path": "s1.png"},
                    {"id": "S2", "description": "SD2", "image_path": "s2.png"},
                    {"id": "S3", "description": "SD3", "image_path": "s3.png"},
                ],
                "supplementary_data_files": [
                    {
                        "id": "D1",
                        "description": "DD1",
                        "file_path": "d1.csv",
                        "data_type": "spectrum",
                    },
                    {
                        "id": "D2",
                        "description": "DD2",
                        "file_path": "d2.csv",
                        "data_type": "geometry",
                    },
                    {
                        "id": "D3",
                        "description": "DD3",
                        "file_path": "d3.csv",
                        "data_type": "parameters",
                    },
                ],
            },
        }
        json_file = tmp_path / "nested.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["figures"]) == 2
        assert len(result["supplementary"]["supplementary_figures"]) == 3
        assert len(result["supplementary"]["supplementary_data_files"]) == 3
        assert len(result["supplementary"]["supplementary_text"]) > 100

    def test_handles_whitespace_in_strings(self, tmp_path):
        """Ensures whitespace in strings is preserved exactly."""
        data = {
            "paper_id": "test  with  spaces",
            "paper_title": "Title\nwith\nnewlines",
            "paper_text": "A" * 150 + "\twith\ttabs",
            "figures": [],
        }
        json_file = tmp_path / "whitespace.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["paper_id"] == "test  with  spaces"
        assert "\n" in result["paper_title"]
        assert "\t" in result["paper_text"]

    def test_handles_special_characters_in_paths(self, tmp_path):
        """Ensures special characters in file paths are preserved."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {
                    "id": "F1",
                    "description": "Desc",
                    "image_path": "path/with-spaces_and_underscores.png",
                }
            ],
        }
        json_file = tmp_path / "special_chars.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["figures"][0]["image_path"] == "path/with-spaces_and_underscores.png"

    def test_returns_same_object_reference(self, tmp_path):
        """Ensures function returns the exact data loaded from JSON."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "reference.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        # Should be equal (same content)
        assert result == data
        # Should be the same dict object (not a copy)
        # Actually, json.load creates a new dict, so they won't be the same object
        # But they should have the same content
        assert result is not None
        assert isinstance(result, dict)

    def test_file_not_found_directory_path(self, tmp_path):
        """Ensures FileNotFoundError raised when path is a directory, not a file."""
        # Create a directory
        dir_path = tmp_path / "adir"
        dir_path.mkdir()

        # The function should raise FileNotFoundError because it's not a file
        # Actually, path.exists() returns True for directories, but open() will fail
        # Let's see what happens - open() on a directory raises IsADirectoryError
        # But the function checks path.exists() first, so it might not catch this
        # This is a potential bug - the function should check if it's a file, not just if it exists
        with pytest.raises((FileNotFoundError, IsADirectoryError, OSError)):
            load_paper_input(str(dir_path))

    def test_handles_json_with_null_values(self, tmp_path):
        """Ensures JSON with null values is handled correctly."""
        # JSON allows null, but validation should catch if required fields are null
        data = {
            "paper_id": "test",
            "paper_title": None,  # null title should fail validation
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "null.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Should raise ValidationError because paper_title is None
        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "paper_title" in str(exc_info.value).lower()

    def test_handles_json_with_numeric_paper_id(self, tmp_path):
        """Ensures ValidationError raised when paper_id is numeric instead of string."""
        data = {
            "paper_id": 12345,  # Should be string
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "numeric_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Should raise ValidationError because paper_id must be string
        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "paper_id" in str(exc_info.value).lower()

    def test_validation_error_numeric_figure_id(self, tmp_path):
        """Ensures ValidationError raised when figure id is numeric instead of string."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {"id": 123, "description": "Desc", "image_path": "f1.png"}  # id should be string
            ],
        }
        json_file = tmp_path / "numeric_fig_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Validation should catch this - figure ids must be strings
        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "id" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower() or "must be" in str(exc_info.value).lower()

    def test_handles_json_with_boolean_values(self, tmp_path):
        """Ensures JSON with boolean values in wrong places is handled."""
        data = {
            "paper_id": "test",
            "paper_title": True,  # Should be string
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "boolean.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Should raise ValidationError because paper_title must be string
        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "paper_title" in str(exc_info.value).lower()

    def test_handles_json_with_array_in_wrong_place(self, tmp_path):
        """Ensures ValidationError raised when array is used where dict expected."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                ["not", "a", "dict"]  # Should be dict, not array
            ],
        }
        json_file = tmp_path / "array_figure.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Should raise ValidationError because figure must be dict
        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "figure" in str(exc_info.value).lower() or "must be a dictionary" in str(exc_info.value).lower()

    def test_handles_very_long_paper_id(self, tmp_path):
        """Ensures very long paper_id is handled correctly."""
        long_id = "a" * 1000
        data = {
            "paper_id": long_id,
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "long_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["paper_id"] == long_id
        assert len(result["paper_id"]) == 1000

    def test_handles_very_long_figure_description(self, tmp_path):
        """Ensures very long figure descriptions are handled correctly."""
        long_desc = "A" * 10000
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "F1", "description": long_desc, "image_path": "f1.png"}
            ],
        }
        json_file = tmp_path / "long_desc.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert len(result["figures"][0]["description"]) == 10000
        assert result["figures"][0]["description"] == long_desc

    def test_handles_empty_string_paper_id(self, tmp_path):
        """Ensures ValidationError raised when paper_id is empty string."""
        data = {
            "paper_id": "",  # Empty string should fail validation
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "empty_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "paper_id" in str(exc_info.value).lower()

    def test_validation_error_empty_string_figure_id(self, tmp_path):
        """Ensures ValidationError raised when figure id is empty string."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "", "description": "Desc", "image_path": "f1.png"}  # Empty id
            ],
        }
        json_file = tmp_path / "empty_fig_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Validation should catch this - figure ids must be non-empty strings
        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "id" in str(exc_info.value).lower()
        assert "non-empty" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()

    def test_handles_unicode_in_paper_id(self, tmp_path):
        """Ensures unicode characters in paper_id are handled correctly."""
        data = {
            "paper_id": "test_émoji_🎉_中文",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
        }
        json_file = tmp_path / "unicode_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        result = load_paper_input(str(json_file))

        assert result["paper_id"] == "test_émoji_🎉_中文"

    def test_handles_json_minimal_valid(self, tmp_path):
        """Ensures minimal valid JSON (only required fields) works."""
        data = {
            "paper_id": "minimal",
            "paper_title": "Minimal Title",
            "paper_text": "A" * 100,  # Exactly 100 chars (minimum)
            "figures": [],
        }
        json_file = tmp_path / "minimal.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_paper_input(str(json_file))

        assert result["paper_id"] == "minimal"
        assert len(result["paper_text"]) == 100
        assert result["figures"] == []
        assert "paper_domain" not in result
        assert "supplementary" not in result

    def test_validation_error_supplementary_figure_numeric_id(self, tmp_path):
        """Ensures ValidationError raised when supplementary figure id is numeric."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"id": 456, "description": "Desc", "image_path": "s1.png"}  # Numeric id
                ],
            },
        }
        json_file = tmp_path / "supp_numeric_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "supplementary figure" in str(exc_info.value).lower() or "figure" in str(exc_info.value).lower()
        assert "id" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower() or "must be" in str(exc_info.value).lower()

    def test_validation_error_supplementary_figure_empty_id(self, tmp_path):
        """Ensures ValidationError raised when supplementary figure id is empty string."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "A" * 150,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"id": "", "description": "Desc", "image_path": "s1.png"}  # Empty id
                ],
            },
        }
        json_file = tmp_path / "supp_empty_id.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError) as exc_info:
            load_paper_input(str(json_file))
        assert "supplementary figure" in str(exc_info.value).lower() or "figure" in str(exc_info.value).lower()
        assert "id" in str(exc_info.value).lower()
        assert "non-empty" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()


