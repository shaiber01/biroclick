"""Unit tests for src/agents/helpers/materials.py"""

import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.agents.helpers.materials import (
    materials_from_stage_outputs,
    load_material_database,
    match_material_from_text,
    format_validated_material,
    extract_materials_from_plan_assumptions,
    deduplicate_materials,
    extract_validated_materials,
    format_material_checkpoint_question,
)


class TestMaterialsFromStageOutputs:
    """Tests for materials_from_stage_outputs function."""

    def test_empty_state_returns_empty(self):
        """Should return empty list for empty state."""
        result = materials_from_stage_outputs({})
        assert result == []
        assert isinstance(result, list)

    def test_missing_stage_outputs_key_returns_empty(self):
        """Should return empty list when stage_outputs key is missing."""
        state = {"plan": {}, "assumptions": {}}
        result = materials_from_stage_outputs(state)
        assert result == []

    def test_empty_files_list_returns_empty(self):
        """Should return empty list when files list is empty."""
        state = {"stage_outputs": {"files": []}}
        result = materials_from_stage_outputs(state)
        assert result == []

    def test_extracts_from_stage_outputs(self):
        """Should extract materials from stage output files."""
        state = {
            "stage_outputs": {
                "files": ["palik_gold.csv", "palik_silver.csv"]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 2
        assert any(m["material_id"] == "palik_gold" for m in result)
        assert any(m["material_id"] == "palik_silver" for m in result)
        
        # Verify all required fields are present
        for mat in result:
            assert "material_id" in mat
            assert "name" in mat
            assert "source" in mat
            assert "path" in mat
            assert "csv_available" in mat
            assert "from" in mat
            assert mat["source"] == "stage0_output"
            assert mat["from"] == "stage0_output"

    def test_extracts_material_id_from_filename(self):
        """Should correctly extract material_id from filename stem."""
        state = {
            "stage_outputs": {
                "files": ["johnson_christy_gold.csv"]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "johnson_christy_gold"
        assert result[0]["name"] == "Johnson Christy Gold"  # Title case with spaces

    def test_filters_non_data_files(self):
        """Should filter out non-data files."""
        state = {
            "stage_outputs": {
                "files": ["gold.csv", "report.pdf", "plot.png", "data.txt"]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"
        assert result[0]["csv_available"] is True

    def test_accepts_all_data_file_formats(self):
        """Should accept CSV, JSON, H5, HDF5, NPZ, and NPY files."""
        state = {
            "stage_outputs": {
                "files": [
                    "gold.csv",
                    "silver.json",
                    "copper.h5",
                    "aluminum.hdf5",
                    "titanium.npz",
                    "platinum.npy",
                ]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 6
        assert result[0]["csv_available"] is True
        assert result[1]["csv_available"] is False
        assert result[2]["csv_available"] is False
        assert result[3]["csv_available"] is False
        assert result[4]["csv_available"] is False
        assert result[5]["csv_available"] is False

    def test_handles_dict_entries_with_path(self):
        """Should handle dict file entries with path key."""
        state = {
            "stage_outputs": {
                "files": [{"path": "materials/silicon.csv"}]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "silicon"
        assert result[0]["path"] == "materials/silicon.csv"

    def test_handles_dict_entries_with_file_key(self):
        """Should handle dict file entries with file key."""
        state = {
            "stage_outputs": {
                "files": [{"file": "materials/gold.csv"}]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    def test_handles_dict_entries_with_filename_key(self):
        """Should handle dict file entries with filename key."""
        state = {
            "stage_outputs": {
                "files": [{"filename": "materials/silver.csv"}]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "silver"

    def test_filters_invalid_dict_entries(self):
        """Should filter out dict entries without valid path/file/filename."""
        state = {
            "stage_outputs": {
                "files": [
                    {"invalid": "data.csv"},
                    {"path": "valid.csv"},
                    {},
                ]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "valid"

    def test_filters_none_entries(self):
        """Should filter out None entries."""
        state = {
            "stage_outputs": {
                "files": [None, "gold.csv", None]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    def test_filters_empty_string_entries(self):
        """Should filter out empty string entries."""
        state = {
            "stage_outputs": {
                "files": ["", "gold.csv", "   "]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    def test_fallback_to_stage0_progress_when_no_files(self):
        """Should fall back to stage0_progress when stage_outputs has no files."""
        state = {
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0_material_validation",
                        "outputs": [
                            {"filename": "palik_gold.csv"},
                            {"filename": "palik_silver.csv"},
                        ]
                    }
                ]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 2
        assert any(m["material_id"] == "palik_gold" for m in result)
        assert any(m["material_id"] == "palik_silver" for m in result)
        assert all(m["source"] == "stage0_progress" for m in result)
        assert all(m["from"] == "stage0_progress" for m in result)

    def test_fallback_to_stage0_progress_when_missing_stage_outputs(self):
        """Should fall back to stage0_progress when stage_outputs is missing."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0_material_validation",
                        "outputs": [
                            {"filename": "copper.csv"},
                        ]
                    }
                ]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "copper"
        assert result[0]["source"] == "stage0_progress"

    def test_fallback_handles_missing_outputs_key(self):
        """Should handle missing outputs key in stage0_progress."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0_material_validation",
                    }
                ]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert result == []

    def test_fallback_handles_missing_filename_in_outputs(self):
        """Should handle missing filename key in output entries."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0_material_validation",
                        "outputs": [
                            {"filename": "gold.csv"},
                            {"invalid": "silver.csv"},
                            {},
                        ]
                    }
                ]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    def test_case_insensitive_file_extensions(self):
        """Should handle case-insensitive file extensions."""
        state = {
            "stage_outputs": {
                "files": ["gold.CSV", "silver.JSON", "copper.H5"]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert len(result) == 3
        assert result[0]["csv_available"] is True
        assert result[1]["csv_available"] is False
        assert result[2]["csv_available"] is False

    def test_name_formatting_with_underscores(self):
        """Should format material names by replacing underscores with spaces and title casing."""
        state = {
            "stage_outputs": {
                "files": ["johnson_christy_gold.csv", "palik_silver.csv"]
            }
        }
        
        result = materials_from_stage_outputs(state)
        
        assert result[0]["name"] == "Johnson Christy Gold"
        assert result[1]["name"] == "Palik Silver"


class TestLoadMaterialDatabase:
    """Tests for load_material_database function."""

    @patch("builtins.open", mock_open(read_data='{"materials": [{"name": "Gold", "material_id": "gold"}]}'))
    @patch("os.path.exists", return_value=True)
    def test_loads_database_from_relative_path(self, mock_exists):
        """Should load material database from relative path."""
        result = load_material_database()
        
        assert isinstance(result, dict)
        assert "materials" in result
        assert len(result["materials"]) == 1
        assert result["materials"][0]["name"] == "Gold"
        assert result["materials"][0]["material_id"] == "gold"

    @patch("builtins.open", side_effect=FileNotFoundError("mocked"))
    @patch("os.path.exists", return_value=False)
    def test_returns_empty_on_missing_file(self, mock_exists, mock_open_func):
        """Should return empty dict when file not found."""
        result = load_material_database()
        assert isinstance(result, dict)
        assert result == {}

    @patch("builtins.open", side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    @patch("os.path.exists", return_value=True)
    def test_handles_invalid_json(self, mock_exists, mock_open):
        """Should return empty dict when JSON is invalid."""
        result = load_material_database()
        assert isinstance(result, dict)
        assert result == {}

    @patch("builtins.open", mock_open(read_data='{}'))
    @patch("os.path.exists", return_value=True)
    def test_handles_empty_json(self, mock_exists):
        """Should return empty dict when JSON file is empty."""
        result = load_material_database()
        assert isinstance(result, dict)
        assert result == {}

    @patch("builtins.open", mock_open(read_data='{"materials": []}'))
    @patch("os.path.exists", return_value=True)
    def test_handles_empty_materials_list(self, mock_exists):
        """Should handle empty materials list."""
        result = load_material_database()
        assert isinstance(result, dict)
        assert "materials" in result
        assert result["materials"] == []

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    @patch("os.path.exists", return_value=True)
    def test_handles_permission_error(self, mock_exists, mock_open):
        """Should return empty dict on permission error."""
        result = load_material_database()
        assert isinstance(result, dict)
        assert result == {}

    @patch("builtins.open", mock_open(read_data='{"materials": [{"material_id": "gold"}]}'))
    @patch("os.path.exists", return_value=False)
    def test_tries_cwd_fallback_when_relative_path_missing(self, mock_exists):
        """Should try CWD fallback when relative path doesn't exist."""
        # Relative path doesn't exist, so function tries CWD path
        result = load_material_database()
        
        assert isinstance(result, dict)
        assert "materials" in result
        # Function checks relative path existence once, then tries to open CWD path
        assert mock_exists.call_count >= 1


class TestMatchMaterialFromText:
    """Tests for match_material_from_text function."""

    def test_exact_material_id_match(self):
        """Should match exact material ID."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold"},
        }
        
        result = match_material_from_text("using palik_gold data", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"
        assert result["name"] == "Gold"

    def test_exact_material_id_match_case_insensitive(self):
        """Should match material ID case-insensitively."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold"},
        }
        
        result = match_material_from_text("using PALIK_GOLD data", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"

    def test_simple_name_match(self):
        """Should match simple material names."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold", "csv_available": True},
            "gold": {"material_id": "palik_gold", "name": "Gold"},
        }
        
        result = match_material_from_text("gold nanoparticle simulation", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"

    def test_simple_name_match_prefers_csv_available(self):
        """Should prefer entries with csv_available=True when multiple matches."""
        lookup = {
            "gold_no_csv": {"material_id": "gold_no_csv", "name": "Gold", "csv_available": False},
            "palik_gold": {"material_id": "palik_gold", "name": "Gold", "csv_available": True},
        }
        
        result = match_material_from_text("gold nanoparticle", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"
        assert result["csv_available"] is True

    def test_chemical_symbol_match_ag(self):
        """Should match Ag chemical symbol to silver."""
        lookup = {
            "palik_silver": {"material_id": "palik_silver", "name": "Silver", "csv_available": True},
            "silver": {"material_id": "palik_silver", "name": "Silver"},
        }
        
        result = match_material_from_text("Ag nanosphere", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_silver"

    def test_chemical_symbol_match_au(self):
        """Should match Au chemical symbol to gold."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold", "csv_available": True},
            "gold": {"material_id": "palik_gold", "name": "Gold"},
        }
        
        result = match_material_from_text("Au nanoparticle", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"

    def test_chemical_symbol_match_al(self):
        """Should match Al chemical symbol to aluminum."""
        lookup = {
            "rakic_aluminum": {"material_id": "rakic_aluminum", "name": "Aluminum", "csv_available": True},
            "aluminum": {"material_id": "rakic_aluminum", "name": "Aluminum"},
        }
        
        result = match_material_from_text("Al film", lookup)
        
        assert result is not None
        assert result["material_id"] == "rakic_aluminum"

    def test_chemical_symbol_match_si(self):
        """Should match Si chemical symbol to silicon."""
        lookup = {
            "palik_silicon": {"material_id": "palik_silicon", "name": "Silicon", "csv_available": True},
            "silicon": {"material_id": "palik_silicon", "name": "Silicon"},
        }
        
        result = match_material_from_text("Si substrate", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_silicon"

    def test_word_boundary_prevents_false_positives(self):
        """Should use word boundaries to prevent false positives."""
        lookup = {
            "custom_au": {"material_id": "custom_au", "name": "Gold"},
        }
        
        # "golden" should not match "gold"
        result = match_material_from_text("golden ratio calculation", lookup)
        assert result is None
        
        # "usage" should not match "ag"
        lookup2 = {
            "custom_ag": {"material_id": "custom_ag", "name": "Silver"},
        }
        result2 = match_material_from_text("usage of material", lookup2)
        assert result2 is None

    def test_no_match_returns_none(self):
        """Should return None when no match found."""
        lookup = {"palik_gold": {"material_id": "palik_gold"}}
        
        result = match_material_from_text("using copper material", lookup)
        
        assert result is None

    def test_empty_text_returns_none(self):
        """Should return None for empty text."""
        lookup = {"palik_gold": {"material_id": "palik_gold"}}
        
        result = match_material_from_text("", lookup)
        
        assert result is None

    def test_empty_lookup_returns_none(self):
        """Should return None when lookup is empty."""
        result = match_material_from_text("gold material", {})
        
        assert result is None

    def test_material_id_match_priority_over_name(self):
        """Should prioritize material_id match over name match."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold"},
            "gold": {"material_id": "gold", "name": "Gold"},
        }
        
        # Contains both "palik_gold" (material_id) and "gold" (name)
        result = match_material_from_text("using palik_gold and gold", lookup)
        
        # Should match palik_gold first (material_id match has priority)
        assert result is not None
        assert result["material_id"] == "palik_gold"

    def test_case_insensitive_name_matching(self):
        """Should match names case-insensitively."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold", "csv_available": True},
            "gold": {"material_id": "palik_gold", "name": "Gold"},
        }
        
        result = match_material_from_text("GOLD nanoparticle", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"

    def test_multiple_matches_returns_first_csv_available(self):
        """Should return first CSV-available match when multiple candidates."""
        lookup = {
            "gold1": {"material_id": "gold1", "name": "Gold", "csv_available": False},
            "gold2": {"material_id": "gold2", "name": "Gold", "csv_available": True},
            "gold3": {"material_id": "gold3", "name": "Gold", "csv_available": True},
        }
        
        result = match_material_from_text("gold material", lookup)
        
        assert result is not None
        assert result["csv_available"] is True
        # Should return first CSV-available match
        assert result["material_id"] == "gold2"

    def test_falls_back_to_non_csv_when_no_csv_available(self):
        """Should fall back to non-CSV entries when no CSV available."""
        lookup = {
            "gold1": {"material_id": "gold1", "name": "Gold", "csv_available": False},
            "gold2": {"material_id": "gold2", "name": "Gold", "csv_available": False},
        }
        
        result = match_material_from_text("gold material", lookup)
        
        assert result is not None
        assert result["material_id"] == "gold1"  # First candidate

    def test_special_characters_in_text(self):
        """Should handle special characters in text."""
        lookup = {
            "palik_gold": {"material_id": "palik_gold", "name": "Gold"},
        }
        
        result = match_material_from_text("using palik_gold (ref. [1])", lookup)
        
        assert result is not None
        assert result["material_id"] == "palik_gold"


class TestFormatValidatedMaterial:
    """Tests for format_validated_material function."""

    def test_formats_basic_material(self):
        """Should format basic material entry."""
        entry = {
            "material_id": "palik_gold",
            "name": "Gold",
            "source": "Palik",
            "data_file": "palik_gold.csv",
            "csv_available": True,
        }
        
        result = format_validated_material(entry, "test_source")
        
        assert result["material_id"] == "palik_gold"
        assert result["name"] == "Gold"
        assert result["source"] == "Palik"
        assert result["path"] == "materials/palik_gold.csv"
        assert result["csv_available"] is True
        assert result["from"] == "test_source"

    def test_handles_missing_data_file(self):
        """Should handle missing data_file."""
        entry = {"material_id": "custom", "name": "Custom"}
        
        result = format_validated_material(entry, "custom_source")
        
        assert result["path"] is None
        assert result["material_id"] == "custom"
        assert result["name"] == "Custom"
        assert result["from"] == "custom_source"

    def test_handles_none_data_file(self):
        """Should handle None data_file."""
        entry = {"material_id": "custom", "name": "Custom", "data_file": None}
        
        result = format_validated_material(entry, "source")
        
        assert result["path"] is None

    def test_handles_empty_string_data_file(self):
        """Should handle empty string data_file."""
        entry = {"material_id": "custom", "name": "Custom", "data_file": ""}
        
        result = format_validated_material(entry, "source")
        
        # Empty string should be treated as None
        assert result["path"] is None

    def test_preserves_all_fields(self):
        """Should preserve all fields from entry."""
        entry = {
            "material_id": "palik_gold",
            "name": "Gold",
            "source": "Palik",
            "data_file": "palik_gold.csv",
            "csv_available": True,
            "drude_lorentz_fit": {"omega_p": 1.0},
            "wavelength_range_nm": [400, 800],
        }
        
        result = format_validated_material(entry, "test_source")
        
        assert result["drude_lorentz_fit"] == {"omega_p": 1.0}
        assert result["wavelength_range_nm"] == [400, 800]

    def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields."""
        entry = {
            "material_id": "custom",
            "name": "Custom",
        }
        
        result = format_validated_material(entry, "source")
        
        assert result["source"] is None
        assert result["csv_available"] is False
        assert result["drude_lorentz_fit"] is None
        assert result["wavelength_range_nm"] is None

    def test_handles_none_optional_fields(self):
        """Should handle None values for optional fields."""
        entry = {
            "material_id": "custom",
            "name": "Custom",
            "source": None,
            "csv_available": None,
            "drude_lorentz_fit": None,
            "wavelength_range_nm": None,
        }
        
        result = format_validated_material(entry, "source")
        
        assert result["source"] is None
        assert result["csv_available"] is False  # get() with default False
        assert result["drude_lorentz_fit"] is None
        assert result["wavelength_range_nm"] is None


class TestExtractMaterialsFromPlanAssumptions:
    """Tests for extract_materials_from_plan_assumptions function."""

    @patch("src.agents.helpers.materials.load_material_database")
    def test_empty_database_returns_empty(self, mock_db):
        """Should return empty list when database is empty."""
        mock_db.return_value = {}
        
        result = extract_materials_from_plan_assumptions({})
        
        assert result == []
        assert isinstance(result, list)

    @patch("src.agents.helpers.materials.load_material_database")
    def test_missing_plan_returns_empty(self, mock_db):
        """Should return empty list when plan is missing."""
        mock_db.return_value = {"materials": [{"material_id": "gold"}]}
        
        result = extract_materials_from_plan_assumptions({})
        
        assert result == []

    @patch("src.agents.helpers.materials.load_material_database")
    def test_missing_extracted_parameters_returns_empty(self, mock_db):
        """Should return empty list when extracted_parameters is missing."""
        mock_db.return_value = {"materials": [{"material_id": "gold"}]}
        
        state = {"plan": {}, "assumptions": {}}
        result = extract_materials_from_plan_assumptions(state)
        
        assert result == []

    @patch("src.agents.helpers.materials.load_material_database")
    def test_extracts_from_parameters(self, mock_db):
        """Should extract materials from plan parameters."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {
                "extracted_parameters": [
                    {"name": "substrate_material", "value": "gold substrate"}
                ]
            },
            "assumptions": {},
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert len(result) > 0
        assert result[0]["material_id"] == "palik_gold"
        assert result[0]["from"].startswith("parameter:")

    @patch("src.agents.helpers.materials.load_material_database")
    def test_extracts_from_assumptions(self, mock_db):
        """Should extract materials from assumptions."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_silver", "name": "Silver", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {"extracted_parameters": []},
            "assumptions": {
                "global_assumptions": {
                    "materials": [
                        {"description": "silver nanoparticle"}
                    ]
                }
            },
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert len(result) > 0
        assert result[0]["material_id"] == "palik_silver"
        assert result[0]["from"].startswith("assumption:")

    @patch("src.agents.helpers.materials.load_material_database")
    def test_deduplicates_materials_from_multiple_sources(self, mock_db):
        """Should deduplicate materials found in both parameters and assumptions."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {
                "extracted_parameters": [
                    {"name": "substrate_material", "value": "gold substrate"}
                ]
            },
            "assumptions": {
                "global_assumptions": {
                    "materials": [
                        {"description": "gold nanoparticle"}
                    ]
                }
            },
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        # Should only appear once despite being in both sources
        assert len(result) == 1
        assert result[0]["material_id"] == "palik_gold"

    @patch("src.agents.helpers.materials.load_material_database")
    def test_filters_non_material_parameters(self, mock_db):
        """Should only process parameters with 'material' in name."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {
                "extracted_parameters": [
                    {"name": "wavelength", "value": "500 nm"},
                    {"name": "substrate_material", "value": "gold"},
                ]
            },
            "assumptions": {},
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "palik_gold"

    @patch("src.agents.helpers.materials.load_material_database")
    def test_handles_missing_assumptions_key(self, mock_db):
        """Should handle missing assumptions key."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {
                "extracted_parameters": [
                    {"name": "substrate_material", "value": "gold"}
                ]
            },
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert len(result) == 1

    @patch("src.agents.helpers.materials.load_material_database")
    def test_handles_missing_global_assumptions(self, mock_db):
        """Should handle missing global_assumptions key."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {"extracted_parameters": []},
            "assumptions": {},
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert result == []

    @patch("src.agents.helpers.materials.load_material_database")
    def test_handles_non_dict_assumption_entries(self, mock_db):
        """Should handle non-dict assumption entries."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {"extracted_parameters": []},
            "assumptions": {
                "global_assumptions": {
                    "materials": [
                        "invalid string entry",
                        {"description": "gold material"},
                    ]
                }
            },
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "palik_gold"

    @patch("src.agents.helpers.materials.load_material_database")
    def test_handles_missing_description_in_assumption(self, mock_db):
        """Should handle missing description key in assumption."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_silver", "name": "Silver", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {"extracted_parameters": []},
            "assumptions": {
                "global_assumptions": {
                    "materials": [
                        {"invalid": "gold material"},
                        {"description": "silver material"},
                    ]
                }
            },
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        # Should only match silver (has description)
        # Note: lookup needs "silver" key, which is built from material_id "palik_silver"
        # The function splits on "_" and uses the last part, so "silver" should be in lookup
        assert len(result) == 1
        assert result[0]["material_id"] == "palik_silver"

    @patch("src.agents.helpers.materials.load_material_database")
    def test_builds_lookup_with_simple_names(self, mock_db):
        """Should build lookup with simple names from material_id."""
        mock_db.return_value = {
            "materials": [
                {"material_id": "palik_gold", "name": "Gold", "csv_available": True}
            ]
        }
        
        state = {
            "plan": {
                "extracted_parameters": [
                    {"name": "substrate_material", "value": "gold"}
                ]
            },
            "assumptions": {},
        }
        
        result = extract_materials_from_plan_assumptions(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "palik_gold"


class TestDeduplicateMaterials:
    """Tests for deduplicate_materials function."""

    def test_empty_list_returns_empty(self):
        """Should return empty list for empty input."""
        assert deduplicate_materials([]) == []
        assert isinstance(deduplicate_materials([]), list)

    def test_removes_duplicates_by_path(self):
        """Should remove duplicates by path."""
        materials = [
            {"material_id": "gold1", "path": "materials/gold.csv"},
            {"material_id": "gold2", "path": "materials/gold.csv"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold1"  # First occurrence kept

    def test_removes_duplicates_by_id(self):
        """Should remove duplicates by material_id when path is None."""
        materials = [
            {"material_id": "gold", "path": None},
            {"material_id": "gold", "path": None},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    def test_keeps_unique_materials(self):
        """Should keep unique materials."""
        materials = [
            {"material_id": "gold", "path": "gold.csv"},
            {"material_id": "silver", "path": "silver.csv"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 2
        assert {m["material_id"] for m in result} == {"gold", "silver"}

    def test_handles_mixed_path_and_id_deduplication(self):
        """Should handle materials with both path and id duplicates."""
        materials = [
            {"material_id": "gold1", "path": "materials/gold.csv"},
            {"material_id": "gold2", "path": "materials/gold.csv"},  # Duplicate path
            {"material_id": "gold", "path": None},
            {"material_id": "gold", "path": None},  # Duplicate id
            {"material_id": "silver", "path": "silver.csv"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 3
        # Should keep first gold with path, first gold without path, and silver
        material_ids = {m["material_id"] for m in result}
        assert "gold1" in material_ids or "gold2" in material_ids
        assert "gold" in material_ids
        assert "silver" in material_ids

    def test_handles_missing_path_key(self):
        """Should handle missing path key."""
        materials = [
            {"material_id": "gold"},
            {"material_id": "gold"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    def test_handles_missing_material_id_key(self):
        """Should handle missing material_id key."""
        materials = [
            {"path": "gold.csv"},
            {"path": "gold.csv"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 1
        assert result[0]["path"] == "gold.csv"

    def test_handles_both_keys_missing(self):
        """Should filter out entries with both path and material_id missing."""
        materials = [
            {"material_id": "gold", "path": "gold.csv"},
            {},  # Both keys missing
            {"material_id": "silver", "path": "silver.csv"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 2
        assert {m["material_id"] for m in result} == {"gold", "silver"}

    def test_handles_empty_string_path(self):
        """Should treat empty string path as None."""
        materials = [
            {"material_id": "gold", "path": ""},
            {"material_id": "gold", "path": None},
        ]
        
        result = deduplicate_materials(materials)
        
        # Empty string should be treated as falsy, so deduplication by material_id
        assert len(result) == 1

    def test_preserves_order_of_first_occurrence(self):
        """Should preserve order, keeping first occurrence."""
        materials = [
            {"material_id": "gold", "path": "gold.csv"},
            {"material_id": "silver", "path": "silver.csv"},
            {"material_id": "gold", "path": "gold.csv"},  # Duplicate
            {"material_id": "copper", "path": "copper.csv"},
        ]
        
        result = deduplicate_materials(materials)
        
        assert len(result) == 3
        assert result[0]["material_id"] == "gold"
        assert result[1]["material_id"] == "silver"
        assert result[2]["material_id"] == "copper"


class TestExtractValidatedMaterials:
    """Tests for extract_validated_materials function."""

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_prefers_stage_outputs(self, mock_plan, mock_stage):
        """Should prefer materials from stage outputs."""
        mock_stage.return_value = [{"material_id": "gold", "path": "gold.csv"}]
        mock_plan.return_value = []
        
        result = extract_validated_materials({})
        
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"
        assert mock_stage.called
        assert not mock_plan.called  # Should not call plan when stage has results

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_falls_back_to_plan_assumptions(self, mock_plan, mock_stage):
        """Should fall back to plan assumptions when stage outputs empty."""
        mock_stage.return_value = []
        mock_plan.return_value = [{"material_id": "silver", "path": "silver.csv"}]
        
        result = extract_validated_materials({})
        
        assert len(result) == 1
        assert result[0]["material_id"] == "silver"
        assert mock_stage.called
        assert mock_plan.called

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_falls_back_to_planned_materials(self, mock_plan, mock_stage):
        """Should fall back to planned_materials as last resort."""
        mock_stage.return_value = []
        mock_plan.return_value = []
        
        state = {
            "planned_materials": [{"material_id": "copper", "path": "copper.csv"}]
        }
        
        result = extract_validated_materials(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "copper"

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_returns_empty_when_all_sources_empty(self, mock_plan, mock_stage):
        """Should return empty list when all sources are empty."""
        mock_stage.return_value = []
        mock_plan.return_value = []
        
        result = extract_validated_materials({})
        
        assert result == []

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_deduplicates_combined_sources(self, mock_plan, mock_stage):
        """Should deduplicate materials from multiple sources."""
        mock_stage.return_value = [
            {"material_id": "gold", "path": "materials/gold.csv"}
        ]
        mock_plan.return_value = [
            {"material_id": "gold", "path": "materials/gold.csv"}  # Duplicate
        ]
        
        result = extract_validated_materials({})
        
        # Should deduplicate
        assert len(result) == 1
        assert result[0]["material_id"] == "gold"

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_handles_missing_planned_materials_key(self, mock_plan, mock_stage):
        """Should handle missing planned_materials key."""
        mock_stage.return_value = []
        mock_plan.return_value = []
        
        result = extract_validated_materials({})
        
        assert result == []

    @patch("src.agents.helpers.materials.materials_from_stage_outputs")
    @patch("src.agents.helpers.materials.extract_materials_from_plan_assumptions")
    def test_handles_empty_planned_materials_list(self, mock_plan, mock_stage):
        """Should handle empty planned_materials list."""
        mock_stage.return_value = []
        mock_plan.return_value = []
        
        state = {"planned_materials": []}
        
        result = extract_validated_materials(state)
        
        assert result == []


class TestFormatMaterialCheckpointQuestion:
    """Tests for format_material_checkpoint_question function."""

    def test_formats_question_with_materials(self):
        """Should format question with material information."""
        state = {"paper_id": "test_paper"}
        stage0_info = {}
        plot_files = ["plot1.png", "plot2.png"]
        validated_materials = [
            {"name": "Gold", "source": "Palik", "path": "gold.csv"}
        ]
        
        result = format_material_checkpoint_question(
            state, stage0_info, plot_files, validated_materials
        )
        
        assert "test_paper" in result
        assert "GOLD" in result
        assert "Palik" in result
        assert "gold.csv" in result
        assert "plot1.png" in result
        assert "plot2.png" in result
        assert "APPROVE" in result
        assert "CHANGE_DATABASE" in result
        assert "CHANGE_MATERIAL" in result
        assert "NEED_HELP" in result

    def test_handles_empty_materials(self):
        """Should handle empty materials list."""
        state = {"paper_id": "test_paper"}
        
        result = format_material_checkpoint_question(state, {}, [], [])
        
        assert "No materials automatically detected" in result
        assert "test_paper" in result

    def test_handles_empty_plots(self):
        """Should handle empty plot list."""
        state = {"paper_id": "test_paper"}
        validated_materials = [
            {"name": "Gold", "source": "Palik", "path": "gold.csv"}
        ]
        
        result = format_material_checkpoint_question(state, {}, [], validated_materials)
        
        assert "No plots generated" in result
        assert "GOLD" in result

    def test_handles_missing_paper_id(self):
        """Should handle missing paper_id."""
        state = {}
        
        result = format_material_checkpoint_question(state, {}, [], [])
        
        assert "unknown" in result

    def test_handles_materials_without_name(self):
        """Should handle materials without name field."""
        state = {"paper_id": "test_paper"}
        validated_materials = [
            {"material_id": "palik_gold", "source": "Palik", "path": "gold.csv"}
        ]
        
        result = format_material_checkpoint_question(
            state, {}, [], validated_materials
        )
        
        assert "PALIK_GOLD" in result or "unknown" in result

    def test_handles_materials_without_material_id_or_name(self):
        """Should handle materials without both name and material_id."""
        state = {"paper_id": "test_paper"}
        validated_materials = [
            {"source": "Palik", "path": "gold.csv"}
        ]
        
        result = format_material_checkpoint_question(
            state, {}, [], validated_materials
        )
        
        # Component uses "unknown" as fallback, which gets uppercased to "UNKNOWN"
        assert "UNKNOWN" in result

    def test_handles_materials_without_path(self):
        """Should handle materials without path."""
        state = {"paper_id": "test_paper"}
        validated_materials = [
            {"name": "Gold", "source": "Palik"}
        ]
        
        result = format_material_checkpoint_question(
            state, {}, [], validated_materials
        )
        
        assert "GOLD" in result
        assert "N/A" in result

    def test_handles_materials_without_source(self):
        """Should handle materials without source."""
        state = {"paper_id": "test_paper"}
        validated_materials = [
            {"name": "Gold", "path": "gold.csv"}
        ]
        
        result = format_material_checkpoint_question(
            state, {}, [], validated_materials
        )
        
        assert "GOLD" in result
        assert "unknown" in result

    def test_formats_multiple_materials(self):
        """Should format multiple materials correctly."""
        state = {"paper_id": "test_paper"}
        validated_materials = [
            {"name": "Gold", "source": "Palik", "path": "gold.csv"},
            {"name": "Silver", "source": "Johnson-Christy", "path": "silver.csv"},
        ]
        
        result = format_material_checkpoint_question(
            state, {}, [], validated_materials
        )
        
        assert "GOLD" in result
        assert "SILVER" in result
        assert "Palik" in result
        assert "Johnson-Christy" in result

    def test_includes_all_required_options(self):
        """Should include all required options in the question."""
        state = {"paper_id": "test_paper"}
        
        result = format_material_checkpoint_question(state, {}, [], [])
        
        assert "APPROVE" in result
        assert "CHANGE_DATABASE" in result
        assert "CHANGE_MATERIAL" in result
        assert "NEED_HELP" in result

    def test_includes_mandatory_checkpoint_header(self):
        """Should include mandatory checkpoint header."""
        state = {"paper_id": "test_paper"}
        
        result = format_material_checkpoint_question(state, {}, [], [])
        
        assert "MANDATORY MATERIAL VALIDATION CHECKPOINT" in result
        assert "Stage 0" in result
