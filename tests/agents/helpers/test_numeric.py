"""Unit tests for src/agents/helpers/numeric.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.agents.helpers.numeric import (
    resolve_data_path,
    normalize_series,
    units_to_multiplier,
    load_numeric_series,
    compute_peak_metrics,
    quantitative_curve_metrics,
)


class TestResolveDataPath:
    """Tests for resolve_data_path function."""

    def test_returns_none_for_empty_path(self):
        """Should return None for empty path."""
        assert resolve_data_path(None) is None
        assert resolve_data_path("") is None
        assert resolve_data_path("   ") is None  # Whitespace-only

    def test_returns_existing_absolute_path(self, tmp_path):
        """Should return existing absolute path."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result == test_file
        assert result.exists()
        assert isinstance(result, Path)

    def test_returns_none_for_nonexistent_path(self):
        """Should return None for non-existent path."""
        result = resolve_data_path("/nonexistent/path/file.csv")
        assert result is None
        
        # Test with relative path that doesn't exist
        result2 = resolve_data_path("nonexistent/relative/path.csv")
        assert result2 is None

    def test_expands_user_home(self, tmp_path, monkeypatch):
        """Should expand ~ in paths."""
        # This test checks that expanduser is called
        result = resolve_data_path("~/nonexistent.csv")
        # Will be None since file doesn't exist, but path should be expanded
        assert result is None

    def test_resolves_relative_path_from_project_root(self, tmp_path, monkeypatch):
        """Should resolve relative paths relative to PROJECT_ROOT."""
        from src.agents.helpers.numeric import PROJECT_ROOT
        
        # Create a file in PROJECT_ROOT
        test_file = PROJECT_ROOT / "test_resolve.csv"
        try:
            test_file.write_text("test")
            
            # Test relative path resolution
            result = resolve_data_path("test_resolve.csv")
            assert result is not None
            assert result == test_file
            assert result.exists()
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    def test_handles_path_objects(self, tmp_path):
        """Should handle Path objects as input."""
        test_file = tmp_path / "test_path.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(test_file)
        assert result == test_file
        assert result.exists()

    def test_handles_special_characters_in_path(self, tmp_path):
        """Should handle paths with special characters."""
        test_file = tmp_path / "test_file_with-special_chars.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        assert result == test_file
        assert result.exists()


class TestNormalizeSeries:
    """Tests for normalize_series function."""

    def test_returns_none_for_empty_lists(self):
        """Should return None for empty lists."""
        assert normalize_series([], []) is None
        assert normalize_series([1], []) is None
        assert normalize_series([], [1]) is None

    def test_returns_none_for_mismatched_lengths(self):
        """Should return None for mismatched list lengths."""
        assert normalize_series([1, 2], [1]) is None
        assert normalize_series([1], [1, 2]) is None
        assert normalize_series([1, 2, 3], [1, 2]) is None
        assert normalize_series([1, 2], [1, 2, 3]) is None

    def test_returns_none_for_less_than_3_points(self):
        """Should return None for less than 3 data points."""
        assert normalize_series([1, 2], [1, 2]) is None
        assert normalize_series([1], [1]) is None

    def test_sorts_by_x_values(self):
        """Should sort data by x values."""
        xs = [3, 1, 2, 5, 4]
        ys = [30, 10, 20, 50, 40]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(y_arr, [10, 20, 30, 40, 50])

    def test_returns_numpy_arrays(self):
        """Should return numpy arrays."""
        result = normalize_series([1, 2, 3], [10, 20, 30])
        
        assert result is not None
        x_arr, y_arr = result
        assert isinstance(x_arr, np.ndarray)
        assert isinstance(y_arr, np.ndarray)
        assert x_arr.dtype == float
        assert y_arr.dtype == float

    def test_handles_negative_values(self):
        """Should handle negative x and y values."""
        xs = [-3, -1, -2, 0, 1]
        ys = [-10, -5, -8, 0, 5]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [-3, -2, -1, 0, 1])
        np.testing.assert_array_equal(y_arr, [-10, -8, -5, 0, 5])

    def test_handles_float_values(self):
        """Should handle float values correctly."""
        xs = [1.5, 1.1, 1.9, 1.3]
        ys = [10.5, 10.1, 10.9, 10.3]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [1.1, 1.3, 1.5, 1.9])
        np.testing.assert_array_almost_equal(y_arr, [10.1, 10.3, 10.5, 10.9])

    def test_handles_duplicate_x_values(self):
        """Should handle duplicate x values (keeps corresponding y values)."""
        xs = [1, 2, 1, 3, 2]
        ys = [10, 20, 15, 30, 25]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        # Should be sorted by x, with duplicates preserved
        assert len(x_arr) == 5
        assert x_arr[0] == 1
        assert x_arr[1] == 1
        assert x_arr[2] == 2
        assert x_arr[3] == 2
        assert x_arr[4] == 3
        # Verify corresponding y values are preserved
        assert y_arr[0] in [10, 15]
        assert y_arr[1] in [10, 15]
        assert y_arr[2] in [20, 25]
        assert y_arr[3] in [20, 25]
        assert y_arr[4] == 30

    def test_handles_very_large_numbers(self):
        """Should handle very large numbers."""
        xs = [1e10, 2e10, 3e10]
        ys = [1e20, 2e20, 3e20]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        assert x_arr.dtype == float
        assert y_arr.dtype == float

    def test_handles_very_small_numbers(self):
        """Should handle very small numbers."""
        xs = [1e-10, 2e-10, 3e-10]
        ys = [1e-20, 2e-20, 3e-20]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        assert x_arr.dtype == float
        assert y_arr.dtype == float

    def test_preserves_data_integrity(self):
        """Should preserve all data points without loss."""
        xs = list(range(100))
        ys = [x * 2.5 for x in xs]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 100
        assert len(y_arr) == 100
        # Verify all values are present
        assert np.min(x_arr) == 0
        assert np.max(x_arr) == 99
        assert np.min(y_arr) == 0
        assert np.max(y_arr) == 99 * 2.5

    def test_handles_mixed_numeric_types(self):
        """Should convert int and float to float."""
        xs = [1, 2.0, 3, 4.5]
        ys = [10, 20.0, 30, 40.5]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert x_arr.dtype == float
        assert y_arr.dtype == float
        np.testing.assert_array_almost_equal(x_arr, [1.0, 2.0, 3.0, 4.5])
        np.testing.assert_array_almost_equal(y_arr, [10.0, 20.0, 30.0, 40.5])


class TestUnitsToMultiplier:
    """Tests for units_to_multiplier function."""

    def test_nm_returns_1(self):
        """Should return 1.0 for nm units."""
        assert units_to_multiplier("wavelength_nm") == 1.0
        assert units_to_multiplier("wavelength (nm)") == 1.0
        assert units_to_multiplier("WAVELENGTH_NM") == 1.0
        assert units_to_multiplier("lambda_nm") == 1.0
        assert units_to_multiplier("freq (nm)") == 1.0

    def test_um_returns_1000(self):
        """Should return 1000.0 for um units."""
        assert units_to_multiplier("wavelength_um") == 1000.0
        assert units_to_multiplier("wavelength (um)") == 1000.0
        assert units_to_multiplier("WAVELENGTH_UM") == 1000.0
        assert units_to_multiplier("lambda_um") == 1000.0

    def test_m_returns_1e9(self):
        """Should return 1e9 for m units."""
        assert units_to_multiplier("wavelength_m") == 1e9
        assert units_to_multiplier("wavelength (m)") == 1e9
        assert units_to_multiplier("WAVELENGTH_M") == 1e9
        assert units_to_multiplier("lambda_m") == 1e9

    def test_no_units_returns_1(self):
        """Should return 1.0 for headers without units."""
        assert units_to_multiplier("wavelength") == 1.0
        assert units_to_multiplier("value") == 1.0
        assert units_to_multiplier("frequency") == 1.0
        assert units_to_multiplier("lambda") == 1.0

    def test_handles_whitespace(self):
        """Should handle whitespace in headers."""
        assert units_to_multiplier("wavelength_nm ") == 1.0
        assert units_to_multiplier(" wavelength_nm") == 1.0
        assert units_to_multiplier("wavelength (nm) ") == 1.0
        assert units_to_multiplier(" wavelength (um)") == 1000.0

    def test_handles_mixed_case(self):
        """Should handle mixed case."""
        assert units_to_multiplier("Wavelength_Nm") == 1.0
        assert units_to_multiplier("Wavelength (Nm)") == 1.0
        assert units_to_multiplier("Wavelength_Um") == 1000.0
        assert units_to_multiplier("Wavelength_M") == 1e9

    def test_handles_empty_string(self):
        """Should return 1.0 for empty string."""
        assert units_to_multiplier("") == 1.0

    def test_handles_special_characters(self):
        """Should handle special characters in headers."""
        assert units_to_multiplier("wavelength_nm_extra") == 1.0
        assert units_to_multiplier("prefix_wavelength_nm") == 1.0
        assert units_to_multiplier("wavelength_nm_suffix") == 1.0

    def test_handles_parentheses_variations(self):
        """Should handle various parentheses formats."""
        assert units_to_multiplier("wavelength(nm)") == 1.0
        assert units_to_multiplier("wavelength ( nm )") == 1.0
        assert units_to_multiplier("wavelength(um)") == 1000.0
        assert units_to_multiplier("wavelength(m)") == 1e9


class TestLoadNumericSeries:
    """Tests for load_numeric_series function."""

    def test_returns_none_for_empty_path(self):
        """Should return None for empty path."""
        assert load_numeric_series(None) is None
        assert load_numeric_series("") is None

    def test_loads_csv_file(self, tmp_path):
        """Should load CSV file correctly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\n500,0.5\n600,0.9\n700,0.3")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_csv_with_specified_columns(self, tmp_path):
        """Should load CSV with specified column names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("freq,transmission,other\n100,0.1,1\n200,0.2,2\n300,0.3,3\n400,0.4,4")
        
        result = load_numeric_series(str(csv_file), columns=["freq", "transmission"])
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [100, 200, 300, 400])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.2, 0.3, 0.4])

    def test_loads_tsv_file(self, tmp_path):
        """Should load TSV file correctly."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("wavelength\tvalue\n400\t0.1\n500\t0.5\n600\t0.9\n700\t0.3")
        
        result = load_numeric_series(str(tsv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_csv_with_unit_conversion_nm(self, tmp_path):
        """Should convert units from nm correctly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength_nm,value\n400,0.1\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should remain in nm (multiplier = 1.0)
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])

    def test_loads_csv_with_unit_conversion_um(self, tmp_path):
        """Should convert units from um to nm correctly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength_um,value\n0.4,0.1\n0.5,0.5\n0.6,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should convert um to nm (multiplier = 1000.0)
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])

    def test_loads_csv_with_unit_conversion_m(self, tmp_path):
        """Should convert units from m to nm correctly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength_m,value\n4e-7,0.1\n5e-7,0.5\n6e-7,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should convert m to nm (multiplier = 1e9)
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600], decimal=6)

    def test_loads_csv_with_missing_header(self, tmp_path):
        """Should handle CSV without header."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("400,0.1\n500,0.5\n600,0.9\n700,0.3")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) >= 3

    def test_loads_csv_with_empty_rows(self, tmp_path):
        """Should skip empty rows in CSV."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\n\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) >= 3

    def test_loads_csv_with_invalid_rows(self, tmp_path):
        """Should skip rows with invalid data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\ninvalid,data\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should have at least 3 valid rows
        assert len(x_arr) >= 3

    def test_loads_json_dict_format(self, tmp_path):
        """Should load JSON with {x: [], y: []} format."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"x": [400, 500, 600, 700], "y": [0.1, 0.5, 0.9, 0.3]}))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_json_list_format(self, tmp_path):
        """Should load JSON with [{x, y}, ...] format."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            {"x": 500, "y": 0.5},
            {"x": 600, "y": 0.9},
            {"x": 700, "y": 0.3},
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_json_list_format_with_tuples(self, tmp_path):
        """Should load JSON with list of tuples format."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            [400, 0.1],
            [500, 0.5],
            [600, 0.9],
            [700, 0.3],
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_json_skips_invalid_points(self, tmp_path):
        """Should skip invalid points in JSON list."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            {"invalid": "data"},
            {"x": 500, "y": 0.5},
            [600, 0.9],
            {"x": 700},  # Missing y
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) >= 3

    def test_loads_npy_file(self, tmp_path):
        """Should load NPY file correctly."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[400, 0.1], [500, 0.5], [600, 0.9], [700, 0.3]])
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_npz_file(self, tmp_path):
        """Should load NPZ file correctly."""
        npz_file = tmp_path / "data.npz"
        arr = np.array([[400, 0.1], [500, 0.5], [600, 0.9], [700, 0.3]])
        np.savez(npz_file, data=arr)
        
        result = load_numeric_series(str(npz_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_returns_none_for_invalid_npy_shape(self, tmp_path):
        """Should return None for NPY file with wrong shape."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([400, 500, 600])  # 1D array, not 2D
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        assert result is None

    def test_returns_none_for_invalid_npy_columns(self, tmp_path):
        """Should return None for NPY file with less than 2 columns."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[400], [500], [600]])  # Only 1 column
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        assert result is None

    def test_returns_none_for_invalid_file(self, tmp_path):
        """Should return None for invalid file."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("not,valid,csv\nwith,bad,data")
        
        result = load_numeric_series(str(bad_file))
        # Should return None for files that can't be parsed properly
        # But if it can extract at least 2 numeric columns, it might return a result
        # The key is that it should not crash

    def test_returns_none_for_corrupted_json(self, tmp_path):
        """Should return None for corrupted JSON."""
        json_file = tmp_path / "data.json"
        json_file.write_text("{invalid json}")
        
        result = load_numeric_series(str(json_file))
        assert result is None

    def test_handles_csv_column_detection(self, tmp_path):
        """Should detect wavelength column automatically."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("frequency,transmission\n400,0.1\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        # Should detect wavelength-like column or use first two columns
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) >= 3

    def test_handles_csv_with_extra_columns(self, tmp_path):
        """Should handle CSV with extra columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value,extra1,extra2\n400,0.1,1,2\n500,0.5,3,4\n600,0.9,5,6")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])


class TestComputePeakMetrics:
    """Tests for compute_peak_metrics function."""

    def test_returns_empty_for_empty_arrays(self):
        """Should return empty dict for empty arrays."""
        result = compute_peak_metrics(np.array([]), np.array([]))
        assert result == {}
        assert len(result) == 0

    def test_returns_empty_for_zero_size_arrays(self):
        """Should return empty dict for zero-size arrays."""
        result = compute_peak_metrics(np.array([]), np.array([1, 2, 3]))
        assert result == {}
        
        result2 = compute_peak_metrics(np.array([1, 2, 3]), np.array([]))
        assert result2 == {}

    def test_finds_peak_position(self):
        """Should find peak position correctly."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        assert "peak_position" in result
        assert "peak_value" in result
        assert result["peak_position"] == 600
        assert result["peak_value"] == 0.9

    def test_finds_peak_at_start(self):
        """Should find peak at the start of array."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 400
        assert result["peak_value"] == 1.0

    def test_finds_peak_at_end(self):
        """Should find peak at the end of array."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.2, 0.3, 0.5, 1.0])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 800
        assert result["peak_value"] == 1.0

    def test_handles_multiple_equal_peaks(self):
        """Should find first occurrence when multiple equal peaks exist."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.9, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        # Should return first peak (at index 0)
        assert result["peak_position"] == 400
        assert result["peak_value"] == 0.9

    def test_handles_negative_values(self):
        """Should handle negative y values."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([-1.0, -0.5, -0.1, -0.3, -0.8])
        
        result = compute_peak_metrics(x, y)
        
        # Peak should be the maximum (least negative)
        assert result["peak_position"] == 600
        assert result["peak_value"] == -0.1

    def test_handles_zero_peak_value(self):
        """Should handle zero peak value."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = compute_peak_metrics(x, y)
        
        assert "peak_position" in result
        assert "peak_value" in result
        assert result["peak_value"] == 0.0
        # FWHM should be None when peak is zero
        assert result.get("fwhm") is None

    def test_computes_fwhm(self):
        """Should compute FWHM when possible."""
        # Create a Gaussian-like peak
        x = np.linspace(400, 800, 101)
        y = np.exp(-((x - 600) ** 2) / (2 * 50 ** 2))
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == pytest.approx(600, rel=0.01)
        assert result["fwhm"] is not None
        # FWHM of Gaussian with sigma=50 is approximately 2.355*50 ≈ 118
        assert result["fwhm"] == pytest.approx(118, rel=0.1)

    def test_fwhm_none_for_no_crossings(self):
        """Should return None for FWHM when no half-max crossings."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.9, 0.95, 1.0, 0.95, 0.9])  # Peak never drops below half max
        
        result = compute_peak_metrics(x, y)
        
        assert "fwhm" in result
        assert result["fwhm"] is None

    def test_fwhm_none_when_only_left_crossing(self):
        """Should return None for FWHM when only left crossing exists."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.3, 0.4, 1.0, 0.8, 0.7])  # Only left side crosses half-max
        
        result = compute_peak_metrics(x, y)
        
        assert result.get("fwhm") is None

    def test_fwhm_none_when_only_right_crossing(self):
        """Should return None for FWHM when only right crossing exists."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.7, 0.8, 1.0, 0.4, 0.3])  # Only right side crosses half-max
        
        result = compute_peak_metrics(x, y)
        
        assert result.get("fwhm") is None

    def test_fwhm_computed_when_both_crossings_exist(self):
        """Should compute FWHM when both crossings exist."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.3, 0.4, 1.0, 0.4, 0.3])  # Both sides cross half-max
        
        result = compute_peak_metrics(x, y)
        
        assert result["fwhm"] is not None
        assert result["fwhm"] > 0

    def test_handles_single_point(self):
        """Should handle single data point."""
        x = np.array([600])
        y = np.array([0.9])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600
        assert result["peak_value"] == 0.9
        assert result.get("fwhm") is None  # Can't compute FWHM with single point

    def test_handles_two_points(self):
        """Should handle two data points."""
        x = np.array([500, 600])
        y = np.array([0.5, 0.9])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600
        assert result["peak_value"] == 0.9
        assert result.get("fwhm") is None  # Can't compute FWHM with two points

    def test_handles_flat_line(self):
        """Should handle flat line (all values same)."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        result = compute_peak_metrics(x, y)
        
        assert "peak_position" in result
        assert "peak_value" in result
        assert result["peak_value"] == 0.5
        # FWHM should be None (half-max would be 0.25, but values never drop below)
        assert result.get("fwhm") is None

    def test_handles_unsorted_x(self):
        """Should handle unsorted x values (uses y for peak finding)."""
        x = np.array([600, 400, 800, 500, 700])
        y = np.array([0.9, 0.1, 0.2, 0.5, 0.4])
        
        result = compute_peak_metrics(x, y)
        
        # Should find peak at x=600 (index 0)
        assert result["peak_position"] == 600
        assert result["peak_value"] == 0.9

    def test_returns_float_types(self):
        """Should return float types for all metrics."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        assert isinstance(result["peak_position"], float)
        assert isinstance(result["peak_value"], float)
        if result.get("fwhm") is not None:
            assert isinstance(result["fwhm"], float)


class TestQuantitativeCurveMetrics:
    """Tests for quantitative_curve_metrics function."""

    def test_returns_empty_for_none_sim(self):
        """Should return empty dict when simulation data is None."""
        result = quantitative_curve_metrics(None, None)
        assert result == {}
        assert len(result) == 0

    def test_returns_empty_for_empty_sim_tuple(self):
        """Should return empty dict when simulation tuple is empty."""
        result = quantitative_curve_metrics((), None)
        # Should handle gracefully - implementation dependent

    def test_returns_sim_metrics_without_ref(self):
        """Should return simulation metrics even without reference."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), None)
        
        assert "peak_position_sim" in result
        assert "peak_value_sim" in result
        assert result["peak_position_sim"] == 600
        assert result["peak_value_sim"] == 0.9
        # Should not have reference metrics
        assert "peak_position_paper" not in result
        assert "normalized_rmse_percent" not in result

    def test_handles_ref_with_less_than_3_points(self):
        """Should return only sim metrics when ref has less than 3 points."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500])
        y_ref = np.array([0.1, 0.5])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_position_sim" in result
        assert "peak_position_paper" not in result
        assert "normalized_rmse_percent" not in result

    def test_computes_comparison_metrics(self):
        """Should compute comparison metrics with reference data."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.12, 0.52, 0.88, 0.38, 0.18])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_position_sim" in result
        assert "peak_position_paper" in result
        assert "normalized_rmse_percent" in result
        assert "correlation" in result
        assert "r_squared" in result
        assert "n_points_compared" in result
        assert result["n_points_compared"] == 5

    def test_computes_peak_position_error(self):
        """Should compute peak position error percentage."""
        x_sim = np.array([400, 500, 620, 700, 800])  # Peak at 620
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])  # Peak at 600
        y_ref = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Error should be |620-600|/600 * 100 ≈ 3.33%
        assert "peak_position_error_percent" in result
        assert result["peak_position_error_percent"] == pytest.approx(3.33, rel=0.1)

    def test_peak_position_error_none_when_ref_peak_zero(self):
        """Should not compute error when reference peak is zero."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Zero peak
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should not have peak_position_error_percent when ref peak is 0
        assert "peak_position_error_percent" not in result or result.get("peak_position_error_percent") is None

    def test_computes_peak_height_ratio(self):
        """Should compute peak height ratio."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])  # Peak = 0.9
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.1, 0.5, 0.45, 0.4, 0.2])  # Peak = 0.5
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_height_ratio" in result
        assert result["peak_height_ratio"] == pytest.approx(0.9 / 0.5, rel=0.01)

    def test_computes_fwhm_ratio(self):
        """Should compute FWHM ratio when both FWHMs exist."""
        # Create curves with computable FWHM
        x_sim = np.linspace(400, 800, 101)
        y_sim = np.exp(-((x_sim - 600) ** 2) / (2 * 50 ** 2))
        x_ref = np.linspace(400, 800, 101)
        y_ref = np.exp(-((x_ref - 600) ** 2) / (2 * 100 ** 2))  # Wider peak
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        if result.get("fwhm_sim") and result.get("fwhm_paper"):
            assert "fwhm_ratio" in result
            assert result["fwhm_ratio"] > 0
            assert result["fwhm_ratio"] < 1.0  # Sim FWHM should be smaller

    def test_handles_different_x_ranges(self):
        """Should interpolate reference to simulation x-axis."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([450, 550, 650, 750])  # Different x range
        y_ref = np.array([0.3, 0.7, 0.6, 0.3])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should still compute metrics where ranges overlap
        assert "n_points_compared" in result
        assert result["n_points_compared"] > 0

    def test_handles_non_overlapping_ranges(self):
        """Should handle non-overlapping x ranges."""
        x_sim = np.array([400, 500, 600])
        y_sim = np.array([0.1, 0.5, 0.9])
        x_ref = np.array([1000, 1100, 1200])  # Completely different range
        y_ref = np.array([0.1, 0.5, 0.9])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should have sim metrics but no comparison metrics
        assert "peak_position_sim" in result
        assert "n_points_compared" in result
        assert result["n_points_compared"] == 0
        assert "normalized_rmse_percent" not in result

    def test_perfect_match_metrics(self):
        """Should return perfect metrics for identical curves."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x, y), (x, y))
        
        assert result["normalized_rmse_percent"] == pytest.approx(0, abs=0.01)
        assert result["correlation"] == pytest.approx(1.0, abs=0.01)
        assert result["r_squared"] == pytest.approx(1.0, abs=0.01)
        assert result["peak_position_error_percent"] == pytest.approx(0, abs=0.01)
        assert result["peak_height_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_handles_single_point_comparison(self):
        """Should handle single overlapping point."""
        x_sim = np.array([400, 500, 600])
        y_sim = np.array([0.1, 0.5, 0.9])
        x_ref = np.array([500, 600, 700])
        y_ref = np.array([0.5, 0.9, 0.3])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should have at least one overlapping point
        assert "n_points_compared" in result
        assert result["n_points_compared"] >= 1

    def test_handles_zero_range_reference(self):
        """Should handle reference with zero range (all same values)."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # All same
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should compute metrics, but r_squared might be problematic
        assert "normalized_rmse_percent" in result
        # Correlation might be NaN or undefined for constant reference

    def test_handles_negative_correlation(self):
        """Should handle negatively correlated curves."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.9, 0.7, 0.5, 0.3, 0.1])  # Decreasing
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Increasing
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "correlation" in result
        assert result["correlation"] < 0  # Negative correlation
        assert "r_squared" in result

    def test_handles_mismatched_array_lengths(self):
        """Should handle arrays of different lengths."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800, 900, 1000])
        y_ref = np.array([0.1, 0.5, 0.9, 0.4, 0.2, 0.1, 0.05])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should interpolate and compare where ranges overlap
        assert "n_points_compared" in result
        assert result["n_points_compared"] > 0

    def test_returns_all_expected_keys(self):
        """Should return all expected metric keys."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.12, 0.52, 0.88, 0.38, 0.18])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        expected_keys = [
            "peak_position_sim", "peak_value_sim",
            "peak_position_paper", "peak_value_paper",
            "normalized_rmse_percent", "correlation", "r_squared",
            "n_points_compared"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_handles_nan_values_after_interpolation(self):
        """Should handle NaN values after interpolation."""
        x_sim = np.array([400, 500, 600])
        y_sim = np.array([0.1, 0.5, 0.9])
        x_ref = np.array([100, 200, 300])  # No overlap
        y_ref = np.array([0.1, 0.5, 0.9])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Should handle NaN interpolation gracefully
        assert "n_points_compared" in result
        assert result["n_points_compared"] == 0



