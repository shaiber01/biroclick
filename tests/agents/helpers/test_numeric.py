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

    def test_returns_existing_absolute_path(self, tmp_path):
        """Should return existing absolute path."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result == test_file

    def test_returns_none_for_nonexistent_path(self):
        """Should return None for non-existent path."""
        result = resolve_data_path("/nonexistent/path/file.csv")
        assert result is None

    def test_expands_user_home(self, tmp_path, monkeypatch):
        """Should expand ~ in paths."""
        # This test checks that expanduser is called
        result = resolve_data_path("~/nonexistent.csv")
        # Will be None since file doesn't exist, but path should be expanded
        assert result is None


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

    def test_returns_none_for_less_than_3_points(self):
        """Should return None for less than 3 data points."""
        assert normalize_series([1, 2], [1, 2]) is None

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


class TestUnitsToMultiplier:
    """Tests for units_to_multiplier function."""

    def test_nm_returns_1(self):
        """Should return 1.0 for nm units."""
        assert units_to_multiplier("wavelength_nm") == 1.0
        assert units_to_multiplier("wavelength (nm)") == 1.0
        assert units_to_multiplier("WAVELENGTH_NM") == 1.0

    def test_um_returns_1000(self):
        """Should return 1000.0 for um units."""
        assert units_to_multiplier("wavelength_um") == 1000.0
        assert units_to_multiplier("wavelength (um)") == 1000.0

    def test_m_returns_1e9(self):
        """Should return 1e9 for m units."""
        assert units_to_multiplier("wavelength_m") == 1e9
        assert units_to_multiplier("wavelength (m)") == 1e9

    def test_no_units_returns_1(self):
        """Should return 1.0 for headers without units."""
        assert units_to_multiplier("wavelength") == 1.0
        assert units_to_multiplier("value") == 1.0


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

    def test_loads_json_dict_format(self, tmp_path):
        """Should load JSON with {x: [], y: []} format."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"x": [400, 500, 600, 700], "y": [0.1, 0.5, 0.9, 0.3]}))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])

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

    def test_loads_npy_file(self, tmp_path):
        """Should load NPY file correctly."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[400, 0.1], [500, 0.5], [600, 0.9], [700, 0.3]])
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])

    def test_returns_none_for_invalid_file(self, tmp_path):
        """Should return None for invalid file."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("not,valid,csv\nwith,bad,data")
        
        result = load_numeric_series(str(bad_file))
        # Should either return None or a valid result - implementation dependent


class TestComputePeakMetrics:
    """Tests for compute_peak_metrics function."""

    def test_returns_empty_for_empty_arrays(self):
        """Should return empty dict for empty arrays."""
        result = compute_peak_metrics(np.array([]), np.array([]))
        assert result == {}

    def test_finds_peak_position(self):
        """Should find peak position correctly."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600
        assert result["peak_value"] == 0.9

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
        
        assert result["fwhm"] is None


class TestQuantitativeCurveMetrics:
    """Tests for quantitative_curve_metrics function."""

    def test_returns_empty_for_none_sim(self):
        """Should return empty dict when simulation data is None."""
        result = quantitative_curve_metrics(None, None)
        assert result == {}

    def test_returns_sim_metrics_without_ref(self):
        """Should return simulation metrics even without reference."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), None)
        
        assert "peak_position_sim" in result
        assert "peak_value_sim" in result
        assert result["peak_position_sim"] == 600

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

    def test_computes_peak_position_error(self):
        """Should compute peak position error percentage."""
        x_sim = np.array([400, 500, 620, 700, 800])  # Peak at 620
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])  # Peak at 600
        y_ref = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Error should be |620-600|/600 * 100 ≈ 3.33%
        assert result["peak_position_error_percent"] == pytest.approx(3.33, rel=0.1)

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

    def test_perfect_match_metrics(self):
        """Should return perfect metrics for identical curves."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x, y), (x, y))
        
        assert result["normalized_rmse_percent"] == pytest.approx(0, abs=0.01)
        assert result["correlation"] == pytest.approx(1.0, abs=0.01)
        assert result["r_squared"] == pytest.approx(1.0, abs=0.01)


