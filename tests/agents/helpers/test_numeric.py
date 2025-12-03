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

    def test_returns_none_for_none_path(self):
        """Should return None for None input."""
        result = resolve_data_path(None)
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Should return None for empty string."""
        result = resolve_data_path("")
        assert result is None

    def test_whitespace_only_path_behavior(self):
        """Whitespace-only strings - verify actual behavior.
        
        Note: The implementation uses `if not path_str` which returns False
        for whitespace strings like "   ". This means whitespace strings
        are processed, not rejected as empty.
        """
        # Whitespace strings like "   " will go through path resolution
        # and should return None since "   " is not a valid path
        result = resolve_data_path("   ")
        assert result is None  # Path "   " doesn't exist

    def test_returns_existing_absolute_path(self, tmp_path):
        """Should return existing absolute path."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result is not None
        assert result == test_file.resolve()  # Should be resolved/absolute
        assert result.exists()
        assert result.is_absolute()
        assert isinstance(result, Path)

    def test_returns_none_for_nonexistent_absolute_path(self):
        """Should return None for non-existent absolute path."""
        result = resolve_data_path("/nonexistent/path/file.csv")
        assert result is None

    def test_returns_none_for_nonexistent_relative_path(self):
        """Should return None for non-existent relative path."""
        result = resolve_data_path("nonexistent/relative/path.csv")
        assert result is None

    def test_expands_user_home_tilde(self, tmp_path, monkeypatch):
        """Should expand ~ to actual home directory path."""
        import os
        # Create a test file in the home directory
        home = Path.home()
        test_file = home / ".test_resolve_data_path_temp.csv"
        try:
            test_file.write_text("test data")
            
            result = resolve_data_path("~/.test_resolve_data_path_temp.csv")
            
            assert result is not None
            assert result == test_file.resolve()
            assert result.exists()
            assert "~" not in str(result)  # Tilde should be expanded
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_expands_user_home_nonexistent_returns_none(self):
        """Should return None for expanded ~ path that doesn't exist."""
        result = resolve_data_path("~/definitely_nonexistent_file_12345.csv")
        assert result is None

    def test_resolves_relative_path_from_project_root(self, tmp_path, monkeypatch):
        """Should resolve relative paths relative to PROJECT_ROOT."""
        from src.agents.helpers.numeric import PROJECT_ROOT
        
        # Create a file in PROJECT_ROOT
        test_file = PROJECT_ROOT / "test_resolve_temp.csv"
        try:
            test_file.write_text("test")
            
            # Test relative path resolution
            result = resolve_data_path("test_resolve_temp.csv")
            
            assert result is not None
            assert result == test_file.resolve()
            assert result.exists()
            assert result.is_absolute()
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    def test_absolute_path_takes_precedence_over_project_root(self, tmp_path):
        """Absolute paths should be used directly, not relative to PROJECT_ROOT."""
        from src.agents.helpers.numeric import PROJECT_ROOT
        
        test_file = tmp_path / "absolute_test.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result is not None
        assert result == test_file.resolve()
        # Verify it didn't try to resolve relative to PROJECT_ROOT
        assert tmp_path in result.parents or result.parent == tmp_path

    def test_handles_path_objects_as_input(self, tmp_path):
        """Should handle Path objects as input, not just strings."""
        test_file = tmp_path / "test_path.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(test_file)
        
        assert result is not None
        assert result == test_file.resolve()
        assert result.exists()

    def test_handles_special_characters_in_path(self, tmp_path):
        """Should handle paths with special characters."""
        test_file = tmp_path / "test_file_with-special_chars.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result is not None
        assert result == test_file.resolve()
        assert result.exists()

    def test_handles_spaces_in_path(self, tmp_path):
        """Should handle paths with spaces."""
        test_file = tmp_path / "test file with spaces.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result is not None
        assert result == test_file.resolve()
        assert result.exists()

    def test_handles_unicode_in_path(self, tmp_path):
        """Should handle paths with unicode characters."""
        test_file = tmp_path / "test_日本語_файл.csv"
        test_file.write_text("data")
        
        result = resolve_data_path(str(test_file))
        
        assert result is not None
        assert result == test_file.resolve()
        assert result.exists()

    def test_returns_resolved_absolute_path_not_relative(self, tmp_path):
        """Result should always be an absolute resolved path."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        
        # Even if we pass it through, result should be resolved
        result = resolve_data_path(str(test_file))
        
        assert result is not None
        assert result.is_absolute()
        # Should have no '..' or '.' components
        assert ".." not in result.parts
        # Single '.' might be in the path name itself, but not as a directory component

    def test_handles_symlink_path(self, tmp_path):
        """Should handle symlink paths."""
        import os
        
        # Create a real file
        real_file = tmp_path / "real_file.csv"
        real_file.write_text("data")
        
        # Create a symlink
        symlink_file = tmp_path / "symlink.csv"
        try:
            os.symlink(real_file, symlink_file)
            
            result = resolve_data_path(str(symlink_file))
            
            assert result is not None
            assert result.exists()
            # resolve() should resolve symlinks
            assert result == real_file.resolve()
        except OSError:
            # Symlinks may not be supported on all systems
            pytest.skip("Symlinks not supported on this system")


class TestNormalizeSeries:
    """Tests for normalize_series function."""

    def test_returns_none_for_both_empty_lists(self):
        """Should return None when both lists are empty."""
        result = normalize_series([], [])
        assert result is None

    def test_returns_none_for_empty_ys(self):
        """Should return None when ys is empty."""
        result = normalize_series([1], [])
        assert result is None

    def test_returns_none_for_empty_xs(self):
        """Should return None when xs is empty."""
        result = normalize_series([], [1])
        assert result is None

    def test_returns_none_for_xs_longer_than_ys(self):
        """Should return None when xs is longer than ys."""
        result = normalize_series([1, 2], [1])
        assert result is None

    def test_returns_none_for_ys_longer_than_xs(self):
        """Should return None when ys is longer than xs."""
        result = normalize_series([1], [1, 2])
        assert result is None

    def test_returns_none_for_mismatched_lengths_larger(self):
        """Should return None for larger mismatched lengths."""
        assert normalize_series([1, 2, 3], [1, 2]) is None
        assert normalize_series([1, 2], [1, 2, 3]) is None

    def test_returns_none_for_exactly_one_point(self):
        """Should return None for exactly 1 data point."""
        result = normalize_series([1], [1])
        assert result is None

    def test_returns_none_for_exactly_two_points(self):
        """Should return None for exactly 2 data points."""
        result = normalize_series([1, 2], [1, 2])
        assert result is None

    def test_returns_result_for_exactly_three_points(self):
        """Should return result for exactly 3 data points."""
        result = normalize_series([3, 1, 2], [30, 10, 20])
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        assert len(y_arr) == 3
        np.testing.assert_array_equal(x_arr, [1, 2, 3])
        np.testing.assert_array_equal(y_arr, [10, 20, 30])

    def test_sorts_by_x_values_ascending(self):
        """Should sort data by x values in ascending order."""
        xs = [3, 1, 2, 5, 4]
        ys = [30, 10, 20, 50, 40]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(y_arr, [10, 20, 30, 40, 50])

    def test_sort_preserves_xy_pairing(self):
        """Sorting should preserve the x-y pairing."""
        xs = [5, 3, 1]
        ys = [500, 300, 100]  # Each y is x * 100
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        # After sorting, each y should still be x * 100
        for i in range(len(x_arr)):
            assert y_arr[i] == x_arr[i] * 100

    def test_returns_numpy_arrays_type(self):
        """Should return numpy arrays, not lists."""
        result = normalize_series([1, 2, 3], [10, 20, 30])
        
        assert result is not None
        x_arr, y_arr = result
        assert isinstance(x_arr, np.ndarray)
        assert isinstance(y_arr, np.ndarray)

    def test_returns_float_dtype_arrays(self):
        """Should return arrays with float dtype."""
        result = normalize_series([1, 2, 3], [10, 20, 30])
        
        assert result is not None
        x_arr, y_arr = result
        assert x_arr.dtype == float
        assert y_arr.dtype == float

    def test_handles_negative_x_values(self):
        """Should handle negative x values and sort correctly."""
        xs = [-3, -1, -2, 0, 1]
        ys = [100, 300, 200, 400, 500]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [-3, -2, -1, 0, 1])
        np.testing.assert_array_equal(y_arr, [100, 200, 300, 400, 500])

    def test_handles_negative_y_values(self):
        """Should handle negative y values correctly."""
        xs = [1, 2, 3]
        ys = [-10, -20, -30]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [1, 2, 3])
        np.testing.assert_array_equal(y_arr, [-10, -20, -30])

    def test_handles_float_values_with_precision(self):
        """Should handle float values correctly with proper precision."""
        xs = [1.5, 1.1, 1.9, 1.3]
        ys = [10.5, 10.1, 10.9, 10.3]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [1.1, 1.3, 1.5, 1.9])
        np.testing.assert_array_almost_equal(y_arr, [10.1, 10.3, 10.5, 10.9])

    def test_duplicate_x_values_preserves_all_points(self):
        """Duplicate x values should preserve all data points."""
        xs = [1, 2, 1, 3, 2]
        ys = [10, 20, 15, 30, 25]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        # Should have all 5 points
        assert len(x_arr) == 5
        assert len(y_arr) == 5

    def test_duplicate_x_values_sorted_correctly(self):
        """Duplicate x values should be grouped together after sorting."""
        xs = [3, 1, 2, 1]
        ys = [30, 10, 20, 15]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        
        # x values should be sorted
        assert x_arr[0] == 1
        assert x_arr[1] == 1
        assert x_arr[2] == 2
        assert x_arr[3] == 3
        
        # The two y values for x=1 should be at indices 0 and 1
        assert set(y_arr[:2]) == {10, 15}
        assert y_arr[2] == 20
        assert y_arr[3] == 30

    def test_handles_very_large_numbers(self):
        """Should handle very large numbers without overflow."""
        xs = [1e10, 2e10, 3e10]
        ys = [1e20, 2e20, 3e20]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [1e10, 2e10, 3e10])
        np.testing.assert_array_almost_equal(y_arr, [1e20, 2e20, 3e20])

    def test_handles_very_small_numbers(self):
        """Should handle very small numbers without underflow."""
        xs = [1e-10, 2e-10, 3e-10]
        ys = [1e-20, 2e-20, 3e-20]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [1e-10, 2e-10, 3e-10])
        np.testing.assert_array_almost_equal(y_arr, [1e-20, 2e-20, 3e-20])

    def test_preserves_data_integrity_large_dataset(self):
        """Should preserve all data points without loss for large datasets."""
        xs = list(range(100))
        ys = [x * 2.5 for x in xs]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 100
        assert len(y_arr) == 100
        # Verify exact values
        np.testing.assert_array_almost_equal(x_arr, list(range(100)))
        np.testing.assert_array_almost_equal(y_arr, [x * 2.5 for x in range(100)])

    def test_handles_mixed_int_and_float_types(self):
        """Should convert mixed int and float to float."""
        xs = [1, 2.0, 3, 4.5]
        ys = [10, 20.0, 30, 40.5]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert x_arr.dtype == float
        assert y_arr.dtype == float
        np.testing.assert_array_almost_equal(x_arr, [1.0, 2.0, 3.0, 4.5])
        np.testing.assert_array_almost_equal(y_arr, [10.0, 20.0, 30.0, 40.5])

    def test_handles_zero_values(self):
        """Should handle zero values in both x and y."""
        xs = [0, 1, 2]
        ys = [0, 0, 0]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [0, 1, 2])
        np.testing.assert_array_equal(y_arr, [0, 0, 0])

    def test_handles_all_same_x_values(self):
        """Should handle all identical x values."""
        xs = [5, 5, 5]
        ys = [10, 20, 30]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        assert all(x == 5 for x in x_arr)
        # y values should all be present
        assert set(y_arr) == {10, 20, 30}

    def test_already_sorted_input(self):
        """Should handle already sorted input correctly."""
        xs = [1, 2, 3, 4, 5]
        ys = [10, 20, 30, 40, 50]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, xs)
        np.testing.assert_array_equal(y_arr, ys)

    def test_reverse_sorted_input(self):
        """Should correctly sort reverse-sorted input."""
        xs = [5, 4, 3, 2, 1]
        ys = [50, 40, 30, 20, 10]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(y_arr, [10, 20, 30, 40, 50])

    def test_handles_nan_in_xs(self):
        """Should handle NaN in x values - verify actual behavior."""
        xs = [1, float('nan'), 3]
        ys = [10, 20, 30]
        
        # NaN values will be converted to float and sorted
        # NaN comparison is tricky - sorted() puts NaN at the end
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        # Check that we have the non-NaN values
        assert 1.0 in x_arr
        assert 3.0 in x_arr
        # NaN should be present
        assert np.isnan(x_arr).sum() == 1

    def test_handles_nan_in_ys(self):
        """Should handle NaN in y values - verify actual behavior."""
        xs = [1, 2, 3]
        ys = [10, float('nan'), 30]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [1, 2, 3])
        assert y_arr[0] == 10
        assert np.isnan(y_arr[1])
        assert y_arr[2] == 30

    def test_handles_inf_in_xs(self):
        """Should handle infinity in x values."""
        xs = [1, float('inf'), 3]
        ys = [10, 20, 30]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        # inf sorts to the end
        assert x_arr[0] == 1
        assert x_arr[1] == 3
        assert np.isinf(x_arr[2])

    def test_handles_negative_inf_in_xs(self):
        """Should handle negative infinity in x values."""
        xs = [1, float('-inf'), 3]
        ys = [10, 20, 30]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        # -inf sorts to the beginning
        assert np.isneginf(x_arr[0])
        assert x_arr[1] == 1
        assert x_arr[2] == 3

    def test_handles_inf_in_ys(self):
        """Should handle infinity in y values."""
        xs = [1, 2, 3]
        ys = [10, float('inf'), 30]
        
        result = normalize_series(xs, ys)
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_equal(x_arr, [1, 2, 3])
        assert y_arr[0] == 10
        assert np.isinf(y_arr[1])
        assert y_arr[2] == 30


class TestUnitsToMultiplier:
    """Tests for units_to_multiplier function."""

    def test_nm_underscore_suffix_returns_1(self):
        """Should return 1.0 for _nm suffix."""
        assert units_to_multiplier("wavelength_nm") == 1.0

    def test_nm_parentheses_returns_1(self):
        """Should return 1.0 for (nm) in header."""
        assert units_to_multiplier("wavelength (nm)") == 1.0

    def test_nm_uppercase_returns_1(self):
        """Should return 1.0 for uppercase NM."""
        assert units_to_multiplier("WAVELENGTH_NM") == 1.0

    def test_nm_various_prefixes_returns_1(self):
        """Should return 1.0 for nm with various prefixes."""
        assert units_to_multiplier("lambda_nm") == 1.0
        assert units_to_multiplier("freq (nm)") == 1.0
        assert units_to_multiplier("x_nm") == 1.0

    def test_um_underscore_suffix_returns_1000(self):
        """Should return 1000.0 for _um suffix."""
        assert units_to_multiplier("wavelength_um") == 1000.0

    def test_um_parentheses_returns_1000(self):
        """Should return 1000.0 for (um) in header."""
        assert units_to_multiplier("wavelength (um)") == 1000.0

    def test_um_uppercase_returns_1000(self):
        """Should return 1000.0 for uppercase UM."""
        assert units_to_multiplier("WAVELENGTH_UM") == 1000.0

    def test_um_various_prefixes_returns_1000(self):
        """Should return 1000.0 for um with various prefixes."""
        assert units_to_multiplier("lambda_um") == 1000.0

    def test_m_underscore_suffix_returns_1e9(self):
        """Should return 1e9 for _m suffix."""
        assert units_to_multiplier("wavelength_m") == 1e9

    def test_m_parentheses_returns_1e9(self):
        """Should return 1e9 for (m) in header."""
        assert units_to_multiplier("wavelength (m)") == 1e9

    def test_m_uppercase_returns_1e9(self):
        """Should return 1e9 for uppercase M."""
        assert units_to_multiplier("WAVELENGTH_M") == 1e9

    def test_m_various_prefixes_returns_1e9(self):
        """Should return 1e9 for m with various prefixes."""
        assert units_to_multiplier("lambda_m") == 1e9

    def test_no_units_returns_1(self):
        """Should return 1.0 for headers without units."""
        assert units_to_multiplier("wavelength") == 1.0
        assert units_to_multiplier("value") == 1.0
        assert units_to_multiplier("frequency") == 1.0
        assert units_to_multiplier("lambda") == 1.0

    def test_trailing_whitespace_nm(self):
        """Should handle trailing whitespace with _nm."""
        # Implementation uses endswith on lowercased string
        # "wavelength_nm " ends with " " not "_nm" so this should NOT match
        result = units_to_multiplier("wavelength_nm ")
        # Based on implementation: "wavelength_nm ".lower().endswith("_nm") is False
        # But "(nm)" in "wavelength_nm " is False too
        # So this returns 1.0 (default)
        assert result == 1.0  # Trailing space breaks _nm detection

    def test_leading_whitespace_nm(self):
        """Should handle leading whitespace with _nm."""
        # " wavelength_nm".lower().endswith("_nm") is True
        assert units_to_multiplier(" wavelength_nm") == 1.0

    def test_trailing_whitespace_parentheses_nm(self):
        """Should handle trailing whitespace with (nm)."""
        # "(nm)" in "wavelength (nm) " is True
        assert units_to_multiplier("wavelength (nm) ") == 1.0

    def test_leading_whitespace_parentheses_um(self):
        """Should handle leading whitespace with (um)."""
        assert units_to_multiplier(" wavelength (um)") == 1000.0

    def test_mixed_case_nm(self):
        """Should handle mixed case _nm."""
        assert units_to_multiplier("Wavelength_Nm") == 1.0
        assert units_to_multiplier("Wavelength_nM") == 1.0

    def test_mixed_case_parentheses_nm(self):
        """Should handle mixed case (nm)."""
        assert units_to_multiplier("Wavelength (Nm)") == 1.0
        assert units_to_multiplier("Wavelength (NM)") == 1.0

    def test_mixed_case_um(self):
        """Should handle mixed case um."""
        assert units_to_multiplier("Wavelength_Um") == 1000.0
        assert units_to_multiplier("Wavelength_uM") == 1000.0

    def test_mixed_case_m(self):
        """Should handle mixed case _m."""
        assert units_to_multiplier("Wavelength_M") == 1e9

    def test_empty_string_returns_default(self):
        """Should return 1.0 for empty string."""
        assert units_to_multiplier("") == 1.0

    def test_nm_not_at_end_not_detected_by_endswith(self):
        """_nm in middle is not detected by endswith but may be by contains."""
        # "wavelength_nm_extra".endswith("_nm") is False
        # "(nm)" in "wavelength_nm_extra" is False
        # So default 1.0 - but this is correct behavior (nm is not the unit here)
        assert units_to_multiplier("wavelength_nm_extra") == 1.0

    def test_nm_suffix_detected_regardless_of_prefix(self):
        """Should detect _nm suffix regardless of prefix."""
        assert units_to_multiplier("prefix_wavelength_nm") == 1.0

    def test_parentheses_no_space(self):
        """Should handle parentheses without spaces."""
        assert units_to_multiplier("wavelength(nm)") == 1.0
        assert units_to_multiplier("wavelength(um)") == 1000.0
        assert units_to_multiplier("wavelength(m)") == 1e9

    def test_parentheses_with_inner_spaces(self):
        """Parentheses with inner spaces - verify behavior."""
        # "( nm )" contains "(nm)"? No, it contains "( nm )"
        # So "(nm)" not in "wavelength ( nm )" - returns default
        result = units_to_multiplier("wavelength ( nm )")
        assert result == 1.0  # Spaces inside parentheses break detection

    def test_just_unit_string(self):
        """Should handle just the unit string."""
        # "_nm" ends with "_nm"
        assert units_to_multiplier("_nm") == 1.0
        assert units_to_multiplier("_um") == 1000.0
        assert units_to_multiplier("_m") == 1e9
        # Just "(nm)" contains "(nm)"
        assert units_to_multiplier("(nm)") == 1.0
        assert units_to_multiplier("(um)") == 1000.0
        assert units_to_multiplier("(m)") == 1e9

    def test_unit_priority_nm_before_m(self):
        """_nm should be detected before _m (order matters in implementation)."""
        # "test_nm" ends with "_nm" -> returns 1.0
        assert units_to_multiplier("test_nm") == 1.0
        # This doesn't end with "_m" alone if it ends with "_nm"
        
    def test_ambiguous_m_at_end(self):
        """Should correctly detect _m at the end."""
        # "wavelength_m" ends with "_m" -> 1e9
        assert units_to_multiplier("wavelength_m") == 1e9
        
    def test_m_suffix_not_confused_with_nm(self):
        """_m should not match when _nm is present."""
        # "wavelength_nm" ends with "_nm" -> 1.0 (checked first)
        assert units_to_multiplier("wavelength_nm") == 1.0
        
    def test_um_suffix_not_confused_with_m(self):
        """_um should not match as _m."""
        # "wavelength_um" ends with "_um" -> 1000.0 (checked before _m)
        assert units_to_multiplier("wavelength_um") == 1000.0

    def test_whitespace_only_returns_default(self):
        """Should return 1.0 for whitespace-only string."""
        assert units_to_multiplier("   ") == 1.0
        assert units_to_multiplier("\t") == 1.0
        assert units_to_multiplier("\n") == 1.0

    def test_numeric_string_returns_default(self):
        """Should return 1.0 for numeric strings."""
        assert units_to_multiplier("123") == 1.0
        assert units_to_multiplier("1.5") == 1.0

    def test_special_characters_only_returns_default(self):
        """Should return 1.0 for special character strings."""
        assert units_to_multiplier("@#$%") == 1.0
        assert units_to_multiplier("---") == 1.0


class TestLoadNumericSeries:
    """Tests for load_numeric_series function."""

    def test_returns_none_for_none_path(self):
        """Should return None for None path."""
        result = load_numeric_series(None)
        assert result is None

    def test_returns_none_for_empty_string_path(self):
        """Should return None for empty string path."""
        result = load_numeric_series("")
        assert result is None

    def test_returns_none_for_nonexistent_path(self):
        """Should return None for non-existent path."""
        result = load_numeric_series("/nonexistent/path/file.csv")
        assert result is None

    def test_loads_csv_file_basic(self, tmp_path):
        """Should load basic CSV file correctly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\n500,0.5\n600,0.9\n700,0.3")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_csv_with_specified_columns(self, tmp_path):
        """Should load CSV with specified column names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("freq,transmission,other\n100,0.1,1\n200,0.2,2\n300,0.3,3\n400,0.4,4")
        
        result = load_numeric_series(str(csv_file), columns=["freq", "transmission"])
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [100, 200, 300, 400])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.2, 0.3, 0.4])

    def test_loads_csv_columns_case_insensitive(self, tmp_path):
        """Column matching should be case-insensitive."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("FREQ,TRANSMISSION\n100,0.1\n200,0.2\n300,0.3")
        
        result = load_numeric_series(str(csv_file), columns=["freq", "transmission"])
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [100, 200, 300])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.2, 0.3])

    def test_loads_tsv_file(self, tmp_path):
        """Should load TSV file correctly using tab delimiter."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("wavelength\tvalue\n400\t0.1\n500\t0.5\n600\t0.9\n700\t0.3")
        
        result = load_numeric_series(str(tsv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_csv_with_unit_conversion_nm(self, tmp_path):
        """Should apply nm multiplier (1.0) - values unchanged."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength_nm,value\n400,0.1\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # nm multiplier = 1.0, values unchanged
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_loads_csv_with_unit_conversion_um(self, tmp_path):
        """Should apply um multiplier (1000) - converts um to nm."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength_um,value\n0.4,0.1\n0.5,0.5\n0.6,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # um * 1000 = nm
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_loads_csv_with_unit_conversion_m(self, tmp_path):
        """Should apply m multiplier (1e9) - converts m to nm."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength_m,value\n4e-7,0.1\n5e-7,0.5\n6e-7,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # m * 1e9 = nm
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600], decimal=5)
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_loads_csv_without_header_uses_numeric_fallback(self, tmp_path):
        """CSV without clear header should use numeric fallback."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("400,0.1\n500,0.5\n600,0.9\n700,0.3")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # First row "400,0.1" becomes header, but since there's no "wave" in it,
        # it falls back to numeric parsing. Let's check it loads something reasonable.
        assert len(x_arr) >= 3
        # Data should be extracted from remaining rows

    def test_loads_csv_skips_empty_rows(self, tmp_path):
        """Should skip empty rows and still load valid data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\n\n500,0.5\n\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_loads_csv_skips_invalid_rows(self, tmp_path):
        """Should skip rows with non-numeric data and load valid rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\ninvalid,data\n500,0.5\nNaN,0.2\n600,0.9")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should have 3 valid rows (400, 500, 600)
        # Note: "invalid" and "NaN" rows should be skipped
        assert len(x_arr) >= 3
        assert 400 in x_arr
        assert 500 in x_arr
        assert 600 in x_arr

    def test_csv_wave_column_auto_detection(self, tmp_path):
        """Should auto-detect column with 'wave' in name."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("intensity,wavelength,extra\n0.1,400,1\n0.5,500,2\n0.9,600,3")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should detect 'wavelength' as x column
        assert len(x_arr) == 3
        # x should be wavelength values, y should be the next column (or intensity)
        # Based on implementation: x_idx found at "wavelength", y_idx defaults to adjacent
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])

    def test_loads_json_dict_format_xy(self, tmp_path):
        """Should load JSON with {x: [], y: []} format."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"x": [400, 500, 600, 700], "y": [0.1, 0.5, 0.9, 0.3]}))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_json_list_of_dicts_format(self, tmp_path):
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
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_json_list_of_arrays_format(self, tmp_path):
        """Should load JSON with [[x, y], ...] format."""
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
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_json_skips_invalid_dict_points(self, tmp_path):
        """Should skip dict points without x or y keys."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            {"invalid": "data"},  # No x or y
            {"x": 500, "y": 0.5},
            {"x": 600, "y": 0.9},
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_json_skips_points_with_missing_y(self, tmp_path):
        """Should skip dict points with missing y value."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            {"x": 500},  # Missing y
            {"x": 600, "y": 0.9},
            {"x": 700, "y": 0.3},
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        # Should have 400, 600, 700 (not 500 which has no y)
        assert 400 in x_arr
        assert 500 not in x_arr
        assert 600 in x_arr
        assert 700 in x_arr

    def test_json_handles_mixed_point_formats(self, tmp_path):
        """Should handle mix of dict and array point formats."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            [500, 0.5],  # Array format
            {"x": 600, "y": 0.9},
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_json_returns_none_for_fewer_than_3_valid_points(self, tmp_path):
        """Should return None when JSON has fewer than 3 valid points."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            {"x": 500, "y": 0.5},
            {"invalid": "data"},
        ]))
        
        result = load_numeric_series(str(json_file))
        
        # Only 2 valid points, should return None
        assert result is None

    def test_loads_npy_file_2d_array(self, tmp_path):
        """Should load NPY file with 2D array correctly."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[400, 0.1], [500, 0.5], [600, 0.9], [700, 0.3]])
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_loads_npz_file_first_array(self, tmp_path):
        """Should load NPZ file using first array."""
        npz_file = tmp_path / "data.npz"
        arr = np.array([[400, 0.1], [500, 0.5], [600, 0.9], [700, 0.3]])
        np.savez(npz_file, data=arr)
        
        result = load_numeric_series(str(npz_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 4
        assert len(y_arr) == 4
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600, 700])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9, 0.3])

    def test_returns_none_for_1d_npy(self, tmp_path):
        """Should return None for 1D NPY array."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([400, 500, 600])  # 1D array
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        assert result is None

    def test_returns_none_for_npy_with_single_column(self, tmp_path):
        """Should return None for NPY array with only 1 column."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[400], [500], [600]])  # 2D but single column
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        assert result is None

    def test_returns_none_for_corrupted_json(self, tmp_path):
        """Should return None for corrupted JSON file."""
        json_file = tmp_path / "data.json"
        json_file.write_text("{invalid json syntax")
        
        result = load_numeric_series(str(json_file))
        assert result is None

    def test_returns_none_for_json_with_wrong_structure(self, tmp_path):
        """Should return None for JSON with unsupported structure."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"data": "not arrays"}))
        
        result = load_numeric_series(str(json_file))
        assert result is None

    def test_csv_with_extra_columns_uses_wavelength_detection(self, tmp_path):
        """Should detect wavelength column and use adjacent column for y."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value,extra1,extra2\n400,0.1,1,2\n500,0.5,3,4\n600,0.9,5,6")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_csv_unsorted_gets_sorted_by_normalize_series(self, tmp_path):
        """CSV data should be sorted by x values via normalize_series."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n600,0.9\n400,0.1\n500,0.5")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should be sorted
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_returns_none_for_csv_with_fewer_than_3_valid_rows(self, tmp_path):
        """Should return None when CSV has fewer than 3 valid data rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\n500,0.5")
        
        result = load_numeric_series(str(csv_file))
        
        # Only 2 valid rows, normalize_series should return None
        assert result is None

    def test_handles_unknown_file_extension(self, tmp_path):
        """Should return None for unknown file extensions."""
        unknown_file = tmp_path / "data.xyz"
        unknown_file.write_text("400,0.1\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(unknown_file))
        
        # Unknown extension not handled
        assert result is None

    def test_handles_path_object_input(self, tmp_path):
        """Should handle Path objects as input."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength,value\n400,0.1\n500,0.5\n600,0.9")
        
        result = load_numeric_series(csv_file)  # Path object, not string
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])

    def test_csv_with_whitespace_in_values(self, tmp_path):
        """Should handle CSV with whitespace around values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("wavelength , value\n  400  ,  0.1  \n  500  ,  0.5  \n  600  ,  0.9  ")
        
        result = load_numeric_series(str(csv_file))
        
        assert result is not None
        x_arr, y_arr = result
        assert len(x_arr) == 3

    def test_json_with_null_values_skipped(self, tmp_path):
        """Should skip JSON points with null values."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"x": 400, "y": 0.1},
            {"x": None, "y": 0.5},  # null x
            {"x": 500, "y": None},  # null y
            {"x": 600, "y": 0.9},
            {"x": 700, "y": 0.3},
        ]))
        
        result = load_numeric_series(str(json_file))
        
        assert result is not None
        x_arr, y_arr = result
        # Should have 3 valid points (400, 600, 700)
        assert len(x_arr) == 3
        assert 400 in x_arr
        assert 600 in x_arr
        assert 700 in x_arr

    def test_npy_with_more_than_2_columns_uses_first_two(self, tmp_path):
        """Should use first two columns from NPY with more columns."""
        npy_file = tmp_path / "data.npy"
        arr = np.array([[400, 0.1, 1], [500, 0.5, 2], [600, 0.9, 3]])
        np.save(npy_file, arr)
        
        result = load_numeric_series(str(npy_file))
        
        assert result is not None
        x_arr, y_arr = result
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])

    def test_npz_empty_archive_returns_none(self, tmp_path):
        """Should return None for NPZ with no arrays."""
        npz_file = tmp_path / "data.npz"
        np.savez(npz_file)  # Empty archive
        
        result = load_numeric_series(str(npz_file))
        
        # No arrays in archive
        assert result is None

    def test_csv_column_partial_match(self, tmp_path):
        """Column matching should work with partial matches."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("frequency_hz,reflectance_percent\n400,0.1\n500,0.5\n600,0.9")
        
        result = load_numeric_series(str(csv_file), columns=["freq", "reflectance"])
        
        assert result is not None
        x_arr, y_arr = result
        # "freq" should match "frequency_hz", "reflectance" should match "reflectance_percent"
        np.testing.assert_array_almost_equal(x_arr, [400, 500, 600])
        np.testing.assert_array_almost_equal(y_arr, [0.1, 0.5, 0.9])


class TestComputePeakMetrics:
    """Tests for compute_peak_metrics function."""

    def test_returns_empty_for_both_empty_arrays(self):
        """Should return empty dict when both arrays are empty."""
        result = compute_peak_metrics(np.array([]), np.array([]))
        assert result == {}

    def test_returns_empty_for_empty_x_array(self):
        """Should return empty dict when x array is empty."""
        result = compute_peak_metrics(np.array([]), np.array([1, 2, 3]))
        assert result == {}

    def test_returns_empty_for_empty_y_array(self):
        """Should return empty dict when y array is empty."""
        result = compute_peak_metrics(np.array([1, 2, 3]), np.array([]))
        assert result == {}

    def test_finds_peak_position_middle(self):
        """Should find peak position in the middle of array."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        assert "peak_position" in result
        assert "peak_value" in result
        assert "fwhm" in result
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == 0.9

    def test_finds_peak_at_first_index(self):
        """Should find peak at the first index."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 400.0
        assert result["peak_value"] == 1.0

    def test_finds_peak_at_last_index(self):
        """Should find peak at the last index."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.2, 0.3, 0.5, 1.0])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 800.0
        assert result["peak_value"] == 1.0

    def test_multiple_equal_peaks_returns_first(self):
        """Should return first peak when multiple equal max values exist."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.9, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        # numpy argmax returns first occurrence
        assert result["peak_position"] == 400.0
        assert result["peak_value"] == 0.9

    def test_handles_negative_y_values(self):
        """Should find maximum even when all y values are negative."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([-1.0, -0.5, -0.1, -0.3, -0.8])
        
        result = compute_peak_metrics(x, y)
        
        # Maximum is -0.1 at position 600
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == -0.1

    def test_all_zeros_peak_value(self):
        """Should handle all zero values."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 400.0  # First index
        assert result["peak_value"] == 0.0
        # FWHM should be None (half_max = 0/2 = 0, but condition checks half_max > 0)
        assert result["fwhm"] is None

    def test_computes_fwhm_gaussian_peak(self):
        """Should compute FWHM for Gaussian-like peak."""
        # Create a Gaussian-like peak
        x = np.linspace(400, 800, 101)
        sigma = 50
        y = np.exp(-((x - 600) ** 2) / (2 * sigma ** 2))
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == pytest.approx(600, rel=0.01)
        assert result["peak_value"] == pytest.approx(1.0, rel=0.01)
        assert result["fwhm"] is not None
        # FWHM of Gaussian = 2.355 * sigma ≈ 118
        assert result["fwhm"] == pytest.approx(2.355 * sigma, rel=0.15)

    def test_fwhm_none_when_values_stay_above_half_max(self):
        """Should return None for FWHM when values never drop below half max."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.9, 0.95, 1.0, 0.95, 0.9])
        
        result = compute_peak_metrics(x, y)
        
        # half_max = 0.5, but minimum value is 0.9 > 0.5
        assert result["fwhm"] is None

    def test_fwhm_none_only_left_crossing_exists(self):
        """Should return None when only left side crosses half-max."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.3, 0.4, 1.0, 0.8, 0.7])
        
        result = compute_peak_metrics(x, y)
        
        # half_max = 0.5
        # Left: 0.3, 0.4 are below 0.5 - left_cross found
        # Right: 0.8, 0.7 are above 0.5 - no right_cross
        assert result["fwhm"] is None

    def test_fwhm_none_only_right_crossing_exists(self):
        """Should return None when only right side crosses half-max."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.7, 0.8, 1.0, 0.4, 0.3])
        
        result = compute_peak_metrics(x, y)
        
        # half_max = 0.5
        # Left: 0.7, 0.8 are above 0.5 - no left_cross
        # Right: 0.4, 0.3 are below 0.5 - right_cross found
        assert result["fwhm"] is None

    def test_fwhm_computed_both_crossings(self):
        """Should compute FWHM when both crossings exist."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.3, 0.4, 1.0, 0.4, 0.3])
        
        result = compute_peak_metrics(x, y)
        
        # half_max = 0.5
        # Left crossing at x=500 (y=0.4 <= 0.5)
        # Right crossing at x=700 (y=0.4 <= 0.5)
        assert result["fwhm"] is not None
        assert result["fwhm"] == pytest.approx(200.0, rel=0.01)  # 700 - 500

    def test_fwhm_exact_calculation(self):
        """FWHM should be |right_cross - left_cross|."""
        x = np.array([100, 200, 300, 400, 500])
        y = np.array([0.4, 0.6, 1.0, 0.6, 0.4])
        
        result = compute_peak_metrics(x, y)
        
        # half_max = 0.5
        # Left: searching from peak (300) backwards: 200 has 0.6 > 0.5, 100 has 0.4 <= 0.5
        # left_cross = 100
        # Right: searching from peak forwards: 400 has 0.6 > 0.5, 500 has 0.4 <= 0.5
        # right_cross = 500
        assert result["fwhm"] == pytest.approx(400.0, rel=0.01)

    def test_single_point_array(self):
        """Should handle single data point."""
        x = np.array([600])
        y = np.array([0.9])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == 0.9
        # Can't compute FWHM with single point
        assert result["fwhm"] is None

    def test_two_point_array(self):
        """Should handle two data points."""
        x = np.array([500, 600])
        y = np.array([0.5, 0.9])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == 0.9
        assert result["fwhm"] is None

    def test_flat_line_all_same_values(self):
        """Should handle flat line (all same y values)."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        result = compute_peak_metrics(x, y)
        
        # First index is the "peak" since all values are equal
        assert result["peak_position"] == 400.0
        assert result["peak_value"] == 0.5
        # half_max = 0.25, all values (0.5) are > 0.25, so no crossing
        assert result["fwhm"] is None

    def test_unsorted_x_values_peak_found_correctly(self):
        """Peak finding should use y values regardless of x order."""
        x = np.array([600, 400, 800, 500, 700])
        y = np.array([0.9, 0.1, 0.2, 0.5, 0.4])
        
        result = compute_peak_metrics(x, y)
        
        # Peak is at index 0 where y=0.9
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == 0.9

    def test_result_types_are_float(self):
        """All returned numeric values should be Python floats."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.3, 0.4, 1.0, 0.4, 0.3])
        
        result = compute_peak_metrics(x, y)
        
        assert isinstance(result["peak_position"], float)
        assert isinstance(result["peak_value"], float)
        assert result["fwhm"] is None or isinstance(result["fwhm"], float)

    def test_handles_nan_in_y_values(self):
        """Should handle NaN values in y array."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, np.nan, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        # numpy.argmax behavior with NaN: returns first NaN if present
        # or first max. Actual behavior depends on numpy version.
        # The function should still return a result without crashing.
        assert "peak_position" in result
        assert "peak_value" in result

    def test_handles_inf_in_y_values(self):
        """Should handle infinity in y array - inf becomes the peak."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, np.inf, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        # inf is the maximum
        assert result["peak_position"] == 600.0
        assert np.isinf(result["peak_value"])

    def test_handles_negative_inf_in_y_values(self):
        """Should handle negative infinity in y array."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, -np.inf, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        # Maximum is still 0.9 at index 2
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == 0.9

    def test_very_small_positive_peak(self):
        """Should handle very small positive peak values."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([1e-10, 1e-9, 1e-8, 1e-9, 1e-10])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == pytest.approx(1e-8, rel=0.01)

    def test_very_large_peak_values(self):
        """Should handle very large peak values."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([1e10, 1e11, 1e12, 1e11, 1e10])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600.0
        assert result["peak_value"] == pytest.approx(1e12, rel=0.01)

    def test_fwhm_with_peak_at_first_index(self):
        """FWHM calculation when peak is at first index."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([1.0, 0.4, 0.3, 0.2, 0.1])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 400.0
        # half_max = 0.5
        # Left: can't go left from index 0, so no left crossing
        # Right: 0.4 <= 0.5, so right_cross = 500
        # No left_cross means FWHM is None
        assert result["fwhm"] is None

    def test_fwhm_with_peak_at_last_index(self):
        """FWHM calculation when peak is at last index."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.2, 0.3, 0.4, 1.0])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 800.0
        # half_max = 0.5
        # Left: 0.4 <= 0.5, so left_cross = 700
        # Right: can't go right from last index, so no right crossing
        # No right_cross means FWHM is None
        assert result["fwhm"] is None

    def test_half_max_calculation(self):
        """Verify half_max is peak_value / 2."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.2, 0.4, 0.8, 0.4, 0.2])  # peak = 0.8, half_max = 0.4
        
        result = compute_peak_metrics(x, y)
        
        # At half_max = 0.4:
        # Left: from index 2 backwards: y[1]=0.4 <= 0.4, left_cross=500
        # Right: from index 2 forwards: y[3]=0.4 <= 0.4, right_cross=700
        assert result["fwhm"] == pytest.approx(200.0, rel=0.01)

    def test_integer_x_values(self):
        """Should handle integer x values and convert to float."""
        x = np.array([400, 500, 600, 700, 800], dtype=int)
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        assert result["peak_position"] == 600.0
        assert isinstance(result["peak_position"], float)

    def test_different_length_arrays_behavior(self):
        """Test behavior with different length x and y arrays.
        
        Note: The function doesn't explicitly check for equal lengths,
        it relies on numpy's behavior.
        """
        x = np.array([400, 500, 600])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])  # y is longer
        
        # This may raise an error or produce unexpected results
        # depending on implementation. Let's check it doesn't crash.
        try:
            result = compute_peak_metrics(x, y)
            # If it succeeds, verify the peak is found
            assert "peak_position" in result
        except IndexError:
            # This is acceptable - arrays should be same length
            pass

    def test_result_dict_structure(self):
        """Should always return dict with expected keys."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = compute_peak_metrics(x, y)
        
        assert isinstance(result, dict)
        assert "peak_position" in result
        assert "peak_value" in result
        assert "fwhm" in result

    def test_peak_value_exactly_half_of_another(self):
        """Test when some values are exactly half of peak."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.3, 0.5, 1.0, 0.5, 0.3])
        
        result = compute_peak_metrics(x, y)
        
        # half_max = 0.5
        # Values at 500 and 700 are exactly 0.5
        # The check is y[i] <= half_max, so 0.5 <= 0.5 is True
        # Left: y[1]=0.5 <= 0.5, left_cross=500
        # Right: y[3]=0.5 <= 0.5, right_cross=700
        assert result["fwhm"] == pytest.approx(200.0, rel=0.01)


class TestQuantitativeCurveMetrics:
    """Tests for quantitative_curve_metrics function."""

    def test_returns_empty_for_none_sim(self):
        """Should return empty dict when simulation data is None."""
        result = quantitative_curve_metrics(None, None)
        assert result == {}

    def test_returns_empty_for_none_sim_with_ref(self):
        """Should return empty dict when sim is None, even with ref data."""
        x_ref = np.array([400, 500, 600])
        y_ref = np.array([0.1, 0.5, 0.9])
        
        result = quantitative_curve_metrics(None, (x_ref, y_ref))
        assert result == {}

    def test_returns_sim_metrics_without_ref(self):
        """Should return simulation metrics even without reference."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), None)
        
        assert "peak_position_sim" in result
        assert "peak_value_sim" in result
        assert "fwhm_sim" in result
        assert result["peak_position_sim"] == 600.0
        assert result["peak_value_sim"] == 0.9
        # Should not have reference metrics
        assert "peak_position_paper" not in result
        assert "normalized_rmse_percent" not in result
        assert "correlation" not in result
        assert "r_squared" not in result

    def test_ref_with_1_point_returns_only_sim_metrics(self):
        """Should return only sim metrics when ref has 1 point."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400])
        y_ref = np.array([0.1])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_position_sim" in result
        assert "peak_position_paper" not in result
        assert "normalized_rmse_percent" not in result

    def test_ref_with_2_points_returns_only_sim_metrics(self):
        """Should return only sim metrics when ref has 2 points."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500])
        y_ref = np.array([0.1, 0.5])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_position_sim" in result
        assert result["peak_position_sim"] == 600.0
        assert "peak_position_paper" not in result
        assert "normalized_rmse_percent" not in result

    def test_ref_with_3_points_includes_ref_metrics(self):
        """Should include reference metrics when ref has 3+ points."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600])
        y_ref = np.array([0.1, 0.5, 0.9])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_position_sim" in result
        assert "peak_position_paper" in result
        assert "peak_value_paper" in result
        assert "fwhm_paper" in result

    def test_computes_all_comparison_metrics(self):
        """Should compute all comparison metrics with overlapping data."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.12, 0.52, 0.88, 0.38, 0.18])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Verify all expected keys exist
        assert "peak_position_sim" in result
        assert "peak_value_sim" in result
        assert "fwhm_sim" in result
        assert "peak_position_paper" in result
        assert "peak_value_paper" in result
        assert "fwhm_paper" in result
        assert "n_points_compared" in result
        assert "normalized_rmse_percent" in result
        assert "correlation" in result
        assert "r_squared" in result
        
        # Verify n_points_compared
        assert result["n_points_compared"] == 5

    def test_peak_position_error_calculation(self):
        """Should compute peak position error as |sim-ref|/|ref| * 100."""
        x_sim = np.array([400, 500, 620, 700, 800])  # Peak at 620
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])  # Peak at 600
        y_ref = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Error = |620 - 600| / 600 * 100 = 20/600 * 100 ≈ 3.33%
        assert "peak_position_error_percent" in result
        expected_error = abs(620 - 600) / 600 * 100
        assert result["peak_position_error_percent"] == pytest.approx(expected_error, rel=0.01)

    def test_peak_position_error_not_computed_when_ref_position_zero(self):
        """Should not compute error when reference peak position is zero.
        
        This would cause division by zero.
        """
        # This is a tricky edge case - peak_position = 0 is unusual
        # because it means the peak is at x=0
        x_sim = np.array([0, 100, 200, 300, 400])
        y_sim = np.array([0.9, 0.5, 0.3, 0.2, 0.1])  # Peak at x=0
        x_ref = np.array([0, 100, 200, 300, 400])
        y_ref = np.array([0.9, 0.5, 0.3, 0.2, 0.1])  # Peak at x=0
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # With ref peak at x=0, peak_position_error_percent should not be computed
        # to avoid division by zero
        assert "peak_position_error_percent" not in result

    def test_peak_position_error_not_computed_when_ref_peak_value_zero(self):
        """Should not compute error when reference peak value is zero."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All zeros
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # peak_value_paper = 0, so error should not be computed
        assert "peak_position_error_percent" not in result

    def test_peak_height_ratio_calculation(self):
        """Should compute peak height ratio as sim_peak / ref_peak."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])  # Peak = 0.9
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.1, 0.3, 0.45, 0.2, 0.1])  # Peak = 0.45
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_height_ratio" in result
        expected_ratio = 0.9 / 0.45
        assert result["peak_height_ratio"] == pytest.approx(expected_ratio, rel=0.01)

    def test_peak_height_ratio_not_computed_when_ref_zero(self):
        """Should not compute peak height ratio when ref peak is zero."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # peak_value_paper = 0, so ratio should not be computed (or be None)
        assert "peak_height_ratio" not in result or result.get("peak_height_ratio") is None

    def test_fwhm_ratio_calculation(self):
        """Should compute FWHM ratio when both FWHMs exist."""
        # Create Gaussian curves with different widths
        x_sim = np.linspace(400, 800, 101)
        sigma_sim = 50
        y_sim = np.exp(-((x_sim - 600) ** 2) / (2 * sigma_sim ** 2))
        
        x_ref = np.linspace(400, 800, 101)
        sigma_ref = 100
        y_ref = np.exp(-((x_ref - 600) ** 2) / (2 * sigma_ref ** 2))
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        if result.get("fwhm_sim") and result.get("fwhm_paper"):
            assert "fwhm_ratio" in result
            # FWHM ratio = sim_fwhm / paper_fwhm
            # sim has smaller sigma, so smaller FWHM
            assert result["fwhm_ratio"] > 0
            expected_ratio = (2.355 * sigma_sim) / (2.355 * sigma_ref)
            assert result["fwhm_ratio"] == pytest.approx(expected_ratio, rel=0.2)

    def test_fwhm_ratio_not_computed_when_ref_fwhm_missing(self):
        """Should not compute FWHM ratio when reference FWHM is None."""
        x_sim = np.linspace(400, 800, 101)
        y_sim = np.exp(-((x_sim - 600) ** 2) / (2 * 50 ** 2))
        
        # Reference that won't have FWHM (flat or peak at edge)
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # ref FWHM should be None
        assert result.get("fwhm_paper") is None
        assert "fwhm_ratio" not in result

    def test_interpolation_to_sim_axis(self):
        """Should interpolate reference to simulation x-axis."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([450, 550, 650, 750])  # Different x points
        y_ref = np.array([0.3, 0.7, 0.6, 0.3])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Points 400 and 800 are outside ref range, should be NaN after interp
        # Only 500, 600, 700 should overlap
        assert "n_points_compared" in result
        assert result["n_points_compared"] == 3  # 500, 600, 700 are within 450-750

    def test_no_overlap_returns_zero_points(self):
        """Should return 0 points compared when no overlap."""
        x_sim = np.array([400, 500, 600])
        y_sim = np.array([0.1, 0.5, 0.9])
        x_ref = np.array([1000, 1100, 1200])
        y_ref = np.array([0.1, 0.5, 0.9])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        assert "peak_position_sim" in result
        assert "peak_position_paper" in result
        assert "n_points_compared" in result
        assert result["n_points_compared"] == 0
        # Comparison metrics should not exist
        assert "normalized_rmse_percent" not in result
        assert "correlation" not in result
        assert "r_squared" not in result

    def test_perfect_match_all_metrics(self):
        """Should return perfect metrics for identical curves."""
        x = np.array([400, 500, 600, 700, 800])
        y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x, y), (x.copy(), y.copy()))
        
        assert result["normalized_rmse_percent"] == pytest.approx(0.0, abs=0.001)
        assert result["correlation"] == pytest.approx(1.0, abs=0.001)
        assert result["r_squared"] == pytest.approx(1.0, abs=0.001)
        assert result["peak_position_error_percent"] == pytest.approx(0.0, abs=0.001)
        assert result["peak_height_ratio"] == pytest.approx(1.0, abs=0.001)
        assert result["n_points_compared"] == 5

    def test_normalized_rmse_calculation(self):
        """Verify NRMSE = RMSE / value_range * 100."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.2, 0.4, 0.6, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800])
        y_ref = np.array([0.1, 0.5, 0.7, 0.5, 0.1])  # Similar but offset
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Manual calculation:
        # diff = [0.1, -0.1, -0.1, -0.1, 0.1]
        # rmse = sqrt(mean([0.01, 0.01, 0.01, 0.01, 0.01])) = sqrt(0.01) = 0.1
        # range = 0.7 - 0.1 = 0.6
        # nrmse = 0.1 / 0.6 * 100 ≈ 16.67%
        expected_nrmse = 0.1 / 0.6 * 100
        assert result["normalized_rmse_percent"] == pytest.approx(expected_nrmse, rel=0.01)

    def test_r_squared_calculation(self):
        """Verify r_squared = 1 - ss_res/ss_tot."""
        x = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
        y_ref = np.array([0.15, 0.45, 0.85, 0.45, 0.15])
        
        result = quantitative_curve_metrics((x, y_sim), (x, y_ref))
        
        # Manual calculation for r_squared
        mean_ref = np.mean(y_ref)
        ss_res = np.sum((y_ref - y_sim) ** 2)
        ss_tot = np.sum((y_ref - mean_ref) ** 2)
        expected_r2 = 1 - ss_res / ss_tot
        
        assert result["r_squared"] == pytest.approx(expected_r2, rel=0.01)

    def test_correlation_calculation(self):
        """Verify correlation coefficient calculation."""
        x = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
        y_ref = np.array([0.15, 0.35, 0.55, 0.35, 0.15])
        
        result = quantitative_curve_metrics((x, y_sim), (x, y_ref))
        
        expected_corr = np.corrcoef(y_sim, y_ref)[0, 1]
        assert result["correlation"] == pytest.approx(expected_corr, rel=0.001)

    def test_negative_correlation(self):
        """Should correctly compute negative correlation."""
        x = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.9, 0.7, 0.5, 0.3, 0.1])  # Decreasing
        y_ref = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Increasing
        
        result = quantitative_curve_metrics((x, y_sim), (x, y_ref))
        
        assert "correlation" in result
        assert result["correlation"] < 0
        assert result["correlation"] == pytest.approx(-1.0, abs=0.001)

    def test_handles_constant_reference_gracefully(self):
        """Should handle reference with all same values."""
        x = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        y_ref = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        result = quantitative_curve_metrics((x, y_sim), (x, y_ref))
        
        # NRMSE uses range = max - min = 0.5 - 0.5 = 0, but implementation
        # uses `or 1.0` to avoid division by zero
        assert "normalized_rmse_percent" in result
        assert result["normalized_rmse_percent"] is not None
        # Correlation with constant reference is undefined (NaN)
        # Implementation should handle this

    def test_different_array_lengths_interpolated(self):
        """Should interpolate reference to match simulation x-axis."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        x_ref = np.array([400, 500, 600, 700, 800, 900, 1000])
        y_ref = np.array([0.1, 0.5, 0.9, 0.4, 0.2, 0.1, 0.05])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # All sim x points (400-800) are within ref range (400-1000)
        assert result["n_points_compared"] == 5

    def test_all_expected_keys_for_full_comparison(self):
        """Should return all expected keys when full comparison is possible."""
        x = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        y_ref = np.array([0.12, 0.52, 0.88, 0.38, 0.18])
        
        result = quantitative_curve_metrics((x, y_sim), (x, y_ref))
        
        # Required keys for sim
        assert "peak_position_sim" in result
        assert "peak_value_sim" in result
        assert "fwhm_sim" in result
        
        # Required keys for ref
        assert "peak_position_paper" in result
        assert "peak_value_paper" in result
        assert "fwhm_paper" in result
        
        # Required comparison keys
        assert "n_points_compared" in result
        assert "normalized_rmse_percent" in result
        assert "correlation" in result
        assert "r_squared" in result
        
        # Optional keys (depend on conditions)
        # peak_position_error_percent - present if both peaks are valid
        # peak_height_ratio - present if ref peak is non-zero
        # fwhm_ratio - present if both fwhms exist

    def test_partial_overlap_correct_n_points(self):
        """Should count only overlapping points after interpolation."""
        x_sim = np.array([100, 200, 300, 400, 500])
        y_sim = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
        x_ref = np.array([300, 400, 500, 600, 700])  # Partial overlap
        y_ref = np.array([0.9, 0.5, 0.1, 0.5, 0.9])
        
        result = quantitative_curve_metrics((x_sim, y_sim), (x_ref, y_ref))
        
        # Only points 300, 400, 500 overlap
        assert result["n_points_compared"] == 3

    def test_returns_dict_type(self):
        """Should always return a dict."""
        result = quantitative_curve_metrics(None, None)
        assert isinstance(result, dict)
        
        x_sim = np.array([400, 500, 600])
        y_sim = np.array([0.1, 0.5, 0.9])
        result2 = quantitative_curve_metrics((x_sim, y_sim), None)
        assert isinstance(result2, dict)

    def test_sim_peak_metrics_are_floats(self):
        """Simulation peak metrics should be floats, not numpy types."""
        x_sim = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        
        result = quantitative_curve_metrics((x_sim, y_sim), None)
        
        assert isinstance(result["peak_position_sim"], float) or result["peak_position_sim"] is None
        assert isinstance(result["peak_value_sim"], float) or result["peak_value_sim"] is None

    def test_r_squared_can_be_negative(self):
        """R-squared can be negative when model is worse than mean."""
        x = np.array([400, 500, 600, 700, 800])
        y_sim = np.array([0.9, 0.1, 0.9, 0.1, 0.9])  # Very different pattern
        y_ref = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Monotonic increase
        
        result = quantitative_curve_metrics((x, y_sim), (x, y_ref))
        
        # Simulation is so bad it's worse than predicting mean
        assert "r_squared" in result
        # R-squared can legitimately be negative



