from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.paper_loader import validate_figure_image
from src.paper_loader.config import DEFAULT_IMAGE_CONFIG


def _build_mock_pil(image_size):
    mock_img = MagicMock()
    mock_img.size = image_size
    mock_img_class = MagicMock()
    mock_img_class.open.return_value = mock_img
    mock_img.close = MagicMock()

    mock_pil = MagicMock()
    mock_pil.Image = mock_img_class
    return mock_pil, mock_img_class, mock_img


class TestValidateFigureImage:
    """Tests for validate_figure_image function."""

    def test_returns_list(self):
        """Verify function returns a list."""
        warnings = validate_figure_image("/nonexistent/image.png")
        assert isinstance(warnings, list)

    def test_nonexistent_image_warns(self):
        """Test that nonexistent file returns exactly one warning with correct message."""
        warnings = validate_figure_image("/nonexistent/image.png")
        assert len(warnings) == 1
        assert warnings[0] == "Image file not found: /nonexistent/image.png"

    def test_nonexistent_image_returns_early(self):
        """Test that function returns early when file doesn't exist (no other checks)."""
        warnings = validate_figure_image("/nonexistent/image.png")
        assert len(warnings) == 1
        # Should not check file size or dimensions if file doesn't exist
        assert all("file size" not in w.lower() for w in warnings)
        assert all("resolution" not in w.lower() for w in warnings)

    def test_existing_image_no_warnings(self, sample_image_path):
        """Test that valid image returns no warnings."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            # File size just under threshold (4.9 MB)
            mock_stat.return_value.st_size = int(4.9 * 1024 * 1024)
            warnings = validate_figure_image(str(sample_image_path))

        assert warnings == []
        mock_img.close.assert_called_once()

    def test_accepts_path_object(self):
        """Test that function accepts Path objects, not just strings."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))
        path_obj = Path("/test/image.png")

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024 * 1024
            warnings = validate_figure_image(path_obj)

        assert warnings == []

    def test_large_file_warns_at_threshold(self):
        """Test that file exactly at size threshold triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))
        threshold_bytes = int(DEFAULT_IMAGE_CONFIG.max_file_size_mb * 1024 * 1024)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            # Exactly at threshold (5.0 MB)
            mock_stat.return_value.st_size = threshold_bytes + 1
            warnings = validate_figure_image("large.png")

        assert len(warnings) == 1
        assert "Large file size" in warnings[0]
        assert "MB" in warnings[0]
        # Verify exact size is included in warning
        size_mb = (threshold_bytes + 1) / (1024 * 1024)
        assert f"{size_mb:.1f}MB" in warnings[0] or f"{size_mb:.1f} MB" in warnings[0]

    def test_large_file_no_warning_below_threshold(self):
        """Test that file just below threshold doesn't warn."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))
        threshold_bytes = int(DEFAULT_IMAGE_CONFIG.max_file_size_mb * 1024 * 1024)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            # Just below threshold
            mock_stat.return_value.st_size = threshold_bytes - 1
            warnings = validate_figure_image("ok.png")

        assert not any("Large file size" in w for w in warnings)

    def test_large_file_warns(self):
        """Test that very large file triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 50 * 1024 * 1024
            warnings = validate_figure_image("large.png")

        assert len(warnings) == 1
        assert "Large file size" in warnings[0]
        assert "50.0" in warnings[0] or "50" in warnings[0]

    def test_zero_file_size_no_warning(self):
        """Test that zero-sized file doesn't trigger size warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 0
            warnings = validate_figure_image("zero.png")

        assert not any("Large file size" in w for w in warnings)

    def test_low_resolution_warns_at_threshold(self):
        """Test that resolution exactly at minimum threshold triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil(
            (DEFAULT_IMAGE_CONFIG.min_resolution - 1, DEFAULT_IMAGE_CONFIG.min_resolution - 1)
        )

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("tiny.png")

        assert len(warnings) == 1
        assert "Low resolution" in warnings[0]
        assert f"{DEFAULT_IMAGE_CONFIG.min_resolution - 1}x{DEFAULT_IMAGE_CONFIG.min_resolution - 1}" in warnings[0]
        assert f"≥{DEFAULT_IMAGE_CONFIG.min_resolution}px" in warnings[0]

    def test_low_resolution_no_warning_at_minimum(self):
        """Test that resolution exactly at minimum doesn't warn."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil(
            (DEFAULT_IMAGE_CONFIG.min_resolution, DEFAULT_IMAGE_CONFIG.min_resolution)
        )

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("ok.png")

        assert not any("Low resolution" in w for w in warnings)

    def test_low_resolution_warns_width_only(self):
        """Test that low width (but sufficient height) triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((100, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("narrow.png")

        assert any("Low resolution" in w for w in warnings)
        assert "100x1000" in warnings[0]

    def test_low_resolution_warns_height_only(self):
        """Test that low height (but sufficient width) triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 100))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("short.png")

        assert any("Low resolution" in w for w in warnings)
        assert "1000x100" in warnings[0]

    def test_low_resolution_warns(self):
        """Test that very low resolution triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((10, 10))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("tiny.png")

        assert len(warnings) == 1
        assert "Low resolution" in warnings[0]
        assert "10x10" in warnings[0]

    def test_high_resolution_warns_at_threshold(self):
        """Test that resolution exactly at maximum threshold triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil(
            (DEFAULT_IMAGE_CONFIG.max_resolution + 1, DEFAULT_IMAGE_CONFIG.max_resolution + 1)
        )

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("giant.png")

        assert len(warnings) == 1
        assert "Very high resolution" in warnings[0]
        assert f"{DEFAULT_IMAGE_CONFIG.max_resolution + 1}x{DEFAULT_IMAGE_CONFIG.max_resolution + 1}" in warnings[0]
        assert f"≤{DEFAULT_IMAGE_CONFIG.max_resolution}px" in warnings[0]

    def test_high_resolution_no_warning_at_maximum(self):
        """Test that resolution exactly at maximum doesn't warn."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil(
            (DEFAULT_IMAGE_CONFIG.max_resolution, DEFAULT_IMAGE_CONFIG.max_resolution)
        )

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("ok.png")

        assert not any("Very high resolution" in w for w in warnings)

    def test_high_resolution_warns_width_only(self):
        """Test that high width (but acceptable height) triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((5000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("wide.png")

        assert any("Very high resolution" in w for w in warnings)
        assert "5000x1000" in warnings[0]

    def test_high_resolution_warns_height_only(self):
        """Test that high height (but acceptable width) triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 5000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("tall.png")

        assert any("Very high resolution" in w for w in warnings)
        assert "1000x5000" in warnings[0]

    def test_high_resolution_warns(self):
        """Test that very high resolution triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((10000, 10000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("giant.png")

        assert len(warnings) == 1
        assert "Very high resolution" in warnings[0]
        assert "10000x10000" in warnings[0]

    def test_extreme_aspect_ratio_warns_at_threshold(self):
        """Test that aspect ratio above threshold triggers warning."""
        # max_aspect_ratio is 5.0, so >5.0 should warn
        # Use valid resolution dimensions (>=512) to avoid low resolution warning
        # 512 x 2560 = 5.0:1 aspect ratio (should NOT warn, as it's exactly at threshold)
        # 512 x 2561 = 5.001:1 aspect ratio (should warn, as it's > threshold)
        mock_pil, mock_img_class, mock_img = _build_mock_pil((512, 2561))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("long.png")

        assert len(warnings) == 1
        assert "Extreme aspect ratio" in warnings[0]
        assert "5.0:1" in warnings[0]  # Should show 5.0:1 (rounded)

    def test_extreme_aspect_ratio_no_warning_at_threshold(self):
        """Test that aspect ratio exactly at threshold doesn't warn."""
        # Exactly 5.0:1 should not warn (threshold is > 5.0)
        # Use valid resolution to avoid low resolution warning
        mock_pil, mock_img_class, mock_img = _build_mock_pil((512, 2560))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("ok.png")

        assert not any("Extreme aspect ratio" in w for w in warnings)

    def test_extreme_aspect_ratio_no_warning_below_threshold(self):
        """Test that aspect ratio below threshold doesn't warn."""
        # Just below 5:1
        # Use valid resolution to avoid low resolution warning
        mock_pil, mock_img_class, mock_img = _build_mock_pil((512, 2559))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("ok.png")

        assert not any("Extreme aspect ratio" in w for w in warnings)

    def test_extreme_aspect_ratio_warns_vertical(self):
        """Test that extreme vertical aspect ratio triggers warning."""
        # Use valid resolution to avoid low resolution warning
        # 512 x 25600 = 50:1 aspect ratio
        mock_pil, mock_img_class, mock_img = _build_mock_pil((512, 25600))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("long.png")

        assert any("Extreme aspect ratio" in w for w in warnings)
        # Find the aspect ratio warning
        aspect_warning = next((w for w in warnings if "Extreme aspect ratio" in w), None)
        assert aspect_warning is not None
        assert "50.0:1" in aspect_warning

    def test_extreme_aspect_ratio_warns_horizontal(self):
        """Test that extreme horizontal aspect ratio triggers warning."""
        # Use valid resolution to avoid low resolution warning
        # 25600 x 512 = 50:1 aspect ratio
        mock_pil, mock_img_class, mock_img = _build_mock_pil((25600, 512))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("wide.png")

        assert any("Extreme aspect ratio" in w for w in warnings)
        # Find the aspect ratio warning (may be mixed with high resolution warning)
        aspect_warning = next((w for w in warnings if "Extreme aspect ratio" in w), None)
        assert aspect_warning is not None
        assert "50.0:1" in aspect_warning

    def test_extreme_aspect_ratio_warns(self):
        """Test that extreme aspect ratio triggers warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((100, 5000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("long.png")

        assert any("Extreme aspect ratio" in w for w in warnings)

    def test_square_image_no_aspect_ratio_warning(self):
        """Test that square image doesn't trigger aspect ratio warning."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("square.png")

        assert not any("Extreme aspect ratio" in w for w in warnings)

    def test_multiple_warnings_simultaneously(self):
        """Test that multiple issues trigger multiple warnings."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((10, 10000))
        threshold_bytes = int(DEFAULT_IMAGE_CONFIG.max_file_size_mb * 1024 * 1024)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            # Large file + low resolution + high resolution + extreme aspect ratio
            mock_stat.return_value.st_size = threshold_bytes + 1
            warnings = validate_figure_image("problematic.png")

        assert len(warnings) >= 3
        assert any("Large file size" in w for w in warnings)
        assert any("Low resolution" in w for w in warnings)
        assert any("Very high resolution" in w for w in warnings)
        assert any("Extreme aspect ratio" in w for w in warnings)

    def test_image_close_called(self):
        """Test that Image.close() is called to free resources."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024 * 1024
            validate_figure_image("test.png")

        mock_img.close.assert_called_once()

    def test_image_close_called_on_exception(self):
        """Test that Image.close() is called even when exception occurs."""
        mock_img = MagicMock()
        mock_img.size = (1000, 1000)
        mock_img.close = MagicMock()
        mock_img_class = MagicMock()
        mock_img_class.open.return_value = mock_img
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class

        # Make close() raise an exception to test error handling
        mock_img.close.side_effect = Exception("Close error")

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024 * 1024
            # Should not raise, but close should be attempted
            warnings = validate_figure_image("test.png")

        # Close should have been called (even if it raised)
        assert mock_img.close.called

    def test_pil_import_error_handled(self):
        """Test that missing PIL module is handled gracefully."""
        with patch.dict("sys.modules", {"PIL": None}), patch(
            "pathlib.Path.exists", return_value=True
        ), patch("pathlib.Path.is_dir", return_value=False), patch(
            "pathlib.Path.stat"
        ) as mock_stat:
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("no_pil.png")

        assert len(warnings) == 1
        assert "PIL/Pillow not installed" in warnings[0]
        assert "pip install Pillow" in warnings[0]

    def test_pil_import_error_does_not_check_dimensions(self):
        """Test that when PIL is missing, dimension checks are skipped."""
        with patch.dict("sys.modules", {"PIL": None}), patch(
            "pathlib.Path.exists", return_value=True
        ), patch("pathlib.Path.is_dir", return_value=False), patch(
            "pathlib.Path.stat"
        ) as mock_stat:
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("no_pil.png")

        # Should only have PIL warning, not dimension warnings
        assert len(warnings) == 1
        assert not any("resolution" in w.lower() for w in warnings)
        assert not any("aspect ratio" in w.lower() for w in warnings)

    def test_image_analysis_exception_handled(self):
        """Test that image analysis exceptions are caught and reported."""
        mock_img_class = MagicMock()
        mock_img_class.open.side_effect = Exception("Corrupt image")
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024
            warnings = validate_figure_image("corrupt.png")

        assert len(warnings) == 1
        assert "Could not analyze image" in warnings[0]
        assert "Corrupt image" in warnings[0]

    def test_image_analysis_exception_preserves_file_size_warning(self):
        """Test that file size warning is preserved even if image analysis fails."""
        mock_img_class = MagicMock()
        mock_img_class.open.side_effect = Exception("Corrupt image")
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class
        threshold_bytes = int(DEFAULT_IMAGE_CONFIG.max_file_size_mb * 1024 * 1024)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = threshold_bytes + 1
            warnings = validate_figure_image("corrupt.png")

        assert len(warnings) == 2
        assert any("Large file size" in w for w in warnings)
        assert any("Could not analyze image" in w for w in warnings)

    def test_empty_string_path(self):
        """Test that empty string path is handled."""
        warnings = validate_figure_image("")
        assert len(warnings) == 1
        assert "Image file not found" in warnings[0]

    def test_relative_path(self):
        """Test that relative paths are handled correctly."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024 * 1024
            warnings = validate_figure_image("relative/path/image.png")

        assert warnings == []

    def test_very_small_file_size(self):
        """Test that very small files don't trigger size warnings."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1
            warnings = validate_figure_image("tiny_file.png")

        assert not any("Large file size" in w for w in warnings)

    def test_warning_messages_are_strings(self):
        """Test that all warnings are strings."""
        mock_pil, mock_img_class, mock_img = _build_mock_pil((10, 10))
        threshold_bytes = int(DEFAULT_IMAGE_CONFIG.max_file_size_mb * 1024 * 1024)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("pathlib.Path.stat") as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = threshold_bytes + 1
            warnings = validate_figure_image("test.png")

        assert all(isinstance(w, str) for w in warnings)
        assert all(len(w) > 0 for w in warnings)

