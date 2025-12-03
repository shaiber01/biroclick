from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.paper_loader import validate_figure_image


def _build_mock_pil(image_size):
    mock_img = MagicMock()
    mock_img.size = image_size
    mock_img_class = MagicMock()
    mock_img_class.open.return_value = mock_img
    mock_img_class.open.return_value.close = MagicMock()

    mock_pil = MagicMock()
    mock_pil.Image = mock_img_class
    return mock_pil, mock_img_class


class TestValidateFigureImage:
    """Tests for validate_figure_image function."""

    def test_nonexistent_image_warns(self):
        warnings = validate_figure_image("/nonexistent/image.png")
        assert len(warnings) == 1
        assert "Image file not found" in warnings[0]

    def test_existing_image_no_critical_warnings(self, sample_image_path):
        mock_pil, mock_img_class = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024 * 1024
            warnings = validate_figure_image(str(sample_image_path))

        assert not warnings

    def test_large_file_warns(self):
        mock_pil, mock_img_class = _build_mock_pil((1000, 1000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 50 * 1024 * 1024
            warnings = validate_figure_image("large.png")

        assert any("Large file size" in w for w in warnings)

    def test_low_resolution_warns(self):
        mock_pil, mock_img_class = _build_mock_pil((10, 10))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("tiny.png")

        assert any("Low resolution" in w for w in warnings)

    def test_high_resolution_warns(self):
        mock_pil, mock_img_class = _build_mock_pil((10000, 10000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("giant.png")

        assert any("Very high resolution" in w for w in warnings)

    def test_extreme_aspect_ratio_warns(self):
        mock_pil, mock_img_class = _build_mock_pil((100, 5000))

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("long.png")

        assert any("Extreme aspect ratio" in w for w in warnings)

    def test_pil_import_error_handled(self):
        with patch.dict("sys.modules", {"PIL": None}), patch(
            "pathlib.Path.exists", return_value=True
        ), patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1000
            warnings = validate_figure_image("no_pil.png")

        assert any("PIL/Pillow not installed" in w for w in warnings)

    def test_image_analysis_exception_handled(self):
        mock_img_class = MagicMock()
        mock_img_class.open.side_effect = Exception("Corrupt image")
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat, patch.dict(
            "sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}
        ):
            mock_stat.return_value.st_size = 1024
            warnings = validate_figure_image("corrupt.png")

        assert any("Could not analyze image" in w for w in warnings)

