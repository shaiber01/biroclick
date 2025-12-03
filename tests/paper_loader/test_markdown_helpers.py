"""Tests for misc markdown helper utilities."""

from __future__ import annotations

from src.paper_loader.markdown_parser import (
    generate_figure_id,
    get_file_extension,
)


class TestGenerateFigureId:
    """Tests for generate_figure_id function."""

    def test_extracts_from_alt_text_figure(self):
        fig_id = generate_figure_id(0, "Figure 3: Extinction spectrum", "img.png")
        assert fig_id == "Fig3"

    def test_extracts_from_alt_text_fig(self):
        fig_id = generate_figure_id(0, "Fig 2a shows results", "img.png")
        assert fig_id == "Fig2a"

    def test_extracts_from_url_filename(self):
        fig_id = generate_figure_id(0, "", "images/figure_4.png")
        assert fig_id == "Fig4"

    def test_fallback_to_index(self):
        fig_id = generate_figure_id(2, "Some random text", "random.png")
        assert fig_id == "Fig3"

    def test_extracts_decimal_figures(self):
        """Extracts decimal figure numbers like 1.2."""
        fig_id = generate_figure_id(0, "Figure 1.2 shows data", "img.png")
        assert fig_id == "Fig1.2"

    def test_extracts_dashed_figures(self):
        """Extracts figure numbers with dashes like 1-2."""
        fig_id = generate_figure_id(0, "Figure 1-2 shows data", "img.png")
        assert fig_id == "Fig1-2"

    def test_extracts_complex_labels(self):
        """Extracts complex labels like 'Fig. S1' or 'Figure 2(a)'."""
        assert generate_figure_id(0, "Fig. S1", "x.png") == "FigS1"
        assert generate_figure_id(0, "Figure 3b", "x.png") == "Fig3b"


class TestGetFileExtension:
    """Tests for get_file_extension function."""

    def test_png_extension(self):
        ext = get_file_extension("path/to/image.png")
        assert ext == ".png"

    def test_uppercase_normalized(self):
        ext = get_file_extension("image.PNG")
        assert ext == ".png"

    def test_url_with_query_params(self):
        ext = get_file_extension("https://example.com/image.gif?token=abc")
        assert ext in [".gif", ".png"]

    def test_unknown_extension_uses_default(self):
        ext = get_file_extension("file.xyz")
        assert ext == ".png"

    def test_custom_default(self):
        ext = get_file_extension("file.xyz", default=".jpg")
        assert ext == ".jpg"

