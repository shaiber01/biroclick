"""Tests for extract_figures_from_markdown."""

from __future__ import annotations

import pytest

from src.paper_loader.markdown_parser import extract_figures_from_markdown
from tests.paper_loader.markdown_shared import assert_single_figure, parse_figures


class TestExtractFiguresFromMarkdown:
    """Tests for extract_figures_from_markdown function."""

    def test_extracts_markdown_image_basic(self):
        """Extracts basic ![alt](url) format."""
        md = "Some text ![Alt text](images/fig1.png) more text"
        figure = assert_single_figure(md, url="images/fig1.png", alt="Alt text")

        assert figure["original_match"] == "![Alt text](images/fig1.png)"

    def test_extracts_markdown_image_with_title(self):
        """Extracts ![alt](url \"title\") format."""
        md = '![Figure 1](fig.png "A title")'
        assert_single_figure(md, url="fig.png", alt="Figure 1")

    def test_extracts_html_img_tag(self):
        """Extracts <img src=\"url\" alt=\"...\"> format."""
        md = '<img src="images/fig2.jpg" alt="Figure 2">'
        assert_single_figure(md, url="images/fig2.jpg", alt="Figure 2")

    def test_extracts_html_img_self_closing(self):
        """Extracts <img src=\"url\" /> self-closing format."""
        md = '<img src="test.png" />'
        assert_single_figure(md, url="test.png", alt="")

    def test_extracts_multiple_figures(self):
        """Extracts multiple figures from markdown."""
        md = """
        # Paper
        ![Fig 1](fig1.png)
        Some text here
        ![Fig 2](fig2.png)
        <img src="fig3.png" alt="Fig 3">
        """
        figures = parse_figures(md)

        assert len(figures) == 3
        assert figures[0]["url"] == "fig1.png"
        assert figures[1]["url"] == "fig2.png"
        assert figures[2]["url"] == "fig3.png"

    def test_no_duplicates_same_url(self):
        """Avoids duplicates when same URL appears in different formats."""
        md = """
        ![Test](image.png)
        <img src="image.png" alt="Test">
        """
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "image.png"

    def test_empty_alt_text(self):
        """Handles empty alt text."""
        md = "![](empty_alt.png)"
        assert_single_figure(md, url="empty_alt.png", alt="")

    def test_no_figures_returns_empty_list(self):
        """Returns empty list when no figures found."""
        md = "# Just text\n\nNo images here."
        assert parse_figures(md) == []

    def test_parentheses_in_url(self):
        """Handles URLs with parentheses correctly."""
        md = "![Alt](path/to/image(1).png)"
        assert_single_figure(md, url="path/to/image(1).png", alt="Alt")

    def test_parentheses_in_url_nested(self):
        """Handles nested parentheses in URL - common in LaTeX exports."""
        md = "![Alt](path/to/image(2023).png)"
        assert_single_figure(md, url="path/to/image(2023).png", alt="Alt")

    def test_brackets_in_alt_text(self):
        """Handles balanced brackets in alt text."""
        md = "![Alt [nested] text](image.png)"
        assert_single_figure(md, url="image.png", alt="Alt [nested] text")

    def test_escaped_brackets(self):
        """Handles escaped brackets in alt text."""
        md = r"![Alt \[brackets\]](image.png)"
        figures = parse_figures(md)

        assert len(figures) == 1
        assert "Alt" in figures[0]["alt"]
        assert "brackets" in figures[0]["alt"]

    def test_multiple_images_on_one_line(self):
        """Handles multiple images on the same line."""
        md = "![Img1](1.png) Text ![Img2](2.png)"
        figures = parse_figures(md)

        assert len(figures) == 2
        assert figures[0]["url"] == "1.png"
        assert figures[1]["url"] == "2.png"

    def test_html_attributes_multiline(self):
        """Handles HTML img tags spanning multiple lines."""
        md = """<img
            src="multiline.png"
            alt="Multiline" />"""
        assert_single_figure(md, url="multiline.png", alt="Multiline")

    def test_html_single_quotes(self):
        """Handles HTML attributes with single quotes."""
        md = "<img src='single.png' alt='Single'>"
        assert_single_figure(md, url="single.png", alt="Single")

    def test_extracts_markdown_image_with_special_chars_in_alt(self):
        """Extracts alt text with brackets and special chars."""
        md = "![Alt [text] with chars](image.png)"
        assert_single_figure(md, url="image.png", alt="Alt [text] with chars")

    def test_none_input_raises_error(self):
        """Ensures None input raises TypeError."""
        with pytest.raises(TypeError):
            extract_figures_from_markdown(None)

