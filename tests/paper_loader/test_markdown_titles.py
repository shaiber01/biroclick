"""Tests for extract_paper_title."""

from __future__ import annotations

import pytest

from src.paper_loader.markdown_parser import extract_paper_title


class TestExtractPaperTitle:
    """Tests for extract_paper_title function."""

    def test_extracts_h1_heading(self):
        """Extracts title from # heading."""
        md = "# My Paper Title\n\nAbstract..."
        title = extract_paper_title(md)
        assert title == "My Paper Title"

    def test_extracts_setext_h1(self):
        """Extracts Setext style H1 heading."""
        md = "My Paper Title\n==============\n\nAbstract..."
        title = extract_paper_title(md)
        assert title == "My Paper Title"

    def test_ignores_code_block_comments(self):
        """Ignores lines that look like headers inside code blocks."""
        md = "```python\n# Not a title\n```\n\n# Real Title"
        title = extract_paper_title(md)
        assert title == "Real Title"

    def test_extracts_html_h1(self):
        """Extracts title from <h1> tag."""
        md = "<h1>HTML Title</h1>\n<p>Content</p>"
        title = extract_paper_title(md)
        assert title == "HTML Title"

    def test_strips_html_tags_inside_h1(self):
        """Strips nested HTML tags inside h1."""
        md = "<h1><b>Bold</b> Title</h1>"
        title = extract_paper_title(md)
        assert title == "Bold Title"

    def test_fallback_first_line(self):
        """Falls back to first non-empty line if no heading."""
        md = "First line as title\n\nMore content"
        title = extract_paper_title(md)
        assert title == "First line as title"

    def test_skips_image_lines(self):
        """Skips lines starting with ! (images)."""
        md = "![Image](img.png)\nActual Title"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_skips_html_tags_fallback(self):
        """Skips lines starting with < (html tags) in fallback."""
        md = "<div class='meta'>Data</div>\nActual Title"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_default_untitled(self):
        """Returns 'Untitled Paper' if nothing found."""
        md = "![]()  \n<img src='x'/>\n"
        title = extract_paper_title(md)
        assert title == "Untitled Paper"

    def test_truncates_long_title(self):
        """Truncates very long titles."""
        md = "A" * 300
        title = extract_paper_title(md)
        assert len(title) <= 200

    def test_h1_with_extra_spaces(self):
        """Handles H1 with extra spaces."""
        md = "#    Spaced Title    \n"
        title = extract_paper_title(md)
        assert title == "Spaced Title"

    def test_h1_not_at_start_of_file(self):
        """Finds H1 even if not first line."""
        md = "Some meta data\n\n# The Real Title\n"
        title = extract_paper_title(md)
        assert title == "The Real Title"

    def test_empty_input(self):
        """Handles empty input string."""
        title = extract_paper_title("")
        assert title == "Untitled Paper"

    def test_none_input_raises_error(self):
        """Ensures None input raises TypeError or AttributeError."""
        with pytest.raises((TypeError, AttributeError)):
            extract_paper_title(None)

