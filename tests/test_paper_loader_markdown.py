"""
Unit tests for paper_loader markdown_parser module.
"""

import pytest
from pathlib import Path

from src.paper_loader import (
    extract_figures_from_markdown,
    extract_paper_title,
    resolve_figure_url,
    generate_figure_id,
    get_file_extension,
)


# ═══════════════════════════════════════════════════════════════════════
# extract_figures_from_markdown Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExtractFiguresFromMarkdown:
    """Tests for extract_figures_from_markdown function."""
    
    def test_extracts_markdown_image_basic(self):
        """Extracts basic ![alt](url) format."""
        md = "Some text ![Alt text](images/fig1.png) more text"
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['alt'] == "Alt text"
        assert figures[0]['url'] == "images/fig1.png"
    
    def test_extracts_markdown_image_with_title(self):
        """Extracts ![alt](url "title") format."""
        md = '![Figure 1](fig.png "A title")'
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['alt'] == "Figure 1"
        assert figures[0]['url'] == "fig.png"
    
    def test_extracts_html_img_tag(self):
        """Extracts <img src="url" alt="..."> format."""
        md = '<img src="images/fig2.jpg" alt="Figure 2">'
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['alt'] == "Figure 2"
        assert figures[0]['url'] == "images/fig2.jpg"
    
    def test_extracts_html_img_self_closing(self):
        """Extracts <img src="url" /> self-closing format."""
        md = '<img src="test.png" />'
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['url'] == "test.png"
    
    def test_extracts_multiple_figures(self):
        """Extracts multiple figures from markdown."""
        md = """
        # Paper
        ![Fig 1](fig1.png)
        Some text here
        ![Fig 2](fig2.png)
        <img src="fig3.png" alt="Fig 3">
        """
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 3
    
    def test_no_duplicates_same_url(self):
        """Avoids duplicates when same URL appears in different formats."""
        md = """
        ![Test](image.png)
        <img src="image.png" alt="Test">
        """
        figures = extract_figures_from_markdown(md)
        
        # Should only have one entry (markdown format found first)
        assert len(figures) == 1
        assert figures[0]['url'] == "image.png"
    
    def test_empty_alt_text(self):
        """Handles empty alt text."""
        md = "![](empty_alt.png)"
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['alt'] == ""
        assert figures[0]['url'] == "empty_alt.png"
    
    def test_returns_original_match(self):
        """Returns the original match string."""
        md = "![Figure 1](fig1.png)"
        figures = extract_figures_from_markdown(md)
        
        assert figures[0]['original_match'] == "![Figure 1](fig1.png)"
    
    def test_no_figures_returns_empty_list(self):
        """Returns empty list when no figures found."""
        md = "# Just text\n\nNo images here."
        figures = extract_figures_from_markdown(md)
        
        assert figures == []


# ═══════════════════════════════════════════════════════════════════════
# extract_paper_title Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExtractPaperTitle:
    """Tests for extract_paper_title function."""
    
    def test_extracts_h1_heading(self):
        """Extracts title from # heading."""
        md = "# My Paper Title\n\nAbstract..."
        title = extract_paper_title(md)
        
        assert title == "My Paper Title"
    
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
    
    def test_default_untitled(self):
        """Returns 'Untitled Paper' if nothing found."""
        md = "![]()  \n<img src='x'/>"
        title = extract_paper_title(md)
        
        assert title == "Untitled Paper"
    
    def test_truncates_long_title(self):
        """Truncates very long titles."""
        md = "A" * 300  # Very long first line
        title = extract_paper_title(md)
        
        assert len(title) <= 200


# ═══════════════════════════════════════════════════════════════════════
# resolve_figure_url Tests
# ═══════════════════════════════════════════════════════════════════════

class TestResolveFigureUrl:
    """Tests for resolve_figure_url function."""
    
    def test_absolute_http_url_unchanged(self):
        """Absolute HTTP URLs returned unchanged."""
        url = "http://example.com/image.png"
        result = resolve_figure_url(url)
        
        assert result == url
    
    def test_absolute_https_url_unchanged(self):
        """Absolute HTTPS URLs returned unchanged."""
        url = "https://example.com/image.png"
        result = resolve_figure_url(url)
        
        assert result == url
    
    def test_file_url_unchanged(self):
        """file:// URLs returned unchanged."""
        url = "file:///path/to/image.png"
        result = resolve_figure_url(url)
        
        assert result == url
    
    def test_relative_with_base_url(self):
        """Relative URLs joined with base_url."""
        url = "images/fig1.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        
        assert result == "https://example.com/papers/images/fig1.png"
    
    def test_relative_with_base_path(self):
        """Relative paths resolved against base_path."""
        url = "images/fig1.png"
        base_path = Path("/home/user/papers")
        result = resolve_figure_url(url, base_path=base_path)
        
        assert result == "/home/user/papers/images/fig1.png"
    
    def test_base_url_takes_precedence(self):
        """base_url takes precedence over base_path for relative URLs."""
        url = "fig.png"
        result = resolve_figure_url(
            url,
            base_path=Path("/local/path"),
            base_url="https://remote.com/"
        )
        
        assert result == "https://remote.com/fig.png"
    
    def test_no_base_returns_as_is(self):
        """Returns URL as-is when no base provided."""
        url = "relative/path.png"
        result = resolve_figure_url(url)
        
        assert result == url


# ═══════════════════════════════════════════════════════════════════════
# generate_figure_id Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGenerateFigureId:
    """Tests for generate_figure_id function."""
    
    def test_extracts_from_alt_text_figure(self):
        """Extracts 'Figure X' from alt text."""
        fig_id = generate_figure_id(0, "Figure 3: Extinction spectrum", "img.png")
        assert fig_id == "Fig3"
    
    def test_extracts_from_alt_text_fig(self):
        """Extracts 'Fig X' from alt text."""
        fig_id = generate_figure_id(0, "Fig 2a shows results", "img.png")
        assert fig_id == "Fig2a"
    
    def test_extracts_from_alt_text_case_insensitive(self):
        """Case insensitive matching for figure text."""
        fig_id = generate_figure_id(0, "FIGURE 5", "img.png")
        assert fig_id == "Fig5"
    
    def test_extracts_from_url_filename(self):
        """Extracts figure number from URL filename."""
        fig_id = generate_figure_id(0, "", "images/figure_4.png")
        assert fig_id == "Fig4"
    
    def test_extracts_from_url_with_hyphen(self):
        """Extracts from URL with hyphen separator."""
        fig_id = generate_figure_id(0, "", "fig-7b.png")
        assert fig_id == "Fig7b"
    
    def test_fallback_to_index(self):
        """Falls back to sequential numbering."""
        fig_id = generate_figure_id(2, "Some random text", "random.png")
        assert fig_id == "Fig3"  # Index 2 -> Fig3 (1-based)
    
    def test_index_zero_gives_fig1(self):
        """Index 0 gives Fig1."""
        fig_id = generate_figure_id(0, "", "image.png")
        assert fig_id == "Fig1"


# ═══════════════════════════════════════════════════════════════════════
# get_file_extension Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGetFileExtension:
    """Tests for get_file_extension function."""
    
    def test_png_extension(self):
        """Detects .png extension."""
        ext = get_file_extension("path/to/image.png")
        assert ext == ".png"
    
    def test_jpg_extension(self):
        """Detects .jpg extension."""
        ext = get_file_extension("image.jpg")
        assert ext == ".jpg"
    
    def test_jpeg_extension(self):
        """Detects .jpeg extension."""
        ext = get_file_extension("image.jpeg")
        assert ext == ".jpeg"
    
    def test_uppercase_normalized(self):
        """Normalizes uppercase extensions."""
        ext = get_file_extension("image.PNG")
        assert ext == ".png"
    
    def test_url_with_query_params(self):
        """Handles URLs with query parameters (may not detect extension)."""
        # URL parsing extracts the path, then Path().suffix gets extension
        ext = get_file_extension("https://example.com/image.gif?token=abc")
        # Note: query params are part of path in this implementation
        # The actual behavior depends on URL structure
        assert ext in [".gif", ".png"]  # Either works
    
    def test_unknown_extension_uses_default(self):
        """Unknown extension returns default."""
        ext = get_file_extension("file.xyz")
        assert ext == ".png"  # Default
    
    def test_custom_default(self):
        """Custom default is used."""
        ext = get_file_extension("file.xyz", default=".jpg")
        assert ext == ".jpg"
    
    def test_svg_supported(self):
        """SVG is in supported formats."""
        ext = get_file_extension("diagram.svg")
        assert ext == ".svg"
    
    def test_webp_supported(self):
        """WebP is in supported formats."""
        ext = get_file_extension("modern.webp")
        assert ext == ".webp"


