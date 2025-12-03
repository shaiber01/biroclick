"""
Unit tests for paper_loader markdown_parser module.
"""

import pytest
from pathlib import Path
from src.paper_loader.markdown_parser import (
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
        assert figures[0]['original_match'] == "![Alt text](images/fig1.png)"
    
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
        assert figures[0]['alt'] == ""
    
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
        assert figures[0]['url'] == "fig1.png"
        assert figures[1]['url'] == "fig2.png"
        assert figures[2]['url'] == "fig3.png"
    
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

    def test_no_figures_returns_empty_list(self):
        """Returns empty list when no figures found."""
        md = "# Just text\n\nNo images here."
        figures = extract_figures_from_markdown(md)
        assert figures == []

    def test_parentheses_in_url(self):
        """Handles URLs with parentheses correctly."""
        md = "![Alt](path/to/image(1).png)"
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['url'] == "path/to/image(1).png"

    def test_html_attributes_multiline(self):
        """Handles HTML img tags spanning multiple lines."""
        md = """<img 
            src="multiline.png" 
            alt="Multiline" />"""
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['url'] == "multiline.png"
        assert figures[0]['alt'] == "Multiline"

    def test_html_single_quotes(self):
        """Handles HTML attributes with single quotes."""
        md = "<img src='single.png' alt='Single'>"
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['url'] == "single.png"
        assert figures[0]['alt'] == "Single"

    def test_extracts_markdown_image_with_special_chars_in_alt(self):
        """Extracts alt text with brackets and special chars."""
        md = "![Alt [text] with chars](image.png)"
        figures = extract_figures_from_markdown(md)
        
        assert len(figures) == 1
        assert figures[0]['url'] == "image.png"
        assert figures[0]['alt'] == "Alt [text] with chars"

    def test_none_input_raises_error(self):
        """Ensures None input raises TypeError."""
        with pytest.raises(TypeError):
            extract_figures_from_markdown(None)

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
        md = "A" * 300  # Very long first line
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
        """Ensures None input raises TypeError."""
        with pytest.raises(TypeError):
            extract_paper_title(None)

# ═══════════════════════════════════════════════════════════════════════
# resolve_figure_url Tests
# ═══════════════════════════════════════════════════════════════════════

class TestResolveFigureUrl:
    """Tests for resolve_figure_url function."""
    
    def test_absolute_http_url_unchanged(self):
        result = resolve_figure_url("http://example.com/image.png")
        assert result == "http://example.com/image.png"
    
    def test_relative_with_base_url(self):
        url = "images/fig1.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig1.png"
    
    def test_relative_with_base_path(self):
        url = "images/fig1.png"
        base_path = Path("/home/user/papers")
        result = resolve_figure_url(url, base_path=base_path)
        assert result == "/home/user/papers/images/fig1.png"
    
    def test_base_url_takes_precedence(self):
        url = "fig.png"
        result = resolve_figure_url(
            url,
            base_path=Path("/local/path"),
            base_url="https://remote.com/"
        )
        assert result == "https://remote.com/fig.png"

    def test_parent_directory_traversal(self):
        """Handles .. in paths."""
        url = "../fig.png"
        base_path = Path("/a/b/c")
        result = resolve_figure_url(url, base_path=base_path)
        # Path resolution should handle ..
        expected = base_path.parent / "fig.png"
        assert Path(result).resolve() == expected.resolve()

# ═══════════════════════════════════════════════════════════════════════
# generate_figure_id Tests
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# get_file_extension Tests
# ═══════════════════════════════════════════════════════════════════════

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
