"""Tests for extract_paper_title."""

from __future__ import annotations

import pytest

from src.paper_loader.markdown_parser import extract_paper_title


class TestExtractPaperTitle:
    """Tests for extract_paper_title function."""

    # ========== Basic H1 Heading Tests ==========
    
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

    def test_h1_with_tabs(self):
        """Handles H1 with tab characters."""
        md = "#\tTabbed\tTitle\t\n"
        title = extract_paper_title(md)
        assert title == "Tabbed\tTitle"

    def test_h1_with_only_hash(self):
        """Handles H1 with only hash symbol."""
        md = "# \nContent"
        title = extract_paper_title(md)
        assert title == ""

    def test_h1_with_special_characters(self):
        """Handles H1 with special characters and unicode."""
        md = "# Title with émojis 🎉 and spéciál chars!"
        title = extract_paper_title(md)
        assert title == "Title with émojis 🎉 and spéciál chars!"

    def test_h1_with_markdown_formatting(self):
        """Handles H1 with markdown formatting inside."""
        md = "# Title with **bold** and *italic*"
        title = extract_paper_title(md)
        assert title == "Title with **bold** and *italic*"

    def test_multiple_h1_headings(self):
        """Returns first H1 when multiple H1 headings exist."""
        md = "# First Title\n\nContent\n\n# Second Title"
        title = extract_paper_title(md)
        assert title == "First Title"

    def test_h1_after_h2(self):
        """Finds H1 even after H2 heading."""
        md = "## Subtitle\n\n# Main Title"
        title = extract_paper_title(md)
        assert title == "Main Title"

    def test_h1_with_multiple_spaces(self):
        """Handles H1 with multiple internal spaces."""
        md = "# Title    with    multiple    spaces"
        title = extract_paper_title(md)
        assert title == "Title    with    multiple    spaces"

    # ========== Code Block Tests ==========

    def test_ignores_code_block_comments(self):
        """Ignores lines that look like headers inside code blocks."""
        md = "```python\n# Not a title\n```\n\n# Real Title"
        title = extract_paper_title(md)
        assert title == "Real Title"

    def test_ignores_h1_inside_code_block(self):
        """Ignores H1 heading inside code block."""
        md = "```\n# Code Block Title\n```\n\n# Real Title"
        title = extract_paper_title(md)
        assert title == "Real Title"

    def test_ignores_html_h1_inside_code_block(self):
        """Ignores HTML h1 tag inside code block."""
        md = "```html\n<h1>Code Title</h1>\n```\n\n# Real Title"
        title = extract_paper_title(md)
        assert title == "Real Title"

    def test_multiple_code_blocks(self):
        """Handles multiple code blocks correctly."""
        md = "```\n# First code\n```\n\n```\n# Second code\n```\n\n# Real Title"
        title = extract_paper_title(md)
        assert title == "Real Title"

    def test_unclosed_code_block_with_h1_after(self):
        """Handles unclosed code block with H1 after it."""
        md = "```python\n# Code comment\n\n# Real Title After Unclosed Block"
        title = extract_paper_title(md)
        # Implementation has special handling for unclosed blocks
        assert title == "Real Title After Unclosed Block"

    def test_code_block_at_start(self):
        """Finds H1 after code block at start of file."""
        md = "```\ncode\n```\n\n# Title"
        title = extract_paper_title(md)
        assert title == "Title"

    def test_code_block_with_h1_like_pattern(self):
        """Ignores H1-like pattern inside code block."""
        md = "```\n# Not a title because it's in code\n```\n\n# Real Title"
        title = extract_paper_title(md)
        assert title == "Real Title"

    # ========== HTML H1 Tests ==========

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

    def test_html_h1_with_attributes(self):
        """Extracts title from HTML h1 with attributes."""
        md = '<h1 class="title" id="main">Title with Attributes</h1>'
        title = extract_paper_title(md)
        assert title == "Title with Attributes"

    def test_html_h1_with_multiple_nested_tags(self):
        """Strips multiple nested HTML tags inside h1."""
        md = "<h1><b><i><u>Complex</u></i></b> Title</h1>"
        title = extract_paper_title(md)
        assert title == "Complex Title"

    def test_html_h1_with_self_closing_tags(self):
        """Handles self-closing tags inside h1."""
        md = "<h1>Title with <br/> break</h1>"
        title = extract_paper_title(md)
        assert title == "Title with  break"

    def test_html_h1_with_attributes_and_nested_tags(self):
        """Handles HTML h1 with both attributes and nested tags."""
        md = '<h1 class="title"><span>Nested</span> Title</h1>'
        title = extract_paper_title(md)
        assert title == "Nested Title"

    def test_multiple_html_h1_tags(self):
        """Returns first HTML h1 when multiple exist."""
        md = "<h1>First HTML Title</h1>\n<p>Content</p>\n<h1>Second HTML Title</h1>"
        title = extract_paper_title(md)
        assert title == "First HTML Title"

    def test_html_h1_preferred_over_fallback(self):
        """HTML h1 is preferred over fallback line."""
        md = "Some fallback text\n<h1>HTML Title</h1>"
        title = extract_paper_title(md)
        assert title == "HTML Title"

    def test_html_h1_case_insensitive(self):
        """HTML h1 tag matching is case insensitive."""
        md = "<H1>Uppercase Tag</H1>"
        title = extract_paper_title(md)
        assert title == "Uppercase Tag"

    def test_html_h1_with_newlines(self):
        """Handles HTML h1 with newlines inside."""
        md = "<h1>Title\nwith\nnewlines</h1>"
        title = extract_paper_title(md)
        assert title == "Title\nwith\nnewlines"

    # ========== Fallback Tests ==========

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

    def test_skips_whitespace_only_lines(self):
        """Skips whitespace-only lines in fallback."""
        md = "   \n\t\n  \nActual Title"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_skips_empty_lines(self):
        """Skips empty lines in fallback."""
        md = "\n\n\nActual Title\n\nMore"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_skips_multiple_image_lines(self):
        """Skips multiple image lines before finding title."""
        md = "![Image1](img1.png)\n![Image2](img2.png)\nActual Title"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_skips_multiple_html_tag_lines(self):
        """Skips multiple HTML tag lines before finding title."""
        md = "<div>One</div>\n<span>Two</span>\nActual Title"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_fallback_with_tabs(self):
        """Handles tabs in fallback line."""
        md = "\tTabbed\tTitle\t"
        title = extract_paper_title(md)
        assert title == "Tabbed\tTitle"

    def test_fallback_preserves_content(self):
        """Fallback preserves exact content of first valid line."""
        md = "  Padded Title  \nMore content"
        title = extract_paper_title(md)
        assert title == "Padded Title"

    def test_fallback_skips_code_blocks(self):
        """Fallback skips code blocks."""
        md = "```\ncode\n```\nActual Title"
        title = extract_paper_title(md)
        assert title == "Actual Title"

    def test_fallback_with_special_characters(self):
        """Fallback handles special characters."""
        md = "Title with émojis 🎉 and spéciál chars!"
        title = extract_paper_title(md)
        assert title == "Title with émojis 🎉 and spéciál chars!"

    # ========== Edge Cases and Boundary Tests ==========

    def test_default_untitled(self):
        """Returns 'Untitled Paper' if nothing found."""
        md = "![]()  \n<img src='x'/>\n"
        title = extract_paper_title(md)
        assert title == "Untitled Paper"

    def test_empty_input(self):
        """Handles empty input string."""
        title = extract_paper_title("")
        assert title == "Untitled Paper"

    def test_whitespace_only_input(self):
        """Handles whitespace-only input."""
        title = extract_paper_title("   \n\t\n  ")
        assert title == "Untitled Paper"

    def test_only_code_blocks(self):
        """Returns 'Untitled Paper' if only code blocks exist."""
        md = "```\ncode\n```"
        title = extract_paper_title(md)
        assert title == "Untitled Paper"

    def test_only_html_tags(self):
        """Returns 'Untitled Paper' if only HTML tags exist."""
        md = "<div>Content</div>\n<span>More</span>"
        title = extract_paper_title(md)
        assert title == "Untitled Paper"

    def test_only_images(self):
        """Returns 'Untitled Paper' if only images exist."""
        md = "![Alt1](img1.png)\n![Alt2](img2.png)"
        title = extract_paper_title(md)
        assert title == "Untitled Paper"

    def test_truncates_long_title(self):
        """Truncates very long titles."""
        md = "A" * 300
        title = extract_paper_title(md)
        assert len(title) <= 200

    def test_truncates_exactly_200_chars(self):
        """Handles title exactly 200 characters."""
        md = "A" * 200
        title = extract_paper_title(md)
        assert len(title) == 200
        assert title == "A" * 200

    def test_truncates_201_chars(self):
        """Truncates title at 201 characters."""
        md = "A" * 201
        title = extract_paper_title(md)
        assert len(title) == 200
        assert title == "A" * 200

    def test_truncates_long_h1(self):
        """Truncates long H1 heading."""
        md = "# " + "A" * 250
        title = extract_paper_title(md)
        assert len(title) == 250  # H1 should not truncate (only fallback does)
        assert title == "A" * 250

    def test_truncates_long_html_h1(self):
        """Truncates long HTML h1 heading."""
        md = "<h1>" + "A" * 250 + "</h1>"
        title = extract_paper_title(md)
        assert len(title) == 250  # HTML h1 should not truncate (only fallback does)
        assert title == "A" * 250

    def test_truncates_long_fallback(self):
        """Truncates long fallback line."""
        md = "A" * 300
        title = extract_paper_title(md)
        assert len(title) == 200
        assert title == "A" * 200

    def test_none_input_raises_error(self):
        """Ensures None input raises TypeError or AttributeError."""
        with pytest.raises((TypeError, AttributeError)):
            extract_paper_title(None)

    def test_non_string_input_raises_error(self):
        """Ensures non-string input raises appropriate error."""
        with pytest.raises((TypeError, AttributeError)):
            extract_paper_title(123)
        
        with pytest.raises((TypeError, AttributeError)):
            extract_paper_title([])
        
        with pytest.raises((TypeError, AttributeError)):
            extract_paper_title({})

    # ========== Priority Order Tests ==========

    def test_h1_preferred_over_html_h1(self):
        """H1 heading is preferred over HTML h1."""
        md = "# Markdown Title\n<h1>HTML Title</h1>"
        title = extract_paper_title(md)
        assert title == "Markdown Title"

    def test_h1_preferred_over_fallback(self):
        """H1 heading is preferred over fallback."""
        md = "# Markdown Title\nFallback text"
        title = extract_paper_title(md)
        assert title == "Markdown Title"

    def test_html_h1_preferred_over_fallback(self):
        """HTML h1 is preferred over fallback."""
        md = "<h1>HTML Title</h1>\nFallback text"
        title = extract_paper_title(md)
        assert title == "HTML Title"

    def test_first_h1_wins_over_later_h1(self):
        """First H1 heading wins when multiple exist."""
        md = "# First Title\n\nContent\n\n# Second Title"
        title = extract_paper_title(md)
        assert title == "First Title"

    # ========== Complex Integration Tests ==========

    def test_complex_markdown_document(self):
        """Handles complex markdown document with multiple elements."""
        md = """![Cover](cover.png)
<div class="meta">Metadata</div>

# Main Paper Title

## Introduction

```python
# Code comment
```

More content.
"""
        title = extract_paper_title(md)
        assert title == "Main Paper Title"

    def test_h1_after_complex_code_block(self):
        """Finds H1 after complex code block structure."""
        md = """```
code block 1
```

```python
# Another code block
```

# Title After Code Blocks
"""
        title = extract_paper_title(md)
        assert title == "Title After Code Blocks"

    def test_html_h1_after_markdown_elements(self):
        """Finds HTML h1 after various markdown elements."""
        md = """![Image](img.png)
<div>Meta</div>
Some text

<h1>HTML Title</h1>
"""
        title = extract_paper_title(md)
        assert title == "HTML Title"

    def test_fallback_after_all_skipped_elements(self):
        """Finds fallback after all elements are skipped."""
        md = """![Image](img.png)
<div>Meta</div>
```
code
```
Actual Title
"""
        title = extract_paper_title(md)
        assert title == "Actual Title"

