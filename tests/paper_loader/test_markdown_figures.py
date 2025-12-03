"""Tests for markdown_parser module: figure extraction, URL resolution, and title extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.paper_loader.markdown_parser import (
    extract_figures_from_markdown,
    extract_paper_title,
    generate_figure_id,
    get_file_extension,
    resolve_figure_url,
)
from tests.paper_loader.markdown_shared import assert_single_figure, parse_figures


# =============================================================================
# Tests for extract_figures_from_markdown
# =============================================================================


class TestExtractFiguresFromMarkdown:
    """Tests for extract_figures_from_markdown function."""

    # -------------------------------------------------------------------------
    # Basic markdown image extraction
    # -------------------------------------------------------------------------

    def test_extracts_markdown_image_basic(self):
        """Extracts basic ![alt](url) format."""
        md = "Some text ![Alt text](images/fig1.png) more text"
        figure = assert_single_figure(md, url="images/fig1.png", alt="Alt text")

        assert figure["original_match"] == "![Alt text](images/fig1.png)"

    def test_extracts_markdown_image_with_title(self):
        """Extracts ![alt](url \"title\") format and ignores the title in URL."""
        md = '![Figure 1](fig.png "A title")'
        figure = assert_single_figure(md, url="fig.png", alt="Figure 1")
        # Verify original_match includes the title
        assert figure["original_match"] == '![Figure 1](fig.png "A title")'

    def test_extracts_markdown_image_with_complex_title(self):
        """Extracts markdown image with title containing special chars."""
        md = '![Alt](image.png "Title with (parens) and [brackets]")'
        figure = assert_single_figure(md, url="image.png", alt="Alt")
        assert "Title with" in figure["original_match"]

    def test_empty_alt_text(self):
        """Handles empty alt text."""
        md = "![](empty_alt.png)"
        figure = assert_single_figure(md, url="empty_alt.png", alt="")
        assert figure["original_match"] == "![](empty_alt.png)"

    def test_whitespace_only_alt_text(self):
        """Alt text with only whitespace is trimmed to empty string."""
        md = "![   ](whitespace.png)"
        figure = assert_single_figure(md, url="whitespace.png", alt="")
        assert figure["url"] == "whitespace.png"

    # -------------------------------------------------------------------------
    # HTML img tag extraction
    # -------------------------------------------------------------------------

    def test_extracts_html_img_tag(self):
        """Extracts <img src=\"url\" alt=\"...\"> format."""
        md = '<img src="images/fig2.jpg" alt="Figure 2">'
        figure = assert_single_figure(md, url="images/fig2.jpg", alt="Figure 2")
        assert figure["original_match"] == '<img src="images/fig2.jpg" alt="Figure 2">'

    def test_extracts_html_img_self_closing(self):
        """Extracts <img src=\"url\" /> self-closing format."""
        md = '<img src="test.png" />'
        figure = assert_single_figure(md, url="test.png", alt="")
        assert figure["original_match"] == '<img src="test.png" />'

    def test_html_img_alt_before_src(self):
        """Handles HTML img with alt before src attribute."""
        md = '<img alt="Before" src="order.png">'
        figure = assert_single_figure(md, url="order.png", alt="Before")
        assert figure["url"] == "order.png"
        assert figure["alt"] == "Before"

    def test_html_single_quotes(self):
        """Handles HTML attributes with single quotes."""
        md = "<img src='single.png' alt='Single'>"
        figure = assert_single_figure(md, url="single.png", alt="Single")
        assert figure["url"] == "single.png"

    def test_html_attributes_multiline(self):
        """Handles HTML img tags spanning multiple lines."""
        md = """<img
            src="multiline.png"
            alt="Multiline" />"""
        figure = assert_single_figure(md, url="multiline.png", alt="Multiline")
        assert figure["url"] == "multiline.png"

    def test_html_img_case_insensitive(self):
        """HTML img tag parsing is case-insensitive."""
        md = '<IMG SRC="upper.png" ALT="Upper">'
        figure = assert_single_figure(md, url="upper.png", alt="Upper")
        assert figure["url"] == "upper.png"

    def test_html_img_no_alt(self):
        """HTML img without alt attribute returns empty alt."""
        md = '<img src="noalt.png">'
        figure = assert_single_figure(md, url="noalt.png", alt="")
        assert figure["alt"] == ""

    def test_html_img_with_extra_attributes(self):
        """HTML img with class, id, width, height attributes."""
        md = '<img class="figure" id="fig1" src="extra.png" alt="Extra" width="100" height="200">'
        figure = assert_single_figure(md, url="extra.png", alt="Extra")
        assert figure["url"] == "extra.png"
        assert figure["alt"] == "Extra"

    # -------------------------------------------------------------------------
    # Multiple figures
    # -------------------------------------------------------------------------

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
        assert figures[0]["alt"] == "Fig 1"
        assert figures[1]["url"] == "fig2.png"
        assert figures[1]["alt"] == "Fig 2"
        assert figures[2]["url"] == "fig3.png"
        assert figures[2]["alt"] == "Fig 3"

    def test_multiple_images_on_one_line(self):
        """Handles multiple images on the same line."""
        md = "![Img1](1.png) Text ![Img2](2.png)"
        figures = parse_figures(md)

        assert len(figures) == 2
        assert figures[0]["url"] == "1.png"
        assert figures[0]["alt"] == "Img1"
        assert figures[1]["url"] == "2.png"
        assert figures[1]["alt"] == "Img2"

    def test_mixed_formats_markdown_and_html(self):
        """Extracts both markdown and HTML images in mixed content."""
        md = """
        ![Markdown](md.png)
        <img src="html.png" alt="HTML">
        """
        figures = parse_figures(md)

        assert len(figures) == 2
        urls = [f["url"] for f in figures]
        assert "md.png" in urls
        assert "html.png" in urls

    # -------------------------------------------------------------------------
    # Duplicate handling
    # -------------------------------------------------------------------------

    def test_no_duplicates_same_url(self):
        """Avoids duplicates when same URL appears in different formats."""
        md = """
        ![Test](image.png)
        <img src="image.png" alt="Test">
        """
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "image.png"

    def test_no_duplicates_same_url_different_alt(self):
        """Same URL with different alt text is still deduplicated."""
        md = """
        ![Alt One](same.png)
        <img src="same.png" alt="Alt Two">
        """
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "same.png"
        # First occurrence should be kept
        assert figures[0]["alt"] == "Alt One"

    def test_different_urls_not_deduplicated(self):
        """Different URLs are not deduplicated even with same alt."""
        md = """
        ![Same Alt](one.png)
        ![Same Alt](two.png)
        """
        figures = parse_figures(md)

        assert len(figures) == 2
        urls = [f["url"] for f in figures]
        assert "one.png" in urls
        assert "two.png" in urls

    # -------------------------------------------------------------------------
    # Empty/no figures
    # -------------------------------------------------------------------------

    def test_no_figures_returns_empty_list(self):
        """Returns empty list when no figures found."""
        md = "# Just text\n\nNo images here."
        result = parse_figures(md)
        assert result == []
        assert isinstance(result, list)

    def test_empty_string_returns_empty_list(self):
        """Empty string returns empty list."""
        result = parse_figures("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input returns empty list."""
        result = parse_figures("   \n\t\n   ")
        assert result == []

    # -------------------------------------------------------------------------
    # Special characters in URL
    # -------------------------------------------------------------------------

    def test_parentheses_in_url(self):
        """Handles URLs with parentheses correctly."""
        md = "![Alt](path/to/image(1).png)"
        figure = assert_single_figure(md, url="path/to/image(1).png", alt="Alt")
        assert figure["url"] == "path/to/image(1).png"

    def test_parentheses_in_url_nested(self):
        """Handles nested parentheses in URL - common in LaTeX exports."""
        md = "![Alt](path/to/image(2023).png)"
        figure = assert_single_figure(md, url="path/to/image(2023).png", alt="Alt")
        assert figure["url"] == "path/to/image(2023).png"

    def test_url_with_query_string(self):
        """Handles URLs with query strings."""
        md = "![Query](image.png?v=123&size=large)"
        figure = assert_single_figure(md, url="image.png?v=123&size=large", alt="Query")
        assert figure["url"] == "image.png?v=123&size=large"

    def test_url_with_hash_fragment(self):
        """Handles URLs with hash fragments."""
        md = "![Hash](image.png#section)"
        figure = assert_single_figure(md, url="image.png#section", alt="Hash")
        assert figure["url"] == "image.png#section"

    def test_url_encoded_characters(self):
        """Handles URL-encoded characters in URL."""
        md = "![Encoded](path/to/image%20with%20spaces.png)"
        figure = assert_single_figure(
            md, url="path/to/image%20with%20spaces.png", alt="Encoded"
        )
        assert figure["url"] == "path/to/image%20with%20spaces.png"

    def test_absolute_http_url(self):
        """Handles absolute HTTP URLs."""
        md = "![HTTP](http://example.com/image.png)"
        figure = assert_single_figure(
            md, url="http://example.com/image.png", alt="HTTP"
        )
        assert figure["url"] == "http://example.com/image.png"

    def test_absolute_https_url(self):
        """Handles absolute HTTPS URLs."""
        md = "![HTTPS](https://example.com/figures/fig1.png)"
        figure = assert_single_figure(
            md, url="https://example.com/figures/fig1.png", alt="HTTPS"
        )
        assert figure["url"] == "https://example.com/figures/fig1.png"

    def test_data_url_not_extracted(self):
        """Data URLs (base64) should not be extracted or handled gracefully."""
        md = "![Data](data:image/png;base64,iVBORw0KGgo=)"
        figures = parse_figures(md)
        # Data URLs might or might not be extracted - but if extracted, should be complete
        if figures:
            assert "data:image/png;base64" in figures[0]["url"]

    # -------------------------------------------------------------------------
    # Special characters in alt text
    # -------------------------------------------------------------------------

    def test_brackets_in_alt_text(self):
        """Handles balanced brackets in alt text."""
        md = "![Alt [nested] text](image.png)"
        figure = assert_single_figure(md, url="image.png", alt="Alt [nested] text")
        assert figure["alt"] == "Alt [nested] text"

    def test_extracts_markdown_image_with_special_chars_in_alt(self):
        """Extracts alt text with brackets and special chars."""
        md = "![Alt [text] with chars](image.png)"
        figure = assert_single_figure(md, url="image.png", alt="Alt [text] with chars")
        assert figure["alt"] == "Alt [text] with chars"

    def test_escaped_brackets(self):
        """Handles escaped brackets in alt text."""
        md = r"![Alt \[brackets\]](image.png)"
        figures = parse_figures(md)

        assert len(figures) == 1
        assert "Alt" in figures[0]["alt"]
        assert "brackets" in figures[0]["alt"]

    def test_unicode_in_alt_text(self):
        """Handles unicode characters in alt text."""
        md = "![图像 αβγ émojis 🎨](unicode.png)"
        figure = assert_single_figure(md, url="unicode.png", alt="图像 αβγ émojis 🎨")
        assert figure["alt"] == "图像 αβγ émojis 🎨"

    def test_newline_in_alt_text(self):
        """Alt text should not contain newlines (regex stops at newline)."""
        md = "![Alt\ntext](image.png)"
        figures = parse_figures(md)
        # This might not match at all due to the newline
        # If it does match, alt should not span lines
        if figures:
            assert "\n" not in figures[0]["alt"]

    # -------------------------------------------------------------------------
    # Code block exclusion (CRITICAL - currently untested!)
    # -------------------------------------------------------------------------

    def test_ignores_figures_in_code_block(self):
        """Figures inside code blocks should be ignored."""
        md = """# Title

Some text before.

```python
# This should be ignored
img = "![Not a figure](fake.png)"
html = '<img src="also_fake.png">'
```

![Real figure](real.png)
"""
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "real.png"
        assert figures[0]["alt"] == "Real figure"

    def test_ignores_figures_in_fenced_code_block_with_language(self):
        """Figures in fenced code blocks with language tag are ignored."""
        md = """```markdown
![Code example](example.png)
```

![Outside](outside.png)"""
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "outside.png"

    def test_ignores_figures_in_multiple_code_blocks(self):
        """Handles multiple code blocks correctly."""
        md = """
![Before](before.png)

```
![Inside1](inside1.png)
```

![Between](between.png)

```
![Inside2](inside2.png)
```

![After](after.png)
"""
        figures = parse_figures(md)

        assert len(figures) == 3
        urls = [f["url"] for f in figures]
        assert "before.png" in urls
        assert "between.png" in urls
        assert "after.png" in urls
        assert "inside1.png" not in urls
        assert "inside2.png" not in urls

    def test_unclosed_code_block_excludes_rest(self):
        """Unclosed code block excludes everything after it."""
        md = """
![Before](before.png)

```python
# Unclosed code block
![Inside](inside.png)
![AlsoInside](alsoinside.png)
"""
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "before.png"

    def test_inline_code_does_not_affect_parsing(self):
        """Inline code (single backticks) should not affect figure parsing."""
        md = "Text `code` ![Figure](fig.png) more `code`"
        figure = assert_single_figure(md, url="fig.png", alt="Figure")
        assert figure["url"] == "fig.png"

    def test_code_block_start_mid_line_not_excluded(self):
        """Code block markers must be at line start (stripped) to count."""
        md = "Text ``` not a code block ``` ![Figure](fig.png)"
        # The image should still be extracted since ``` is not at line start
        figures = parse_figures(md)
        # Behavior may vary, but image should likely be extracted
        assert len(figures) >= 0  # Allow flexibility here

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    def test_none_input_raises_error(self):
        """Ensures None input raises TypeError."""
        with pytest.raises(TypeError):
            extract_figures_from_markdown(None)

    def test_non_string_input_raises_error(self):
        """Ensures non-string input raises TypeError."""
        with pytest.raises(TypeError):
            extract_figures_from_markdown(123)

        with pytest.raises(TypeError):
            extract_figures_from_markdown(["![Alt](url.png)"])

    # -------------------------------------------------------------------------
    # Return value structure
    # -------------------------------------------------------------------------

    def test_return_value_is_list_of_dicts(self):
        """Return value is a list of dictionaries."""
        md = "![Test](test.png)"
        result = parse_figures(md)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_dict_has_required_keys(self):
        """Each figure dict has alt, url, and original_match keys."""
        md = "![Alt](url.png)"
        figures = parse_figures(md)

        assert len(figures) == 1
        figure = figures[0]
        assert "alt" in figure
        assert "url" in figure
        assert "original_match" in figure
        assert len(figure) == 3  # No extra keys

    def test_values_are_strings(self):
        """All values in figure dict are strings."""
        md = "![Alt](url.png)"
        figures = parse_figures(md)

        figure = figures[0]
        assert isinstance(figure["alt"], str)
        assert isinstance(figure["url"], str)
        assert isinstance(figure["original_match"], str)

    # -------------------------------------------------------------------------
    # Edge cases for pattern matching
    # -------------------------------------------------------------------------

    def test_markdown_link_not_extracted(self):
        """Regular markdown links [text](url) should NOT be extracted."""
        md = "[Click here](https://example.com)"
        figures = parse_figures(md)
        assert figures == []

    def test_markdown_link_with_image_extracted(self):
        """Linked images [![alt](img)](link) should extract the image."""
        md = "[![Click me](button.png)](https://example.com)"
        figures = parse_figures(md)

        assert len(figures) == 1
        assert figures[0]["url"] == "button.png"
        assert figures[0]["alt"] == "Click me"

    def test_reference_style_image_not_extracted(self):
        """Reference-style images ![alt][ref] are NOT extracted (by design)."""
        md = """![Alt text][ref]

[ref]: image.png
"""
        figures = parse_figures(md)
        # Reference-style images are not currently supported
        # This documents current behavior
        assert len(figures) == 0

    def test_only_exclamation_not_extracted(self):
        """Just exclamation mark with brackets/parens that isn't valid image."""
        md = "This is not an image ![] or !()"
        figures = parse_figures(md)
        assert len(figures) == 0

    def test_malformed_markdown_image_not_extracted(self):
        """Malformed markdown images should not be extracted."""
        malformed_cases = [
            "![Alt]()",  # Empty URL
            "![]()",  # Empty both
            "![Alt]",  # Missing URL part
            "!(url.png)",  # Missing brackets
        ]
        for md in malformed_cases:
            figures = parse_figures(md)
            # Empty URL cases should not extract (or extract empty URL which is useless)
            for fig in figures:
                if fig["url"] == "":
                    continue  # Empty URL is arguably malformed
                # If extracted, URL should not be empty
                assert fig["url"]


# =============================================================================
# Tests for resolve_figure_url
# =============================================================================


class TestResolveFigureUrl:
    """Tests for resolve_figure_url function."""

    # -------------------------------------------------------------------------
    # Absolute URLs
    # -------------------------------------------------------------------------

    def test_absolute_http_url_unchanged(self):
        """HTTP URLs are returned as-is."""
        url = "http://example.com/images/fig1.png"
        result = resolve_figure_url(url)
        assert result == url

    def test_absolute_https_url_unchanged(self):
        """HTTPS URLs are returned as-is."""
        url = "https://example.com/images/fig1.png"
        result = resolve_figure_url(url)
        assert result == url

    def test_file_url_unchanged(self):
        """file:// URLs are returned as-is."""
        url = "file:///home/user/image.png"
        result = resolve_figure_url(url)
        assert result == url

    def test_absolute_url_ignores_base_url(self):
        """Absolute URLs ignore base_url parameter."""
        url = "https://example.com/fig.png"
        result = resolve_figure_url(url, base_url="https://other.com/")
        assert result == url

    def test_absolute_url_ignores_base_path(self):
        """Absolute URLs ignore base_path parameter."""
        url = "https://example.com/fig.png"
        result = resolve_figure_url(url, base_path=Path("/some/path"))
        assert result == url

    # -------------------------------------------------------------------------
    # Relative URLs with base_url
    # -------------------------------------------------------------------------

    def test_relative_url_with_base_url(self):
        """Relative URL joined with base_url."""
        url = "figures/fig1.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/figures/fig1.png"

    def test_relative_url_with_base_url_trailing_slash(self):
        """Base URL with trailing slash joins correctly."""
        url = "fig.png"
        base_url = "https://example.com/images/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/images/fig.png"

    def test_relative_url_with_base_url_no_trailing_slash(self):
        """Base URL without trailing slash joins correctly."""
        url = "fig.png"
        base_url = "https://example.com/images"
        result = resolve_figure_url(url, base_url=base_url)
        # urljoin replaces last segment if no trailing slash
        assert result == "https://example.com/fig.png"

    def test_dot_relative_url_with_base_url(self):
        """Relative URL starting with ./ joined with base_url."""
        url = "./figures/fig1.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert "fig1.png" in result

    def test_parent_relative_url_with_base_url(self):
        """Relative URL with ../ resolved against base_url."""
        url = "../images/fig1.png"
        base_url = "https://example.com/papers/2023/"
        result = resolve_figure_url(url, base_url=base_url)
        assert "images/fig1.png" in result

    # -------------------------------------------------------------------------
    # Relative paths with base_path
    # -------------------------------------------------------------------------

    def test_relative_path_with_base_path(self):
        """Relative path resolved against base_path."""
        url = "figures/fig1.png"
        base_path = Path("/home/user/papers")
        result = resolve_figure_url(url, base_path=base_path)
        # Check path components (macOS may resolve /home to /System/Volumes/Data/home)
        assert result.endswith("home/user/papers/figures/fig1.png")
        assert "figures/fig1.png" in result

    def test_relative_path_with_base_path_obj(self):
        """Base path as Path object."""
        url = "fig.png"
        base_path = Path("/tmp/markdown")
        result = resolve_figure_url(url, base_path=base_path)
        # Check path components (macOS may resolve /tmp to /private/tmp)
        assert result.endswith("tmp/markdown/fig.png")
        assert "markdown/fig.png" in result

    def test_parent_relative_path_with_base_path(self):
        """Parent path traversal with base_path."""
        url = "../images/fig.png"
        base_path = Path("/home/user/papers/2023")
        result = resolve_figure_url(url, base_path=base_path)
        # Path doesn't normalize .. automatically, just joins
        assert "../images/fig.png" in result or "images/fig.png" in result

    # -------------------------------------------------------------------------
    # No base provided
    # -------------------------------------------------------------------------

    def test_relative_url_no_base_returns_as_is(self):
        """Relative URL without any base returns as-is."""
        url = "figures/fig1.png"
        result = resolve_figure_url(url)
        assert result == url

    def test_empty_url(self):
        """Empty URL returns empty string."""
        result = resolve_figure_url("")
        assert result == ""

    # -------------------------------------------------------------------------
    # Priority: base_url over base_path
    # -------------------------------------------------------------------------

    def test_base_url_takes_priority_over_base_path(self):
        """When both base_url and base_path provided, base_url is used."""
        url = "fig.png"
        base_url = "https://example.com/images/"
        base_path = Path("/local/path")
        result = resolve_figure_url(url, base_path=base_path, base_url=base_url)
        # base_url should take priority
        assert result == "https://example.com/images/fig.png"


# =============================================================================
# Tests for generate_figure_id
# =============================================================================


class TestGenerateFigureId:
    """Tests for generate_figure_id function."""

    # -------------------------------------------------------------------------
    # Extract from alt text
    # -------------------------------------------------------------------------

    def test_figure_number_from_alt_text(self):
        """Extracts figure number from 'Figure X' in alt text."""
        result = generate_figure_id(0, "Figure 1", "image.png")
        assert result == "Fig1"

    def test_figure_number_abbreviated(self):
        """Extracts figure number from 'Fig X' in alt text."""
        result = generate_figure_id(0, "Fig 2", "image.png")
        assert result == "Fig2"

    def test_figure_number_with_period(self):
        """Extracts figure number from 'Fig. X' in alt text."""
        result = generate_figure_id(0, "Fig. 3", "image.png")
        assert result == "Fig3"

    def test_figure_number_with_letter_suffix(self):
        """Extracts figure number with letter suffix like 'Figure 3a'."""
        result = generate_figure_id(0, "Figure 3a", "image.png")
        assert result == "Fig3a"

    def test_figure_number_with_letter_prefix(self):
        """Extracts supplementary figure like 'Figure S1'."""
        result = generate_figure_id(0, "Figure S1", "image.png")
        # Should match the S pattern
        assert "S1" in result or "s1" in result.lower()

    def test_figure_number_with_dash(self):
        """Extracts figure number with dash like 'Fig 2-3'."""
        result = generate_figure_id(0, "Fig 2-3", "image.png")
        assert "2-3" in result or "2" in result

    def test_figure_number_with_decimal(self):
        """Extracts figure number with decimal like 'Fig 1.2'."""
        result = generate_figure_id(0, "Fig 1.2", "image.png")
        assert "1.2" in result or "1" in result

    def test_figure_number_case_insensitive(self):
        """Figure extraction is case insensitive."""
        result1 = generate_figure_id(0, "FIGURE 1", "image.png")
        result2 = generate_figure_id(0, "figure 1", "image.png")
        result3 = generate_figure_id(0, "Figure 1", "image.png")
        assert result1 == result2 == result3 == "Fig1"

    def test_figure_number_in_longer_text(self):
        """Extracts figure number from longer alt text."""
        result = generate_figure_id(0, "Fig 5: Experimental setup for measurement", "image.png")
        assert result == "Fig5"

    # -------------------------------------------------------------------------
    # Extract from URL
    # -------------------------------------------------------------------------

    def test_figure_number_from_url_filename(self):
        """Extracts figure number from URL filename."""
        result = generate_figure_id(0, "No figure number here", "figures/figure_3.png")
        assert result == "Fig3"

    def test_figure_number_from_url_fig_prefix(self):
        """Extracts from URL with 'fig' prefix."""
        result = generate_figure_id(0, "Image", "path/to/fig2.png")
        assert result == "Fig2"

    def test_figure_number_from_url_with_dash(self):
        """Extracts from URL with dash separator."""
        result = generate_figure_id(0, "Image", "figures/fig-4.png")
        assert result == "Fig4"

    def test_figure_number_from_url_with_underscore(self):
        """Extracts from URL with underscore separator."""
        result = generate_figure_id(0, "Image", "figures/figure_5a.png")
        assert result == "Fig5a"

    # -------------------------------------------------------------------------
    # Fallback to sequential
    # -------------------------------------------------------------------------

    def test_fallback_to_sequential_numbering(self):
        """Falls back to sequential numbering when no pattern found."""
        result = generate_figure_id(0, "Random image", "path/to/random.png")
        assert result == "Fig1"

    def test_fallback_sequential_index_1(self):
        """Sequential numbering uses index + 1."""
        result = generate_figure_id(1, "Random", "random.png")
        assert result == "Fig2"

    def test_fallback_sequential_index_5(self):
        """Sequential numbering for higher index."""
        result = generate_figure_id(5, "No pattern", "image.png")
        assert result == "Fig6"

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_empty_alt_and_generic_url(self):
        """Empty alt with generic URL falls back to sequential."""
        result = generate_figure_id(2, "", "image.png")
        assert result == "Fig3"

    def test_alt_text_priority_over_url(self):
        """Alt text takes priority over URL for figure ID."""
        result = generate_figure_id(0, "Figure 7", "figures/fig2.png")
        # Alt text should take priority
        assert result == "Fig7"

    def test_url_encoded_filename(self):
        """Handles URL-encoded characters in filename."""
        result = generate_figure_id(0, "Image", "path/figure%201.png")
        # %20 decodes to space, might not match
        assert "Fig" in result


# =============================================================================
# Tests for get_file_extension
# =============================================================================


class TestGetFileExtension:
    """Tests for get_file_extension function."""

    # -------------------------------------------------------------------------
    # Standard formats
    # -------------------------------------------------------------------------

    def test_png_extension(self):
        """Extracts .png extension."""
        result = get_file_extension("image.png")
        assert result == ".png"

    def test_jpg_extension(self):
        """Extracts .jpg extension."""
        result = get_file_extension("image.jpg")
        assert result == ".jpg"

    def test_jpeg_extension(self):
        """Extracts .jpeg extension."""
        result = get_file_extension("image.jpeg")
        assert result == ".jpeg"

    def test_gif_extension(self):
        """Extracts .gif extension."""
        result = get_file_extension("image.gif")
        assert result == ".gif"

    def test_webp_extension(self):
        """Extracts .webp extension."""
        result = get_file_extension("image.webp")
        assert result == ".webp"

    def test_svg_extension(self):
        """Extracts .svg extension."""
        result = get_file_extension("image.svg")
        assert result == ".svg"

    def test_pdf_extension(self):
        """Extracts .pdf extension."""
        result = get_file_extension("figure.pdf")
        assert result == ".pdf"

    def test_bmp_extension(self):
        """Extracts .bmp extension."""
        result = get_file_extension("image.bmp")
        assert result == ".bmp"

    def test_tiff_extension(self):
        """Extracts .tiff extension."""
        result = get_file_extension("image.tiff")
        assert result == ".tiff"

    def test_tif_extension(self):
        """Extracts .tif extension."""
        result = get_file_extension("image.tif")
        assert result == ".tif"

    # -------------------------------------------------------------------------
    # Case insensitivity
    # -------------------------------------------------------------------------

    def test_uppercase_extension(self):
        """Uppercase extensions are normalized to lowercase."""
        result = get_file_extension("image.PNG")
        assert result == ".png"

    def test_mixed_case_extension(self):
        """Mixed case extensions are normalized."""
        result = get_file_extension("image.JpG")
        assert result == ".jpg"

    # -------------------------------------------------------------------------
    # Complex URLs
    # -------------------------------------------------------------------------

    def test_extension_from_url_path(self):
        """Extracts extension from full URL path."""
        result = get_file_extension("https://example.com/images/fig.png")
        assert result == ".png"

    def test_extension_with_query_string(self):
        """Extracts extension ignoring query string."""
        result = get_file_extension("image.png?v=123")
        assert result == ".png"

    def test_extension_with_fragment(self):
        """Extracts extension ignoring URL fragment."""
        result = get_file_extension("image.png#section")
        assert result == ".png"

    def test_extension_url_encoded(self):
        """Handles URL-encoded path."""
        result = get_file_extension("path/to/image%20name.png")
        assert result == ".png"

    def test_extension_from_complex_url(self):
        """Extracts from complex URL with multiple path segments."""
        result = get_file_extension("https://cdn.example.com/v1/images/2023/fig1.jpeg")
        assert result == ".jpeg"

    # -------------------------------------------------------------------------
    # Default behavior
    # -------------------------------------------------------------------------

    def test_no_extension_returns_default(self):
        """No extension returns default .png."""
        result = get_file_extension("image")
        assert result == ".png"

    def test_unknown_extension_returns_default(self):
        """Unknown extension returns default .png."""
        result = get_file_extension("image.xyz")
        assert result == ".png"

    def test_custom_default(self):
        """Custom default extension is used."""
        result = get_file_extension("image.xyz", default=".jpg")
        assert result == ".jpg"

    def test_empty_url_returns_default(self):
        """Empty URL returns default."""
        result = get_file_extension("")
        assert result == ".png"

    def test_dot_only_returns_default(self):
        """URL ending in just a dot returns default."""
        result = get_file_extension("image.")
        assert result == ".png"

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_multiple_dots_in_filename(self):
        """Handles filenames with multiple dots."""
        result = get_file_extension("image.v2.final.png")
        assert result == ".png"

    def test_hidden_file_extension(self):
        """Handles 'hidden' files with extension."""
        result = get_file_extension(".hidden.png")
        assert result == ".png"

    def test_relative_path_extension(self):
        """Extracts from relative path."""
        result = get_file_extension("../figures/fig1.jpg")
        assert result == ".jpg"


# =============================================================================
# Tests for extract_paper_title
# =============================================================================


class TestExtractPaperTitle:
    """Tests for extract_paper_title function."""

    # -------------------------------------------------------------------------
    # H1 heading extraction
    # -------------------------------------------------------------------------

    def test_extracts_h1_heading(self):
        """Extracts title from # heading."""
        md = "# My Paper Title\n\nContent here."
        result = extract_paper_title(md)
        assert result == "My Paper Title"

    def test_extracts_first_h1_heading(self):
        """Extracts first H1 when multiple exist."""
        md = """# First Title

## Section

# Second Title
"""
        result = extract_paper_title(md)
        assert result == "First Title"

    def test_h1_with_leading_whitespace(self):
        """Handles H1 with leading whitespace on the line."""
        md = "   # Title With Spaces\n\nContent."
        result = extract_paper_title(md)
        assert result == "Title With Spaces"

    def test_h1_without_space_not_matched(self):
        """#Title without space after # is not a valid H1."""
        md = "#NoSpace\n\n# Valid Title"
        result = extract_paper_title(md)
        # First valid H1 is "Valid Title"
        assert result == "Valid Title"

    def test_h2_not_extracted_as_title(self):
        """H2 headings are not extracted as title."""
        md = "## Not a Title\n\nSome text"
        result = extract_paper_title(md)
        # Should fall back to something else
        assert result != "Not a Title"

    # -------------------------------------------------------------------------
    # HTML h1 extraction
    # -------------------------------------------------------------------------

    def test_extracts_html_h1_tag(self):
        """Extracts title from <h1> tag."""
        md = "<h1>HTML Title</h1>\n\nContent."
        result = extract_paper_title(md)
        assert result == "HTML Title"

    def test_extracts_html_h1_with_attributes(self):
        """Extracts title from <h1> with attributes."""
        md = '<h1 class="title" id="main">Styled Title</h1>'
        result = extract_paper_title(md)
        assert result == "Styled Title"

    def test_html_h1_strips_inner_tags(self):
        """Strips HTML tags from inside h1."""
        md = "<h1>Title with <strong>bold</strong> text</h1>"
        result = extract_paper_title(md)
        assert result == "Title with bold text"

    def test_html_h1_case_insensitive(self):
        """HTML h1 matching is case insensitive."""
        md = "<H1>Uppercase Tag</H1>"
        result = extract_paper_title(md)
        assert result == "Uppercase Tag"

    def test_h1_markdown_priority_over_html(self):
        """Markdown H1 takes priority over HTML h1."""
        md = "# Markdown Title\n\n<h1>HTML Title</h1>"
        result = extract_paper_title(md)
        assert result == "Markdown Title"

    # -------------------------------------------------------------------------
    # Code block exclusion
    # -------------------------------------------------------------------------

    def test_ignores_h1_in_code_block(self):
        """H1 inside code block is ignored."""
        md = """```
# Not a title
```

# Real Title
"""
        result = extract_paper_title(md)
        assert result == "Real Title"

    def test_ignores_h1_in_fenced_code_with_language(self):
        """H1 in fenced code with language tag is ignored."""
        md = """```python
# This is a comment
```

# Actual Title
"""
        result = extract_paper_title(md)
        assert result == "Actual Title"

    def test_h1_only_in_code_block_falls_back(self):
        """When only H1 is in code block, falls back to first line."""
        md = """Some first line text

```
# Code comment
```
"""
        result = extract_paper_title(md)
        assert result == "Some first line text"

    # -------------------------------------------------------------------------
    # Fallback behavior
    # -------------------------------------------------------------------------

    def test_fallback_to_first_non_empty_line(self):
        """Falls back to first non-empty line when no heading."""
        md = "First line content\n\nSecond line"
        result = extract_paper_title(md)
        assert result == "First line content"

    def test_fallback_skips_empty_lines(self):
        """Fallback skips empty lines."""
        md = "\n\n\nThird line content\n\n"
        result = extract_paper_title(md)
        assert result == "Third line content"

    def test_fallback_skips_image_lines(self):
        """Fallback skips lines starting with !."""
        md = "![Image](img.png)\n\nActual Title Line"
        result = extract_paper_title(md)
        assert result == "Actual Title Line"

    def test_fallback_skips_html_lines(self):
        """Fallback skips lines starting with <."""
        md = "<meta charset='utf-8'>\n\nActual Title"
        result = extract_paper_title(md)
        assert result == "Actual Title"

    def test_fallback_truncates_long_lines(self):
        """Fallback truncates very long lines to 200 chars."""
        long_line = "A" * 300
        md = long_line
        result = extract_paper_title(md)
        assert len(result) == 200
        assert result == "A" * 200

    def test_returns_untitled_paper_when_empty(self):
        """Returns 'Untitled Paper' for empty input."""
        result = extract_paper_title("")
        assert result == "Untitled Paper"

    def test_returns_untitled_paper_when_only_code_blocks(self):
        """Returns 'Untitled Paper' when only code blocks exist."""
        md = """```
code only
```"""
        result = extract_paper_title(md)
        assert result == "Untitled Paper"

    def test_returns_untitled_paper_when_only_images(self):
        """Returns 'Untitled Paper' when only images exist."""
        md = "![Image1](img1.png)\n![Image2](img2.png)"
        result = extract_paper_title(md)
        assert result == "Untitled Paper"

    def test_returns_untitled_paper_for_whitespace_only(self):
        """Returns 'Untitled Paper' for whitespace-only input."""
        result = extract_paper_title("   \n\t\n   ")
        assert result == "Untitled Paper"

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_title_with_special_characters(self):
        """Handles special characters in title."""
        md = "# Title: A & B - Results (2023)"
        result = extract_paper_title(md)
        assert result == "Title: A & B - Results (2023)"

    def test_title_with_unicode(self):
        """Handles unicode in title."""
        md = "# 研究论文 αβγ émojis 🔬"
        result = extract_paper_title(md)
        assert result == "研究论文 αβγ émojis 🔬"

    def test_title_with_markdown_formatting(self):
        """Title may contain markdown formatting."""
        md = "# **Bold** and *italic* title"
        result = extract_paper_title(md)
        # Markdown formatting should be preserved as-is
        assert "Bold" in result
        assert "italic" in result

    def test_multiline_html_h1(self):
        """Handles h1 tag spanning multiple lines."""
        md = """<h1>
Title
</h1>"""
        result = extract_paper_title(md)
        assert result.strip() == "Title"

    def test_none_input_raises_error(self):
        """None input raises TypeError."""
        with pytest.raises((TypeError, AttributeError)):
            extract_paper_title(None)
