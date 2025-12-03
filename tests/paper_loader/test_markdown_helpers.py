"""Tests for misc markdown helper utilities."""

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


class TestGenerateFigureId:
    """Tests for generate_figure_id function."""

    def test_extracts_from_alt_text_figure(self):
        """Extracts figure number from alt text with 'Figure' prefix."""
        fig_id = generate_figure_id(0, "Figure 3: Extinction spectrum", "img.png")
        assert fig_id == "Fig3"

    def test_extracts_from_alt_text_fig(self):
        """Extracts figure number from alt text with 'Fig' prefix."""
        fig_id = generate_figure_id(0, "Fig 2a shows results", "img.png")
        assert fig_id == "Fig2a"

    def test_extracts_from_alt_text_fig_with_period(self):
        """Extracts figure number from alt text with 'Fig.' prefix."""
        fig_id = generate_figure_id(0, "Fig. 5 shows the results", "img.png")
        assert fig_id == "Fig5"

    def test_extracts_from_url_filename(self):
        """Extracts figure number from URL filename when alt text is empty."""
        fig_id = generate_figure_id(0, "", "images/figure_4.png")
        assert fig_id == "Fig4"

    def test_extracts_from_url_filename_with_dash(self):
        """Extracts figure number from URL filename with dash separator."""
        fig_id = generate_figure_id(0, "", "images/figure-5.png")
        assert fig_id == "Fig5"

    def test_extracts_from_url_filename_with_underscore(self):
        """Extracts figure number from URL filename with underscore separator."""
        fig_id = generate_figure_id(0, "", "images/figure_6a.png")
        assert fig_id == "Fig6a"

    def test_fallback_to_index(self):
        """Falls back to index-based numbering when no figure number found."""
        fig_id = generate_figure_id(2, "Some random text", "random.png")
        assert fig_id == "Fig3"

    def test_fallback_to_index_zero_based(self):
        """Index is 0-based, so index 0 produces Fig1."""
        fig_id = generate_figure_id(0, "No figure number here", "random.png")
        assert fig_id == "Fig1"

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

    def test_extracts_supplementary_figures(self):
        """Extracts supplementary figure numbers."""
        assert generate_figure_id(0, "Figure S2", "x.png") == "FigS2"
        assert generate_figure_id(0, "Fig S3a", "x.png") == "FigS3a"

    def test_extracts_figure_with_parentheses(self):
        """Extracts figure numbers with parenthetical labels."""
        fig_id = generate_figure_id(0, "Figure 2(a) shows", "img.png")
        assert fig_id == "Fig2(a)"

    def test_alt_text_takes_priority_over_url(self):
        """Alt text extraction takes priority over URL filename."""
        fig_id = generate_figure_id(0, "Figure 7", "images/figure_10.png")
        assert fig_id == "Fig7"

    def test_url_extraction_when_alt_empty(self):
        """URL extraction works when alt text is empty."""
        fig_id = generate_figure_id(0, "", "images/figure_8.png")
        assert fig_id == "Fig8"

    def test_url_extraction_when_alt_no_match(self):
        """URL extraction works when alt text has no figure number."""
        fig_id = generate_figure_id(0, "Some description", "images/figure_9.png")
        assert fig_id == "Fig9"

    def test_case_insensitive_alt_text(self):
        """Figure extraction is case-insensitive for alt text."""
        assert generate_figure_id(0, "FIGURE 1", "x.png") == "Fig1"
        assert generate_figure_id(0, "fig 2", "x.png") == "Fig2"
        assert generate_figure_id(0, "FiGuRe 3", "x.png") == "Fig3"

    def test_case_insensitive_url(self):
        """Figure extraction is case-insensitive for URL."""
        fig_id = generate_figure_id(0, "", "images/FIGURE_10.png")
        assert fig_id == "Fig10"

    def test_multiple_figures_in_alt_text_takes_first(self):
        """When multiple figure numbers exist, takes the first one."""
        fig_id = generate_figure_id(0, "See Figure 5 and Figure 6", "img.png")
        assert fig_id == "Fig5"

    def test_figure_number_with_letter_suffix(self):
        """Extracts figure numbers with letter suffixes."""
        assert generate_figure_id(0, "Figure 2a", "x.png") == "Fig2a"
        assert generate_figure_id(0, "Figure 2b", "x.png") == "Fig2b"
        assert generate_figure_id(0, "Figure 2z", "x.png") == "Fig2z"

    def test_empty_alt_and_url_falls_back_to_index(self):
        """Falls back to index when both alt and URL are empty."""
        fig_id = generate_figure_id(5, "", "")
        assert fig_id == "Fig6"

    def test_url_with_query_params(self):
        """URL extraction works with query parameters."""
        fig_id = generate_figure_id(0, "", "https://example.com/figure_11.png?token=abc")
        assert fig_id == "Fig11"

    def test_url_with_fragment(self):
        """URL extraction works with URL fragments."""
        fig_id = generate_figure_id(0, "", "https://example.com/figure_12.png#section")
        assert fig_id == "Fig12"

    def test_url_with_encoded_characters(self):
        """URL extraction handles URL-encoded characters."""
        fig_id = generate_figure_id(0, "", "images/figure%2013.png")
        assert fig_id == "Fig13"

    def test_negative_index_handled(self):
        """Negative index is handled (though unusual)."""
        fig_id = generate_figure_id(-1, "No match", "no_match.png")
        assert fig_id == "Fig0"


class TestGetFileExtension:
    """Tests for get_file_extension function."""

    def test_png_extension(self):
        """Extracts .png extension from file path."""
        ext = get_file_extension("path/to/image.png")
        assert ext == ".png"

    def test_jpg_extension(self):
        """Extracts .jpg extension."""
        ext = get_file_extension("path/to/image.jpg")
        assert ext == ".jpg"

    def test_jpeg_extension(self):
        """Extracts .jpeg extension."""
        ext = get_file_extension("path/to/image.jpeg")
        assert ext == ".jpeg"

    def test_gif_extension(self):
        """Extracts .gif extension."""
        ext = get_file_extension("path/to/image.gif")
        assert ext == ".gif"

    def test_webp_extension(self):
        """Extracts .webp extension."""
        ext = get_file_extension("path/to/image.webp")
        assert ext == ".webp"

    def test_svg_extension(self):
        """Extracts .svg extension."""
        ext = get_file_extension("path/to/image.svg")
        assert ext == ".svg"

    def test_pdf_extension(self):
        """Extracts .pdf extension."""
        ext = get_file_extension("path/to/image.pdf")
        assert ext == ".pdf"

    def test_uppercase_normalized(self):
        """Uppercase extensions are normalized to lowercase."""
        assert get_file_extension("image.PNG") == ".png"
        assert get_file_extension("image.JPG") == ".jpg"
        assert get_file_extension("image.GIF") == ".gif"

    def test_mixed_case_normalized(self):
        """Mixed case extensions are normalized to lowercase."""
        assert get_file_extension("image.PnG") == ".png"
        assert get_file_extension("image.JpEg") == ".jpeg"

    def test_url_with_query_params(self):
        """Extension extraction works with URL query parameters."""
        ext = get_file_extension("https://example.com/image.gif?token=abc")
        assert ext == ".gif"

    def test_url_with_fragment(self):
        """Extension extraction works with URL fragments."""
        ext = get_file_extension("https://example.com/image.png#section")
        assert ext == ".png"

    def test_url_with_both_query_and_fragment(self):
        """Extension extraction works with both query and fragment."""
        ext = get_file_extension("https://example.com/image.jpg?token=abc#section")
        assert ext == ".jpg"

    def test_unknown_extension_uses_default(self):
        """Unknown extensions return default (.png)."""
        ext = get_file_extension("file.xyz")
        assert ext == ".png"

    def test_custom_default(self):
        """Custom default extension is used when provided."""
        ext = get_file_extension("file.xyz", default=".jpg")
        assert ext == ".jpg"

    def test_custom_default_with_unknown_extension(self):
        """Custom default is used for unknown extensions."""
        ext = get_file_extension("file.unknown", default=".gif")
        assert ext == ".gif"

    def test_no_extension_uses_default(self):
        """Files without extension use default."""
        ext = get_file_extension("file_without_extension")
        assert ext == ".png"

    def test_no_extension_with_custom_default(self):
        """Files without extension use custom default."""
        ext = get_file_extension("file_without_extension", default=".svg")
        assert ext == ".svg"

    def test_url_encoded_path(self):
        """URL-encoded paths are decoded before extraction."""
        ext = get_file_extension("https://example.com/image%20name.png")
        assert ext == ".png"

    def test_multiple_dots_in_filename(self):
        """Extension is taken from the last dot."""
        ext = get_file_extension("file.name.with.dots.png")
        assert ext == ".png"

    def test_extension_at_end_of_path(self):
        """Extension extraction works at end of path."""
        ext = get_file_extension("/very/long/path/to/file.png")
        assert ext == ".png"

    def test_relative_path(self):
        """Extension extraction works with relative paths."""
        ext = get_file_extension("../images/figure.jpg")
        assert ext == ".jpg"

    def test_absolute_path(self):
        """Extension extraction works with absolute paths."""
        ext = get_file_extension("/absolute/path/to/image.gif")
        assert ext == ".gif"

    def test_file_protocol_url(self):
        """Extension extraction works with file:// URLs."""
        ext = get_file_extension("file:///path/to/image.png")
        assert ext == ".png"

    def test_empty_string_uses_default(self):
        """Empty string uses default extension."""
        ext = get_file_extension("")
        assert ext == ".png"

    def test_empty_string_with_custom_default(self):
        """Empty string uses custom default."""
        ext = get_file_extension("", default=".jpg")
        assert ext == ".jpg"

    def test_just_extension(self):
        """String with just extension returns the extension."""
        ext = get_file_extension(".png")
        assert ext == ".png"

    def test_just_extension_uppercase(self):
        """String with just uppercase extension is normalized."""
        ext = get_file_extension(".PNG")
        assert ext == ".png"


class TestExtractFiguresFromMarkdown:
    """Tests for extract_figures_from_markdown function."""

    def test_simple_markdown_image(self):
        """Extracts simple markdown image syntax."""
        markdown = "![Alt text](image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"
        assert figures[0]["url"] == "image.png"
        assert "original_match" in figures[0]

    def test_markdown_image_with_title(self):
        """Extracts markdown image with title attribute."""
        markdown = '![Alt text](image.png "Title")'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"
        assert figures[0]["url"] == "image.png"

    def test_multiple_markdown_images(self):
        """Extracts multiple markdown images."""
        markdown = "![Fig1](img1.png)\n![Fig2](img2.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 2
        assert figures[0]["url"] == "img1.png"
        assert figures[1]["url"] == "img2.png"

    def test_html_img_tag(self):
        """Extracts HTML img tag."""
        markdown = '<img src="image.png" alt="Alt text" />'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"
        assert figures[0]["url"] == "image.png"

    def test_html_img_tag_no_alt(self):
        """Extracts HTML img tag without alt attribute."""
        markdown = '<img src="image.png" />'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == ""
        assert figures[0]["url"] == "image.png"

    def test_html_img_tag_case_insensitive(self):
        """HTML img tag matching is case-insensitive."""
        markdown = '<IMG SRC="image.png" ALT="Alt text" />'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"
        assert figures[0]["url"] == "image.png"

    def test_html_img_tag_alt_before_src(self):
        """Extracts alt text when alt comes before src."""
        markdown = '<img alt="Alt text" src="image.png" />'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"
        assert figures[0]["url"] == "image.png"

    def test_markdown_and_html_both_present(self):
        """Extracts both markdown and HTML images."""
        markdown = "![Fig1](img1.png)\n<img src=\"img2.png\" alt=\"Fig2\" />"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 2
        assert figures[0]["url"] == "img1.png"
        assert figures[1]["url"] == "img2.png"

    def test_duplicate_url_only_added_once(self):
        """Same URL in markdown and HTML is only added once."""
        markdown = "![Alt](image.png)\n<img src=\"image.png\" alt=\"Alt\" />"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "image.png"

    def test_ignores_images_in_code_blocks(self):
        """Images inside code blocks are ignored."""
        markdown = "```\n![Fig1](img1.png)\n```\n![Fig2](img2.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "img2.png"

    def test_ignores_html_in_code_blocks(self):
        """HTML images inside code blocks are ignored."""
        markdown = "```\n<img src=\"img1.png\" />\n```\n<img src=\"img2.png\" />"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "img2.png"

    def test_multiple_code_blocks(self):
        """Handles multiple code blocks correctly."""
        markdown = "![Fig1](img1.png)\n```\ncode\n```\n![Fig2](img2.png)\n```\nmore code\n```"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 2
        assert figures[0]["url"] == "img1.png"
        assert figures[1]["url"] == "img2.png"

    def test_unclosed_code_block(self):
        """Unclosed code block excludes until end."""
        markdown = "![Fig1](img1.png)\n```\ncode without closing"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "img1.png"

    def test_nested_brackets_in_alt_text(self):
        """Handles nested brackets in alt text."""
        markdown = "![Alt [with] brackets](image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt [with] brackets"
        assert figures[0]["url"] == "image.png"

    def test_parentheses_in_url(self):
        """Handles parentheses in URL."""
        markdown = "![Alt](image(1).png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "image(1).png"

    def test_url_with_spaces(self):
        """Handles URLs with spaces (though unusual)."""
        markdown = "![Alt](image name.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "image name.png"

    def test_empty_alt_text(self):
        """Handles empty alt text."""
        markdown = "![](image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == ""
        assert figures[0]["url"] == "image.png"

    def test_empty_markdown(self):
        """Empty markdown returns empty list."""
        figures = extract_figures_from_markdown("")
        assert figures == []

    def test_no_images(self):
        """Markdown with no images returns empty list."""
        markdown = "Some text without images."
        figures = extract_figures_from_markdown(markdown)
        assert figures == []

    def test_url_with_http(self):
        """Extracts HTTP URLs."""
        markdown = "![Alt](http://example.com/image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "http://example.com/image.png"

    def test_url_with_https(self):
        """Extracts HTTPS URLs."""
        markdown = "![Alt](https://example.com/image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "https://example.com/image.png"

    def test_html_img_with_single_quotes(self):
        """Extracts HTML img with single quotes."""
        markdown = "<img src='image.png' alt='Alt text' />"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"
        assert figures[0]["url"] == "image.png"

    def test_html_img_self_closing(self):
        """Extracts self-closing HTML img tag."""
        markdown = '<img src="image.png" alt="Alt" />'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1

    def test_html_img_not_self_closing(self):
        """Extracts non-self-closing HTML img tag."""
        markdown = '<img src="image.png" alt="Alt">'
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1

    def test_whitespace_in_alt_text_stripped(self):
        """Whitespace in alt text is stripped."""
        markdown = "![  Alt text  ](image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["alt"] == "Alt text"

    def test_whitespace_in_url_stripped(self):
        """Whitespace in URL is stripped."""
        markdown = "![Alt](  image.png  )"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert figures[0]["url"] == "image.png"

    def test_all_figures_have_required_keys(self):
        """All extracted figures have required keys."""
        markdown = "![Alt](image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert len(figures) == 1
        assert "alt" in figures[0]
        assert "url" in figures[0]
        assert "original_match" in figures[0]

    def test_original_match_contains_full_match(self):
        """original_match contains the full matched string."""
        markdown = "![Alt text](image.png)"
        figures = extract_figures_from_markdown(markdown)
        assert figures[0]["original_match"] == "![Alt text](image.png)"


class TestResolveFigureUrl:
    """Tests for resolve_figure_url function."""

    def test_absolute_http_url(self):
        """Absolute HTTP URL is returned as-is."""
        url = resolve_figure_url("http://example.com/image.png")
        assert url == "http://example.com/image.png"

    def test_absolute_https_url(self):
        """Absolute HTTPS URL is returned as-is."""
        url = resolve_figure_url("https://example.com/image.png")
        assert url == "https://example.com/image.png"

    def test_absolute_file_url(self):
        """Absolute file:// URL is returned as-is."""
        url = resolve_figure_url("file:///path/to/image.png")
        assert url == "file:///path/to/image.png"

    def test_relative_url_with_base_url(self):
        """Relative URL is joined with base_url."""
        url = resolve_figure_url("image.png", base_url="https://example.com/path/")
        assert url == "https://example.com/path/image.png"

    def test_relative_url_with_base_path(self):
        """Relative URL is resolved against base_path."""
        base_path = Path("/absolute/path/to")
        url = resolve_figure_url("image.png", base_path=base_path)
        assert url == str(base_path / "image.png")

    def test_base_url_takes_priority_over_base_path(self):
        """base_url takes priority over base_path."""
        base_path = Path("/absolute/path/to")
        url = resolve_figure_url("image.png", base_path=base_path, base_url="https://example.com/")
        assert url == "https://example.com/image.png"

    def test_relative_url_no_base_returns_as_is(self):
        """Relative URL without base returns as-is."""
        url = resolve_figure_url("image.png")
        assert url == "image.png"

    def test_absolute_url_ignores_base(self):
        """Absolute URL ignores base_url and base_path."""
        url = resolve_figure_url("https://example.com/image.png", base_url="https://other.com/")
        assert url == "https://example.com/image.png"

    def test_relative_path_with_parent(self):
        """Relative path with .. is resolved correctly."""
        base_path = Path("/absolute/path/to")
        url = resolve_figure_url("../other/image.png", base_path=base_path)
        assert url == str(base_path.parent / "other" / "image.png")

    def test_relative_path_with_current_dir(self):
        """Relative path with ./ is resolved correctly."""
        base_path = Path("/absolute/path/to")
        url = resolve_figure_url("./image.png", base_path=base_path)
        assert url == str(base_path / "image.png")

    def test_base_url_with_trailing_slash(self):
        """base_url with trailing slash joins correctly."""
        url = resolve_figure_url("image.png", base_url="https://example.com/path/")
        assert url == "https://example.com/path/image.png"

    def test_base_url_without_trailing_slash(self):
        """base_url without trailing slash joins correctly."""
        url = resolve_figure_url("image.png", base_url="https://example.com/path")
        assert url == "https://example.com/image.png"

    def test_base_url_with_subpath(self):
        """base_url with subpath joins correctly."""
        url = resolve_figure_url("subdir/image.png", base_url="https://example.com/base/")
        assert url == "https://example.com/base/subdir/image.png"

    def test_empty_url(self):
        """Empty URL returns empty string."""
        url = resolve_figure_url("")
        assert url == ""

    def test_empty_url_with_base_url(self):
        """Empty URL with base_url returns base_url."""
        url = resolve_figure_url("", base_url="https://example.com/")
        assert url == "https://example.com/"

    def test_empty_url_with_base_path(self):
        """Empty URL with base_path returns base_path."""
        base_path = Path("/path/to")
        url = resolve_figure_url("", base_path=base_path)
        assert url == str(base_path)

    def test_relative_url_with_query_params(self):
        """Relative URL with query params is preserved."""
        url = resolve_figure_url("image.png?token=abc", base_url="https://example.com/")
        assert url == "https://example.com/image.png?token=abc"

    def test_relative_url_with_fragment(self):
        """Relative URL with fragment is preserved."""
        url = resolve_figure_url("image.png#section", base_url="https://example.com/")
        assert url == "https://example.com/image.png#section"

    def test_relative_url_with_both_query_and_fragment(self):
        """Relative URL with both query and fragment is preserved."""
        url = resolve_figure_url("image.png?token=abc#section", base_url="https://example.com/")
        assert url == "https://example.com/image.png?token=abc#section"


class TestExtractPaperTitle:
    """Tests for extract_paper_title function."""

    def test_h1_heading(self):
        """Extracts title from H1 markdown heading."""
        markdown = "# Paper Title"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_h1_heading_with_whitespace(self):
        """H1 heading whitespace is stripped."""
        markdown = "#  Paper Title  "
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_h1_heading_not_first_line(self):
        """Extracts first H1 heading even if not first line."""
        markdown = "Some text\n# Paper Title\nMore text"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_h1_heading_ignores_code_blocks(self):
        """H1 headings in code blocks are ignored."""
        markdown = "```\n# Code comment\n```\n# Paper Title"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_html_h1_tag(self):
        """Extracts title from HTML h1 tag."""
        markdown = "<h1>Paper Title</h1>"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_html_h1_tag_with_attributes(self):
        """Extracts title from HTML h1 tag with attributes."""
        markdown = '<h1 class="title">Paper Title</h1>'
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_html_h1_tag_case_insensitive(self):
        """HTML h1 tag matching is case-insensitive."""
        markdown = "<H1>Paper Title</H1>"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_html_h1_tag_strips_inner_html(self):
        """HTML tags inside h1 are stripped."""
        markdown = "<h1>Paper <em>Title</em></h1>"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_h1_markdown_takes_priority_over_html(self):
        """Markdown H1 takes priority over HTML h1."""
        markdown = "# Markdown Title\n<h1>HTML Title</h1>"
        title = extract_paper_title(markdown)
        assert title == "Markdown Title"

    def test_fallback_to_first_non_empty_line(self):
        """Falls back to first non-empty line when no heading."""
        markdown = "Paper Title\nMore content"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_fallback_ignores_code_blocks(self):
        """Fallback ignores code blocks."""
        markdown = "```\ncode\n```\nPaper Title"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_fallback_ignores_images(self):
        """Fallback ignores image lines."""
        markdown = "![Alt](img.png)\nPaper Title"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_fallback_ignores_html_images(self):
        """Fallback ignores HTML image tags."""
        markdown = '<img src="img.png" />\nPaper Title'
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_fallback_truncates_long_lines(self):
        """Fallback truncates very long lines to 200 characters."""
        long_title = "A" * 300
        markdown = f"{long_title}\nMore content"
        title = extract_paper_title(markdown)
        assert len(title) == 200
        assert title == "A" * 200

    def test_empty_markdown_returns_untitled(self):
        """Empty markdown returns 'Untitled Paper'."""
        title = extract_paper_title("")
        assert title == "Untitled Paper"

    def test_only_whitespace_returns_untitled(self):
        """Markdown with only whitespace returns 'Untitled Paper'."""
        title = extract_paper_title("   \n\n  ")
        assert title == "Untitled Paper"

    def test_only_code_blocks_returns_untitled(self):
        """Markdown with only code blocks returns 'Untitled Paper'."""
        title = extract_paper_title("```\ncode\n```")
        assert title == "Untitled Paper"

    def test_only_images_returns_untitled(self):
        """Markdown with only images returns 'Untitled Paper'."""
        title = extract_paper_title("![Alt](img.png)")
        assert title == "Untitled Paper"

    def test_h2_heading_not_used(self):
        """H2 headings are not used for title."""
        markdown = "## Not a Title\n# Paper Title"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_h3_heading_not_used(self):
        """H3 headings are not used for title."""
        markdown = "### Not a Title\n# Paper Title"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_multiple_h1_uses_first(self):
        """Multiple H1 headings use the first one."""
        markdown = "# First Title\n# Second Title"
        title = extract_paper_title(markdown)
        assert title == "First Title"

    def test_h1_with_markdown_formatting(self):
        """H1 heading preserves markdown formatting in text."""
        markdown = "# Paper *Title*"
        title = extract_paper_title(markdown)
        assert title == "Paper *Title*"

    def test_html_h1_multiline(self):
        """HTML h1 tag can span multiple lines."""
        markdown = "<h1>\nPaper Title\n</h1>"
        title = extract_paper_title(markdown)
        assert title == "Paper Title"

    def test_unclosed_code_block(self):
        """Unclosed code block excludes content."""
        markdown = "```\ncode without closing\n# Paper Title"
        title = extract_paper_title(markdown)
        # Title should be found if code block detection works correctly
        # This tests the edge case of unclosed blocks
        assert title == "Paper Title"
