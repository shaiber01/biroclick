"""Tests for resolve_figure_url."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.paper_loader.markdown_parser import resolve_figure_url


class TestResolveFigureUrl:
    """Tests for resolve_figure_url function."""

    # ═══════════════════════════════════════════════════════════════════════
    # Absolute URL Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_absolute_http_url_unchanged(self):
        """HTTP URLs should be returned unchanged regardless of base parameters."""
        url = "http://example.com/image.png"
        result = resolve_figure_url(url)
        assert result == "http://example.com/image.png"
        # Should still be unchanged even with base_url/base_path provided
        result2 = resolve_figure_url(url, base_url="https://other.com/", base_path=Path("/tmp"))
        assert result2 == "http://example.com/image.png"

    def test_absolute_https_url_unchanged(self):
        """HTTPS URLs should be returned unchanged."""
        url = "https://example.com/image.png"
        result = resolve_figure_url(url)
        assert result == "https://example.com/image.png"
        # Should still be unchanged even with base_url/base_path provided
        result2 = resolve_figure_url(url, base_url="http://other.com/", base_path=Path("/tmp"))
        assert result2 == "https://example.com/image.png"

    def test_absolute_file_url_unchanged(self):
        """File URLs should be returned unchanged."""
        url = "file:///absolute/path/image.png"
        result = resolve_figure_url(url)
        assert result == "file:///absolute/path/image.png"
        # Should still be unchanged even with base_url/base_path provided
        result2 = resolve_figure_url(url, base_url="https://other.com/", base_path=Path("/tmp"))
        assert result2 == "file:///absolute/path/image.png"

    def test_absolute_url_with_query_params(self):
        """Absolute URLs with query parameters should be preserved."""
        url = "https://example.com/image.png?version=1&size=large"
        result = resolve_figure_url(url)
        assert result == "https://example.com/image.png?version=1&size=large"

    def test_absolute_url_with_fragment(self):
        """Absolute URLs with fragments should be preserved."""
        url = "https://example.com/image.png#section1"
        result = resolve_figure_url(url)
        assert result == "https://example.com/image.png#section1"

    def test_absolute_url_with_port(self):
        """Absolute URLs with port numbers should be preserved."""
        url = "http://example.com:8080/image.png"
        result = resolve_figure_url(url)
        assert result == "http://example.com:8080/image.png"

    def test_absolute_url_with_authentication(self):
        """Absolute URLs with authentication should be preserved."""
        url = "https://user:pass@example.com/image.png"
        result = resolve_figure_url(url)
        assert result == "https://user:pass@example.com/image.png"

    # ═══════════════════════════════════════════════════════════════════════
    # Relative URL with base_url Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_relative_with_base_url(self):
        """Relative URLs should be joined with base_url."""
        url = "images/fig1.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig1.png"

    def test_relative_with_base_url_no_trailing_slash(self):
        """base_url without trailing slash should still work correctly."""
        url = "images/fig1.png"
        base_url = "https://example.com/papers"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig1.png"

    def test_relative_with_base_url_leading_slash(self):
        """Relative URL starting with / should replace path component."""
        url = "/images/fig1.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/images/fig1.png"

    def test_relative_with_base_url_query_params(self):
        """Query parameters in relative URL should be preserved."""
        url = "images/fig1.png?v=1"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig1.png?v=1"

    def test_relative_with_base_url_fragment(self):
        """Fragments in relative URL should be preserved."""
        url = "images/fig1.png#zoom"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig1.png#zoom"

    def test_relative_with_base_url_encoded_characters(self):
        """URL-encoded characters should be handled correctly."""
        url = "images/fig%201.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig%201.png"

    def test_relative_with_base_url_special_characters(self):
        """Special characters in URL should be handled correctly."""
        url = "images/fig-1_2.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig-1_2.png"

    # ═══════════════════════════════════════════════════════════════════════
    # Relative Path with base_path Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_relative_with_base_path(self, paper_loader_base_path):
        """Relative paths should be resolved against base_path."""
        url = "images/fig1.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path / "images/fig1.png"
        assert result == str(expected.resolve())

    def test_relative_with_base_path_leading_slash(self, paper_loader_base_path):
        """Relative path starting with / should be treated as absolute."""
        url = "/images/fig1.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        # Should resolve to absolute path
        assert Path(result).is_absolute()
        assert result == str(Path("/images/fig1.png").resolve())

    def test_relative_with_base_path_current_directory(self, paper_loader_base_path):
        """Paths starting with ./ should work correctly."""
        url = "./images/fig1.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path / "images/fig1.png"
        assert result == str(expected.resolve())

    def test_parent_directory_traversal(self, paper_loader_base_path):
        """Parent directory traversal (..) should work correctly."""
        url = "../fig.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path.parent / "fig.png"
        assert Path(result).resolve() == expected.resolve()

    def test_multiple_parent_directory_traversal(self, paper_loader_base_path):
        """Multiple parent directory traversals should work correctly."""
        url = "../../fig.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path.parent.parent / "fig.png"
        assert Path(result).resolve() == expected.resolve()

    def test_security_path_traversal_outside_root(self, paper_loader_base_path):
        """Path traversal outside root should be resolved (documents current behavior)."""
        url = "../../../../etc/passwd"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        # Should resolve to absolute path ending with etc/passwd
        assert Path(result).is_absolute()
        assert result.endswith("etc/passwd")
        # Verify it actually traverses up
        resolved_path = Path(result).resolve()
        assert "etc" in str(resolved_path)
        assert "passwd" in str(resolved_path)

    def test_relative_with_base_path_encoded_spaces(self, paper_loader_base_path):
        """URL-encoded spaces should be handled correctly."""
        url = "images/fig%201.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path / "images/fig 1.png"
        assert result == str(expected.resolve())

    # ═══════════════════════════════════════════════════════════════════════
    # Precedence Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_base_url_takes_precedence(self, paper_loader_base_path):
        """base_url should take precedence over base_path."""
        url = "fig.png"
        result = resolve_figure_url(
            url,
            base_path=paper_loader_base_path,
            base_url="https://remote.com/",
        )
        assert result == "https://remote.com/fig.png"
        # Verify it's a URL, not a path
        assert result.startswith("https://")
        assert not Path(result).exists() or Path(result).is_absolute() == False

    def test_base_url_takes_precedence_even_with_absolute_path(self, paper_loader_base_path):
        """base_url should take precedence even when base_path would create absolute path."""
        url = "/fig.png"
        result = resolve_figure_url(
            url,
            base_path=paper_loader_base_path,
            base_url="https://remote.com/",
        )
        assert result == "https://remote.com/fig.png"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: No base provided
    # ═══════════════════════════════════════════════════════════════════════

    def test_relative_url_no_base_returns_as_is(self):
        """Relative URL without base_url or base_path should return as-is."""
        url = "images/fig1.png"
        result = resolve_figure_url(url)
        assert result == "images/fig1.png"

    def test_relative_url_with_none_base_url(self):
        """Relative URL with explicit None base_url should return as-is."""
        url = "images/fig1.png"
        result = resolve_figure_url(url, base_url=None)
        assert result == "images/fig1.png"

    def test_relative_url_with_none_base_path(self):
        """Relative URL with explicit None base_path should return as-is."""
        url = "images/fig1.png"
        result = resolve_figure_url(url, base_path=None)
        assert result == "images/fig1.png"

    def test_relative_url_with_both_none(self):
        """Relative URL with both base_url and base_path as None should return as-is."""
        url = "images/fig1.png"
        result = resolve_figure_url(url, base_url=None, base_path=None)
        assert result == "images/fig1.png"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Empty and special inputs
    # ═══════════════════════════════════════════════════════════════════════

    def test_empty_string_url(self):
        """Empty string URL should return empty string."""
        result = resolve_figure_url("")
        assert result == ""

    def test_empty_string_url_with_base_url(self):
        """Empty string URL with base_url should return base_url."""
        result = resolve_figure_url("", base_url="https://example.com/papers/")
        assert result == "https://example.com/papers/"

    def test_empty_string_url_with_base_path(self, paper_loader_base_path):
        """Empty string URL with base_path should return base_path."""
        result = resolve_figure_url("", base_path=paper_loader_base_path)
        assert result == str(paper_loader_base_path.resolve())

    def test_resolves_absolute_path_input(self):
        """Absolute path inputs should be returned as-is when no base provided."""
        url = "/absolute/path/img.png"
        result = resolve_figure_url(url)
        assert result == "/absolute/path/img.png"

    def test_absolute_path_with_base_url(self):
        """Absolute path with base_url should still use base_url (urljoin behavior)."""
        url = "/absolute/path/img.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        # urljoin treats absolute paths as replacing the path component
        assert result == "https://example.com/absolute/path/img.png"

    def test_absolute_path_with_base_path(self, paper_loader_base_path):
        """Absolute path with base_path should resolve to absolute path."""
        url = "/absolute/path/img.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        assert Path(result).is_absolute()
        assert result == str(Path("/absolute/path/img.png").resolve())

    # ═══════════════════════════════════════════════════════════════════════
    # Error Handling Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_none_url_raises_type_error(self):
        """None URL should raise TypeError (urlparse requires string)."""
        with pytest.raises((TypeError, AttributeError)):
            resolve_figure_url(None)

    def test_non_string_url_raises_type_error(self):
        """Non-string URL should raise TypeError."""
        with pytest.raises((TypeError, AttributeError)):
            resolve_figure_url(123)
        with pytest.raises((TypeError, AttributeError)):
            resolve_figure_url([])
        with pytest.raises((TypeError, AttributeError)):
            resolve_figure_url({})

    # ═══════════════════════════════════════════════════════════════════════
    # Complex URL Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_url_with_all_components(self):
        """URL with scheme, netloc, path, query, and fragment should be preserved."""
        url = "https://example.com:8080/path/to/image.png?version=1&size=large#section1"
        result = resolve_figure_url(url)
        assert result == "https://example.com:8080/path/to/image.png?version=1&size=large#section1"

    def test_relative_url_with_all_components_and_base(self):
        """Relative URL with query and fragment should preserve them when joined."""
        url = "images/fig1.png?version=1#zoom"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/fig1.png?version=1#zoom"

    def test_unicode_characters_in_url(self):
        """Unicode characters in URL should be handled."""
        url = "https://example.com/图像.png"
        result = resolve_figure_url(url)
        assert result == "https://example.com/图像.png"

    def test_unicode_characters_in_relative_url(self):
        """Unicode characters in relative URL should be handled."""
        url = "images/图像.png"
        base_url = "https://example.com/papers/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/images/图像.png"

    # ═══════════════════════════════════════════════════════════════════════
    # base_url Edge Cases
    # ═══════════════════════════════════════════════════════════════════════

    def test_base_url_with_path_component(self):
        """base_url with existing path component should join correctly."""
        url = "fig1.png"
        base_url = "https://example.com/papers/paper1/"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/papers/paper1/fig1.png"

    def test_base_url_ending_with_file(self):
        """base_url ending with a file should replace it when joining."""
        url = "fig1.png"
        base_url = "https://example.com/index.html"
        result = resolve_figure_url(url, base_url=base_url)
        assert result == "https://example.com/fig1.png"

    def test_base_url_with_query_params(self):
        """base_url with query parameters should be handled correctly."""
        url = "fig1.png"
        base_url = "https://example.com/papers/?section=1"
        result = resolve_figure_url(url, base_url=base_url)
        # Query params from base_url may or may not be preserved depending on urljoin behavior
        assert result.startswith("https://example.com/papers/")
        assert "fig1.png" in result

    # ═══════════════════════════════════════════════════════════════════════
    # base_path Edge Cases
    # ═══════════════════════════════════════════════════════════════════════

    def test_base_path_resolves_symlinks(self, paper_loader_base_path):
        """base_path should resolve symlinks correctly."""
        url = "fig1.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        # Should be resolved (no symlinks in path)
        resolved_result = Path(result).resolve()
        assert str(resolved_result) == result

    def test_base_path_with_special_characters(self, tmp_path):
        """base_path with special characters should work correctly."""
        base_path = tmp_path / "test-dir_with.special"
        base_path.mkdir(parents=True, exist_ok=True)
        url = "fig1.png"
        result = resolve_figure_url(url, base_path=base_path)
        expected = base_path / "fig1.png"
        assert result == str(expected.resolve())

