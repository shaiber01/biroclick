"""Tests for resolve_figure_url."""

from __future__ import annotations

from pathlib import Path

from src.paper_loader.markdown_parser import resolve_figure_url


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

    def test_relative_with_base_path(self, paper_loader_base_path):
        url = "images/fig1.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path / "images/fig1.png"
        assert result == str(expected)

    def test_base_url_takes_precedence(self, paper_loader_base_path):
        url = "fig.png"
        result = resolve_figure_url(
            url,
            base_path=paper_loader_base_path,
            base_url="https://remote.com/",
        )
        assert result == "https://remote.com/fig.png"

    def test_parent_directory_traversal(self, paper_loader_base_path):
        """Handles .. in paths."""
        url = "../fig.png"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        expected = paper_loader_base_path.parent / "fig.png"
        assert Path(result).resolve() == expected.resolve()

    def test_security_path_traversal_outside_root(self, paper_loader_base_path):
        """Documents current traversal behavior."""
        url = "../../../../etc/passwd"
        result = resolve_figure_url(url, base_path=paper_loader_base_path)
        assert result.endswith("etc/passwd")

    def test_resolves_absolute_path_input(self):
        """Handles absolute path inputs correctly."""
        url = "/absolute/path/img.png"
        result = resolve_figure_url(url)
        assert result == "/absolute/path/img.png"

