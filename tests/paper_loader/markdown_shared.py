from __future__ import annotations

from typing import List, Mapping

from src.paper_loader.markdown_parser import extract_figures_from_markdown


def parse_figures(md: str) -> List[Mapping[str, str]]:
    """Convenience wrapper for figure extraction."""
    return extract_figures_from_markdown(md)


def assert_single_figure(md: str, *, url: str, alt: str = "") -> Mapping[str, str]:
    """Extract a single figure and assert its key fields."""
    figures = parse_figures(md)
    assert len(figures) == 1, f"Expected 1 figure but got {len(figures)}"
    figure = figures[0]
    assert figure["url"] == url
    if alt is not None:
        assert figure["alt"] == alt
    return figure



