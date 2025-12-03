"""Shared fixtures and helpers for paper loader test modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "paper_loader"


@pytest.fixture
def paper_loader_base_path(tmp_path: Path) -> Path:
    """Provide a deterministic base path for markdown resolver tests."""
    base = tmp_path / "papers" / "paper-under-test"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def basic_paper_input() -> Dict[str, Any]:
    """Create a representative paper payload for cost estimation tests."""
    return {
        "paper_id": "test",
        "paper_title": "Test",
        "paper_text": "A" * 10_000,
        "figures": _build_figures(["Fig1", "Fig2"]),
    }


def _build_figures(ids: List[str]) -> List[Dict[str, Any]]:
    """Helper to keep dummy figure construction in a single place."""
    return [
        {
            "id": figure_id,
            "description": f"Figure {index}",
            "image_path": f"{figure_id.lower()}.png",
        }
        for index, figure_id in enumerate(ids, start=1)
    ]


def create_valid_paper_input() -> Dict[str, Any]:
    """Create a minimal valid paper input for validation-heavy tests."""
    return {
        "paper_id": "test_paper",
        "paper_title": "Test Paper Title",
        "paper_text": "A" * 150,
        "figures": [
            {
                "id": "Fig1",
                "description": "Test figure",
                "image_path": str(FIXTURES_DIR / "sample_images" / "test_figure.png"),
            }
        ],
    }


@pytest.fixture
def valid_paper_input() -> Dict[str, Any]:
    """Return a fresh valid paper input per test."""
    return create_valid_paper_input()


@pytest.fixture
def paper_input_factory() -> Callable[..., Dict[str, Any]]:
    """Return a factory for customized paper inputs."""

    def _factory(**overrides: Any) -> Dict[str, Any]:
        paper_input = create_valid_paper_input()
        paper_input.update(overrides)
        return paper_input

    return _factory


@pytest.fixture
def sample_image_path() -> Path:
    """Return path to a real sample figure image."""
    return FIXTURES_DIR / "sample_images" / "test_figure.png"


def make_sample_paper_input() -> Dict[str, Any]:
    """Return a reusable sample paper input payload."""
    return {
        "paper_id": "test",
        "paper_title": "Test",
        "paper_text": "A" * 150,
        "figures": [
            {"id": "Fig1", "description": "First", "image_path": "fig1.png"},
            {"id": "Fig2", "description": "Second", "image_path": "fig2.png"},
        ],
        "supplementary": {
            "supplementary_text": "Supplementary content",
            "supplementary_figures": [
                {"id": "S1", "description": "Supp fig", "image_path": "s1.png"}
            ],
            "supplementary_data_files": [
                {
                    "id": "D1",
                    "description": "Data 1",
                    "file_path": "d1.csv",
                    "data_type": "spectrum",
                },
                {
                    "id": "D2",
                    "description": "Data 2",
                    "file_path": "d2.csv",
                    "data_type": "geometry",
                },
            ],
        },
    }


@pytest.fixture
def sample_paper_input() -> Dict[str, Any]:
    """Fixture that provides the reusable sample paper input."""
    return make_sample_paper_input()


@pytest.fixture(scope="session")
def paper_loader_fixtures_dir() -> Path:
    """Expose the fixtures directory path for tests that need raw assets."""
    return FIXTURES_DIR
