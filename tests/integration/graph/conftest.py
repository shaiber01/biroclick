"""Shared fixtures for graph integration tests."""

from __future__ import annotations

from typing import Callable

import pytest

from schemas.state import ReproState
from tests.integration.graph.state_utils import build_state

StateFactory = Callable[..., ReproState]


@pytest.fixture
def repro_state_factory() -> StateFactory:
    """Factory fixture for creating mutable state dictionaries on demand."""

    def _factory(**kwargs) -> ReproState:
        return build_state(**kwargs)

    return _factory


@pytest.fixture
def test_state(repro_state_factory: StateFactory) -> ReproState:
    """Convenience fixture that returns a default state per test."""
    return repro_state_factory()


