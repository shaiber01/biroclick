"""Shared fixtures for routing tests."""

import pytest
from unittest.mock import patch

from schemas.state import create_initial_state, ReproState


@pytest.fixture
def base_state() -> ReproState:
    """Create a baseline repro state for routing tests."""
    state = create_initial_state(
        paper_id="test_paper",
        paper_text="Test paper content",
    )
    return state


@pytest.fixture
def mock_save_checkpoint():
    """Mock save_checkpoint to avoid touching disk in routing tests."""
    with patch("src.routing.save_checkpoint") as mock:
        yield mock



