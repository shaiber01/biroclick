"""Shared fixtures for state-machine test suites."""

import pytest

from schemas.state import create_initial_state


@pytest.fixture
def initial_state():
    return create_initial_state(
        paper_id="test_paper",
        paper_text="Gold nanorods have LSPR at 700nm." * 10,
    )



