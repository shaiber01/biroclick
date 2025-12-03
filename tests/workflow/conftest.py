import pytest

from schemas.state import create_initial_state

from tests.workflow.fixtures import load_fixture


@pytest.fixture
def paper_input():
    """Load sample paper input."""
    return load_fixture("sample_paper_input.json")


@pytest.fixture
def base_state(paper_input):
    """Create base state from paper input."""
    state = create_initial_state(
        paper_id=paper_input["paper_id"],
        paper_text=paper_input["paper_text"],
        paper_domain=paper_input.get("paper_domain", "other"),
    )
    state["paper_figures"] = paper_input.get("paper_figures", [])
    return state


