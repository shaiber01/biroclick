"""Shared fixtures for real integration tests."""

import pytest

from schemas.state import create_initial_state


@pytest.fixture
def minimal_state():
    """Create minimal valid state for node execution tests."""
    paper_text = (
        "We study the optical properties of gold nanorods with length 100nm and "
        "diameter 40nm. Using FDTD simulations, we calculate extinction spectra "
        "and near-field enhancement patterns. The localized surface plasmon "
        "resonance is observed at 650nm wavelength. This text is long enough to "
        "pass the planner's validation check which requires 100 characters."
    )
    state = create_initial_state(
        paper_id="test_paper",
        paper_text=paper_text,
        paper_domain="plasmonics",
    )
    state["plan"] = {
        "paper_id": "test_paper",
        "title": "Test Plan",
        "stages": [
            {
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Test stage",
                "targets": ["Fig1"],
                "dependencies": [],
            }
        ],
        "targets": [{"figure_id": "Fig1", "description": "Test"}],
        "extracted_parameters": [],
    }
    state["progress"] = {
        "stages": [
            {
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "status": "not_started",
                "dependencies": [],
            }
        ]
    }
    return state


@pytest.fixture
def mock_llm_response():
    """Return canned responses emulating the LLM for different agents."""

    def _mock(*args, **kwargs):
        agent = kwargs.get("agent_name", "unknown")
        if "reviewer" in agent:
            return {"verdict": "approve", "issues": [], "summary": "OK"}
        if "validator" in agent:
            return {"verdict": "pass", "issues": [], "summary": "OK"}
        if agent == "planner":
            return {
                "paper_id": "test",
                "title": "Test",
                "stages": [],
                "targets": [],
                "extracted_parameters": [],
            }
        if agent == "supervisor":
            return {"verdict": "ok_continue", "reasoning": "OK"}
        if agent == "report":
            return {
                "executive_summary": {"overall_assessment": []},
                "conclusions": ["Done"],
            }
        if agent == "physics_sanity":
            return {"verdict": "pass", "reasoning": "Physics looks good"}
        if agent == "prompt_adaptor":
            return {"adaptations": []}
        return {}

    return _mock


@pytest.fixture
def analysis_state():
    """State ready for analysis nodes that expect file outputs."""
    state = create_initial_state(
        paper_id="test_file_validation",
        paper_text="Gold nanorod optical simulation study." * 20,
        paper_domain="plasmonics",
    )
    state["plan"] = {
        "paper_id": "test_file_validation",
        "title": "File Validation Test",
        "stages": [
            {
                "stage_id": "stage_0",
                "stage_type": "SINGLE_STRUCTURE",
                "description": "Test stage",
                "targets": ["Fig1"],
                "dependencies": [],
            }
        ],
        "targets": [{"figure_id": "Fig1", "description": "Test spectrum"}],
    }
    state["progress"] = {
        "stages": [
            {
                "stage_id": "stage_0",
                "stage_type": "SINGLE_STRUCTURE",
                "status": "running",
                "dependencies": [],
            }
        ]
    }
    state["current_stage_id"] = "stage_0"
    state["current_stage_type"] = "SINGLE_STRUCTURE"
    return state




