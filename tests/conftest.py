"""Global shared fixtures for all tests."""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from schemas.state import create_initial_state


# ═══════════════════════════════════════════════════════════════════════
# Validated Mock Response Loading
# ═══════════════════════════════════════════════════════════════════════

# Path to the mock responses that are validated by tests/test_llm_contracts.py
MOCK_RESPONSES_DIR = Path(__file__).parent / "fixtures" / "mock_responses"

def load_validated_mock(agent_name: str) -> Dict[str, Any]:
    """Load a validated mock response from the fixtures directory."""
    path = MOCK_RESPONSES_DIR / f"{agent_name}_response.json"
    if not path.exists():
        # Fallback or error if mock doesn't exist yet
        raise FileNotFoundError(f"Validated mock not found for {agent_name} at {path}")
    with open(path, "r") as f:
        return json.load(f)

@pytest.fixture
def validated_planner_response():
    return load_validated_mock("planner")

@pytest.fixture
def validated_plan_reviewer_response():
    return load_validated_mock("plan_reviewer")

@pytest.fixture
def validated_simulation_designer_response():
    return load_validated_mock("simulation_designer")

@pytest.fixture
def validated_design_reviewer_response():
    return load_validated_mock("design_reviewer")

@pytest.fixture
def validated_code_generator_response():
    return load_validated_mock("code_generator")

@pytest.fixture
def validated_code_reviewer_response():
    return load_validated_mock("code_reviewer")

@pytest.fixture
def validated_execution_validator_response():
    return load_validated_mock("execution_validator")

@pytest.fixture
def validated_physics_sanity_response():
    return load_validated_mock("physics_sanity")

@pytest.fixture
def validated_results_analyzer_response():
    return load_validated_mock("results_analyzer")

@pytest.fixture
def validated_supervisor_response():
    return load_validated_mock("supervisor")


# ═══════════════════════════════════════════════════════════════════════
# Shared State Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Create a base state with realistic paper content."""
    paper_text = (
        """
    We study the optical properties of gold nanorods with length 100nm and 
    diameter 40nm using FDTD simulations. The localized surface plasmon 
    resonance (LSPR) is observed at approximately 650nm wavelength.
    
    Materials: Gold optical constants from Johnson & Christy (1972).
    Surrounding medium: Water (n=1.33).
    
    Figure 1 shows the extinction spectrum with the longitudinal mode peak.
    Figure 2 shows near-field enhancement maps at resonance.
    """
        * 5
    )

    state = create_initial_state(
        paper_id="test_integration",
        paper_text=paper_text,
        paper_domain="plasmonics",
    )
    state["paper_figures"] = [
        {"figure_id": "Fig1", "description": "Extinction spectrum"},
        {"figure_id": "Fig2", "description": "Near-field map"},
    ]
    return state


@pytest.fixture
def valid_plan():
    """A valid plan structure for testing."""
    return {
        "paper_id": "test_integration",
        "title": "Gold Nanorod Optical Properties",
        "stages": [
            {
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Validate gold optical constants",
                "targets": ["material_gold"],
                "dependencies": [],
            }
        ],
        "targets": [{"figure_id": "Fig1", "description": "Test"}],
        "extracted_parameters": [
            {"name": "length", "value": 100, "unit": "nm", "source": "text"},
        ],
    }

