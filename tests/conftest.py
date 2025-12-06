"""Global shared fixtures for all tests."""

import json
import os
import pytest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from schemas.state import create_initial_state


# ═══════════════════════════════════════════════════════════════════════
# Disable LangSmith Tracing - Prevents tracing overhead/costs in tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True, scope="session")
def disable_langsmith_tracing():
    """
    Disable LangSmith tracing for all tests.
    
    This runs once at the start of the test session and prevents:
    - Tracing overhead slowing down tests
    - Accidental tracing costs
    - Network calls to LangSmith servers
    """
    # Store original values
    original_values = {}
    keys_to_disable = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGSMITH_TRACING",
    ]
    
    for key in keys_to_disable:
        original_values[key] = os.environ.get(key)
        os.environ[key] = "false"
    
    yield
    
    # Restore original values
    for key in keys_to_disable:
        if original_values[key] is not None:
            os.environ[key] = original_values[key]
        elif key in os.environ:
            del os.environ[key]


# ═══════════════════════════════════════════════════════════════════════
# LLM Call Protection - Prevents accidental real API calls in tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def auto_mock_llm(request, monkeypatch):
    """
    Auto-mock LLM client for all non-smoke tests.
    
    This provides a safety net - if a test forgets to mock the LLM client,
    it will fail immediately with a clear error instead of making a real
    (slow, costly) API call.
    
    Tests marked with @pytest.mark.smoke are excluded and CAN make real calls.
    """
    # Skip this protection for smoke tests (they intentionally make real calls)
    if "smoke" in [marker.name for marker in request.node.iter_markers()]:
        yield
        return
    
    def mock_get_llm_client(*args, **kwargs):
        raise RuntimeError(
            "\n\n"
            "═══════════════════════════════════════════════════════════════════\n"
            "  REAL LLM CALL ATTEMPTED IN NON-SMOKE TEST!\n"
            "═══════════════════════════════════════════════════════════════════\n"
            "\n"
            "  This test tried to call the real LLM API without proper mocking.\n"
            "\n"
            "  To fix, either:\n"
            "    1. Add @patch('src.llm_client.call_agent_with_metrics') to mock LLM\n"
            "    2. Mark test with @pytest.mark.smoke if it needs real LLM calls\n"
            "\n"
            "═══════════════════════════════════════════════════════════════════\n"
        )
    
    monkeypatch.setattr("src.llm_client.get_llm_client", mock_get_llm_client)
    yield


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

