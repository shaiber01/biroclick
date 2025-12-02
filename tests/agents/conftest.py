"""Shared fixtures for agent tests."""

import pytest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch


@pytest.fixture
def minimal_state() -> Dict[str, Any]:
    """Minimal ReproState for testing."""
    return {
        "paper_id": "test_paper",
        "paper_text": "Test paper content about optical simulations.",
        "target_figures": ["Figure 1"],
        "plan": None,
        "prompt_adaptations": None,
        "current_stage": None,
        "stage_results": {},
        "execution_attempts": {},
        "discrepancies": [],
        "user_responses": {},
        "plan_revisions": 0,
        "design_revision_count": 0,
        "code_revision_count": 0,
        "execution_attempt_count": 0,
        "physics_revision_count": 0,
        "supervisor_call_count": 0,
        "comparison_failures": 0,
        "comparison_succeeded": [],
        "comparison_failed": [],
        "completed_stages": [],
        "skipped_stages": [],
        "awaiting_user_input": False,
        "user_question": None,
        "final_report": None,
        "context_overflow_triggered": False,
        "escalation_message": None,
        "backtrack_stages": [],
        "supervisor_verdict": None,
        "validated_materials": [],
        "flagged_materials": [],
        "material_resolution_complete": False,
        "material_usage": {},
    }


@pytest.fixture
def state_with_plan(minimal_state) -> Dict[str, Any]:
    """State with a valid plan."""
    minimal_state["plan"] = {
        "title": "Test Plan",
        "stages": [
            {
                "name": "Stage 1",
                "goal": "Test goal",
                "dependencies": [],
                "target_figures": ["Figure 1"],
            }
        ],
        "assumptions": {},
        "materials": [],
    }
    return minimal_state


@pytest.fixture
def state_with_stage(state_with_plan) -> Dict[str, Any]:
    """State with a current stage selected."""
    state_with_plan["current_stage"] = "Stage 1"
    state_with_plan["stage_results"] = {
        "Stage 1": {
            "design": None,
            "code": None,
            "execution_result": None,
            "physics_result": None,
            "analysis_result": None,
        }
    }
    return state_with_plan


@pytest.fixture
def mock_llm_response():
    """Factory fixture for creating mock LLM responses."""
    def _make_response(response_data: Dict[str, Any]) -> MagicMock:
        mock = MagicMock()
        mock.return_value = response_data
        return mock
    return _make_response


@pytest.fixture
def mock_call_agent():
    """Fixture to mock call_agent_with_metrics across all agent modules."""
    agent_modules = [
        "src.agents.planning",
        "src.agents.stage_selection",
        "src.agents.design",
        "src.agents.code",
        "src.agents.execution",
        "src.agents.analysis",
        "src.agents.supervision",
        "src.agents.user_interaction",
        "src.agents.reporting",
    ]
    
    class MultiPatcher:
        def __init__(self, mock_fn):
            self.mock_fn = mock_fn
            self.patchers = []
            self.mocks = []
        
        def __enter__(self):
            for module in agent_modules:
                try:
                    patcher = patch(f"{module}.call_agent_with_metrics", self.mock_fn)
                    mock = patcher.start()
                    self.patchers.append(patcher)
                    self.mocks.append(mock)
                except AttributeError:
                    pass
            return self.mocks
        
        def __exit__(self, *args):
            for patcher in self.patchers:
                patcher.stop()
    
    return MultiPatcher


@pytest.fixture
def sample_numeric_data() -> str:
    """Sample CSV-like numeric data for testing."""
    return """wavelength,value
400,0.1
500,0.5
600,0.9
700,0.3"""


@pytest.fixture
def sample_material_database() -> Dict[str, Any]:
    """Sample material database for testing."""
    return {
        "materials": [
            {
                "name": "Silicon",
                "aliases": ["Si", "silicon", "c-Si"],
                "category": "semiconductor",
                "data_file": "palik_silicon.csv",
            },
            {
                "name": "Gold",
                "aliases": ["Au", "gold"],
                "category": "metal",
                "data_file": "palik_gold.csv",
            },
            {
                "name": "Silver",
                "aliases": ["Ag", "silver"],
                "category": "metal",
                "data_file": "palik_silver.csv",
            },
            {
                "name": "SiO2",
                "aliases": ["silica", "silicon dioxide", "glass"],
                "category": "dielectric",
                "data_file": "malitson_sio2.csv",
            },
        ]
    }


@pytest.fixture
def sample_plan() -> Dict[str, Any]:
    """A comprehensive sample plan for testing."""
    return {
        "title": "Reproduce Figure 1: Gold Nanoparticle Absorption",
        "stages": [
            {
                "name": "stage_1_spherical_particle",
                "goal": "Simulate absorption spectrum of 50nm gold sphere",
                "dependencies": [],
                "target_figures": ["Figure 1a"],
                "parameters": {
                    "particle_radius": "50nm",
                    "wavelength_range": "400-800nm",
                },
            },
            {
                "name": "stage_2_ellipsoidal_particle",
                "goal": "Simulate absorption spectrum of ellipsoidal gold particle",
                "dependencies": ["stage_1_spherical_particle"],
                "target_figures": ["Figure 1b"],
                "parameters": {
                    "major_axis": "75nm",
                    "minor_axis": "25nm",
                },
            },
        ],
        "assumptions": {
            "simulation_method": "FDTD",
            "boundary_conditions": "PML",
        },
        "materials": [
            {
                "name": "Gold",
                "source": "Palik",
            }
        ],
    }


@pytest.fixture
def sample_design() -> Dict[str, Any]:
    """A sample simulation design for testing."""
    return {
        "simulation_type": "FDTD",
        "geometry": {
            "type": "sphere",
            "radius": "50nm",
            "material": "Gold",
        },
        "sources": [
            {
                "type": "plane_wave",
                "direction": "x",
                "wavelength_range": [400, 800],
            }
        ],
        "monitors": [
            {
                "type": "flux",
                "name": "absorption",
            }
        ],
        "boundary_conditions": "PML",
        "mesh": {
            "resolution": 10,
        },
        "expected_outputs": [
            {
                "filename": "absorption.csv",
                "columns": ["wavelength", "absorption"],
            }
        ],
    }


@pytest.fixture
def sample_code() -> str:
    """Sample simulation code for testing."""
    return '''import meep as mp
import numpy as np

# Define simulation
sim = mp.Simulation(cell_size=mp.Vector3(2, 2, 2), resolution=10)

# Run simulation
sim.run(until=200)

# Save results
np.savetxt("absorption.csv", results, delimiter=",", header="wavelength,absorption")
'''


@pytest.fixture
def sample_execution_result() -> Dict[str, Any]:
    """Sample execution result for testing."""
    return {
        "success": True,
        "output_files": ["absorption.csv"],
        "stdout": "Simulation completed successfully.",
        "stderr": "",
        "execution_time": 5.2,
    }


@pytest.fixture  
def sample_analysis_result() -> Dict[str, Any]:
    """Sample analysis result for testing."""
    return {
        "verdict": "good_match",
        "metrics": {
            "peak_wavelength_error": 2.5,
            "peak_amplitude_error": 5.1,
            "rmse": 0.03,
        },
        "explanation": "Simulation matches reference data within acceptable tolerance.",
    }


