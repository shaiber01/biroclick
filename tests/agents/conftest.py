"""Shared fixtures for agent tests."""

import pytest
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from schemas.state import (
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_REPLANS,
)
from tests.agents.shared_objects import (
    CLI_MATERIAL_CHECKPOINT_PROMPT,
    LONG_FALLBACK_JSON,
    LONG_FALLBACK_PAYLOAD,
    NonSerializable,
)


# ═══════════════════════════════════════════════════════════════════════
# State Fixtures (Preserved as agent-specific)
# ═══════════════════════════════════════════════════════════════════════

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
        "ask_user_trigger": None,  # Single mechanism for user interaction routing
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
                "stage_id": "stage_1",
                "stage_type": "MATERIAL_VALIDATION",
                "name": "Stage 1",
                "description": "Test goal",
                "dependencies": [],
                "targets": ["Figure 1"],
            }
        ],
        "assumptions": {},
        "materials": [],
    }
    return minimal_state


@pytest.fixture
def state_with_stage(state_with_plan) -> Dict[str, Any]:
    """State with a current stage selected."""
    state_with_plan["current_stage"] = "stage_1"
    state_with_plan["stage_results"] = {
        "stage_1": {
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
                "stage_id": "stage_1_spherical_particle",
                "stage_type": "SINGLE_STRUCTURE",
                "name": "stage_1_spherical_particle",
                "description": "Simulate absorption spectrum of 50nm gold sphere",
                "dependencies": [],
                "targets": ["Figure 1a"],
                "parameters": {
                    "particle_radius": "50nm",
                    "wavelength_range": "400-800nm",
                },
            },
            {
                "stage_id": "stage_2_ellipsoidal_particle",
                "stage_type": "SINGLE_STRUCTURE",
                "name": "stage_2_ellipsoidal_particle",
                "description": "Simulate absorption spectrum of ellipsoidal gold particle",
                "dependencies": ["stage_1_spherical_particle"],
                "targets": ["Figure 1b"],
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


# ═══════════════════════════════════════════════════════════════════════
# Shared state templates & helpers
# ═══════════════════════════════════════════════════════════════════════

def _clone(template: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(template)


_ANALYSIS_STATE_TEMPLATE = {
    "paper_id": "test_paper",
    "current_stage_id": "stage_1_sim",
    "paper_figures": [{"id": "Fig1", "image_path": "fig1.png"}],
    "plan": {
        "stages": [
            {
                "stage_id": "stage_1_sim",
                "targets": ["Fig1"],
                "target_details": [
                    {"figure_id": "Fig1", "precision_requirement": "acceptable"}
                ],
                "expected_outputs": [
                    {
                        "target_figure": "Fig1",
                        "filename_pattern": "output.csv",
                        "columns": ["x", "y"],
                    }
                ],
            }
        ],
        "targets": [{"figure_id": "Fig1", "precision_requirement": "acceptable"}],
    },
    "stage_outputs": {"files": ["simulation_stage_1_sim.py", "output.csv"]},
    "analysis_revision_count": 0,
    "analysis_result_reports": [],
    "figure_comparisons": [],
}

_CODE_STATE_TEMPLATE = {
    "paper_id": "test_paper",
    "current_stage_id": "stage_1_sim",
    "current_stage_type": "SINGLE_STRUCTURE",
    "design_description": (
        "Simulate a gold nanorod with length 100nm and diameter 40nm using FDTD. "
        "This description is long enough to pass validation checks (>50 chars)."
    ),
    "plan": {"stages": [{"stage_id": "stage_1_sim", "targets": ["Fig1"]}]},
    "validated_materials": [{"material_id": "gold", "path": "/materials/gold.csv"}],
    "code": "import meep as mp\n# Valid simulation code structure\n# ... more lines ...",
    "code_revision_count": 0,
    "design_revision_count": 0,
    "runtime_config": {
        "max_code_revisions": MAX_CODE_REVISIONS,
        "max_design_revisions": MAX_DESIGN_REVISIONS,
    },
}

_DESIGN_STATE_TEMPLATE = {
    "paper_id": "test_paper",
    "current_stage_id": "stage_1_sim",
    "plan": {
        "stages": [
            {
                "stage_id": "stage_1_sim",
                "type": "simulation",
                "name": "Simulation Stage",
                "targets": ["Fig1"],
                "complexity_class": "standard",
            }
        ]
    },
    "design_revision_count": 0,
    "runtime_config": {"max_design_revisions": MAX_DESIGN_REVISIONS},
    "assumptions": {
        "global_assumptions": [
            {"id": "existing_1", "description": "Existing assumption"}
        ]
    },
    "validated_materials": {},
    "paper_text": "Full paper text...",
    "paper_domain": "nanophotonics",
}

_EXECUTION_STATE_TEMPLATE = {
    "paper_id": "test_paper",
    "current_stage_id": "stage_1_sim",
    "stage_outputs": {
        "stdout": "Simulation running...",
        "stderr": "",
        "exit_code": 0,
        "files": ["output.csv"],
        "timeout_exceeded": False,
    },
    "run_error": None,
    "execution_failure_count": 0,
    "physics_failure_count": 0,
    "design_revision_count": 0,
    "design_description": {"parameters": {"p1": 10}},
    "runtime_config": {
        "max_execution_failures": MAX_EXECUTION_FAILURES,
        "max_physics_failures": MAX_PHYSICS_FAILURES,
        "max_design_revisions": MAX_DESIGN_REVISIONS,
    },
}

_PLAN_STATE_TEMPLATE = {
    "paper_text": "x" * 500,
    "paper_id": "test_paper",
    "replan_count": 0,
    "runtime_config": {"max_replans": MAX_REPLANS},
}

_PLANNER_LLM_OUTPUT_TEMPLATE = {
    "paper_id": "test_paper",
    "paper_domain": "plasmonics",
    "title": "Test Paper",
    "summary": "A test paper summary.",
    "stages": [
        {"stage_id": "stage_0_materials", "stage_type": "MATERIAL_VALIDATION", "targets": ["mat1"]},
        {"stage_id": "stage_1_sim", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig1"]},
    ],
    "targets": [{"figure_id": "Fig1"}],
    "extracted_parameters": [{"name": "param1", "value": 10}],
    "planned_materials": ["Gold"],
    "assumptions": {"assumption1": "true"},
}

_REPORT_STATE_TEMPLATE = {
    "paper_id": "test_paper",
    "paper_title": "Test Paper",
    "progress": {"stages": [{"stage_id": "stage1", "status": "completed_success"}]},
    "metrics": {"agent_calls": []},
    "figure_comparisons": [{"fig_id": "fig1", "status": "match"}],
    "assumptions": {"temp": "300K"},
    "discrepancies": [
        {"parameter": "gap", "classification": "minor", "likely_cause": "noise"}
    ],
}


@pytest.fixture
def analysis_state():
    """Shared analysis-state fixture for analyzer + validator suites."""
    return _clone(_ANALYSIS_STATE_TEMPLATE)


@pytest.fixture
def code_state():
    """Shared code-state fixture for code generator/reviewer."""
    return _clone(_CODE_STATE_TEMPLATE)


@pytest.fixture
def design_state():
    """Shared design-state fixture."""
    return _clone(_DESIGN_STATE_TEMPLATE)


@pytest.fixture
def execution_state():
    """Shared execution-state fixture for validator + physics tests."""
    return _clone(_EXECUTION_STATE_TEMPLATE)


@pytest.fixture
def plan_state():
    """Shared planning-state fixture."""
    return _clone(_PLAN_STATE_TEMPLATE)


@pytest.fixture
def planner_llm_output():
    """Canonical successful planner response."""
    return _clone(_PLANNER_LLM_OUTPUT_TEMPLATE)


@pytest.fixture
def report_state():
    """State used by reporting/handle_backtrack tests."""
    return _clone(_REPORT_STATE_TEMPLATE)


@pytest.fixture
def non_serializable_object():
    """Reusable NonSerializable helper."""
    return NonSerializable()


@pytest.fixture
def fallback_payload():
    """Reusable LONG_FALLBACK payload."""
    return deepcopy(LONG_FALLBACK_PAYLOAD)


# ═══════════════════════════════════════════════════════════════════════
# Patch helpers
# ═══════════════════════════════════════════════════════════════════════

@contextmanager
def patch_agent_stack(
    module_path: str,
    *,
    prompt: Optional[str] = "System Prompt",
    llm_response: Optional[Any] = None,
    context_response: Optional[Any] = None,
    user_content_attr: Optional[str] = None,
    user_content: Optional[Any] = None,
):
    """Patch the standard prompt/context/LLM stack for an agent module."""

    prompt_patch = patch(f"{module_path}.build_agent_prompt", return_value=prompt)
    context_patch = patch(
        f"{module_path}.check_context_or_escalate", return_value=context_response
    )
    call_patch = patch(
        f"{module_path}.call_agent_with_metrics", return_value=llm_response
    )

    user_content_patch = (
        patch(f"{module_path}.{user_content_attr}", return_value=user_content)
        if user_content_attr
        else nullcontext(None)
    )

    with prompt_patch as prompt_mock, context_patch as context_mock, call_patch as call_mock, user_content_patch as uc_mock:
        yield {
            "build_agent_prompt": prompt_mock,
            "check_context_or_escalate": context_mock,
            "call_agent_with_metrics": call_mock,
            "user_content_builder": uc_mock,
        }


@pytest.fixture
def agent_stack_patcher():
    """Fixture exposing the patch_agent_stack context manager for reuse."""

    def _patch(**kwargs):
        return patch_agent_stack(**kwargs)

    return _patch


@contextmanager
def patched_call_agent_with_metrics(module_path: str, response: Optional[Any] = None):
    """Utility to patch call_agent_with_metrics only."""
    with patch(f"{module_path}.call_agent_with_metrics", return_value=response) as mock_call:
        yield mock_call


@pytest.fixture
def material_checkpoint_questions():
    """Standardized CLI prompt list for material checkpoint tests."""
    return list(CLI_MATERIAL_CHECKPOINT_PROMPT)

