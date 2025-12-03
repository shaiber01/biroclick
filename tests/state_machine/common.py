"""Shared helpers for the LangGraph state-machine E2E tests."""

import uuid
from typing import List
from unittest.mock import patch

# Ensure src.graph is imported so patch targets such as `src.graph.ask_user_node`
# exist before tests attempt to monkeypatch them.
import src.graph  # noqa: F401


# All modules that import call_agent_with_metrics
LLM_PATCH_LOCATIONS = [
    "src.agents.planning.call_agent_with_metrics",
    "src.agents.design.call_agent_with_metrics",
    "src.agents.code.call_agent_with_metrics",
    "src.agents.execution.call_agent_with_metrics",
    "src.agents.analysis.call_agent_with_metrics",
    "src.agents.supervision.supervisor.call_agent_with_metrics",
    "src.agents.reporting.call_agent_with_metrics",
]

CHECKPOINT_PATCH_LOCATIONS = [
    "schemas.state.save_checkpoint",
    "src.graph.save_checkpoint",
    "src.routing.save_checkpoint",
]


def unique_thread_id(prefix: str = "test") -> str:
    """Generate a unique thread ID to prevent state pollution between tests."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_mock_ask_user_node():
    """
    Create a mock ask_user_node that returns user_responses from state.

    This avoids interactive stdin reads in tests while still exercising
    the graph routing logic.
    """

    def mock_ask_user(state):
        """Mock ask_user that returns already-injected user_responses."""
        user_responses = state.get("user_responses", {})
        trigger = state.get("ask_user_trigger", "unknown")
        print(
            f"    [MOCK ask_user] trigger={trigger}, responses={list(user_responses.keys())}",
            flush=True,
        )

        return {
            "awaiting_user_input": False,
            "pending_user_questions": [],
        }

    return mock_ask_user


class MultiPatch:
    """Context manager to patch multiple locations with the same mock."""

    def __init__(self, locations: List[str], side_effect=None, return_value=None):
        self.locations = locations
        self.side_effect = side_effect
        self.return_value = return_value
        self.patches = []
        self.mocks = []

    def __enter__(self):
        for loc in self.locations:
            p = patch(loc, side_effect=self.side_effect, return_value=self.return_value)
            mock = p.start()
            self.patches.append(p)
            self.mocks.append(mock)
        return self.mocks

    def __exit__(self, *args):
        for p in self.patches:
            p.stop()


class MockLLMResponses:
    """Mock LLM responses that satisfy node validation logic."""

    @staticmethod
    def prompt_adaptor() -> dict:
        return {
            "adaptations": [],
            "paper_domain": "plasmonics",
        }

    @staticmethod
    def planner(paper_id: str = "test_gold_nanorod") -> dict:
        """A minimal valid plan with one stage."""
        return {
            "paper_id": paper_id,
            "paper_domain": "plasmonics",
            "title": "Gold Nanorod Optical Properties",
            "summary": "Reproduce extinction spectrum using FDTD",
            "main_system": "Gold nanorod in water",
            "main_claims": ["Longitudinal plasmon at 700nm"],
            "simulation_approach": "FDTD with Meep",
            "extracted_parameters": [
                {"name": "length", "value": 100, "unit": "nm", "source": "text", "location": "p1"},
                {"name": "diameter", "value": 40, "unit": "nm", "source": "text", "location": "p1"},
            ],
            "planned_materials": [
                {"material_id": "gold_jc", "name": "Gold (Johnson-Christy)"},
            ],
            "targets": [
                {
                    "figure_id": "Fig1",
                    "description": "Extinction spectrum",
                    "type": "spectrum",
                    "simulation_class": "FDTD_DIRECT",
                    "precision_requirement": "acceptable",
                }
            ],
            "stages": [
                {
                    "stage_id": "stage_0_materials",
                    "stage_type": "MATERIAL_VALIDATION",
                    "name": "Material Validation",
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                    "runtime_budget_minutes": 5,
                }
            ],
            "assumptions": {},
            "reproduction_scope": {
                "total_figures": 1,
                "reproducible_figures": 1,
                "attempted_figures": ["Fig1"],
                "skipped_figures": [],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage_0_materials", "status": "not_started", "summary": ""}
                ]
            },
        }

    @staticmethod
    def plan_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {
                    "status": "pass",
                    "figures_covered": ["Fig1"],
                    "figures_missing": [],
                    "notes": "All figures covered",
                },
                "digitized_data": {
                    "status": "pass",
                    "excellent_targets": [],
                    "have_digitized": [],
                    "missing_digitized": [],
                    "notes": "N/A",
                },
                "staging": {
                    "status": "pass",
                    "stage_0_present": True,
                    "stage_1_present": True,
                    "validation_hierarchy_followed": True,
                    "dependency_issues": [],
                    "notes": "OK",
                },
                "parameter_extraction": {
                    "status": "pass",
                    "extracted_count": 2,
                    "cross_checked_count": 2,
                    "missing_critical": [],
                    "notes": "OK",
                },
                "assumptions": {
                    "status": "pass",
                    "assumption_count": 0,
                    "risky_assumptions": [],
                    "undocumented_gaps": [],
                    "notes": "OK",
                },
                "performance": {
                    "status": "pass",
                    "total_estimated_runtime_min": 5,
                    "budget_min": 15,
                    "risky_stages": [],
                    "notes": "OK",
                },
                "material_validation_setup": {
                    "status": "pass",
                    "materials_covered": ["gold"],
                    "materials_missing": [],
                    "validation_criteria_clear": True,
                    "notes": "OK",
                },
                "output_specifications": {
                    "status": "pass",
                    "all_stages_have_outputs": True,
                    "figure_mappings_complete": True,
                    "notes": "OK",
                },
            },
            "issues": [],
            "strengths": ["Good parameter extraction"],
            "summary": "Plan is valid",
        }

    @staticmethod
    def plan_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {
                    "status": "pass",
                    "figures_covered": ["Fig1"],
                    "figures_missing": [],
                    "notes": "OK",
                },
                "digitized_data": {
                    "status": "pass",
                    "excellent_targets": [],
                    "have_digitized": [],
                    "missing_digitized": [],
                    "notes": "N/A",
                },
                "staging": {
                    "status": "pass",
                    "stage_0_present": True,
                    "stage_1_present": True,
                    "validation_hierarchy_followed": True,
                    "dependency_issues": [],
                    "notes": "OK",
                },
                "parameter_extraction": {
                    "status": "fail",
                    "extracted_count": 2,
                    "cross_checked_count": 0,
                    "missing_critical": ["material source"],
                    "notes": "Missing material source",
                },
                "assumptions": {
                    "status": "pass",
                    "assumption_count": 0,
                    "risky_assumptions": [],
                    "undocumented_gaps": [],
                    "notes": "OK",
                },
                "performance": {
                    "status": "pass",
                    "total_estimated_runtime_min": 5,
                    "budget_min": 15,
                    "risky_stages": [],
                    "notes": "OK",
                },
                "material_validation_setup": {
                    "status": "fail",
                    "materials_covered": [],
                    "materials_missing": ["gold"],
                    "validation_criteria_clear": False,
                    "notes": "Missing material source",
                },
                "output_specifications": {
                    "status": "pass",
                    "all_stages_have_outputs": True,
                    "figure_mappings_complete": True,
                    "notes": "OK",
                },
            },
            "issues": [
                {
                    "severity": "blocking",
                    "category": "material_validation",
                    "description": "Missing material source",
                    "suggested_fix": "Add material database reference",
                }
            ],
            "strengths": [],
            "summary": "Plan needs work",
        }

    @staticmethod
    def simulation_designer(stage_id: str = "stage_0_materials") -> dict:
        return {
            "stage_id": stage_id,
            "design_description": "Gold nanorod FDTD simulation for extinction spectrum",
            "unit_system": {
                "characteristic_length_m": 1e-9,
                "length_unit": "nm",
                "example_conversions": {"100nm_to_meep": 100},
            },
            "geometry": {
                "dimensionality": "3D",
                "cell_size": {"x": 400, "y": 400, "z": 800},
                "resolution": 20,
                "structures": [
                    {
                        "name": "nanorod",
                        "type": "cylinder",
                        "material_ref": "gold_jc",
                        "center": {"x": 0, "y": 0, "z": 0},
                        "dimensions": {"radius": 20, "height": 100},
                        "real_dimensions": {"radius_nm": 20, "height_nm": 100},
                    }
                ],
                "symmetries": [],
            },
            "materials": [
                {
                    "id": "gold_jc",
                    "name": "Gold (Johnson-Christy)",
                    "model_type": "drude_lorentz",
                    "source": "johnson_christy",
                    "data_file": "materials/johnson_christy_gold.csv",
                    "parameters": {},
                    "wavelength_range": {"min_nm": 400, "max_nm": 900},
                },
                {
                    "id": "water",
                    "name": "Water",
                    "model_type": "constant",
                    "source": "assumed",
                    "data_file": None,
                    "parameters": {"epsilon": 1.77},
                },
            ],
            "sources": [
                {
                    "type": "gaussian",
                    "component": "Ex",
                    "center": {"x": 0, "y": 0, "z": -300},
                    "size": {"x": 400, "y": 400, "z": 0},
                    "wavelength_center_nm": 650,
                    "wavelength_width_nm": 400,
                    "frequency_center_meep": 1.54,
                    "frequency_width_meep": 0.95,
                }
            ],
            "monitors": [
                {
                    "name": "flux_monitor",
                    "type": "flux",
                    "center": {"x": 0, "y": 0, "z": 300},
                    "size": {"x": 200, "y": 200, "z": 0},
                }
            ],
            "post_processing": [
                {
                    "type": "spectrum",
                    "input_monitor": "flux_monitor",
                    "output_name": "extinction_spectrum",
                }
            ],
        }

    @staticmethod
    def design_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "checklist_results": {
                "geometry_valid": True,
                "materials_defined": True,
                "sources_valid": True,
                "monitors_valid": True,
                "resolution_sufficient": True,
            },
            "issues": [],
            "summary": "Design is solid",
        }

    @staticmethod
    def design_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "checklist_results": {
                "geometry_valid": False,
                "materials_defined": True,
                "sources_valid": True,
                "monitors_valid": True,
                "resolution_sufficient": True,
            },
            "issues": [
                {"severity": "blocking", "description": "Geometry exceeds cell boundaries"}
            ],
            "summary": "Geometry issue",
        }

    @staticmethod
    def code_generator() -> dict:
        return {
            "code": "print('Simulating...')",
            "filename": "simulation.py",
            "execution_command": "python simulation.py",
            "required_packages": ["meep"],
            "description": "FDTD simulation script",
        }

    @staticmethod
    def code_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good",
        }

    @staticmethod
    def code_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "issues": [{"severity": "blocking", "description": "Syntax error on line 5"}],
            "summary": "Fix syntax error",
        }

    @staticmethod
    def execution_validator_pass() -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Execution successful",
        }

    @staticmethod
    def execution_validator_fail() -> dict:
        return {
            "verdict": "fail",
            "issues": [{"severity": "blocking", "description": "RuntimeError: out of memory"}],
            "summary": "Execution failed",
        }

    @staticmethod
    def physics_sanity_pass() -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Physics looks reasonable",
        }

    @staticmethod
    def physics_sanity_fail() -> dict:
        return {
            "verdict": "fail",
            "issues": [{"severity": "blocking", "description": "Negative energy detected"}],
            "summary": "Physics violation",
        }

    @staticmethod
    def physics_sanity_design_flaw() -> dict:
        return {
            "verdict": "design_flaw",
            "issues": [
                {
                    "severity": "blocking",
                    "description": "Resolution too low for plasmon",
                }
            ],
            "summary": "Fundamental design issue",
        }

    @staticmethod
    def results_analyzer() -> dict:
        return {
            "key_findings": ["Peak at 705nm"],
            "extracted_values": [{"name": "LSPR_peak", "value": 705, "unit": "nm"}],
            "figures_generated": ["spectrum.png"],
            "summary": "Analysis complete",
        }

    @staticmethod
    def comparison_validator() -> dict:
        return {
            "verdict": "approve",
            "match_quality": "good",
            "discrepancies": [],
            "summary": "Good match with target",
        }

    @staticmethod
    def supervisor_continue() -> dict:
        return {
            "verdict": "ok_continue",
            "reasoning": "Stage complete, proceed to next",
            "suggested_next_step": "select_stage",
        }

    @staticmethod
    def supervisor_replan() -> dict:
        return {
            "verdict": "replan_needed",
            "reasoning": "Results inconsistent, need new approach",
            "suggested_next_step": "plan",
        }

    @staticmethod
    def report_generator() -> dict:
        return {
            "report_path": "reproduction_report.md",
            "summary": "Reproduction successful",
        }


__all__ = [
    "LLM_PATCH_LOCATIONS",
    "CHECKPOINT_PATCH_LOCATIONS",
    "MultiPatch",
    "MockLLMResponses",
    "create_mock_ask_user_node",
    "unique_thread_id",
]


