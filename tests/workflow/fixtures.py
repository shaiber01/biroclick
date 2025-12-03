import json
from pathlib import Path

from src.agents.constants import AnalysisClassification


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / name
    with open(path, "r") as f:
        return json.load(f)


class MockResponseFactory:
    """Factory for creating coordinated mock LLM responses."""

    @staticmethod
    def planner_response(paper_id: str = "test_gold_nanorod") -> dict:
        """Create a valid planner response."""
        return {
            "paper_id": paper_id,
            "paper_domain": "plasmonics",
            "title": "Gold Nanorod Optical Properties",
            "summary": "Reproduce extinction spectrum of gold nanorod",
            "extracted_parameters": [
                {"name": "length", "value": 100, "unit": "nm", "source": "text"},
                {"name": "diameter", "value": 40, "unit": "nm", "source": "text"},
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
                    "description": "Validate gold optical properties",
                    "targets": ["material_gold"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1_extinction",
                    "stage_type": "SINGLE_STRUCTURE",
                    "name": "Extinction Spectrum",
                    "description": "Simulate nanorod extinction",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0_materials"],
                },
            ],
            "assumptions": {
                "global_assumptions": [
                    {
                        "id": "A1",
                        "category": "material",
                        "description": "Gold from Johnson & Christy",
                        "reason": "Standard database",
                        "source": "literature_default",
                    }
                ]
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage_0_materials",
                        "status": "not_started",
                        "stage_type": "MATERIAL_VALIDATION",
                    },
                    {
                        "stage_id": "stage_1_extinction",
                        "status": "not_started",
                        "stage_type": "SINGLE_STRUCTURE",
                    },
                ]
            },
            "planned_materials": [
                {
                    "material_id": "gold",
                    "name": "Gold",
                    "suggested_source": "johnson_christy",
                }
            ],
        }

    @staticmethod
    def reviewer_approve() -> dict:
        """Create an approve response for any reviewer."""
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Approved - meets all requirements",
        }

    @staticmethod
    def reviewer_needs_revision(
        feedback: str = "Needs improvement",
    ) -> dict:
        """Create a needs_revision response."""
        return {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": feedback}],
            "summary": feedback,
            "feedback": feedback,
        }

    @staticmethod
    def designer_response() -> dict:
        """Create a simulation designer response."""
        return {
            "design_description": "FDTD simulation of gold nanorod",
            "simulation_parameters": {
                "cell_size": [400, 200, 200],
                "resolution": 2,
                "pml_layers": 20,
            },
            "geometry_definitions": [
                {
                    "name": "nanorod",
                    "type": "cylinder",
                    "material": "gold",
                    "dimensions": {"length": 100, "radius": 20},
                }
            ],
            "source_configuration": {
                "type": "plane_wave",
                "polarization": "x",
                "wavelength_range": [400, 900],
            },
            "boundary_conditions": "PML",
            "output_configuration": {
                "monitors": ["flux"],
                "output_files": ["extinction.csv"],
            },
        }

    @staticmethod
    def code_generator_response() -> dict:
        """Create a code generator response."""
        return {
            "code": """import meep as mp
import numpy as np

# Gold nanorod FDTD simulation
cell = mp.Vector3(0.4, 0.2, 0.2)
resolution = 50

geometry = [
    mp.Cylinder(radius=0.02, height=0.1, material=mp.Medium(epsilon=1))
]

sim = mp.Simulation(cell_size=cell, geometry=geometry, resolution=resolution)
sim.run(until=100)

# Save extinction data
np.savetxt("extinction.csv", [[400, 0.5], [700, 1.0], [900, 0.3]])
""",
            "explanation": "FDTD simulation using Meep for gold nanorod extinction",
            "expected_outputs": ["extinction.csv"],
            "runtime_estimate_minutes": 5,
        }

    @staticmethod
    def execution_validator_pass() -> dict:
        """Create a pass response for execution validator."""
        return {
            "verdict": "pass",
            "summary": "Simulation completed successfully",
            "output_files_found": ["extinction.csv"],
            "runtime_actual_seconds": 120,
        }

    @staticmethod
    def execution_validator_fail(
        error: str = "Simulation crashed",
    ) -> dict:
        """Create a fail response for execution validator."""
        return {
            "verdict": "fail",
            "summary": error,
            "output_files_found": [],
            "error_classification": "runtime_error",
        }

    @staticmethod
    def physics_sanity_pass() -> dict:
        """Create a pass response for physics sanity."""
        return {
            "verdict": "pass",
            "summary": "Physics check passed - results are plausible",
            "checks_performed": ["energy_conservation", "value_ranges"],
            "backtrack_suggestion": {"suggest_backtrack": False},
        }

    @staticmethod
    def analyzer_response(
        classification: str = AnalysisClassification.ACCEPTABLE_MATCH,
    ) -> dict:
        """Create a results analyzer response."""
        return {
            "overall_classification": classification,
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "classification": AnalysisClassification.MATCH
                    if classification == AnalysisClassification.EXCELLENT_MATCH
                    else AnalysisClassification.PARTIAL_MATCH,
                    "shape_comparison": ["Peak position within 5%"],
                    "reason_for_difference": "Minor numerical differences",
                }
            ],
            "summary": f"Results classified as {classification}",
        }

    @staticmethod
    def comparison_validator_approve() -> dict:
        """Create an approve response for comparison validator."""
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Comparison validated successfully",
        }

    @staticmethod
    def supervisor_continue() -> dict:
        """Create an ok_continue response for supervisor."""
        return {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Stage completed successfully, continuing to next stage",
            "reasoning": "All checks passed",
        }

    @staticmethod
    def supervisor_complete() -> dict:
        """Create an all_complete response for supervisor."""
        return {
            "verdict": "all_complete",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "All stages completed successfully",
            "should_stop": True,
            "stop_reason": "All reproduction targets achieved",
        }

    @staticmethod
    def report_response() -> dict:
        """Create a report generator response."""
        return {
            "title": "Reproduction Report: Gold Nanorod",
            "overall_assessment": "SUCCESSFUL",
            "executive_summary": "Successfully reproduced extinction spectrum",
            "stage_summaries": [
                {
                    "stage_id": "stage_1_extinction",
                    "status": "completed_success",
                    "summary": "Extinction spectrum matches paper within 5%",
                }
            ],
            "recommendations": ["Consider extending to near-field calculations"],
        }


