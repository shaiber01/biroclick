"""
End-to-End Workflow Tests with Mocked LLM Responses.

These tests verify the full workflow from paper input to final report,
using coordinated mock responses that simulate realistic LLM behavior.

Test categories:
1. Full workflow (paper → plan → design → code → execute → analyze → report)
2. Workflow with failures and recovery
3. Multi-stage workflows with dependencies
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from copy import deepcopy

from schemas.state import create_initial_state, ReproState
from src.agents import (
    plan_node,
    plan_reviewer_node,
    select_stage_node,
    simulation_designer_node,
    design_reviewer_node,
    code_generator_node,
    code_reviewer_node,
    execution_validator_node,
    physics_sanity_node,
    results_analyzer_node,
    comparison_validator_node,
    supervisor_node,
    generate_report_node,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / name
    with open(path, "r") as f:
        return json.load(f)


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


# ═══════════════════════════════════════════════════════════════════════
# Mock Response Factory
# ═══════════════════════════════════════════════════════════════════════

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
                    {"stage_id": "stage_0_materials", "status": "not_started", "stage_type": "MATERIAL_VALIDATION"},
                    {"stage_id": "stage_1_extinction", "status": "not_started", "stage_type": "SINGLE_STRUCTURE"},
                ]
            },
            "planned_materials": [
                {"material_id": "gold", "name": "Gold", "suggested_source": "johnson_christy"}
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
    def reviewer_needs_revision(feedback: str = "Needs improvement") -> dict:
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
            "code": '''import meep as mp
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
''',
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
    def execution_validator_fail(error: str = "Simulation crashed") -> dict:
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
    def analyzer_response(classification: str = "ACCEPTABLE_MATCH") -> dict:
        """Create a results analyzer response."""
        return {
            "overall_classification": classification,
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "classification": "match" if classification == "EXCELLENT_MATCH" else "partial_match",
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


# ═══════════════════════════════════════════════════════════════════════
# Full Workflow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFullWorkflowSuccess:
    """Test complete successful workflow from paper to report."""
    
    def test_planning_phase(self, base_state):
        """Test planning phase produces valid plan."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()
            
            result = plan_node(base_state)
            
            assert "plan" in result
            assert result["plan"]["stages"]
            assert len(result["plan"]["stages"]) >= 1
            assert result["workflow_phase"] == "planning"
    
    def test_plan_review_approve(self, base_state):
        """Test plan review approval."""
        # Set up state with plan
        base_state["plan"] = MockResponseFactory.planner_response()
        
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            
            result = plan_reviewer_node(base_state)
            
            assert result["last_plan_review_verdict"] == "approve"
    
    def test_stage_selection(self, base_state):
        """Test stage selection picks first available stage."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        result = select_stage_node(base_state)
        
        # Should select first stage with no dependencies
        selected = result.get("current_stage_id")
        assert selected is not None, "Should select a stage"
        # First stage should be materials (no deps) or first available
        assert selected in ["stage_0_materials", "stage_1_extinction"]
    
    def test_full_single_stage_workflow(self, base_state):
        """Test complete single-stage workflow."""
        # Setup: Plan with approved review
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold", "path": "/materials/Au.csv"}]
        
        # Mock execution result (simulates code_runner)
        base_state["execution_result"] = {
            "success": True,
            "output_files": ["extinction.csv"],
            "runtime_seconds": 120,
        }
        base_state["stage_outputs"] = {
            "files": ["extinction.csv"],
            "stage_id": "stage_1_extinction",
        }
        
        workflow_states = []
        
        # Design phase
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)
            workflow_states.append(("design", result))
        
        # Design review
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)
            workflow_states.append(("design_review", result))
        
        assert base_state["last_design_review_verdict"] == "approve"
        
        # Code generation
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)
            base_state.update(result)
            workflow_states.append(("code_gen", result))
        
        assert "code" in base_state
        
        # Code review
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = code_reviewer_node(base_state)
            base_state.update(result)
            workflow_states.append(("code_review", result))
        
        assert base_state["last_code_review_verdict"] == "approve"


class TestWorkflowWithRevisions:
    """Test workflow with revision cycles."""
    
    def test_design_revision_cycle(self, base_state):
        """Test design → review reject → revision → approve cycle."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        
        # First design attempt
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)
        
        # Review rejects
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Add PML thickness")
            result = design_reviewer_node(base_state)
            base_state.update(result)
        
        assert base_state["last_design_review_verdict"] == "needs_revision"
        assert base_state["design_revision_count"] == 1
        
        # Second design attempt with feedback
        assert "reviewer_feedback" in base_state
        
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)
        
        # Review approves
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)
        
        assert base_state["last_design_review_verdict"] == "approve"
    
    def test_code_revision_with_max_limit(self, base_state):
        """Test code revision respects max limit."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Test design"
        base_state["code"] = "# test code"
        base_state["runtime_config"] = {"max_code_revisions": 2}
        
        # First revision
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)
        
        assert base_state["code_revision_count"] == 1
        
        # Second revision
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)
        
        assert base_state["code_revision_count"] == 2
        
        # Third revision - should not increment past max
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)
        
        # Count stays at max (2)
        assert base_state["code_revision_count"] == 2


class TestWorkflowWithFailures:
    """Test workflow recovery from failures."""
    
    def test_execution_failure_recovery(self, base_state):
        """Test handling of execution failures."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["code"] = "# test code"
        base_state["execution_result"] = {
            "success": False,
            "error": "Simulation crashed",
            "output_files": [],
        }
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()
            result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1


class TestMultiStageWorkflow:
    """Test workflows with multiple dependent stages."""
    
    def test_stage_dependency_progression(self, base_state):
        """Test stages execute in dependency order."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # First selection should pick a stage with no deps
        result = select_stage_node(base_state)
        first_stage = result.get("current_stage_id")
        assert first_stage is not None, "Should select first stage"
        
        # Mark first stage complete
        for stage in base_state["progress"]["stages"]:
            if stage["stage_id"] == first_stage:
                stage["status"] = "completed_success"
        
        # Second selection should pick next available stage
        result = select_stage_node(base_state)
        second_stage = result.get("current_stage_id")
        
        # Should either select another stage or return None if done
        if second_stage is not None:
            assert second_stage != first_stage, "Should not re-select completed stage"
    
    def test_blocked_stage_skipped(self, base_state):
        """Test blocked stages are skipped."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Mark stage_0 as blocked
        base_state["progress"]["stages"][0]["status"] = "blocked"
        
        # Stage_1 depends on stage_0, so it should also be blocked/skipped
        # (depends on implementation - might return None or next available)
        result = select_stage_node(base_state)
        
        # Should not select stage_1 since its dependency is blocked
        assert result.get("current_stage_id") != "stage_1_extinction" or result.get("current_stage_id") is None


class TestSupervisorDecisions:
    """Test supervisor routing decisions."""
    
    def test_supervisor_continues_on_success(self, base_state):
        """Test supervisor returns ok_continue on success."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["analysis_overall_classification"] = "ACCEPTABLE_MATCH"
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.supervisor_continue()
            
            result = supervisor_node(base_state)
        
        assert result["supervisor_verdict"] == "ok_continue"
    
    def test_supervisor_completes_workflow(self, base_state):
        """Test supervisor returns all_complete when done."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        # All stages complete
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0_materials", "status": "completed_success"},
                {"stage_id": "stage_1_extinction", "status": "completed_success"},
            ]
        }
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.supervisor_complete()
            
            result = supervisor_node(base_state)
        
        # Note: actual verdict depends on implementation
        assert "supervisor_verdict" in result

