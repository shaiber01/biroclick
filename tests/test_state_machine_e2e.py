"""
End-to-End State Machine Tests.

These tests run the ACTUAL LangGraph with real node code, mocking only
external dependencies (LLM calls, file I/O).

Strategy: Start simple (planning phase only), then incrementally add nodes.

Usage:
    pytest tests/test_state_machine_e2e.py -v -s
"""

import json
import signal
import uuid
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from schemas.state import create_initial_state, ReproState
from src.graph import create_repro_graph


def unique_thread_id(prefix: str = "test") -> str:
    """Generate a unique thread ID to prevent state pollution between tests."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ═══════════════════════════════════════════════════════════════════════
# Multi-Patch Helper for LLM Mocking
# ═══════════════════════════════════════════════════════════════════════

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
        print(f"    [MOCK ask_user] trigger={trigger}, responses={list(user_responses.keys())}", flush=True)
        
        return {
            "awaiting_user_input": False,
            "pending_user_questions": [],  # Clear questions
            # user_responses are already in state from update_state()
        }
    return mock_ask_user


# Import src.graph early so we can patch ask_user_node at the right location
# The graph imports ask_user_node from src.agents, creating a local binding
import src.graph


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


# ═══════════════════════════════════════════════════════════════════════
# Mock Responses
# ═══════════════════════════════════════════════════════════════════════

class MockLLMResponses:
    """Mock LLM responses that satisfy node validation logic."""
    
    # ─────────────────────────────────────────────────────────────────────
    # Planning Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
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
                "coverage": {"status": "pass", "figures_covered": ["Fig1"], "figures_missing": [], "notes": "All figures covered"},
                "digitized_data": {"status": "pass", "excellent_targets": [], "have_digitized": [], "missing_digitized": [], "notes": "N/A"},
                "staging": {"status": "pass", "stage_0_present": True, "stage_1_present": True, "validation_hierarchy_followed": True, "dependency_issues": [], "notes": "OK"},
                "parameter_extraction": {"status": "pass", "extracted_count": 2, "cross_checked_count": 2, "missing_critical": [], "notes": "OK"},
                "assumptions": {"status": "pass", "assumption_count": 0, "risky_assumptions": [], "undocumented_gaps": [], "notes": "OK"},
                "performance": {"status": "pass", "total_estimated_runtime_min": 5, "budget_min": 15, "risky_stages": [], "notes": "OK"},
                "material_validation_setup": {"status": "pass", "materials_covered": ["gold"], "materials_missing": [], "validation_criteria_clear": True, "notes": "OK"},
                "output_specifications": {"status": "pass", "all_stages_have_outputs": True, "figure_mappings_complete": True, "notes": "OK"}
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
                "coverage": {"status": "pass", "figures_covered": ["Fig1"], "figures_missing": [], "notes": "OK"},
                "digitized_data": {"status": "pass", "excellent_targets": [], "have_digitized": [], "missing_digitized": [], "notes": "N/A"},
                "staging": {"status": "pass", "stage_0_present": True, "stage_1_present": True, "validation_hierarchy_followed": True, "dependency_issues": [], "notes": "OK"},
                "parameter_extraction": {"status": "fail", "extracted_count": 2, "cross_checked_count": 0, "missing_critical": ["material source"], "notes": "Missing material source"},
                "assumptions": {"status": "pass", "assumption_count": 0, "risky_assumptions": [], "undocumented_gaps": [], "notes": "OK"},
                "performance": {"status": "pass", "total_estimated_runtime_min": 5, "budget_min": 15, "risky_stages": [], "notes": "OK"},
                "material_validation_setup": {"status": "fail", "materials_covered": [], "materials_missing": ["gold"], "validation_criteria_clear": False, "notes": "Missing material source"},
                "output_specifications": {"status": "pass", "all_stages_have_outputs": True, "figure_mappings_complete": True, "notes": "OK"}
            },
            "issues": [{"severity": "blocking", "category": "material_validation", "description": "Missing material source", "suggested_fix": "Add material database reference"}],
            "strengths": [],
            "summary": "Plan needs work",
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Design Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def simulation_designer(stage_id: str = "stage_0_materials") -> dict:
        return {
            "stage_id": stage_id,
            "design_description": "Gold nanorod FDTD simulation for extinction spectrum",
            "unit_system": {
                "characteristic_length_m": 1e-9,
                "length_unit": "nm",
                "example_conversions": {"100nm_to_meep": 100}
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
                        "real_dimensions": {"radius_nm": 20, "height_nm": 100}
                    }
                ],
                "symmetries": []
            },
            "materials": [
                {"id": "gold_jc", "name": "Gold (Johnson-Christy)", "model_type": "drude_lorentz", "source": "johnson_christy", "data_file": "materials/johnson_christy_gold.csv", "parameters": {}, "wavelength_range": {"min_nm": 400, "max_nm": 900}},
                {"id": "water", "name": "Water", "model_type": "constant", "source": "assumed", "data_file": None, "parameters": {"epsilon": 1.77}}
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
                    "frequency_width_meep": 0.95
                }
            ],
            "monitors": [
                {
                    "name": "flux_monitor",
                    "type": "flux",
                    "center": {"x": 0, "y": 0, "z": 300},
                    "size": {"x": 200, "y": 200, "z": 0}
                }
            ],
            "post_processing": [
                {"type": "spectrum", "input_monitor": "flux_monitor", "output_name": "extinction_spectrum"}
            ]
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
                "resolution_sufficient": True
            },
            "issues": [],
            "summary": "Design is solid"
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
                "resolution_sufficient": True
            },
            "issues": [{"severity": "blocking", "description": "Geometry exceeds cell boundaries"}],
            "summary": "Geometry issue"
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Code Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def code_generator() -> dict:
        return {
            "code": "print('Simulating...')",
            "filename": "simulation.py",
            "execution_command": "python simulation.py",
            "required_packages": ["meep"],
            "description": "FDTD simulation script"
        }
    
    @staticmethod
    def code_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good"
        }
    
    @staticmethod
    def code_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "issues": [{"severity": "blocking", "description": "Syntax error on line 5"}],
            "summary": "Fix syntax error"
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Execution & Validation Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def execution_validator_pass() -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Execution successful"
        }
    
    @staticmethod
    def execution_validator_fail() -> dict:
        return {
            "verdict": "fail",
            "issues": [{"severity": "blocking", "description": "RuntimeError: out of memory"}],
            "summary": "Execution failed"
        }
    
    @staticmethod
    def physics_sanity_pass() -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Physics looks reasonable"
        }
    
    @staticmethod
    def physics_sanity_fail() -> dict:
        return {
            "verdict": "fail",
            "issues": [{"severity": "blocking", "description": "Negative energy detected"}],
            "summary": "Physics violation"
        }
        
    @staticmethod
    def physics_sanity_design_flaw() -> dict:
        return {
            "verdict": "design_flaw",
            "issues": [{"severity": "blocking", "description": "Resolution too low for plasmon"}],
            "summary": "Fundamental design issue"
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Analysis & Comparison Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def results_analyzer() -> dict:
        return {
            "key_findings": ["Peak at 705nm"],
            "extracted_values": [{"name": "LSPR_peak", "value": 705, "unit": "nm"}],
            "figures_generated": ["spectrum.png"],
            "summary": "Analysis complete"
        }
    
    @staticmethod
    def comparison_validator() -> dict:
        return {
            "verdict": "approve",
            "match_quality": "good",
            "discrepancies": [],
            "summary": "Good match with target"
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Supervisor & Reporting
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def supervisor_continue() -> dict:
        return {
            "verdict": "ok_continue",
            "reasoning": "Stage complete, proceed to next",
            "suggested_next_step": "select_stage"
        }
        
    @staticmethod
    def supervisor_replan() -> dict:
        return {
            "verdict": "replan_needed",
            "reasoning": "Results inconsistent, need new approach",
            "suggested_next_step": "plan"
        }
    
    @staticmethod
    def report_generator() -> dict:
        return {
            "report_path": "reproduction_report.md",
            "summary": "Reproduction successful"
        }


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def initial_state():
    return create_initial_state(
        paper_id="test_paper",
        paper_text="Gold nanorods have LSPR at 700nm." * 10,  # Ensure text is long enough
    )


# ═══════════════════════════════════════════════════════════════════════
# Test: Planning Phase
# ═══════════════════════════════════════════════════════════════════════

class TestPlanningPhase:
    """Test planning agent flow."""
    
    def test_planning_approve_flow(self, initial_state):
        """Test: adapt_prompts → plan → plan_review(approve) → select_stage"""
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}")
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60)
            print("TEST: Planning Phase (approve flow)")
            print("=" * 60)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("planning")}}
            
            print("\n--- Running graph ---")
            final_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    print(f"  → {node_name}")
                    
                    if node_name == "plan":
                        # Verify plan structure
                        pass
                    
                    if node_name == "select_stage":
                        state = graph.get_state(config)
                        final_state = state.values
                        break
                else:
                    continue
                break
            
            # Assertions
            assert final_state is not None
            assert "plan" in final_state
            assert len(final_state["plan"]["stages"]) == 1
            assert final_state["last_plan_review_verdict"] == "approve"
            
            print("\n✅ Planning phase test passed!")

    def test_planning_revision_flow(self, initial_state):
        """Test: plan_review rejects → routes back to plan"""
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}")
            
            if agent == "plan_reviewer":
                # First: reject, Second: approve
                review_count = sum(1 for v in visited if v == "plan_reviewer")
                if review_count <= 1:
                    print("    [Rejecting plan]")
                    return MockLLMResponses.plan_reviewer_reject()
                else:
                    print("    [Approving plan]")
                    return MockLLMResponses.plan_reviewer_approve()
            
            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
            }
            return responses.get(agent, {})
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60)
            print("TEST: Planning Phase (revision flow)")
            print("=" * 60)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("planning_rev")}}
            
            print("\n--- Running graph ---")
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "select_stage":
                        break
                else:
                    continue
                break
            
            # Should see plan twice
            assert nodes_visited.count("plan") == 2
            assert nodes_visited.count("plan_review") == 2
            
            print("\n✅ Revision flow test passed!")


# ═══════════════════════════════════════════════════════════════════════
# Test: Planning + Stage Selection
# ═══════════════════════════════════════════════════════════════════════

class TestStageSelection:
    """Test planning through stage selection."""
    
    def test_stage_selection_picks_first_stage(self, initial_state):
        """After plan approval, select_stage should pick the first available stage."""
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}")
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Stage Selection", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("stage_select")}}
            
            print("\n--- Running graph ---")
            final_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    print(f"  → {node_name}")
                    
                    if node_name == "select_stage":
                        # Get state after select_stage
                        state = graph.get_state(config)
                        final_state = state.values
                        print(f"\n  Selected stage: {final_state.get('current_stage_id')}")
                        print(f"  Stage type: {final_state.get('current_stage_type')}")
                        
                        # Enhanced Assertions inside the loop
                        assert final_state.get("progress") is not None, "Progress should be initialized"
                        assert len(final_state.get("progress", {}).get("stages", [])) > 0, "Progress stages should be populated"
                        break
                else:
                    continue
                break
            
            # Assertions
            assert final_state is not None
            assert final_state.get("current_stage_id") == "stage_0_materials"
            assert final_state.get("current_stage_type") == "MATERIAL_VALIDATION"
            
            # Verify progress initialization
            progress = final_state.get("progress", {})
            stages = progress.get("stages", [])
            assert len(stages) == 1, "Should have 1 stage in progress"
            assert stages[0]["stage_id"] == "stage_0_materials"
            assert stages[0]["status"] == "not_started"
            
            print("\n✅ Stage selection test passed!")


# ═══════════════════════════════════════════════════════════════════════
# Test: Design Phase
# ═══════════════════════════════════════════════════════════════════════

class TestDesignPhase:
    """Test design → design_review flow."""
    
    def test_design_approve_flow(self, initial_state):
        """Test: select_stage → design → design_review(approve) → generate_code"""
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
            }
            return responses.get(agent, {})
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Design Phase (approve flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("design")}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Verify state updates during flow
                    if node_name == "design_review":
                        # Check updates directly as they contain the immediate output
                        assert updates.get("last_design_review_verdict") == "approve"
                    
                    # Stop after generate_code starts
                    if node_name == "generate_code":
                        break
                else:
                    continue
                break
            
            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            
            # Verify we went through design phase
            assert "design" in nodes_visited
            assert "design_review" in nodes_visited
            assert "generate_code" in nodes_visited
            
            # Check that design revision count is still 0 (clean pass)
            state = graph.get_state(config).values
            assert state.get("design_revision_count") == 0
            
            print("\n✅ Design phase test passed!", flush=True)
    
    def test_design_revision_flow(self, initial_state):
        """Test: design_review rejects → routes back to design"""
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "design_reviewer":
                # First: reject, Second: approve
                review_count = sum(1 for v in visited if v == "design_reviewer")
                if review_count <= 1:
                    print("    [Rejecting design]", flush=True)
                    return MockLLMResponses.design_reviewer_reject()
                else:
                    print("    [Approving design]", flush=True)
                    return MockLLMResponses.design_reviewer_approve()
            
            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
            }
            return responses.get(agent, {})
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Design Phase (revision flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("design_rev")}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    if node_name == "generate_code":
                        break
                else:
                    continue
                break
            
            # Should see design twice
            assert nodes_visited.count("design") == 2, f"Expected design twice: {nodes_visited}"
            assert nodes_visited.count("design_review") == 2
            
            print("\n✅ Design revision flow test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Test: Code Generation Phase
# ═══════════════════════════════════════════════════════════════════════

class TestCodePhase:
    """Test generate_code → code_review flow."""
    
    def test_code_approve_flow(self, initial_state):
        """Test: generate_code → code_review(approve) → run_code"""
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
            }
            return responses.get(agent, {})
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Code Phase (approve flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("code")}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    if node_name == "run_code":
                        break
                else:
                    continue
                break
            
            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            
            assert "generate_code" in nodes_visited
            assert "code_review" in nodes_visited
            assert "run_code" in nodes_visited
            
            print("\n✅ Code phase test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Test: Execution Phase
# ═══════════════════════════════════════════════════════════════════════

class TestExecutionPhase:
    """Test run_code → execution_check flow."""
    
    def test_execution_success_flow(self, initial_state):
        """Test: run_code → execution_check(pass) → physics_check"""
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            # print(f"    [LLM] {agent}", flush=True)
            
            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
            }
            return responses.get(agent, {})
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.code_runner.run_code_node", return_value={"workflow_phase": "running_code", "execution_output": "Simulating..."}):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Execution Phase (success flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("execution")}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    if node_name == "physics_check":
                        break
                else:
                    continue
                break
            
            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            
            assert "run_code" in nodes_visited
            assert "execution_check" in nodes_visited
            assert "physics_check" in nodes_visited
            
            print("\n✅ Execution phase test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Test: Full Single Stage Flow
# ═══════════════════════════════════════════════════════════════════════

class TestFullSingleStageHappyPath:
    """Run a full single stage from start to finish."""
    
    def test_full_single_stage_happy_path(self, initial_state):
        """
        Full Happy Path:
        Planning → Design → Code → Execution → Analysis → Supervisor → Report
        """
        visited_nodes = []
        llm_calls = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            llm_calls.append(f"LLM:{agent}")
            print(f"    [LLM] {agent}")
            
            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
                "results_analyzer": MockLLMResponses.results_analyzer(),
                "comparison_validator": MockLLMResponses.comparison_validator(),
                "supervisor": MockLLMResponses.supervisor_continue(),
                "report_generator": MockLLMResponses.report_generator(),
            }
            
            # Dynamic supervisor response to avoid loop
            if agent == "supervisor":
                # Count occurrences of supervisor in llm_calls
                supervisor_calls = sum(1 for c in llm_calls if "supervisor" in c)
                print(f"    [DEBUG] Supervisor called {supervisor_calls} times")
                if supervisor_calls == 1:
                    # First call: continue (triggers checkpoint)
                    print("    [DEBUG] Returning supervisor_continue (trigger checkpoint)")
                    return MockLLMResponses.supervisor_continue()
                else:
                    # Second call (after user input): all complete
                    print("    [DEBUG] Returning all_complete")
                    return {"verdict": "all_complete", "reasoning": "User approved, done."}
                
            return responses.get(agent, {})

        # Create mock ask_user
        mock_ask_user = create_mock_ask_user_node()

        # Patch run_code to avoid actual execution and ask_user to avoid stdin
        # NOTE: REMOVE patch on supervisor_node so it uses the real logic (which calls mock_llm)
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.code_runner.run_code_node", return_value={"workflow_phase": "running_code", "execution_output": "Success"}), \
             patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), \
             patch("src.graph.ask_user_node", side_effect=mock_ask_user):
            
            print("\n" + "=" * 60)
            print("TEST: Full Single-Stage Happy Path")
            print("=" * 60)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("happy_path")}}
            
            print("\n--- Running graph ---")
            
            # Step counter to prevent infinite loops
            steps = 0
            max_steps = 30
            
            for event in graph.stream(initial_state, config):
                steps += 1
                for node_name, updates in event.items():
                    visited_nodes.append(node_name)
                    print(f"  [{steps}] → {node_name}")
                    
                    # Detect interrupt and resume
                    if node_name == "material_checkpoint":
                        print("  [Interrupt Detected] - Resuming with mock user approval")
                        # Update state to mimic user approval
                        graph.update_state(config, {"user_responses": {"material_checkpoint": "approved"}})
                        # Resume streaming will happen in next iteration? No, stream() stops on interrupt.
                        break 
                    
                    # Stop when we hit report
                    if node_name == "generate_report":
                        break
                else:
                    if steps >= max_steps:
                        print("⚠️ Max steps reached!")
                        break
                    continue
                # Break from outer loop if we hit interrupt
                if "material_checkpoint" in visited_nodes[-1:]:
                     # Resume loop
                     continue
                break
            
            # If stopped at material_checkpoint, resume
            if "material_checkpoint" in visited_nodes and "generate_report" not in visited_nodes:
                 print("--- Resuming after interrupt ---")
                 graph.update_state(config, {"user_responses": {"material_checkpoint": "approved"}})
                 for event in graph.stream(None, config):
                    steps += 1
                    for node_name, updates in event.items():
                        visited_nodes.append(node_name)
                        print(f"  [{steps}] → {node_name}")
                        if node_name == "generate_report":
                            break
            
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Nodes visited: {len(visited_nodes)}")
            print(f"Unique nodes: {len(set(visited_nodes))}")
            print(f"LLM agents called: {len(set(llm_calls))}")
            print(f"Flow: {' → '.join(visited_nodes)}")
            
            # Assertions
            assert "plan" in visited_nodes
            assert "design" in visited_nodes
            assert "generate_code" in visited_nodes
            assert "execution_check" in visited_nodes
            assert "analyze" in visited_nodes
            assert "generate_report" in visited_nodes
            
            print("\n✅ Full single-stage test passed!")
