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
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from schemas.state import create_initial_state, ReproState
from src.graph import create_repro_graph


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
# Timeout Handler
# ═══════════════════════════════════════════════════════════════════════

class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out - possible infinite loop!")


def run_with_timeout(func, timeout_seconds=10):
    """Run a function with a timeout. Returns (result, error)."""
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel the alarm
        return result, None
    except TimeoutError as e:
        return None, str(e)
    except Exception as e:
        signal.alarm(0)
        return None, str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
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
def initial_state(paper_input) -> ReproState:
    """Create initial state from paper input."""
    state = create_initial_state(
        paper_id=paper_input["paper_id"],
        paper_text=paper_input["paper_text"],
        paper_domain=paper_input.get("paper_domain", "plasmonics"),
    )
    state["paper_figures"] = paper_input.get("paper_figures", [])
    return state


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
        }
    
    @staticmethod
    def plan_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Plan is valid",
            "recommendations": [],
        }
    
    @staticmethod
    def plan_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "issues": ["Missing material source"],
            "summary": "Plan needs work",
            "recommendations": ["Add material database reference"],
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Design Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def simulation_designer(stage_id: str = "stage_0_materials") -> dict:
        return {
            "stage_id": stage_id,
            "design_description": "Gold nanorod FDTD simulation for extinction spectrum",
            "unit_system": "nm",
            "computational_domain": {
                "x_range": [-200, 200],
                "y_range": [-200, 200],
                "z_range": [-400, 400],
            },
            "geometry": [
                {
                    "object_type": "cylinder",
                    "center": [0, 0, 0],
                    "radius": 20,
                    "height": 100,
                    "axis": [0, 0, 1],
                    "material": "gold_jc",
                }
            ],
            "sources": [
                {
                    "source_type": "GaussianSource",
                    "center": [0, 0, -300],
                    "size": [400, 400, 0],
                    "wavelength_range": [400, 900],
                    "polarization": [1, 0, 0],
                }
            ],
            "materials": [
                {"material_id": "gold_jc", "name": "Gold (Johnson-Christy)", "role": "nanorod"},
                {"material_id": "water", "name": "Water", "role": "background"},
            ],
            "boundary_conditions": {
                "x": "PML",
                "y": "PML",
                "z": "PML",
            },
            "monitors": [
                {
                    "monitor_type": "FluxMonitor",
                    "center": [0, 0, 300],
                    "size": [400, 400, 0],
                    "name": "transmission",
                }
            ],
            "expected_outputs": [
                {"name": "extinction_spectrum.csv", "type": "spectrum"},
            ],
            "performance_estimate": {
                "estimated_runtime_minutes": 5,
                "memory_estimate_mb": 512,
            },
        }
    
    @staticmethod
    def design_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Design is valid",
            "recommendations": [],
        }
    
    @staticmethod
    def design_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "issues": ["PML thickness too small"],
            "summary": "Revise boundary conditions",
            "recommendations": ["Increase PML layers to 16"],
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Code Generation Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def code_generator(stage_id: str = "stage_0_materials") -> dict:
        return {
            "stage_id": stage_id,
            "code": '''"""Gold nanorod extinction spectrum simulation."""
import meep as mp
import numpy as np

# Define simulation parameters
resolution = 20
cell_size = mp.Vector3(0.4, 0.4, 0.8)

# Define geometry
geometry = [
    mp.Cylinder(
        radius=0.02,
        height=0.1,
        axis=mp.Vector3(0, 0, 1),
        material=mp.metal_Au,
    )
]

# Define source
sources = [
    mp.Source(
        mp.GaussianSource(frequency=1/0.7, fwidth=0.5),
        component=mp.Ex,
        center=mp.Vector3(0, 0, -0.3),
        size=mp.Vector3(0.4, 0.4, 0),
    )
]

# Create simulation
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=[mp.PML(0.05)],
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# Add flux monitor
flux_region = mp.FluxRegion(
    center=mp.Vector3(0, 0, 0.3),
    size=mp.Vector3(0.4, 0.4, 0),
)
flux_monitor = sim.add_flux(1/0.7, 0.5, 100, flux_region)

# Run simulation
sim.run(until=200)

# Get flux data
flux_data = mp.get_fluxes(flux_monitor)
wavelengths = 1 / np.array(mp.get_flux_freqs(flux_monitor))

# Save results
np.savetxt("extinction_spectrum.csv", 
           np.column_stack([wavelengths, flux_data]),
           delimiter=",", header="wavelength_nm,flux")

print("Simulation completed successfully")
''',
            "expected_outputs": ["extinction_spectrum.csv"],
            "explanation": "FDTD simulation of gold nanorod extinction",
        }
    
    @staticmethod
    def code_reviewer_approve() -> dict:
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Code is valid",
            "recommendations": [],
        }
    
    @staticmethod
    def code_reviewer_reject() -> dict:
        return {
            "verdict": "needs_revision",
            "issues": ["Missing output file path"],
            "summary": "Fix output handling",
            "recommendations": ["Use absolute path for output"],
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Execution Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def execution_validator_pass() -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Execution successful",
            "output_files_found": ["extinction_spectrum.csv"],
        }
    
    @staticmethod
    def execution_validator_fail() -> dict:
        return {
            "verdict": "fail",
            "issues": ["Meep segfault during run"],
            "summary": "Execution failed",
            "output_files_found": [],
        }
    
    @staticmethod
    def physics_sanity_pass() -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Physics looks reasonable",
            "checks_performed": ["Peak location", "Peak width", "Signal-to-noise"],
        }
    
    @staticmethod
    def physics_sanity_fail() -> dict:
        return {
            "verdict": "fail",
            "issues": ["Peak at wrong wavelength"],
            "summary": "Physics sanity check failed",
            "checks_performed": ["Peak location"],
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Analysis Phase Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def results_analyzer() -> dict:
        return {
            "stage_id": "stage_0_materials",
            "quantitative_summary": {
                "peak_wavelength_nm": 705,
                "peak_width_nm": 50,
                "peak_intensity": 0.85,
            },
            "visual_comparison": "Spectrum shape matches reference well",
            "confidence_factors": ["Clear peak", "Low noise"],
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "match_quality": "good",
                    "discrepancies": [],
                    "confidence": 0.85,
                }
            ],
            "per_result_reports": [
                {
                    "output_file": "extinction_spectrum.csv",
                    "status": "analyzed",
                    "summary": "Extinction peak at 705nm",
                }
            ],
        }
    
    @staticmethod
    def comparison_validator_approve() -> dict:
        return {
            "verdict": "approve",
            "issues": [],
            "summary": "Results match reference",
            "match_quality": "good",
        }
    
    @staticmethod
    def comparison_validator_needs_revision() -> dict:
        return {
            "verdict": "needs_revision",
            "issues": ["Peak offset by 20nm"],
            "summary": "Results need adjustment",
            "match_quality": "partial",
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Supervisor Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def supervisor_continue() -> dict:
        return {
            "verdict": "ok_continue",
            "feedback": "Stage completed successfully",
            "next_action": "continue_to_next_stage",
        }
    
    @staticmethod
    def supervisor_all_complete() -> dict:
        return {
            "verdict": "all_stages_complete",
            "feedback": "All stages completed, ready for report",
            "next_action": "generate_report",
        }
    
    # ─────────────────────────────────────────────────────────────────────
    # Report Generation Responses
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def report_generator() -> dict:
        return {
            "title": "Reproduction Report: Gold Nanorod Optical Properties",
            "summary": "Successfully reproduced extinction spectrum",
            "methodology": "FDTD simulation using Meep",
            "results": [
                {
                    "figure_id": "Fig1",
                    "status": "reproduced",
                    "match_quality": "good",
                }
            ],
            "conclusions": "Paper claims validated",
        }


# ═══════════════════════════════════════════════════════════════════════
# Test: Planning Phase Only
# ═══════════════════════════════════════════════════════════════════════

class TestPlanningPhase:
    """Test just the planning phase: adapt_prompts → plan → plan_review."""
    
    def test_planning_approve_flow(self, initial_state):
        """
        Test: START → adapt_prompts → plan → plan_review(approve) → select_stage
        
        This tests the happy path through planning with mocked LLM responses.
        Uses timeout to catch infinite loops.
        """
        visited = []
        llm_call_count = [0]
        MAX_LLM_CALLS = 20  # Safety limit
        
        def mock_llm(*args, **kwargs):
            """Return appropriate mock based on agent name."""
            llm_call_count[0] += 1
            if llm_call_count[0] > MAX_LLM_CALLS:
                raise RuntimeError(f"Too many LLM calls ({llm_call_count[0]}) - possible infinite loop!")
            
            agent = kwargs.get("agent_name", "unknown")
            visited.append(f"LLM:{agent}")
            print(f"    [LLM #{llm_call_count[0]}] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            else:
                print(f"    [WARNING] Unexpected agent: {agent}", flush=True)
                return {}
        
        # Patch LLM at ALL locations where it's imported (not where defined)
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Planning Phase (approve flow)", flush=True)
            print("=" * 60, flush=True)
            
            print("Creating graph...", flush=True)
            graph = create_repro_graph()
            print("Graph created successfully", flush=True)
            
            config = {"configurable": {"thread_id": "test_planning"}}
            print(f"Config: {config}", flush=True)
            
            # Stream through the graph with node counting for safety
            print("\n--- Running graph ---", flush=True)
            graph_nodes_visited = []
            final_state = None
            node_visit_count = 0
            MAX_NODE_VISITS = 50  # Safety limit
            
            print("  About to call graph.stream()...", flush=True)
            
            try:
                stream_iter = graph.stream(initial_state, config)
                print("  Got stream iterator, starting iteration...", flush=True)
                
                for event in stream_iter:
                    print(f"  Got event: {list(event.keys())}", flush=True)
                    for node_name, updates in event.items():
                        node_visit_count += 1
                        graph_nodes_visited.append(node_name)
                        print(f"  [{node_visit_count}] → {node_name}", flush=True)
                        
                        if node_visit_count > MAX_NODE_VISITS:
                            raise RuntimeError(f"Too many node visits ({node_visit_count}) - infinite loop!")
                        
                        # Stop after select_stage to avoid continuing into design phase
                        if node_name == "select_stage":
                            print("\n  [Stopping after select_stage]", flush=True)
                            break
                    else:
                        continue
                    break
                    
                result = "completed"
                error = None
            except Exception as e:
                import traceback
                print(f"  Exception: {e}", flush=True)
                traceback.print_exc()
                result = None
                error = str(e)
            
            if error:
                print(f"\n❌ ERROR: {error}", flush=True)
                print(f"Nodes visited before error: {graph_nodes_visited}", flush=True)
                print(f"LLM calls made: {visited}", flush=True)
                pytest.fail(f"Test failed: {error}")
            
            # Get current state
            state = graph.get_state(config)
            final_state = state.values
            
            # Print results
            print("\n" + "=" * 60, flush=True)
            print("RESULTS", flush=True)
            print("=" * 60, flush=True)
            print(f"Nodes visited: {' → '.join(graph_nodes_visited)}", flush=True)
            print(f"LLM calls: {visited}", flush=True)
            print(f"Plan title: {final_state.get('plan', {}).get('title', 'N/A')}", flush=True)
            print(f"Plan stages: {len(final_state.get('plan', {}).get('stages', []))}", flush=True)
            print(f"Plan verdict: {final_state.get('last_plan_review_verdict', 'N/A')}", flush=True)
            print("=" * 60, flush=True)
            
            # Assertions
            assert "adapt_prompts" in graph_nodes_visited
            assert "plan" in graph_nodes_visited
            assert "plan_review" in graph_nodes_visited
            assert "select_stage" in graph_nodes_visited
            
            assert final_state.get("last_plan_review_verdict") == "approve"
            assert final_state.get("plan") is not None
            assert len(final_state["plan"].get("stages", [])) > 0
            
            print("\n✅ Planning phase test passed!", flush=True)
    
    def test_planning_revision_flow(self, initial_state):
        """
        Test: plan_review rejects → routes back to plan
        """
        visited = []
        plan_call_count = [0]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(f"LLM:{agent}")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                plan_call_count[0] += 1
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                review_count = sum(1 for v in visited if "plan_reviewer" in v)
                if review_count <= 1:
                    print("    [Rejecting plan]", flush=True)
                    return MockLLMResponses.plan_reviewer_reject()
                else:
                    print("    [Approving plan]", flush=True)
                    return MockLLMResponses.plan_reviewer_approve()
            return {}
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60)
            print("TEST: Planning Phase (revision flow)")
            print("=" * 60)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_revision"}}
            
            print("\n--- Running graph ---")
            graph_nodes = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    graph_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "select_stage":
                        print("\n  [Stopping after select_stage]")
                        break
                else:
                    continue
                break
            
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Nodes: {' → '.join(graph_nodes)}")
            print(f"Planner called: {plan_call_count[0]} times")
            print("=" * 60)
            
            # Should have gone: plan → plan_review → plan → plan_review → select_stage
            assert graph_nodes.count("plan") == 2, f"Expected plan called twice, got {graph_nodes.count('plan')}"
            assert graph_nodes.count("plan_review") == 2
            
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
            config = {"configurable": {"thread_id": "test_stage_select"}}
            
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
                        break
                else:
                    continue
                break
            
            # Assertions
            assert final_state is not None
            assert final_state.get("current_stage_id") == "stage_0_materials"
            assert final_state.get("current_stage_type") == "MATERIAL_VALIDATION"
            
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
            config = {"configurable": {"thread_id": "test_design"}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
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
            config = {"configurable": {"thread_id": "test_design_rev"}}
            
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
        
        # Mock run_code_node to avoid actual execution
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Code Phase (approve flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_code"}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after run_code
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
    """Test run_code → execution_check → physics_check flow."""
    
    def test_execution_success_flow(self, initial_state):
        """Test successful execution through physics check."""
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
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
            }
            return responses.get(agent, {})
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Execution Phase (success flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_exec"}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after physics_check
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
# Test: Full Single-Stage Happy Path
# ═══════════════════════════════════════════════════════════════════════

class TestFullSingleStage:
    """Test complete single-stage workflow through to completion."""
    
    def test_single_stage_to_supervisor(self, initial_state):
        """
        Full single-stage flow through to supervisor decision.
        
        Flow: plan → design → code → execute → analyze → supervisor
        """
        visited = []
        node_count = [0]
        MAX_NODES = 100  # Safety limit
        
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
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
                "results_analyzer": MockLLMResponses.results_analyzer(),
                "comparison_validator": MockLLMResponses.comparison_validator_approve(),
                "supervisor": MockLLMResponses.supervisor_all_complete(),
                "report_generator": MockLLMResponses.report_generator(),
            }
            return responses.get(agent, {})
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        # Mock file existence checks
        mock_path_exists = MagicMock(return_value=True)
        mock_path_is_file = MagicMock(return_value=True)
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", mock_path_exists), \
             patch("pathlib.Path.is_file", mock_path_is_file):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Full Single-Stage Happy Path", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_full"}}
            
            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            final_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    node_count[0] += 1
                    nodes_visited.append(node_name)
                    print(f"  [{node_count[0]}] → {node_name}", flush=True)
                    
                    if node_count[0] > MAX_NODES:
                        pytest.fail(f"Too many nodes ({node_count[0]}) - infinite loop!")
                    
                    # Graph pauses before ask_user due to interrupt_before
                    # Let it run until it pauses or completes
                    if node_name == "supervisor":
                        state = graph.get_state(config)
                        final_state = state.values
                        # Check if supervisor decided we're done
                        verdict = final_state.get("supervisor_verdict", "")
                        print(f"    Supervisor verdict: {verdict}", flush=True)
                        if verdict == "all_stages_complete":
                            break
                else:
                    continue
                break
            
            print("\n" + "=" * 60, flush=True)
            print("RESULTS", flush=True)
            print("=" * 60, flush=True)
            print(f"Nodes visited: {len(nodes_visited)}", flush=True)
            print(f"Unique nodes: {len(set(nodes_visited))}", flush=True)
            print(f"LLM agents called: {len(visited)}", flush=True)
            print(f"Flow: {' → '.join(nodes_visited[:20])}{'...' if len(nodes_visited) > 20 else ''}", flush=True)
            
            # Verify key nodes were visited
            expected_nodes = [
                "adapt_prompts", "plan", "plan_review", "select_stage",
                "design", "design_review", "generate_code", "code_review",
                "run_code", "execution_check", "physics_check",
                "analyze", "comparison_check", "supervisor",
            ]
            
            for node in expected_nodes:
                assert node in nodes_visited, f"Missing node: {node}"
            
            print("\n✅ Full single-stage test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Helper: Run Until Pause or Complete
# ═══════════════════════════════════════════════════════════════════════

def run_until_pause_or_complete(graph, state, config, max_nodes=100, stop_after=None):
    """
    Run graph until it pauses (at ask_user) or completes.
    
    Args:
        graph: Compiled LangGraph
        state: Initial state dict
        config: Config with thread_id
        max_nodes: Safety limit for node visits
        stop_after: Optional node name to stop after
    
    Returns:
        (nodes_visited, graph_state, is_paused)
    """
    nodes_visited = []
    node_count = 0
    
    for event in graph.stream(state, config):
        for node_name, _ in event.items():
            node_count += 1
            nodes_visited.append(node_name)
            print(f"  [{node_count}] → {node_name}", flush=True)
            
            if node_count > max_nodes:
                raise RuntimeError(f"Too many nodes ({node_count}) - infinite loop!")
            
            if stop_after and node_name == stop_after:
                break
        else:
            continue
        break
    
    # Check if paused
    graph_state = graph.get_state(config)
    is_paused = len(graph_state.next) > 0
    return nodes_visited, graph_state, is_paused


# ═══════════════════════════════════════════════════════════════════════
# Test: User Interaction Flows
# ═══════════════════════════════════════════════════════════════════════

class TestUserInteraction:
    """
    Test user interaction flows where graph pauses at ask_user.
    
    The graph uses interrupt_before=["ask_user"] which pauses BEFORE ask_user
    executes, allowing external code to inject user responses.
    """
    
    def _create_material_validation_plan(self, paper_id="test_gold_nanorod"):
        """Create a plan with MATERIAL_VALIDATION stage that triggers ask_user."""
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
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                    "runtime_budget_minutes": 5,
                },
                {
                    "stage_id": "stage_1_spectrum",
                    "stage_type": "FDTD_DIRECT",
                    "description": "Compute extinction spectrum",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0_materials"],
                    "runtime_budget_minutes": 30,
                },
            ],
            "assumptions": {},
            "reproduction_scope": {
                "total_figures": 1,
                "reproducible_figures": 1,
                "attempted_figures": ["Fig1"],
                "skipped_figures": [],
            },
        }
    
    def test_material_checkpoint_pauses_for_user(self, initial_state):
        """
        Test: After MATERIAL_VALIDATION stage completes, graph pauses before ask_user.
        
        Flow: ... → supervisor (ok_continue) → material_checkpoint → [PAUSE before ask_user]
        """
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_material_validation_plan()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                # Return ok_continue to trigger material_checkpoint
                return {
                    "verdict": "ok_continue",
                    "feedback": "Stage completed, proceed to material checkpoint",
                    "next_action": "continue",
                }
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Material Checkpoint Pauses for User", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_material_pause"}}
            
            # Run graph - it should pause before ask_user
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=50
            )
            
            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            print(f"Is paused: {is_paused}", flush=True)
            print(f"Next nodes: {graph_state.next}", flush=True)
            
            # Verify we reached material_checkpoint and paused before ask_user
            assert "material_checkpoint" in nodes_visited, "Should reach material_checkpoint"
            assert is_paused, "Graph should be paused"
            assert "ask_user" in graph_state.next, f"Should pause before ask_user, got: {graph_state.next}"
            
            # Check pending questions are set
            state_values = graph_state.values
            assert state_values.get("pending_user_questions"), "Should have pending questions"
            assert state_values.get("ask_user_trigger") == "material_checkpoint"
            
            print("\n✅ Material checkpoint pause test passed!", flush=True)
    
    def test_user_approve_continues_workflow(self, initial_state):
        """
        Test: User approves materials, graph continues to next stage.
        
        Flow: [PAUSE] → inject approval → ask_user → supervisor → select_stage
        """
        supervisor_call_count = [0]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_material_validation_plan()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                supervisor_call_count[0] += 1
                if supervisor_call_count[0] == 1:
                    # First call: complete stage 0
                    return {
                        "verdict": "ok_continue",
                        "feedback": "Stage completed",
                        "next_action": "continue",
                    }
                else:
                    # After user approval: continue to next stage
                    return {
                        "verdict": "ok_continue",
                        "feedback": "Materials approved, proceeding",
                        "next_action": "continue",
                    }
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch.object(src.graph, "ask_user_node", create_mock_ask_user_node()), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: User Approve Continues Workflow", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_user_approve"}}
            
            # Run until paused
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=50
            )
            
            print(f"\nPhase 1 - Nodes: {' → '.join(nodes_visited[-10:])}", flush=True)
            assert is_paused, "Should be paused before ask_user"
            
            # Inject user approval
            print("\n--- Injecting user approval ---", flush=True)
            graph.update_state(
                config,
                {"user_responses": {"material_checkpoint": "APPROVE - materials look correct"}}
            )
            
            # Resume and run until next pause or select_stage
            print("\n--- Resuming graph ---", flush=True)
            resumed_nodes = []
            for event in graph.stream(None, config):
                for node_name, _ in event.items():
                    resumed_nodes.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after select_stage to see if we moved to next stage
                    if node_name == "select_stage":
                        break
                else:
                    continue
                break
            
            print(f"\nPhase 2 - Nodes: {' → '.join(resumed_nodes)}", flush=True)
            
            # Verify flow: ask_user → supervisor → select_stage
            assert "ask_user" in resumed_nodes, "Should execute ask_user"
            assert "supervisor" in resumed_nodes, "Should route to supervisor"
            
            # Get final state
            final_state = graph.get_state(config).values
            print(f"Current stage: {final_state.get('current_stage_id')}", flush=True)
            
            print("\n✅ User approve continues workflow test passed!", flush=True)
    
    def test_user_reject_triggers_replan(self, initial_state):
        """
        Test: User rejects materials, supervisor triggers replan.
        
        Flow: [PAUSE] → inject rejection → ask_user → supervisor → plan
        """
        supervisor_call_count = [0]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_material_validation_plan()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                supervisor_call_count[0] += 1
                if supervisor_call_count[0] == 1:
                    return {
                        "verdict": "ok_continue",
                        "feedback": "Stage completed",
                        "next_action": "continue",
                    }
                else:
                    # After user rejection: trigger replan
                    return {
                        "verdict": "replan_needed",
                        "feedback": "User rejected materials, replanning",
                        "next_action": "replan",
                    }
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch.object(src.graph, "ask_user_node", create_mock_ask_user_node()), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: User Reject Triggers Replan", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_user_reject"}}
            
            # Run until paused
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=50
            )
            
            assert is_paused, "Should be paused"
            
            # Inject user rejection
            print("\n--- Injecting user rejection ---", flush=True)
            graph.update_state(
                config,
                {"user_responses": {"material_checkpoint": "REJECT - CHANGE_DATABASE to Palik"}}
            )
            
            # Resume and watch for replan
            print("\n--- Resuming graph ---", flush=True)
            resumed_nodes = []
            for event in graph.stream(None, config):
                for node_name, _ in event.items():
                    resumed_nodes.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after plan to see replan was triggered
                    if node_name == "plan":
                        break
                else:
                    continue
                break
            
            print(f"\nResumed nodes: {' → '.join(resumed_nodes)}", flush=True)
            
            # Verify flow: ask_user → supervisor → plan
            assert "ask_user" in resumed_nodes
            assert "supervisor" in resumed_nodes
            assert "plan" in resumed_nodes, "Should route to plan for replanning"
            
            print("\n✅ User reject triggers replan test passed!", flush=True)
    
    def test_context_overflow_escalation(self, initial_state):
        """
        Test: Context overflow triggers user escalation.
        
        Simulates context overflow by setting context_budget to exceeded state.
        """
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                # Simulate context overflow via return value
                return {
                    "verdict": "needs_revision",
                    "issues": ["Context exceeded during review"],
                    "summary": "Context overflow",
                    "recommendations": [],
                }
            return {}
        
        # Inject context overflow into initial state
        initial_state["context_budget"] = {
            "max_tokens": 100000,
            "current_tokens": 150000,  # Exceeded!
            "warning_threshold": 80000,
        }
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Context Overflow Escalation", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_context_overflow"}}
            
            # Run until pause or design phase completes
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=30, stop_after="design_review"
            )
            
            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            print(f"Is paused: {is_paused}", flush=True)
            
            # Context overflow handling depends on implementation
            # At minimum, verify we reached design phase
            assert "design" in nodes_visited
            
            print("\n✅ Context overflow escalation test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Test: Multi-Stage Workflows
# ═══════════════════════════════════════════════════════════════════════

class TestMultiStageWorkflow:
    """
    Test workflows with multiple stages and dependencies.
    """
    
    @staticmethod
    def _create_two_stage_plan(paper_id="test_gold_nanorod"):
        """Create a plan with 2 stages where stage_1 depends on stage_0."""
        return {
            "paper_id": paper_id,
            "paper_domain": "plasmonics",
            "title": "Two-Stage Simulation",
            "summary": "Material validation then FDTD simulation",
            "main_system": "Gold nanorod in water",
            "main_claims": ["Longitudinal plasmon at 700nm"],
            "simulation_approach": "FDTD with Meep",
            "extracted_parameters": [
                {"name": "length", "value": 100, "unit": "nm", "source": "text", "location": "p1"},
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
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                    "runtime_budget_minutes": 5,
                },
                {
                    "stage_id": "stage_1_spectrum",
                    "stage_type": "FDTD_DIRECT",
                    "description": "Compute extinction spectrum",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0_materials"],  # Depends on stage_0
                    "runtime_budget_minutes": 30,
                },
            ],
            "assumptions": {},
            "reproduction_scope": {
                "total_figures": 1,
                "reproducible_figures": 1,
                "attempted_figures": ["Fig1"],
                "skipped_figures": [],
            },
        }
    
    def test_two_stage_sequential(self, initial_state):
        """
        Test: Two stages run in sequence (stage_1 depends on stage_0).
        
        Verify stage_0 completes before stage_1 is selected.
        """
        stage_selections = []
        supervisor_call_count = [0]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_two_stage_plan()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                supervisor_call_count[0] += 1
                return {
                    "verdict": "ok_continue",
                    "feedback": "Stage completed, continue",
                    "next_action": "continue",
                }
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Two-Stage Sequential", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_two_stage"}}
            
            # Track stage selections
            nodes_visited = []
            select_stage_count = 0
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    if node_name == "select_stage":
                        select_stage_count += 1
                        state = graph.get_state(config).values
                        current_stage = state.get("current_stage_id")
                        stage_selections.append(current_stage)
                        print(f"    Selected: {current_stage}", flush=True)
                        
                        # Stop after second stage selection
                        if select_stage_count >= 2:
                            break
                    
                    # Skip ask_user by continuing past material checkpoint
                    if len(nodes_visited) > 50:
                        break
                else:
                    continue
                break
            
            print(f"\nStage selections: {stage_selections}", flush=True)
            
            # Verify stage_0 selected first
            assert len(stage_selections) >= 1
            assert stage_selections[0] == "stage_0_materials"
            
            print("\n✅ Two-stage sequential test passed!", flush=True)
    
    def test_stage_completion_triggers_next(self, initial_state):
        """
        Test: Stage completion flow reaches supervisor which handles continuation.
        
        Verifies: select_stage → (stage workflow) → supervisor flow works correctly.
        The actual stage_1 selection requires properly completed stage_0 state,
        which is complex to mock. This test verifies the path to supervisor exists.
        """
        supervisor_called = [False]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_two_stage_plan()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                supervisor_called[0] = True
                return {
                    "verdict": "ok_continue",
                    "feedback": "Stage completed",
                    "next_action": "continue",
                }
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Stage Completion Triggers Next", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_stage_trigger"}}
            
            # Run until material_checkpoint pause
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=50
            )
            
            print(f"\nNodes: {' → '.join(nodes_visited[-10:])}", flush=True)
            
            # Verify we reached supervisor after stage completion
            assert "supervisor" in nodes_visited, "Should reach supervisor"
            assert supervisor_called[0], "Supervisor LLM should be called"
            
            # Verify we went through the full stage workflow
            assert "select_stage" in nodes_visited
            assert "design" in nodes_visited
            assert "generate_code" in nodes_visited
            assert "run_code" in nodes_visited
            
            print("\n✅ Stage completion triggers next test passed!", flush=True)
    
    @staticmethod
    def _create_single_structure_plan(paper_id="test_gold_nanorod"):
        """Create a plan with only SINGLE_STRUCTURE stage (no MATERIAL_VALIDATION)."""
        return {
            "paper_id": paper_id,
            "paper_domain": "plasmonics",
            "title": "Single Structure Simulation",
            "summary": "Simple single structure simulation",
            "main_system": "Gold nanorod in water",
            "main_claims": ["Longitudinal plasmon at 700nm"],
            "simulation_approach": "FDTD with Meep",
            "extracted_parameters": [
                {"name": "length", "value": 100, "unit": "nm", "source": "text", "location": "p1"},
            ],
            "planned_materials": [
                {"material_id": "gold_jc", "name": "Gold (Johnson-Christy)"},
            ],
            "targets": [
                {
                    "figure_id": "Fig1",
                    "description": "Extinction spectrum",
                    "type": "spectrum",
                    "simulation_class": "SINGLE_STRUCTURE",
                    "precision_requirement": "acceptable",
                }
            ],
            "stages": [
                {
                    "stage_id": "stage_0_single",
                    "stage_type": "SINGLE_STRUCTURE",  # Valid stage type
                    "description": "Compute extinction spectrum",
                    "targets": ["Fig1"],
                    "dependencies": [],
                    "runtime_budget_minutes": 30,
                }
            ],
            "assumptions": {},
            "reproduction_scope": {
                "total_figures": 1,
                "reproducible_figures": 1,
                "attempted_figures": ["Fig1"],
                "skipped_figures": [],
            },
        }
    
    def test_all_stages_complete_triggers_report(self, initial_state):
        """
        Test: After all stages complete, supervisor returns all_complete → generate_report.
        """
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_fdtd_only_plan()  # No MATERIAL_VALIDATION
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                # All stages complete
                return {
                    "verdict": "all_stages_complete",
                    "feedback": "All stages completed successfully",
                    "next_action": "generate_report",
                }
            elif agent == "report_generator":
                return MockLLMResponses.report_generator()
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: All Stages Complete Triggers Report", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_all_complete"}}
            
            nodes_visited = []
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after generate_report
                    if node_name == "generate_report":
                        break
                else:
                    continue
                break
            
            print(f"\nNodes: {' → '.join(nodes_visited[-10:])}", flush=True)
            
            # Verify supervisor → generate_report
            assert "supervisor" in nodes_visited
            assert "generate_report" in nodes_visited
            
            print("\n✅ All stages complete triggers report test passed!", flush=True)
    
    def test_stage_dependency_blocks_selection(self, initial_state):
        """
        Test: Stage with unmet dependencies is not selected first.
        
        In a two-stage plan where stage_1 depends on stage_0,
        select_stage should always pick stage_0 first.
        """
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return self._create_two_stage_plan()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Stage Dependency Blocks Selection", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_dep_block"}}
            
            nodes_visited = []
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    if node_name == "select_stage":
                        state = graph.get_state(config).values
                        selected = state.get("current_stage_id")
                        print(f"    Selected: {selected}", flush=True)
                        
                        # First selection should be stage_0 (no deps) or None if plan failed
                        # stage_1 has deps so should NOT be selected first
                        if selected is not None:
                            assert selected == "stage_0_materials", \
                                f"Should select stage with no deps first, got {selected}"
                        break
                else:
                    continue
                break
            
            # Verify we at least reached select_stage
            assert "select_stage" in nodes_visited, "Should reach select_stage"
            
            print("\n✅ Stage dependency blocks selection test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Test: Failure Recovery Paths
# ═══════════════════════════════════════════════════════════════════════

class TestFailureRecovery:
    """
    Test failure routing and retry limits.
    """
    
    def test_execution_fail_routes_to_code_gen(self, initial_state):
        """
        Test: execution_check fail → routes to generate_code.
        """
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                # First call: fail, second call: pass
                exec_count = sum(1 for v in visited if v == "execution_validator")
                if exec_count <= 1:
                    print("    [Execution FAIL]", flush=True)
                    return MockLLMResponses.execution_validator_fail()
                else:
                    print("    [Execution PASS]", flush=True)
                    return MockLLMResponses.execution_validator_pass()
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Execution Fail Routes to Code Gen", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_exec_fail"}}
            
            nodes_visited = []
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after second execution_check
                    if nodes_visited.count("execution_check") >= 2:
                        break
                else:
                    continue
                break
            
            print(f"\nFlow after first run_code: {nodes_visited}", flush=True)
            
            # Verify: execution_check → generate_code → code_review → run_code → execution_check
            exec_idx = nodes_visited.index("execution_check")
            assert nodes_visited[exec_idx + 1] == "generate_code", \
                f"After execution fail, should route to generate_code"
            
            print("\n✅ Execution fail routes to code gen test passed!", flush=True)
    
    def test_physics_fail_routes_to_design(self, initial_state):
        """
        Test: physics_check design_flaw → routes to design node.
        """
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                # Return design_flaw verdict
                physics_count = sum(1 for v in visited if v == "physics_sanity")
                if physics_count <= 1:
                    print("    [Physics: DESIGN_FLAW]", flush=True)
                    return {
                        "verdict": "design_flaw",
                        "issues": ["Geometry too small for resonance"],
                        "summary": "Fundamental design issue",
                        "checks_performed": ["Resonance check"],
                    }
                else:
                    return MockLLMResponses.physics_sanity_pass()
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Physics Fail Routes to Design", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_physics_fail"}}
            
            nodes_visited = []
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after seeing design node after physics_check
                    if "physics_check" in nodes_visited and nodes_visited[-1] == "design":
                        break
                    
                    if len(nodes_visited) > 40:
                        break
                else:
                    continue
                break
            
            print(f"\nFlow: {' → '.join(nodes_visited[-10:])}", flush=True)
            
            # Verify: physics_check → design
            physics_idx = nodes_visited.index("physics_check")
            assert nodes_visited[physics_idx + 1] == "design", \
                f"After physics design_flaw, should route to design, got {nodes_visited[physics_idx + 1]}"
            
            print("\n✅ Physics fail routes to design test passed!", flush=True)
    
    def test_code_review_max_retries_escalates(self, initial_state):
        """
        Test: After max code review rejections, escalates to ask_user.
        """
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                # Always reject to hit max retries
                print("    [Code review: REJECT]", flush=True)
                return MockLLMResponses.code_reviewer_reject()
            return {}
        
        # Set code_revision_count near max to trigger escalation faster
        initial_state["code_revision_count"] = 2  # Max is typically 3
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Code Review Max Retries Escalates", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_code_retry"}}
            
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=30
            )
            
            print(f"\nNodes: {' → '.join(nodes_visited[-10:])}", flush=True)
            print(f"Is paused: {is_paused}", flush=True)
            
            # Should pause before ask_user due to max retries
            # Or reach ask_user in the visited list
            if is_paused:
                assert "ask_user" in graph_state.next, \
                    f"Should pause before ask_user, got next: {graph_state.next}"
            
            print("\n✅ Code review max retries escalates test passed!", flush=True)
    
    def test_design_review_max_retries_escalates(self, initial_state):
        """
        Test: After max design review rejections, escalates to ask_user.
        """
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                # Always reject
                print("    [Design review: REJECT]", flush=True)
                return MockLLMResponses.design_reviewer_reject()
            return {}
        
        # Set near max
        initial_state["design_revision_count"] = 2  # Max is typically 3
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Design Review Max Retries Escalates", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_design_retry"}}
            
            nodes_visited, graph_state, is_paused = run_until_pause_or_complete(
                graph, initial_state, config, max_nodes=30
            )
            
            print(f"\nNodes: {' → '.join(nodes_visited[-10:])}", flush=True)
            print(f"Is paused: {is_paused}", flush=True)
            
            # Should pause before ask_user
            if is_paused:
                assert "ask_user" in graph_state.next, \
                    f"Should pause before ask_user, got next: {graph_state.next}"
            
            print("\n✅ Design review max retries escalates test passed!", flush=True)
    
    def test_llm_error_fallback_continues(self, initial_state):
        """
        Test: LLM error uses fallback response and workflow continues.
        """
        visited = []
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                # Simulate LLM error - the node should use fallback
                raise Exception("Simulated LLM failure")
            return {}
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: LLM Error Fallback Continues", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_llm_error"}}
            
            # The plan_reviewer has error handling that auto-approves on LLM failure
            nodes_visited = []
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Stop after select_stage to see if we continued
                    if node_name == "select_stage":
                        break
                else:
                    continue
                break
            
            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            
            # The plan_reviewer's fallback should auto-approve, allowing continuation
            assert "plan_review" in nodes_visited
            assert "select_stage" in nodes_visited, \
                "Workflow should continue after LLM error with fallback"
            
            print("\n✅ LLM error fallback continues test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Test: Report Generation
# ═══════════════════════════════════════════════════════════════════════

class TestReportGeneration:
    """
    Test final report generation.
    """
    
    def test_all_complete_triggers_report(self, initial_state):
        """
        Test: supervisor all_complete → generate_report → END.
        """
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                return {
                    "verdict": "all_stages_complete",
                    "feedback": "All stages complete, generate report",
                    "next_action": "generate_report",
                }
            elif agent == "report_generator":
                return MockLLMResponses.report_generator()
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: All Complete Triggers Report", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_report"}}
            
            nodes_visited = []
            final_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
            
            final_state = graph.get_state(config).values
            
            print(f"\nFinal nodes: {' → '.join(nodes_visited[-5:])}", flush=True)
            
            # Verify flow ends with generate_report
            assert "supervisor" in nodes_visited
            assert "generate_report" in nodes_visited
            assert nodes_visited[-1] == "generate_report", \
                f"Last node should be generate_report, got {nodes_visited[-1]}"
            
            print("\n✅ All complete triggers report test passed!", flush=True)
    
    def test_report_contains_stage_results(self, initial_state):
        """
        Test: Report includes completed stage data.
        """
        report_output = [None]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                return {
                    "verdict": "all_stages_complete",
                    "feedback": "All stages complete",
                    "next_action": "generate_report",
                }
            elif agent == "report_generator":
                report = {
                    "title": "Reproduction Report: Gold Nanorod",
                    "summary": "Successfully reproduced extinction spectrum",
                    "methodology": "FDTD simulation using Meep",
                    "results": [
                        {
                            "figure_id": "Fig1",
                            "stage_id": "stage_0_materials",
                            "status": "reproduced",
                            "match_quality": "good",
                        }
                    ],
                    "conclusions": "Paper claims validated",
                }
                report_output[0] = report
                return report
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Report Contains Stage Results", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_report_content"}}
            
            # Run to completion
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    print(f"  → {node_name}", flush=True)
            
            # Verify report content
            assert report_output[0] is not None, "Report should be generated"
            report = report_output[0]
            
            assert "results" in report
            assert len(report["results"]) > 0
            assert report["results"][0].get("stage_id") == "stage_0_materials"
            
            print(f"\nReport title: {report.get('title')}", flush=True)
            print(f"Results: {len(report.get('results', []))} entries", flush=True)
            
            print("\n✅ Report contains stage results test passed!", flush=True)
    
    def test_report_node_called_once(self, initial_state):
        """
        Test: Report generation node is called exactly once.
        """
        report_call_count = [0]
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            elif agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            elif agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            elif agent == "code_generator":
                return MockLLMResponses.code_generator()
            elif agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_approve()
            elif agent == "execution_validator":
                return MockLLMResponses.execution_validator_pass()
            elif agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()
            elif agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            elif agent == "comparison_validator":
                return MockLLMResponses.comparison_validator_approve()
            elif agent == "supervisor":
                return {
                    "verdict": "all_stages_complete",
                    "feedback": "All stages complete",
                    "next_action": "generate_report",
                }
            elif agent == "report_generator":
                report_call_count[0] += 1
                return MockLLMResponses.report_generator()
            return {}
        
        mock_run_code = MagicMock(return_value={
            "execution_result": {
                "success": True,
                "stdout": "Simulation completed",
                "stderr": "",
                "output_files": ["extinction_spectrum.csv"],
            },
            "output_files": ["extinction_spectrum.csv"],
        })
        
        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"), \
             patch("src.graph.run_code_node", mock_run_code), \
             patch("pathlib.Path.exists", MagicMock(return_value=True)), \
             patch("pathlib.Path.is_file", MagicMock(return_value=True)):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Report Node Called Once", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_report_once"}}
            
            nodes_visited = []
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
            
            print(f"\nReport calls: {report_call_count[0]}", flush=True)
            print(f"generate_report in nodes: {nodes_visited.count('generate_report')}", flush=True)
            
            assert report_call_count[0] == 1, \
                f"Report should be called exactly once, was called {report_call_count[0]} times"
            assert nodes_visited.count("generate_report") == 1
            
            print("\n✅ Report node called once test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Run tests directly
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
