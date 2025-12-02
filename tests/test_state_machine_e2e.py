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
# Run tests directly
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
