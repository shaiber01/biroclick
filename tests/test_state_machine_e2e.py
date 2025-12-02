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
        
        # Patch LLM and checkpoints
        with patch("src.llm_client.call_agent_with_metrics", side_effect=mock_llm), \
             patch("schemas.state.save_checkpoint", return_value="/tmp/cp.json"), \
             patch("src.graph.save_checkpoint", return_value="/tmp/cp.json"), \
             patch("src.routing.save_checkpoint", return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60, flush=True)
            print("TEST: Planning Phase (approve flow)", flush=True)
            print("=" * 60, flush=True)
            
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": "test_planning"}}
            
            # Stream through the graph with node counting for safety
            print("\n--- Running graph ---", flush=True)
            graph_nodes_visited = []
            final_state = None
            node_visit_count = 0
            MAX_NODE_VISITS = 50  # Safety limit
            
            def run_graph():
                nonlocal final_state, node_visit_count
                for event in graph.stream(initial_state, config):
                    for node_name, updates in event.items():
                        node_visit_count += 1
                        graph_nodes_visited.append(node_name)
                        print(f"  [{node_visit_count}] → {node_name}", flush=True)
                        
                        if node_visit_count > MAX_NODE_VISITS:
                            raise RuntimeError(f"Too many node visits ({node_visit_count}) - infinite loop!")
                        
                        # Stop after select_stage to avoid continuing into design phase
                        if node_name == "select_stage":
                            print("\n  [Stopping after select_stage]", flush=True)
                            return "stopped"
                return "completed"
            
            # Run with timeout
            result, error = run_with_timeout(run_graph, timeout_seconds=15)
            
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
            print(f"    [LLM] {agent}")
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            elif agent == "planner":
                plan_call_count[0] += 1
                # Return same plan each time
                return MockLLMResponses.planner()
            elif agent == "plan_reviewer":
                # First review: reject. Second review: approve.
                review_count = sum(1 for v in visited if "plan_reviewer" in v)
                if review_count <= 1:
                    print("    [Rejecting plan]")
                    return MockLLMResponses.plan_reviewer_reject()
                else:
                    print("    [Approving plan]")
                    return MockLLMResponses.plan_reviewer_approve()
            return {}
        
        with patch("src.llm_client.call_agent_with_metrics", side_effect=mock_llm), \
             patch("schemas.state.save_checkpoint", return_value="/tmp/cp.json"), \
             patch("src.graph.save_checkpoint", return_value="/tmp/cp.json"), \
             patch("src.routing.save_checkpoint", return_value="/tmp/cp.json"):
            
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
        
        with patch("src.llm_client.call_agent_with_metrics", side_effect=mock_llm), \
             patch("schemas.state.save_checkpoint", return_value="/tmp/cp.json"), \
             patch("src.graph.save_checkpoint", return_value="/tmp/cp.json"), \
             patch("src.routing.save_checkpoint", return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60)
            print("TEST: Stage Selection")
            print("=" * 60)
            
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
# Run tests directly
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
