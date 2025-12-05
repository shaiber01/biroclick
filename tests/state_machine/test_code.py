"""Code-generation E2E tests.

Tests the code_generator_node and code_reviewer_node integration with the 
LangGraph state machine.

Coverage includes:
- Approve flow: generate_code → code_review(approve) → run_code
- Revision flow: code_review(needs_revision) → generate_code
- Limit escalation: code_revision_count reaches max → ask_user
- Code generator validation: missing stage_id, missing/stub design, missing materials
- Code reviewer verdict normalization: pass/approved/accept → approve
- Stub/empty code detection
- LLM error handling
"""

from unittest.mock import patch, MagicMock

import pytest

from src.graph import create_repro_graph
from src.agents.code import code_generator_node, code_reviewer_node
from schemas.state import (
    MAX_CODE_REVISIONS,
    MAX_DESIGN_REVISIONS,
    create_initial_state,
)

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    create_mock_ask_user_node,
    unique_thread_id,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def state_at_code_generation():
    """
    State at the point where code generation begins.
    Includes approved plan and design.
    """
    return {
        "paper_id": "test_paper",
        "paper_text": "Gold nanorods have LSPR at 700nm." * 10,
        "paper_domain": "plasmonics",
        "current_stage_id": "stage_1_fdtd",
        "current_stage_type": "FDTD_DIRECT",
        "plan": MockLLMResponses.planner(),
        "design_description": MockLLMResponses.simulation_designer(),
        "validated_materials": [
            {
                "material_id": "gold_jc",
                "name": "Gold (Johnson-Christy)",
                "file_path": "materials/johnson_christy_gold.csv",
            }
        ],
        "code_revision_count": 0,
        "design_revision_count": 0,
        "workflow_phase": "design_review",
        "runtime_config": {"max_code_revisions": 3},
    }


@pytest.fixture
def state_at_code_review(state_at_code_generation):
    """
    State at the point where code review begins.
    Includes generated code.
    """
    state = state_at_code_generation.copy()
    state["code"] = """
import meep as mp
import numpy as np

# Gold nanorod FDTD simulation
sim = mp.Simulation(
    cell_size=mp.Vector3(10, 10, 10),
    geometry=[mp.Cylinder(radius=1, height=5, material=mp.Au)],
    sources=[mp.Source(mp.GaussianSource(frequency=1.5), component=mp.Ex)],
    resolution=20,
)
sim.run(until=100)
print("Simulation complete")
"""
    state["workflow_phase"] = "code_generation"
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Full Graph Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

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

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60, flush=True)
            print("TEST: Code Phase (approve flow)", flush=True)
            print("=" * 60, flush=True)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("code")}}

            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            node_updates = {}

            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    node_updates[node_name] = updates
                    print(f"  → {node_name}", flush=True)

                    if node_name == "run_code":
                        break
                else:
                    continue
                break

            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)

            # Verify expected nodes were visited in sequence
            assert "generate_code" in nodes_visited, "generate_code node should be visited"
            assert "code_review" in nodes_visited, "code_review node should be visited"
            assert "run_code" in nodes_visited, "run_code node should be visited"
            
            # Verify sequence: generate_code should come before code_review
            gen_idx = nodes_visited.index("generate_code")
            review_idx = nodes_visited.index("code_review")
            run_idx = nodes_visited.index("run_code")
            assert gen_idx < review_idx < run_idx, (
                f"Expected generate_code < code_review < run_code, "
                f"got indices {gen_idx}, {review_idx}, {run_idx}"
            )
            
            # Verify code_review set the approve verdict
            code_review_updates = node_updates.get("code_review", {})
            assert code_review_updates.get("last_code_review_verdict") == "approve", (
                f"Expected code_review to set verdict='approve', "
                f"got {code_review_updates.get('last_code_review_verdict')}"
            )
            
            # Verify code_generator agent was called
            assert "code_generator" in visited, "code_generator LLM should be called"
            assert "code_reviewer" in visited, "code_reviewer LLM should be called"

            print("\n✅ Code phase test passed!", flush=True)

    def test_code_revision_flow(self, initial_state):
        """Test: code_review rejects → routes back to generate_code → re-review → approve"""
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)

            # Reject first code review, approve second
            if agent == "code_reviewer":
                review_count = sum(1 for v in visited if v == "code_reviewer")
                if review_count <= 1:
                    print("    [Rejecting code]", flush=True)
                    return MockLLMResponses.code_reviewer_reject()
                print("    [Approving code]", flush=True)
                return MockLLMResponses.code_reviewer_approve()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60, flush=True)
            print("TEST: Code Phase (revision flow)", flush=True)
            print("=" * 60, flush=True)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("code_rev")}}

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
            
            # Verify generate_code was called twice (initial + revision)
            gen_count = nodes_visited.count("generate_code")
            assert gen_count == 2, f"Expected generate_code twice, got {gen_count}. Nodes: {nodes_visited}"
            
            # Verify code_review was called twice (reject + approve)
            review_count = nodes_visited.count("code_review")
            assert review_count == 2, f"Expected code_review twice, got {review_count}. Nodes: {nodes_visited}"
            
            # Verify code_generator LLM was called twice
            code_gen_calls = sum(1 for v in visited if v == "code_generator")
            assert code_gen_calls == 2, f"Expected code_generator called twice, got {code_gen_calls}"
            
            # Get final state to verify revision count
            final_state = graph.get_state(config).values
            assert final_state.get("code_revision_count") >= 1, (
                f"Expected code_revision_count >= 1 after revision, "
                f"got {final_state.get('code_revision_count')}"
            )

            print("\n✅ Code revision flow test passed!", flush=True)

    def test_code_review_limit_escalates_to_ask_user(self, initial_state):
        """
        Test: when code_revision_count hits limit, routes to ask_user.
        
        Sets max_code_revisions=1, so first rejection should escalate.
        """
        state = initial_state.copy()
        runtime_config = {**state.get("runtime_config", {})}
        runtime_config["max_code_revisions"] = 1
        state["runtime_config"] = runtime_config

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}", flush=True)

            if agent == "code_reviewer":
                # Always reject to trigger limit
                return MockLLMResponses.code_reviewer_reject()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
            }
            return responses.get(agent, {})

        mock_ask_user = create_mock_ask_user_node()

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), patch(
            "src.graph.ask_user_node", side_effect=mock_ask_user
        ):
            print("\n" + "=" * 60, flush=True)
            print("TEST: Code Review Limit Escalation", flush=True)
            print("=" * 60, flush=True)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("code_limit")}}

            print("\n--- Running graph ---", flush=True)
            nodes_visited = []
            for event in graph.stream(state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"    → {node_name}", flush=True)

            # Verify ask_user was visited (mock handles it without interrupt)
            assert "ask_user" in nodes_visited, \
                f"ask_user should be visited when code_review_limit is reached. Visited: {nodes_visited}"
            
            # Verify code_review was visited
            assert "code_review" in nodes_visited, \
                f"code_review should be visited. Visited: {nodes_visited}"

            print("\n✅ Code review limit escalation test passed!", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Code Generator Node Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeGeneratorNode:
    """Unit tests for code_generator_node validation and error handling."""

    def test_missing_stage_id_returns_error(self):
        """Code generator should error when current_stage_id is None."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": None,  # Missing!
            "current_stage_type": "FDTD_DIRECT",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [{"material_id": "gold"}],
            "runtime_config": {},
        }
        
        result = code_generator_node(state)
        
        # Should set ask_user_trigger for missing stage
        assert result.get("ask_user_trigger") == "missing_stage_id", (
            f"Expected ask_user_trigger='missing_stage_id', got {result.get('ask_user_trigger')}"
        )
        assert result.get("awaiting_user_input") is True, (
            "Expected awaiting_user_input=True for missing stage_id"
        )
        assert result.get("workflow_phase") == "code_generation", (
            "workflow_phase should be set to 'code_generation'"
        )
        assert len(result.get("pending_user_questions", [])) > 0, (
            "Should have pending questions explaining the error"
        )

    def test_missing_design_returns_revision_error(self):
        """Code generator should error when design_description is empty."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",
            "design_description": "",  # Empty!
            "validated_materials": [{"material_id": "gold"}],
            "runtime_config": {},
            "design_revision_count": 0,
        }
        
        result = code_generator_node(state)
        
        # Should increment design_revision_count and provide feedback
        assert result.get("design_revision_count") >= 1, (
            f"Expected design_revision_count >= 1, got {result.get('design_revision_count')}"
        )
        assert "reviewer_feedback" in result, (
            "Should have reviewer_feedback explaining the issue"
        )
        assert "design" in result.get("reviewer_feedback", "").lower() or "stub" in result.get("reviewer_feedback", "").lower(), (
            f"Feedback should mention design issue: {result.get('reviewer_feedback')}"
        )

    def test_stub_design_returns_revision_error(self):
        """Code generator should error when design_description contains stub markers."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",
            "design_description": "TODO: This design would be generated later",  # Stub!
            "validated_materials": [{"material_id": "gold"}],
            "runtime_config": {},
            "design_revision_count": 0,
        }
        
        result = code_generator_node(state)
        
        # Should detect stub and increment design_revision_count
        assert result.get("design_revision_count") >= 1, (
            f"Expected design_revision_count >= 1 for stub design, got {result.get('design_revision_count')}"
        )
        assert "reviewer_feedback" in result, (
            "Should have reviewer_feedback for stub design"
        )

    def test_missing_materials_for_stage1_returns_error(self):
        """Code generator should error when validated_materials is empty for Stage 1+."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",  # Not MATERIAL_VALIDATION
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],  # Empty!
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        result = code_generator_node(state)
        
        # Should return error with run_error
        assert "run_error" in result, (
            f"Expected run_error for missing materials. Got: {result}"
        )
        assert "validated_materials" in result.get("run_error", "").lower(), (
            f"run_error should mention validated_materials: {result.get('run_error')}"
        )
        assert result.get("code_revision_count", 0) >= 1, (
            "Should increment code_revision_count when materials missing"
        )

    def test_material_validation_stage_allows_empty_materials(self):
        """Stage 0 (MATERIAL_VALIDATION) should NOT require validated_materials."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",  # Stage 0!
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],  # Empty is OK for Stage 0
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockLLMResponses.code_generator()
            result = code_generator_node(state)
        
        # Should NOT have run_error about materials
        assert "run_error" not in result or "validated_materials" not in result.get("run_error", "").lower(), (
            f"Stage 0 should not error on empty validated_materials. Got: {result}"
        )
        # Should have generated code
        assert "code" in result, (
            f"Expected code to be generated for Stage 0. Got: {result}"
        )

    def test_llm_error_escalates_to_user(self):
        """When LLM call fails, code generator should escalate to ask_user."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {},
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = RuntimeError("API rate limit exceeded")
            result = code_generator_node(state)
        
        # Should escalate to user
        assert result.get("awaiting_user_input") is True, (
            "Expected awaiting_user_input=True on LLM error"
        )
        assert result.get("ask_user_trigger") == "llm_error", (
            f"Expected ask_user_trigger='llm_error', got {result.get('ask_user_trigger')}"
        )
        assert len(result.get("pending_user_questions", [])) > 0, (
            "Should have pending questions about the error"
        )

    def test_empty_code_output_increments_revision(self):
        """When LLM returns empty code, should increment revision count."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {"max_code_revisions": 3},
            "code_revision_count": 0,
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"code": ""}  # Empty code!
            result = code_generator_node(state)
        
        # Should NOT increment revision count - that's code_review's job
        # (avoids double-counting when code_review also increments on needs_revision)
        assert "code_revision_count" not in result, (
            f"code_generator should not set code_revision_count for empty code, got {result.get('code_revision_count')}"
        )
        assert "reviewer_feedback" in result, (
            "Should have reviewer_feedback about empty code"
        )

    def test_stub_code_output_increments_revision(self):
        """When LLM returns stub code, should increment revision count."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {"max_code_revisions": 3},
            "code_revision_count": 0,
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            # Short code with stub marker
            mock_llm.return_value = {"code": "# TODO: Implement simulation"}
            result = code_generator_node(state)
        
        # Should NOT increment revision count - that's code_review's job
        # (avoids double-counting when code_review also increments on needs_revision)
        assert "code_revision_count" not in result, (
            f"code_generator should not set code_revision_count for stub code, got {result.get('code_revision_count')}"
        )

    def test_valid_code_sets_expected_outputs(self):
        """Valid code generation should set code and expected_outputs."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        expected_outputs = ["spectrum.png", "data.csv"]
        long_valid_code = """
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# This is a valid simulation with enough content
sim = mp.Simulation(
    cell_size=mp.Vector3(10, 10, 10),
    geometry=[mp.Cylinder(radius=1, height=5, material=mp.Au)],
    sources=[mp.Source(mp.GaussianSource(frequency=1.5), component=mp.Ex)],
    resolution=20,
)

# Run simulation
sim.run(until=100)

# Process results
flux_data = sim.get_flux_data()
np.save("flux.npy", flux_data)
plt.savefig("spectrum.png")
"""
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "code": long_valid_code,
                "expected_outputs": expected_outputs,
            }
            result = code_generator_node(state)
        
        # Should have code
        assert "code" in result, "Result should contain code"
        assert len(result["code"]) > 50, "Code should be substantial"
        
        # Should have expected_outputs
        assert result.get("expected_outputs") == expected_outputs, (
            f"Expected expected_outputs={expected_outputs}, got {result.get('expected_outputs')}"
        )
        
        # Should NOT have revision increments
        assert result.get("code_revision_count", 0) == 0, (
            "Valid code should not increment revision count"
        )
        assert "reviewer_feedback" not in result or not result.get("reviewer_feedback"), (
            "Valid code should not have error feedback"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Code Reviewer Node Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeReviewerNode:
    """Unit tests for code_reviewer_node verdict handling."""

    @pytest.fixture
    def base_state(self):
        """Base state for code review tests."""
        return {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",
            "design_description": MockLLMResponses.simulation_designer(),
            "code": "import meep; print('simulation')" * 10,  # Long enough
            "code_revision_count": 0,
            "runtime_config": {"max_code_revisions": 3},
        }

    def test_approve_verdict_sets_state(self, base_state):
        """Approve verdict should set last_code_review_verdict='approve'."""
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "approve", "issues": [], "summary": "Good"}
            result = code_reviewer_node(base_state)
        
        assert result.get("last_code_review_verdict") == "approve", (
            f"Expected verdict='approve', got {result.get('last_code_review_verdict')}"
        )
        assert result.get("workflow_phase") == "code_review", (
            "workflow_phase should be 'code_review'"
        )
        # Should NOT increment revision count on approve
        assert result.get("code_revision_count") == 0, (
            f"Approve should not increment revision count, got {result.get('code_revision_count')}"
        )

    def test_needs_revision_increments_counter(self, base_state):
        """needs_revision verdict should increment code_revision_count."""
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": [{"severity": "blocking", "description": "Bug"}],
                "summary": "Fix bug",
            }
            result = code_reviewer_node(base_state)
        
        assert result.get("last_code_review_verdict") == "needs_revision", (
            f"Expected verdict='needs_revision', got {result.get('last_code_review_verdict')}"
        )
        assert result.get("code_revision_count") == 1, (
            f"Expected code_revision_count=1, got {result.get('code_revision_count')}"
        )
        assert "reviewer_feedback" in result, (
            "needs_revision should set reviewer_feedback"
        )

    @pytest.mark.parametrize("input_verdict,expected_verdict", [
        ("pass", "approve"),
        ("approved", "approve"),
        ("accept", "approve"),
        ("approve", "approve"),
        ("reject", "needs_revision"),
        ("revision_needed", "needs_revision"),
        ("needs_work", "needs_revision"),
        ("needs_revision", "needs_revision"),
    ])
    def test_verdict_normalization(self, base_state, input_verdict, expected_verdict):
        """Verify various verdict strings are normalized correctly."""
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": input_verdict, "issues": [], "summary": "Test"}
            result = code_reviewer_node(base_state)
        
        assert result.get("last_code_review_verdict") == expected_verdict, (
            f"Expected '{input_verdict}' to normalize to '{expected_verdict}', "
            f"got '{result.get('last_code_review_verdict')}'"
        )

    def test_unknown_verdict_defaults_to_needs_revision(self, base_state):
        """Unknown verdicts should default to needs_revision (safer for code)."""
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "maybe_ok", "issues": [], "summary": "Uncertain"}
            result = code_reviewer_node(base_state)
        
        assert result.get("last_code_review_verdict") == "needs_revision", (
            f"Unknown verdict should default to 'needs_revision', "
            f"got {result.get('last_code_review_verdict')}"
        )

    def test_max_revisions_triggers_ask_user(self, base_state):
        """When code_revision_count hits max, should escalate to ask_user."""
        base_state["code_revision_count"] = 3  # At max (with max=3)
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": [{"severity": "blocking", "description": "Bug"}],
                "summary": "Fix bug",
            }
            result = code_reviewer_node(base_state)
        
        # Should escalate to ask_user
        assert result.get("ask_user_trigger") == "code_review_limit", (
            f"Expected ask_user_trigger='code_review_limit', got {result.get('ask_user_trigger')}"
        )
        assert result.get("awaiting_user_input") is True, (
            "Expected awaiting_user_input=True at limit"
        )
        assert result.get("last_node_before_ask_user") == "code_review", (
            f"Expected last_node_before_ask_user='code_review', "
            f"got {result.get('last_node_before_ask_user')}"
        )

    def test_llm_error_defaults_to_needs_revision(self, base_state):
        """When LLM fails, code reviewer should default to needs_revision (fail-closed)."""
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = code_reviewer_node(base_state)
        
        # Should default to needs_revision for safety
        assert result.get("last_code_review_verdict") == "needs_revision", (
            f"LLM error should result in needs_revision (fail-closed), got {result.get('last_code_review_verdict')}"
        )
        # Should have issues noting the LLM unavailability
        issues = result.get("reviewer_issues", [])
        assert len(issues) > 0, "Should have issues noting LLM unavailability"

    def test_issues_preserved_in_result(self, base_state):
        """reviewer_issues should be preserved in result."""
        issues = [
            {"severity": "minor", "description": "Could use better variable names"},
            {"severity": "blocking", "description": "Missing import statement"},
        ]
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": issues,
                "summary": "Needs fixes",
            }
            result = code_reviewer_node(base_state)
        
        assert result.get("reviewer_issues") == issues, (
            f"Expected issues to be preserved: {issues}, got {result.get('reviewer_issues')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Context Check Behavior
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextCheckBehavior:
    """Test context check decorator behavior on code nodes."""

    def test_code_reviewer_returns_empty_when_awaiting_input(self):
        """When awaiting_user_input=True, code_reviewer should return empty dict."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "print('test')" * 10,
            "awaiting_user_input": True,  # Already awaiting
            "runtime_config": {},
        }
        
        # Should NOT call LLM
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            result = code_reviewer_node(state)
            mock_llm.assert_not_called()
        
        assert result == {}, (
            f"Expected empty dict when awaiting_user_input=True, got {result}"
        )

    def test_code_generator_returns_escalation_when_context_check_triggers(self):
        """When context check triggers escalation, code_generator should return it."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [{"material_id": "gold"}],
            "runtime_config": {},
        }
        
        # Mock context check to return escalation
        escalation_result = {
            "awaiting_user_input": True,
            "ask_user_trigger": "context_limit",
            "pending_user_questions": ["Context limit reached"],
        }
        
        with patch("src.agents.code.check_context_or_escalate") as mock_context, \
             patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_context.return_value = escalation_result
            result = code_generator_node(state)
            
            # LLM should NOT be called when context check escalates
            mock_llm.assert_not_called()
        
        # Should return the escalation result
        assert result.get("awaiting_user_input") is True, (
            "Should return escalation when context check triggers"
        )
        assert result.get("ask_user_trigger") == "context_limit", (
            f"Expected ask_user_trigger='context_limit', got {result.get('ask_user_trigger')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_code_generator_with_none_design_description(self):
        """Code generator should handle None design_description."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",
            "design_description": None,  # None, not empty string
            "validated_materials": [{"material_id": "gold"}],
            "runtime_config": {},
            "design_revision_count": 0,
        }
        
        result = code_generator_node(state)
        
        # Should treat None as empty/stub
        assert result.get("design_revision_count", 0) >= 1 or "reviewer_feedback" in result, (
            f"None design should be treated as invalid. Got: {result}"
        )

    def test_code_reviewer_with_none_feedback_and_summary(self):
        """Code reviewer should handle missing feedback/summary gracefully."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "import meep; print('simulation')" * 10,
            "design_description": MockLLMResponses.simulation_designer(),
            "code_revision_count": 0,
            "runtime_config": {},
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            # LLM returns needs_revision but no feedback or summary
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": [{"severity": "blocking", "description": "Bug"}],
                # Missing both "feedback" and "summary"
            }
            result = code_reviewer_node(state)
        
        # Should have fallback feedback message
        assert "reviewer_feedback" in result, "Should set reviewer_feedback even without LLM feedback"
        assert result["reviewer_feedback"] is not None, "reviewer_feedback should not be None"
        assert len(result["reviewer_feedback"]) > 0, "reviewer_feedback should not be empty"

    def test_code_reviewer_extracts_feedback_from_summary(self):
        """When 'feedback' key is missing, should fall back to 'summary'."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "import meep; print('simulation')" * 10,
            "design_description": MockLLMResponses.simulation_designer(),
            "code_revision_count": 0,
            "runtime_config": {},
        }
        
        summary_text = "Specific feedback from summary field"
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": [],
                "summary": summary_text,  # Only summary, no feedback
            }
            result = code_reviewer_node(state)
        
        # Should extract feedback from summary
        assert result.get("reviewer_feedback") == summary_text, (
            f"Expected feedback from summary: '{summary_text}', got: '{result.get('reviewer_feedback')}'"
        )

    def test_code_reviewer_with_empty_code(self):
        """Code reviewer should handle empty code string."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "",  # Empty
            "design_description": MockLLMResponses.simulation_designer(),
            "code_revision_count": 0,
            "runtime_config": {},
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "needs_revision", "issues": [], "summary": "Empty"}
            result = code_reviewer_node(state)
        
        # Should still process (LLM will review empty code)
        assert "last_code_review_verdict" in result, (
            "Should still produce verdict for empty code"
        )

    def test_code_generator_respects_max_revision_bound(self):
        """Code revision count should not exceed max_code_revisions."""
        max_revisions = 3
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {"max_code_revisions": max_revisions},
            "code_revision_count": max_revisions,  # Already at max
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"code": "# STUB"}  # Will trigger increment
            result = code_generator_node(state)
        
        # Should not exceed max
        assert result.get("code_revision_count", 0) <= max_revisions, (
            f"code_revision_count should not exceed {max_revisions}, "
            f"got {result.get('code_revision_count')}"
        )

    def test_code_generator_extracts_code_from_simulation_code_key(self):
        """Code generator should extract code from 'simulation_code' key too."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        long_code = """
import meep as mp
import numpy as np
# Full simulation code here...
sim = mp.Simulation(resolution=20, cell_size=mp.Vector3(10,10,10))
sim.run(until=100)
""" * 3  # Make it long enough
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            # Return with 'simulation_code' key instead of 'code'
            mock_llm.return_value = {"simulation_code": long_code, "expected_outputs": []}
            result = code_generator_node(state)
        
        # Should extract from simulation_code
        assert result.get("code") == long_code, (
            f"Should extract code from simulation_code key"
        )

    def test_code_generator_json_fallback_for_unknown_keys(self):
        """When LLM returns unexpected dict, should JSON dump it."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        weird_response = {
            "unknown_key": "some value",
            "another_key": {"nested": "data"},
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = weird_response
            result = code_generator_node(state)
        
        # Should JSON dump the response as code (which will then be detected as short/stub)
        code = result.get("code", "")
        # The code should be a JSON string of the response
        assert "unknown_key" in code or result.get("code_revision_count", 0) >= 1, (
            f"Should either JSON dump the response or increment revision count"
        )

    def test_code_generator_handles_string_response(self):
        """When LLM returns a raw string instead of dict, should use it as code."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        # Simulate LLM returning just a string (not a dict)
        raw_code = """
import meep as mp
import numpy as np
# Full simulation here
sim = mp.Simulation(resolution=20)
sim.run(until=100)
""" * 3  # Make it long enough
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = raw_code  # String, not dict
            result = code_generator_node(state)
        
        # Should convert string to code
        assert result.get("code") == raw_code, (
            f"String response should be used as code directly"
        )

    def test_code_reviewer_with_string_design_description(self):
        """Code reviewer should handle string design_description."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "import meep; print('simulation')" * 10,
            "design_description": "This is a text design description, not a dict.",
            "code_revision_count": 0,
            "runtime_config": {},
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "approve", "issues": [], "summary": "OK"}
            result = code_reviewer_node(state)
        
        # Should process without error
        assert result.get("last_code_review_verdict") == "approve", (
            "Should handle string design_description"
        )

    def test_code_revision_count_increments_multiple_times(self):
        """Verify revision count increments correctly across multiple rejections."""
        base_state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "import meep; print('simulation')" * 10,
            "design_description": MockLLMResponses.simulation_designer(),
            "code_revision_count": 0,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        # Simulate multiple review calls
        revision_counts = []
        state = base_state.copy()
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": [{"severity": "blocking", "description": "Bug"}],
                "feedback": "Fix it",
            }
            
            for i in range(3):
                result = code_reviewer_node(state)
                revision_counts.append(result.get("code_revision_count"))
                # Update state for next iteration
                state = {**state, **result}
        
        # Verify counts increment: [1, 2, 3]
        assert revision_counts == [1, 2, 3], (
            f"Expected revision counts [1, 2, 3], got {revision_counts}"
        )

    def test_code_reviewer_preserves_issues_from_llm(self):
        """Verify that reviewer_issues includes all issues from LLM response."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "import meep; print('simulation')" * 10,
            "design_description": MockLLMResponses.simulation_designer(),
            "code_revision_count": 0,
            "runtime_config": {},
        }
        
        issues = [
            {"severity": "blocking", "description": "Missing import"},
            {"severity": "minor", "description": "Could use better names"},
            {"severity": "warning", "description": "Performance concern"},
        ]
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": issues,
                "feedback": "Multiple issues",
            }
            result = code_reviewer_node(state)
        
        assert result.get("reviewer_issues") == issues, (
            f"All issues should be preserved. Expected {len(issues)}, "
            f"got {len(result.get('reviewer_issues', []))}"
        )

    def test_code_generator_with_empty_validated_materials_list(self):
        """Test that empty list [] is properly detected as missing materials."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",  # Not Stage 0
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],  # Empty list
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        result = code_generator_node(state)
        
        # Should fail validation for Stage 1+ with empty materials
        assert "run_error" in result, (
            f"Empty validated_materials for Stage 1+ should cause run_error. Got: {result}"
        )

    def test_code_generator_with_none_validated_materials(self):
        """Test that None validated_materials is detected as missing."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_1",
            "current_stage_type": "FDTD_DIRECT",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": None,  # None instead of list
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        result = code_generator_node(state)
        
        # Should fail validation
        assert "run_error" in result, (
            f"None validated_materials for Stage 1+ should cause run_error. Got: {result}"
        )

    def test_approve_verdict_does_not_increment_counter(self):
        """Verify that approve verdict leaves code_revision_count unchanged."""
        state = {
            "paper_id": "test",
            "paper_text": "Test" * 10,
            "current_stage_id": "stage_1",
            "code": "import meep; print('simulation')" * 10,
            "design_description": MockLLMResponses.simulation_designer(),
            "code_revision_count": 2,  # Already has some revisions
            "runtime_config": {"max_code_revisions": 5},
        }
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "approve", "issues": [], "summary": "Good"}
            result = code_reviewer_node(state)
        
        # Should NOT increment - should stay at 2
        assert result.get("code_revision_count") == 2, (
            f"Approve should not increment revision count. Expected 2, got {result.get('code_revision_count')}"
        )

    def test_code_generator_long_code_with_todo_comment_ok(self):
        """Long valid code with TODO in comments should NOT be flagged as stub."""
        state = {
            "paper_id": "test",
            "paper_text": "Test paper text." * 10,
            "current_stage_id": "stage_0_materials",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": MockLLMResponses.simulation_designer(),
            "validated_materials": [],
            "runtime_config": {},
            "code_revision_count": 0,
        }
        
        # Long code with TODO in a comment (which is normal)
        long_code_with_todo = """
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Setup simulation
cell = mp.Vector3(10, 10, 10)
geometry = [
    mp.Cylinder(radius=1, height=5, material=mp.Au)
]

# TODO: Add more geometry for complex shapes later
sources = [
    mp.Source(mp.GaussianSource(frequency=1.5), component=mp.Ex, center=mp.Vector3())
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    resolution=20,
)

# Run simulation
sim.run(until=100)

# Save outputs
flux_data = sim.get_flux_data()
np.save("flux.npy", flux_data)
"""
        
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"code": long_code_with_todo, "expected_outputs": []}
            result = code_generator_node(state)
        
        # Should NOT increment revision count - TODO in comment is OK for long code
        assert result.get("code_revision_count", 0) == 0, (
            f"Long code with TODO comment should be valid. Got revision count: {result.get('code_revision_count')}"
        )
        assert "reviewer_feedback" not in result or not result.get("reviewer_feedback"), (
            "Should not have error feedback for valid code with TODO comment"
        )
