"""Execution-phase E2E tests."""

from unittest.mock import patch

from src.graph import create_repro_graph

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


class TestExecutionPhase:
    """Test run_code → execution_check flow."""

    def test_execution_success_flow(self, initial_state):
        """Test: run_code → execution_check(pass) → physics_check"""
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

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

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Simulating..."},
        ):
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


