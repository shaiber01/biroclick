"""Code-generation E2E tests."""

from src.graph import create_repro_graph

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


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


