"""Design-phase E2E tests."""

from src.graph import create_repro_graph

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


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

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
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

                    if node_name == "design_review":
                        assert updates.get("last_design_review_verdict") == "approve"

                    if node_name == "generate_code":
                        break
                else:
                    continue
                break

            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)

            assert "design" in nodes_visited
            assert "design_review" in nodes_visited
            assert "generate_code" in nodes_visited

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
                review_count = sum(1 for v in visited if v == "design_reviewer")
                if review_count <= 1:
                    print("    [Rejecting design]", flush=True)
                    return MockLLMResponses.design_reviewer_reject()
                print("    [Approving design]", flush=True)
                return MockLLMResponses.design_reviewer_approve()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
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

            assert nodes_visited.count("design") == 2, f"Expected design twice: {nodes_visited}"
            assert nodes_visited.count("design_review") == 2

            print("\n✅ Design revision flow test passed!", flush=True)


