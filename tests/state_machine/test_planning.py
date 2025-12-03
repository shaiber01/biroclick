"""Planning and stage-selection E2E tests."""

from src.graph import create_repro_graph

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


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
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
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
                        pass

                    if node_name == "select_stage":
                        state = graph.get_state(config)
                        final_state = state.values
                        break
                else:
                    continue
                break

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
                review_count = sum(1 for v in visited if v == "plan_reviewer")
                if review_count <= 1:
                    print("    [Rejecting plan]")
                    return MockLLMResponses.plan_reviewer_reject()
                print("    [Approving plan]")
                return MockLLMResponses.plan_reviewer_approve()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
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

            assert nodes_visited.count("plan") == 2
            assert nodes_visited.count("plan_review") == 2

            print("\n✅ Revision flow test passed!")


class TestStageSelection:
    """Test planning through stage selection."""

    def test_stage_selection_picks_first_stage(self, initial_state):
        """After plan approval, select_stage should pick the first available stage."""

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}")

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
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
                        state = graph.get_state(config)
                        final_state = state.values
                        print(f"\n  Selected stage: {final_state.get('current_stage_id')}")
                        print(f"  Stage type: {final_state.get('current_stage_type')}")

                        assert final_state.get("progress") is not None
                        assert (
                            len(final_state.get("progress", {}).get("stages", [])) > 0
                        )
                        break
                else:
                    continue
                break

            assert final_state is not None
            assert final_state.get("current_stage_id") == "stage_0_materials"
            assert final_state.get("current_stage_type") == "MATERIAL_VALIDATION"

            progress = final_state.get("progress", {})
            stages = progress.get("stages", [])
            assert len(stages) == 1
            assert stages[0]["stage_id"] == "stage_0_materials"
            assert stages[0]["status"] == "not_started"

            print("\n✅ Stage selection test passed!")


