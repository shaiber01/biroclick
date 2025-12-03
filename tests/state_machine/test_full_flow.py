"""Full single-stage happy-path E2E test."""

from unittest.mock import patch

from src.graph import create_repro_graph

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    create_mock_ask_user_node,
    unique_thread_id,
)


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

            if agent == "supervisor":
                supervisor_calls = sum(1 for c in llm_calls if "supervisor" in c)
                print(f"    [DEBUG] Supervisor called {supervisor_calls} times")
                if supervisor_calls == 1:
                    print("    [DEBUG] Returning supervisor_continue (trigger checkpoint)")
                    return MockLLMResponses.supervisor_continue()
                print("    [DEBUG] Returning all_complete")
                return {"verdict": "all_complete", "reasoning": "User approved, done."}

            return responses.get(agent, {})

        mock_ask_user = create_mock_ask_user_node()

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Success"},
        ), patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), patch(
            "src.graph.ask_user_node", side_effect=mock_ask_user
        ):
            print("\n" + "=" * 60)
            print("TEST: Full Single-Stage Happy Path")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("happy_path")}}

            print("\n--- Running graph ---")

            steps = 0
            max_steps = 40
            pending_state = initial_state
            completed = False

            while steps < max_steps and not completed:
                interrupted = False
                for event in graph.stream(pending_state, config):
                    for node_name, updates in event.items():
                        steps += 1
                        visited_nodes.append(node_name)
                        print(f"  [{steps}] → {node_name}")

                        if node_name == "generate_report":
                            completed = True
                            break

                        if node_name == "__interrupt__":
                            interrupted = True
                            break
                    if completed or interrupted:
                        break

                if completed:
                    break

                if not interrupted:
                    print("⚠️ Workflow exited without reaching generate_report")
                    break

                state_snapshot = graph.get_state(config).values
                trigger = state_snapshot.get("ask_user_trigger") or "material_checkpoint"
                questions = state_snapshot.get("pending_user_questions") or []
                response_key = questions[0] if questions else trigger

                print("  [Interrupt Detected] - Resuming with mock user approval")
                print(f"  [DEBUG] trigger={trigger}, responding to {response_key}")

                graph.update_state(
                    config,
                    {
                        "user_responses": {
                            response_key: "approved",
                        }
                    },
                )

                pending_state = None

            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Nodes visited: {len(visited_nodes)}")
            print(f"Unique nodes: {len(set(visited_nodes))}")
            print(f"LLM agents called: {len(set(llm_calls))}")
            print(f"Flow: {' → '.join(visited_nodes)}")

            assert "plan" in visited_nodes
            assert "design" in visited_nodes
            assert "generate_code" in visited_nodes
            assert "execution_check" in visited_nodes
            assert "analyze" in visited_nodes
            assert "generate_report" in visited_nodes

            print("\n✅ Full single-stage test passed!")


