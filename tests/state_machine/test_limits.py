"""Limit-escalation E2E tests."""

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


class TestLimitEscalation:
    """Tests for limit-induced ask_user flows and resumption."""

    def test_code_review_limit_interrupts_and_resumes(self, initial_state):
        """
        Code revision loop should hit the configured limit, trigger ask_user with the
        code_review_limit contract, and resume once the user provides a hint.
        """
        state = initial_state
        runtime_config = {**state.get("runtime_config", {})}
        runtime_config["max_code_revisions"] = 1
        state["runtime_config"] = runtime_config

        hint_message = "PROVIDE_HINT: tighten meshing near the nanorod."

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            if agent == "simulation_designer":
                return MockLLMResponses.simulation_designer()
            if agent == "design_reviewer":
                return MockLLMResponses.design_reviewer_approve()
            if agent == "code_generator":
                return MockLLMResponses.code_generator()
            if agent == "code_reviewer":
                return MockLLMResponses.code_reviewer_reject()
            if agent == "supervisor":
                return MockLLMResponses.supervisor_continue()
            return {}

        mock_ask_user = create_mock_ask_user_node()

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "noop"},
        ), patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), patch(
            "src.graph.ask_user_node", side_effect=mock_ask_user
        ):
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("code_limit")}}

            interrupt_event = None
            for event in graph.stream(state, config):
                if "__interrupt__" in event:
                    interrupt_event = event
                    break

            assert interrupt_event is not None

            limit_state = graph.get_state(config).values

            assert limit_state.get("ask_user_trigger") == "code_review_limit"
            assert limit_state.get("awaiting_user_input") is True
            questions = limit_state.get("pending_user_questions", [])
            assert questions
            assert limit_state.get("last_node_before_ask_user") == "code_review"

            graph.update_state(
                config,
                {
                    "user_responses": {questions[0]: hint_message},
                },
            )

            resumed = False
            for event in graph.stream(None, config):
                if "supervisor" in event:
                    resumed = True
                    break
            assert resumed

            post_state = graph.get_state(config).values
            assert post_state.get("code_revision_count") == 0
            assert post_state.get("supervisor_verdict") == "ok_continue"
            assert post_state.get("ask_user_trigger") is None
            assert post_state.get("awaiting_user_input") is False
            feedback = post_state.get("reviewer_feedback", "")
            assert "hint" in feedback.lower()


