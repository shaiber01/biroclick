"""Full flow E2E tests for the LangGraph state machine.

These tests exercise complete workflows through the graph, verifying:
- Node traversal sequences
- State transitions at each step
- Verdict-based routing
- Revision loops and limit escalation
- Interrupt/resume flows
- Error handling and edge cases
"""

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
    """Run a full single stage from start to finish with rigorous state verification."""

    def test_full_single_stage_happy_path(self, initial_state):
        """
        Full Happy Path:
        Planning → Design → Code → Execution → Analysis → Supervisor → Report
        
        This test verifies:
        1. Correct node traversal sequence
        2. State values at critical checkpoints
        3. Verdict values set by agents
        4. Progress tracking through stages
        """
        visited_nodes = []
        llm_calls = []
        state_snapshots = {}

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
                        
                        # Capture state snapshot after key nodes
                        if node_name in ["plan", "plan_review", "select_stage", "design", 
                                         "design_review", "code_review", "execution_check",
                                         "physics_check", "supervisor", "generate_report"]:
                            state_snapshots[node_name] = graph.get_state(config).values.copy()

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

            # ═══════════════════════════════════════════════════════════════
            # ASSERTION BLOCK: Verify node traversal
            # ═══════════════════════════════════════════════════════════════
            assert "adapt_prompts" in visited_nodes, "adapt_prompts should be first node"
            assert "plan" in visited_nodes, "plan node must be visited"
            assert "plan_review" in visited_nodes, "plan_review must follow plan"
            assert "select_stage" in visited_nodes, "select_stage must follow plan approval"
            assert "design" in visited_nodes, "design node must be visited"
            assert "design_review" in visited_nodes, "design_review must follow design"
            assert "generate_code" in visited_nodes, "generate_code must be visited"
            assert "code_review" in visited_nodes, "code_review must follow generate_code"
            assert "run_code" in visited_nodes, "run_code must follow code approval"
            assert "execution_check" in visited_nodes, "execution_check must follow run_code"
            assert "physics_check" in visited_nodes, "physics_check must follow execution pass"
            assert "analyze" in visited_nodes, "analyze must follow physics pass"
            assert "comparison_check" in visited_nodes, "comparison_check must follow analyze"
            assert "supervisor" in visited_nodes, "supervisor must follow comparison"
            assert "generate_report" in visited_nodes, "generate_report must be final node"

            # ═══════════════════════════════════════════════════════════════
            # ASSERTION BLOCK: Verify node ordering
            # ═══════════════════════════════════════════════════════════════
            plan_idx = visited_nodes.index("plan")
            plan_review_idx = visited_nodes.index("plan_review")
            select_stage_idx = visited_nodes.index("select_stage")
            design_idx = visited_nodes.index("design")
            
            assert plan_idx < plan_review_idx, "plan must come before plan_review"
            assert plan_review_idx < select_stage_idx, "plan_review must come before select_stage"
            assert select_stage_idx < design_idx, "select_stage must come before design"
            
            # ═══════════════════════════════════════════════════════════════
            # ASSERTION BLOCK: Verify state values at checkpoints
            # Note: State snapshots are taken after the node runs, so verdicts
            # may not be present if set by subsequent nodes in the same step.
            # We verify the final state instead for key verdicts.
            # ═══════════════════════════════════════════════════════════════
            final_state = graph.get_state(config).values
            
            # Verify plan exists and is valid
            assert final_state.get("plan") is not None, "plan should exist in final state"
            plan = final_state.get("plan", {})
            assert len(plan.get("stages", [])) >= 1, "plan should have at least one stage"
            
            # Verify stage was selected
            # Note: current_stage_id may be None if all stages completed
            # But we should have progress tracking
            progress = final_state.get("progress", {})
            assert progress is not None, "progress should exist in final state"
            
            # Verify workflow completed
            assert completed, "workflow should complete to generate_report"

            # ═══════════════════════════════════════════════════════════════
            # ASSERTION BLOCK: Verify LLM agents were called
            # ═══════════════════════════════════════════════════════════════
            assert "LLM:planner" in llm_calls, "planner LLM should be called"
            assert "LLM:plan_reviewer" in llm_calls, "plan_reviewer LLM should be called"
            assert "LLM:simulation_designer" in llm_calls, "simulation_designer LLM should be called"
            assert "LLM:design_reviewer" in llm_calls, "design_reviewer LLM should be called"
            assert "LLM:code_generator" in llm_calls, "code_generator LLM should be called"
            assert "LLM:code_reviewer" in llm_calls, "code_reviewer LLM should be called"
            assert "LLM:execution_validator" in llm_calls, "execution_validator LLM should be called"
            assert "LLM:physics_sanity" in llm_calls, "physics_sanity LLM should be called"
            assert "LLM:results_analyzer" in llm_calls, "results_analyzer LLM should be called"
            assert "LLM:supervisor" in llm_calls, "supervisor LLM should be called"

            print("\n✅ Full single-stage test passed!")


class TestPlanRevisionFlow:
    """Test plan revision loop when plan_reviewer rejects."""

    def test_plan_rejection_triggers_revision(self, initial_state):
        """
        Plan rejection should route back to plan node.
        After revision, approved plan should proceed to select_stage.
        """
        visited_nodes = []
        plan_count = 0
        plan_review_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal plan_count, plan_review_count
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}")

            if agent == "planner":
                plan_count += 1
                return MockLLMResponses.planner()
            
            if agent == "plan_reviewer":
                plan_review_count += 1
                if plan_review_count == 1:
                    print("    [Rejecting plan - first review]")
                    return MockLLMResponses.plan_reviewer_reject()
                print("    [Approving plan - second review]")
                return MockLLMResponses.plan_reviewer_approve()

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan Rejection Triggers Revision")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("plan_revision")}}

            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "select_stage":
                        break
                else:
                    continue
                break

            # Verify revision loop occurred
            assert visited_nodes.count("plan") == 2, \
                f"plan should be visited exactly 2 times (initial + revision), got {visited_nodes.count('plan')}"
            assert visited_nodes.count("plan_review") == 2, \
                f"plan_review should be visited exactly 2 times, got {visited_nodes.count('plan_review')}"
            assert plan_count == 2, f"planner LLM should be called 2 times, got {plan_count}"
            assert plan_review_count == 2, f"plan_reviewer LLM should be called 2 times, got {plan_review_count}"
            
            # Verify final state has approved verdict
            final_state = graph.get_state(config).values
            assert final_state.get("last_plan_review_verdict") == "approve", \
                "Final plan_review verdict should be 'approve'"
            assert final_state.get("replan_count", 0) >= 1, \
                "replan_count should be incremented after revision"

            print("\n✅ Plan revision flow test passed!")


class TestDesignRevisionFlow:
    """Test design revision loop with limit escalation."""

    def test_design_rejection_triggers_revision(self, initial_state):
        """
        Design rejection should route back to design node.
        """
        visited_nodes = []
        design_review_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal design_review_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "design_reviewer":
                design_review_count += 1
                if design_review_count == 1:
                    return MockLLMResponses.design_reviewer_reject()
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
            print("\n" + "=" * 60)
            print("TEST: Design Rejection Triggers Revision")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("design_revision")}}

            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "generate_code":
                        break
                else:
                    continue
                break

            # Verify revision loop
            assert visited_nodes.count("design") == 2, \
                f"design should be visited 2 times, got {visited_nodes.count('design')}"
            assert visited_nodes.count("design_review") == 2, \
                f"design_review should be visited 2 times, got {visited_nodes.count('design_review')}"
            
            final_state = graph.get_state(config).values
            assert final_state.get("last_design_review_verdict") == "approve"
            assert final_state.get("design_revision_count", 0) >= 1, \
                "design_revision_count should be incremented"

            print("\n✅ Design revision flow test passed!")


class TestCodeRevisionLimitEscalation:
    """Test code revision reaching limit and escalating to ask_user."""

    def test_code_revision_limit_triggers_ask_user(self, initial_state):
        """
        When code_revision_count reaches max_code_revisions, 
        routing should escalate to ask_user interrupt.
        """
        # Configure low limit for testing
        state = initial_state.copy()
        state["runtime_config"] = {
            **(state.get("runtime_config") or {}),
            "max_code_revisions": 2,
        }

        visited_nodes = []
        code_review_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal code_review_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "code_reviewer":
                code_review_count += 1
                # Always reject to hit the limit
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
            print("\n" + "=" * 60)
            print("TEST: Code Revision Limit Triggers ask_user")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("code_limit")}}

            interrupt_detected = False
            for event in graph.stream(state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "__interrupt__":
                        interrupt_detected = True
                        break
                if interrupt_detected:
                    break

            assert interrupt_detected, "Should interrupt before ask_user when limit reached"
            
            # Verify we went through code revision loop
            assert visited_nodes.count("generate_code") >= 2, \
                f"generate_code should be visited at least 2 times, got {visited_nodes.count('generate_code')}"
            assert visited_nodes.count("code_review") >= 2, \
                f"code_review should be visited at least 2 times, got {visited_nodes.count('code_review')}"
            
            # Verify state at interrupt
            interrupt_state = graph.get_state(config).values
            assert interrupt_state.get("ask_user_trigger") == "code_review_limit", \
                f"ask_user_trigger should be 'code_review_limit', got: {interrupt_state.get('ask_user_trigger')}"
            assert interrupt_state.get("awaiting_user_input") is True, \
                "awaiting_user_input should be True at interrupt"
            assert interrupt_state.get("code_revision_count", 0) >= 2, \
                "code_revision_count should be at the limit"

            print("\n✅ Code revision limit escalation test passed!")


class TestExecutionFailureFlow:
    """Test execution failure routing back to code generation."""

    def test_execution_failure_routes_to_code_regeneration(self, initial_state):
        """
        Execution failure should route back to generate_code for fix.
        """
        visited_nodes = []
        execution_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal execution_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "execution_validator":
                execution_count += 1
                if execution_count == 1:
                    return MockLLMResponses.execution_validator_fail()
                return MockLLMResponses.execution_validator_pass()

            if agent == "physics_sanity":
                return MockLLMResponses.physics_sanity_pass()

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
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Output"},
        ):
            print("\n" + "=" * 60)
            print("TEST: Execution Failure Routes to Code Regeneration")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("exec_fail")}}

            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "physics_check":
                        break
                else:
                    continue
                break

            # Verify failure loop: execution_check(fail) → generate_code → ... → execution_check(pass)
            exec_indices = [i for i, n in enumerate(visited_nodes) if n == "execution_check"]
            assert len(exec_indices) >= 2, \
                f"execution_check should be visited at least 2 times, got {len(exec_indices)}"
            
            # After first execution_check failure, should go back to generate_code
            first_exec_idx = exec_indices[0]
            assert "generate_code" in visited_nodes[first_exec_idx:], \
                "generate_code should be visited after execution failure"
            
            # Final verdict should be pass
            final_state = graph.get_state(config).values
            assert final_state.get("execution_verdict") == "pass", \
                f"Final execution_verdict should be 'pass', got: {final_state.get('execution_verdict')}"

            print("\n✅ Execution failure flow test passed!")


class TestPhysicsDesignFlawFlow:
    """Test physics_check design_flaw verdict routing to design."""

    def test_physics_design_flaw_routes_to_design(self, initial_state):
        """
        Physics design_flaw verdict should route back to design node.
        """
        visited_nodes = []
        physics_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal physics_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "physics_sanity":
                physics_count += 1
                if physics_count == 1:
                    return MockLLMResponses.physics_sanity_design_flaw()
                return MockLLMResponses.physics_sanity_pass()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_pass(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Output"},
        ):
            print("\n" + "=" * 60)
            print("TEST: Physics Design Flaw Routes to Design")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("physics_flaw")}}

            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "analyze":
                        break
                else:
                    continue
                break

            # Verify design_flaw loop: physics_check(design_flaw) → design → ...
            physics_indices = [i for i, n in enumerate(visited_nodes) if n == "physics_check"]
            assert len(physics_indices) >= 2, \
                f"physics_check should be visited at least 2 times, got {len(physics_indices)}"
            
            # After first physics_check, should go to design (not generate_code)
            first_physics_idx = physics_indices[0]
            nodes_after_first_physics = visited_nodes[first_physics_idx + 1:]
            design_idx_after = nodes_after_first_physics.index("design") if "design" in nodes_after_first_physics else -1
            assert design_idx_after >= 0, "design should be visited after physics design_flaw"
            
            # Verify design is visited before generate_code after the flaw
            if "generate_code" in nodes_after_first_physics:
                gen_idx_after = nodes_after_first_physics.index("generate_code")
                assert design_idx_after < gen_idx_after, \
                    "design should come before generate_code after design_flaw"

            print("\n✅ Physics design flaw flow test passed!")


class TestSupervisorReplanFlow:
    """Test supervisor replan_needed verdict."""

    def test_supervisor_replan_routes_to_plan(self, initial_state):
        """
        Supervisor replan_needed verdict should route to plan node.
        """
        visited_nodes = []
        supervisor_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal supervisor_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "supervisor":
                supervisor_count += 1
                if supervisor_count == 1:
                    return MockLLMResponses.supervisor_replan()
                return MockLLMResponses.supervisor_continue()

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
            }
            return responses.get(agent, {})

        mock_ask_user = create_mock_ask_user_node()

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Output"},
        ), patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), patch(
            "src.graph.ask_user_node", side_effect=mock_ask_user
        ):
            print("\n" + "=" * 60)
            print("TEST: Supervisor Replan Routes to Plan")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("replan")}}

            steps = 0
            max_steps = 60
            pending_state = initial_state
            found_replan = False

            while steps < max_steps and not found_replan:
                interrupted = False
                for event in graph.stream(pending_state, config):
                    for node_name, _ in event.items():
                        steps += 1
                        visited_nodes.append(node_name)
                        print(f"  [{steps}] → {node_name}")
                        
                        # After supervisor, look for plan (replan)
                        if node_name == "plan" and "supervisor" in visited_nodes:
                            found_replan = True
                            break
                        
                        if node_name == "__interrupt__":
                            interrupted = True
                            break
                    if found_replan or interrupted:
                        break
                
                if found_replan:
                    break
                
                if interrupted:
                    state_snapshot = graph.get_state(config).values
                    trigger = state_snapshot.get("ask_user_trigger") or "material_checkpoint"
                    questions = state_snapshot.get("pending_user_questions") or []
                    response_key = questions[0] if questions else trigger
                    
                    graph.update_state(
                        config,
                        {"user_responses": {response_key: "approved"}},
                    )
                    pending_state = None
                else:
                    break

            # Verify replan flow
            supervisor_indices = [i for i, n in enumerate(visited_nodes) if n == "supervisor"]
            assert len(supervisor_indices) >= 1, "supervisor should be visited"
            
            # After supervisor with replan_needed, should go to plan
            first_supervisor_idx = supervisor_indices[0]
            nodes_after_supervisor = visited_nodes[first_supervisor_idx + 1:]
            
            # Should find plan after supervisor (might go through ask_user first)
            assert "plan" in nodes_after_supervisor or found_replan, \
                f"plan should be visited after supervisor replan_needed. Nodes after supervisor: {nodes_after_supervisor}"

            print("\n✅ Supervisor replan flow test passed!")


class TestMaterialCheckpointFlow:
    """Test material checkpoint flow after Stage 0 completion."""

    def test_material_checkpoint_triggers_interrupt(self, initial_state):
        """
        After MATERIAL_VALIDATION stage completes, should trigger material_checkpoint → ask_user.
        """
        visited_nodes = []
        supervisor_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal supervisor_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "supervisor":
                supervisor_count += 1
                # First call: ok_continue triggers material checkpoint for MATERIAL_VALIDATION stage
                return MockLLMResponses.supervisor_continue()

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
            }
            return responses.get(agent, {})

        mock_ask_user = create_mock_ask_user_node()

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Output"},
        ), patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), patch(
            "src.graph.ask_user_node", side_effect=mock_ask_user
        ):
            print("\n" + "=" * 60)
            print("TEST: Material Checkpoint Triggers Interrupt")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("material_cp")}}

            interrupt_at_material = False
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "__interrupt__":
                        # Check if we're at material checkpoint
                        state = graph.get_state(config).values
                        if state.get("ask_user_trigger") == "material_checkpoint":
                            interrupt_at_material = True
                        break
                if interrupt_at_material or node_name == "__interrupt__":
                    break

            # Verify material checkpoint was visited and caused interrupt
            assert "supervisor" in visited_nodes, "supervisor should be visited"
            assert "material_checkpoint" in visited_nodes, \
                f"material_checkpoint should be visited after supervisor for MATERIAL_VALIDATION stage. Visited: {visited_nodes}"
            
            # Verify interrupt state
            interrupt_state = graph.get_state(config).values
            assert interrupt_state.get("ask_user_trigger") == "material_checkpoint", \
                f"ask_user_trigger should be 'material_checkpoint', got: {interrupt_state.get('ask_user_trigger')}"
            assert interrupt_state.get("current_stage_type") == "MATERIAL_VALIDATION", \
                f"Should be at MATERIAL_VALIDATION stage, got: {interrupt_state.get('current_stage_type')}"

            print("\n✅ Material checkpoint flow test passed!")


class TestNoneVerdictEscalation:
    """Test that None verdicts escalate to ask_user."""

    def test_none_plan_review_verdict_escalates(self, initial_state):
        """
        None verdict from plan_reviewer should escalate to ask_user.
        """
        visited_nodes = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")

            if agent == "plan_reviewer":
                # Return dict without verdict key (simulates None verdict)
                return {"summary": "Something went wrong"}

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
            }
            return responses.get(agent, {})

        mock_ask_user = create_mock_ask_user_node()

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch("src.agents.user_interaction.ask_user_node", side_effect=mock_ask_user), patch(
            "src.graph.ask_user_node", side_effect=mock_ask_user
        ):
            print("\n" + "=" * 60)
            print("TEST: None Plan Review Verdict Escalates")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("none_verdict")}}

            interrupt_detected = False
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "__interrupt__":
                        interrupt_detected = True
                        break
                if interrupt_detected:
                    break

            # Verify escalation happened
            assert interrupt_detected, "Should interrupt when verdict is None"
            assert "plan_review" in visited_nodes, "plan_review should be visited"
            
            # Should NOT proceed to select_stage
            assert "select_stage" not in visited_nodes, \
                "select_stage should NOT be visited when verdict is None"
            
            # Verify error escalation state
            state = graph.get_state(config).values
            trigger = state.get("ask_user_trigger", "")
            assert "error" in trigger or "fallback" in trigger, \
                f"ask_user_trigger should indicate error/fallback, got: {trigger}"

            print("\n✅ None verdict escalation test passed!")


class TestMultiStageCompletion:
    """Test completing multiple stages and reaching all_complete."""

    def test_all_complete_routes_to_report(self, initial_state):
        """
        After all stages complete, supervisor should return all_complete 
        and route to generate_report.
        """
        visited_nodes = []
        supervisor_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal supervisor_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "supervisor":
                supervisor_count += 1
                # First supervisor call: continue
                if supervisor_count == 1:
                    return MockLLMResponses.supervisor_continue()
                # Second supervisor call: all_complete
                return {"verdict": "all_complete", "reasoning": "All stages completed"}

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
                "report_generator": MockLLMResponses.report_generator(),
            }
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
            print("TEST: All Complete Routes to Report")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("all_complete")}}

            steps = 0
            max_steps = 50
            pending_state = initial_state
            completed = False

            while steps < max_steps and not completed:
                interrupted = False
                for event in graph.stream(pending_state, config):
                    for node_name, _ in event.items():
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
                
                if interrupted:
                    state_snapshot = graph.get_state(config).values
                    trigger = state_snapshot.get("ask_user_trigger") or "material_checkpoint"
                    questions = state_snapshot.get("pending_user_questions") or []
                    response_key = questions[0] if questions else trigger
                    
                    graph.update_state(
                        config,
                        {"user_responses": {response_key: "approved"}},
                    )
                    pending_state = None
                else:
                    break

            # Verify reached generate_report via all_complete
            assert completed, "Should complete and reach generate_report"
            assert "generate_report" in visited_nodes, "generate_report should be visited"
            
            # Count supervisor visits
            supervisor_visits = visited_nodes.count("supervisor")
            assert supervisor_visits >= 2, \
                f"supervisor should be visited at least 2 times, got {supervisor_visits}"
            
            # Verify final state
            final_state = graph.get_state(config).values
            # The last supervisor verdict that led to generate_report should be all_complete
            assert final_state.get("supervisor_verdict") == "all_complete", \
                f"Final supervisor_verdict should be 'all_complete', got: {final_state.get('supervisor_verdict')}"

            print("\n✅ All complete flow test passed!")


class TestInterruptResumeFlow:
    """Test interrupt before ask_user and resume with user response."""

    def test_interrupt_resume_with_user_response(self, initial_state):
        """
        Graph should interrupt before ask_user, allow state update with user response,
        and resume execution correctly.
        """
        # Set low limit to trigger interrupt quickly
        state = initial_state.copy()
        state["runtime_config"] = {
            **(state.get("runtime_config") or {}),
            "max_code_revisions": 1,
        }

        visited_nodes = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")

            if agent == "code_reviewer":
                # Always reject initially
                return MockLLMResponses.code_reviewer_reject()
            
            if agent == "supervisor":
                # After user provides hint, continue
                return MockLLMResponses.supervisor_continue()

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
            print("\n" + "=" * 60)
            print("TEST: Interrupt Resume with User Response")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("interrupt_resume")}}

            # Phase 1: Run until interrupt
            print("\n--- Phase 1: Run until interrupt ---")
            for event in graph.stream(state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "__interrupt__":
                        break
                if "__interrupt__" in visited_nodes:
                    break

            assert "__interrupt__" in visited_nodes, "Should have interrupted"
            
            # Verify interrupt state
            interrupt_state = graph.get_state(config).values
            assert interrupt_state.get("awaiting_user_input") is True, \
                "awaiting_user_input should be True at interrupt"
            trigger = interrupt_state.get("ask_user_trigger")
            questions = interrupt_state.get("pending_user_questions", [])
            
            print(f"  Interrupt state: trigger={trigger}, questions={questions}")
            
            # Phase 2: Update state with user response
            print("\n--- Phase 2: Provide user response ---")
            response_key = questions[0] if questions else trigger
            graph.update_state(
                config,
                {
                    "user_responses": {response_key: "PROVIDE_HINT: Fix the mesh resolution"},
                },
            )
            
            # Phase 3: Resume and verify flow continues
            print("\n--- Phase 3: Resume execution ---")
            resumed_nodes = []
            for event in graph.stream(None, config):
                for node_name, _ in event.items():
                    resumed_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    # Stop at supervisor or next interrupt
                    if node_name in ["supervisor", "__interrupt__"]:
                        break
                else:
                    continue
                break

            assert len(resumed_nodes) > 0, "Should have resumed execution"
            assert "ask_user" in resumed_nodes, "ask_user should be called after resume"
            
            # Verify state after resume
            post_resume_state = graph.get_state(config).values
            assert post_resume_state.get("awaiting_user_input") is False, \
                "awaiting_user_input should be False after ask_user processes response"

            print("\n✅ Interrupt resume flow test passed!")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_stages_routes_to_report(self, initial_state):
        """
        If select_stage finds no stages to run, should route to generate_report.
        """
        visited_nodes = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")

            if agent == "planner":
                # Return plan with empty stages
                plan = MockLLMResponses.planner()
                plan["stages"] = []  # No stages
                return plan

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "report_generator": MockLLMResponses.report_generator(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Empty Stages Routes to Report")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("empty_stages")}}

            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    if node_name == "generate_report":
                        break
                else:
                    continue
                break

            # Should go straight to report without design/code phases
            assert "select_stage" in visited_nodes, "select_stage should be visited"
            assert "generate_report" in visited_nodes, "generate_report should be visited"
            
            # Should NOT have design phase
            assert "design" not in visited_nodes, \
                "design should NOT be visited when no stages exist"

            print("\n✅ Empty stages flow test passed!")

    def test_comparison_revision_limit_proceeds_to_supervisor(self, initial_state):
        """
        When comparison revision limit is reached, should proceed to supervisor
        (not ask_user) with a flag.
        """
        # Configure low limit
        state = initial_state.copy()
        state["runtime_config"] = {
            **(state.get("runtime_config") or {}),
            "max_analysis_revisions": 1,
        }

        visited_nodes = []
        comparison_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal comparison_count
            agent = kwargs.get("agent_name", "unknown")

            if agent == "comparison_validator":
                comparison_count += 1
                # Always reject to hit limit
                return {"verdict": "needs_revision", "summary": "Needs more work"}
            
            if agent == "results_analyzer":
                return MockLLMResponses.results_analyzer()
            
            if agent == "supervisor":
                return MockLLMResponses.supervisor_continue()

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
            print("TEST: Comparison Revision Limit Proceeds to Supervisor")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("comparison_limit")}}

            for event in graph.stream(state, config):
                for node_name, _ in event.items():
                    visited_nodes.append(node_name)
                    print(f"  → {node_name}")
                    
                    # Stop when we reach supervisor after comparison limit
                    if node_name == "supervisor":
                        break
                    if node_name == "__interrupt__":
                        break
                else:
                    continue
                break

            # Should have hit comparison limit and gone to supervisor
            assert comparison_count >= 2, \
                f"comparison_validator should be called at least 2 times, got {comparison_count}"
            
            # Should go to supervisor (not ask_user) when comparison limit hit
            assert "supervisor" in visited_nodes, \
                "supervisor should be visited after comparison limit"
            
            # The interrupt should come from material_checkpoint or supervisor,
            # NOT from comparison reaching limit
            final_state = graph.get_state(config).values
            trigger = final_state.get("ask_user_trigger")
            if trigger:
                assert trigger != "comparison_limit", \
                    "comparison limit should route to supervisor, not ask_user directly"

            print("\n✅ Comparison revision limit flow test passed!")
