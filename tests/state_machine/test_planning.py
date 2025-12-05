"""Planning and stage-selection E2E tests.

Tests cover:
- adapt_prompts_node: Prompt customization for paper-specific needs
- plan_node: Reproduction plan creation with validation
- plan_reviewer_node: Plan review with structural validation
- select_stage_node: Stage selection based on dependencies and validation hierarchy

These tests verify the complete planning phase flow, including edge cases,
error handling, and counter management.

IMPORTANT NOTE ON STATE CAPTURE:
When using graph.stream(), the state from graph.get_state() represents the state
BEFORE the current event's updates are applied. To get state AFTER a node runs:
1. Wait for the NEXT event and then call get_state(), OR
2. Merge the updates dict from the event into the current state, OR
3. Consume all events and check final state
"""

import pytest
from unittest.mock import patch

from langgraph.types import Command

from src.graph import create_repro_graph
from schemas.state import create_initial_state, MAX_REPLANS

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


def stream_until_node(graph, initial_state, config, target_node: str, break_on_target: bool = False):
    """Stream graph until reaching target node.
    
    Args:
        graph: LangGraph graph instance
        initial_state: Initial state dict
        config: Config dict with thread_id
        target_node: Node name to stop at
        break_on_target: If True, break when reaching target. If False, break on next node.
    
    Returns:
        Tuple of (final_state_dict, nodes_visited_list, updates_from_target_node)
    
    Note: State from get_state() is the state BEFORE the current event's updates.
    To get state after target_node, set break_on_target=False which waits for next node.
    """
    nodes_visited = []
    target_updates = None
    
    for event in graph.stream(initial_state, config):
        for node_name, updates in event.items():
            nodes_visited.append(node_name)
            
            # If we already saw target, current state includes target's updates
            if len(nodes_visited) > 1 and nodes_visited[-2] == target_node:
                state = graph.get_state(config)
                return state.values, nodes_visited, target_updates
            
            if node_name == target_node:
                target_updates = updates
                if break_on_target:
                    state = graph.get_state(config)
                    return state.values, nodes_visited, target_updates
    
    # End of stream - get final state
    state = graph.get_state(config)
    return state.values, nodes_visited, target_updates


class TestPlanningPhase:
    """Test planning agent flow."""

    def test_planning_approve_flow(self, initial_state):
        """Test: adapt_prompts → plan → plan_review(approve) → select_stage
        
        Verifies:
        - All expected nodes are visited in order
        - workflow_phase transitions correctly
        - plan structure is complete with all required fields
        - progress is initialized from plan
        - last_plan_review_verdict is 'approve'
        - paper_domain is preserved from prompt_adaptor
        - prompt_adaptations are stored (even if empty)
        - current_stage_id and current_stage_type are set
        """
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
            
            # Stream until after select_stage (wait for design node to get state after select_stage)
            final_state, nodes_visited, select_updates = stream_until_node(
                graph, initial_state, config, "select_stage", break_on_target=False
            )
            
            for node in nodes_visited:
                print(f"  → {node}")

            # Verify graph execution
            assert final_state is not None, "Graph should complete"
            
            # Verify node visitation order
            assert "adapt_prompts" in nodes_visited, "adapt_prompts node should be visited"
            assert "planning" in nodes_visited, "planning node should be visited"
            assert "plan_review" in nodes_visited, "plan_review node should be visited"
            assert "select_stage" in nodes_visited, "select_stage node should be visited"
            
            # Verify LLM agents were called
            assert "prompt_adaptor" in visited, "prompt_adaptor agent should be called"
            assert "planner" in visited, "planner agent should be called"
            assert "plan_reviewer" in visited, "plan_reviewer agent should be called"
            
            # Verify plan structure (the core deliverable of planning phase)
            assert "plan" in final_state, "State should contain plan"
            plan = final_state["plan"]
            assert isinstance(plan, dict), "Plan should be a dictionary"
            
            # Verify required plan fields
            assert "paper_id" in plan, "Plan should have paper_id"
            assert "stages" in plan, "Plan should have stages"
            assert len(plan["stages"]) == 1, "Plan should have exactly 1 stage"
            
            # Verify stage structure
            stage = plan["stages"][0]
            assert "stage_id" in stage, "Stage should have stage_id"
            assert stage["stage_id"] == "stage_0_materials", "First stage should be materials validation"
            assert "stage_type" in stage, "Stage should have stage_type"
            assert stage["stage_type"] == "MATERIAL_VALIDATION", "First stage should be MATERIAL_VALIDATION type"
            assert "targets" in stage, "Stage should have targets"
            assert len(stage["targets"]) > 0, "Stage should have at least one target"
            
            # Verify verdict
            assert final_state["last_plan_review_verdict"] == "approve", \
                "Verdict should be 'approve' for approval flow"
            
            # Verify progress initialization
            assert "progress" in final_state, "State should have progress"
            progress = final_state["progress"]
            assert progress is not None, "Progress should not be None"
            assert "stages" in progress, "Progress should have stages"
            assert len(progress["stages"]) == 1, "Progress should have same number of stages as plan"
            
            # Verify progress stage matches plan stage
            progress_stage = progress["stages"][0]
            assert progress_stage["stage_id"] == "stage_0_materials", \
                "Progress stage_id should match plan stage_id"
            assert progress_stage["status"] == "not_started", \
                "Progress stage status should be 'not_started'"
            
            # Verify current stage selection (state after select_stage runs)
            assert final_state.get("current_stage_id") == "stage_0_materials", \
                "current_stage_id should be set to first stage"
            assert final_state.get("current_stage_type") == "MATERIAL_VALIDATION", \
                "current_stage_type should match stage type"
            
            # Verify paper_domain is preserved
            assert final_state.get("paper_domain") == "plasmonics", \
                "paper_domain should be 'plasmonics' from prompt_adaptor"
            
            # Verify prompt_adaptations is set (even if empty list)
            assert "prompt_adaptations" in final_state, \
                "prompt_adaptations should be in state"
            assert isinstance(final_state["prompt_adaptations"], list), \
                "prompt_adaptations should be a list"

            print("\n✅ Planning phase test passed!")

    def test_planning_revision_flow(self, initial_state):
        """Test: plan_review rejects → routes back to plan → re-review → approve
        
        Verifies:
        - Plan revision flow is triggered on rejection
        - replan_count is incremented on rejection
        - planner_feedback is set on rejection
        - Plan node is visited twice (initial + revision)
        - Plan review node is visited twice
        - Flow completes after approval
        """
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
            rejection_state_captured = None
            
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    # Capture state AFTER first plan_review (when we see the next node)
                    if len(nodes_visited) > 1 and nodes_visited.count("plan_review") == 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and rejection_state_captured is None:
                            rejection_state_captured = graph.get_state(config).values.copy()
                            
                            # Verify rejection state has feedback
                            assert rejection_state_captured.get("last_plan_review_verdict") == "needs_revision", \
                                f"First review should reject with 'needs_revision', got {rejection_state_captured.get('last_plan_review_verdict')}"
                            assert rejection_state_captured.get("planner_feedback") is not None, \
                                "Rejected plan should have planner_feedback"
                            assert len(rejection_state_captured.get("planner_feedback", "")) > 0, \
                                "planner_feedback should not be empty"
                            print(f"    [Captured rejection state - feedback: {rejection_state_captured.get('planner_feedback')[:50]}...]")

                    if node_name == "select_stage":
                        break
                else:
                    continue
                break
            
            # Get final state after select_stage
            final_state = graph.get_state(config).values

            # Verify revision flow occurred
            assert nodes_visited.count("planning") == 2, \
                f"Planning node should be visited twice (initial + revision), got {nodes_visited.count('planning')}"
            assert nodes_visited.count("plan_review") == 2, \
                f"Plan review node should be visited twice, got {nodes_visited.count('plan_review')}"
            
            # Verify planner was called twice
            planner_calls = sum(1 for v in visited if v == "planner")
            assert planner_calls == 2, f"Planner should be called twice, was called {planner_calls} times"
            
            # Verify final state is properly approved
            assert final_state.get("last_plan_review_verdict") == "approve", \
                "Final verdict should be 'approve'"
            
            # Verify replan_count was incremented
            assert final_state.get("replan_count", 0) >= 1, \
                f"replan_count should be >= 1 after revision, got {final_state.get('replan_count', 0)}"

            print("\n✅ Revision flow test passed!")

    def test_planning_max_replan_limit_triggers_ask_user(self, initial_state):
        """Test: Max replans exceeded → escalates to ask_user
        
        Verifies:
        - After MAX_REPLANS rejections, routes to ask_user
        - Graph interrupts before ask_user node
        - State indicates awaiting user input
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}")

            if agent == "plan_reviewer":
                # Always reject to trigger max replan limit
                print("    [Rejecting plan]")
                return MockLLMResponses.plan_reviewer_reject()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Planning Phase (max replan limit)")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("planning_limit")}}

            print("\n--- Running graph ---")
            nodes_visited = []

            # Stream until interrupt (ask_user is interrupt_before)
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")

            # Graph should interrupt before ask_user
            state = graph.get_state(config)
            final_state = state.values
            
            # Verify we hit the replan limit
            # MAX_REPLANS is 2, so we should see: initial planning + 2 replans = 3 planning visits
            # But routing checks count BEFORE incrementing, so we get MAX_REPLANS + 1 iterations
            plan_count = nodes_visited.count("planning")
            review_count = nodes_visited.count("plan_review")
            
            print(f"\n  planning visits: {plan_count}, plan_review visits: {review_count}")
            print(f"  replan_count: {final_state.get('replan_count')}")
            
            assert plan_count >= MAX_REPLANS, \
                f"Should have at least {MAX_REPLANS} planning visits, got {plan_count}"
            
            # Verify replan_count reached limit
            assert final_state.get("replan_count", 0) >= MAX_REPLANS, \
                f"replan_count should be >= {MAX_REPLANS}"
            
            # Verify graph is waiting for user input (ask_user has interrupt_before)
            # The next node should be ask_user
            next_nodes = state.next
            assert "ask_user" in next_nodes, \
                f"Next node should be 'ask_user', got {next_nodes}"

            print("\n✅ Max replan limit test passed!")

    def test_interrupt_resume_routes_through_supervisor(self, initial_state):
        """Test: Resuming from interrupt properly routes through ask_user → supervisor.
        
        This is a CRITICAL test that verifies the interrupt-resume flow works correctly.
        
        With the interrupt() pattern (not interrupt_before):
        - ask_user node runs and calls interrupt() to pause
        - We resume with Command(resume=user_response)
        - The interrupt() call returns the user_response
        - ask_user completes and routes to supervisor
        
        Verifies:
        - After interrupt inside ask_user, resume with Command(resume=...)
        - Flow continues: ask_user completes → supervisor
        - Supervisor processes the user response correctly
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}")

            if agent == "plan_reviewer":
                # Always reject to trigger max replan limit
                return MockLLMResponses.plan_reviewer_reject()

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), \
             MultiPatch(CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"):
            
            print("\n" + "=" * 60)
            print("TEST: Interrupt resume routes through supervisor")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("interrupt_resume")}}

            print("\n--- Phase 1: Run until interrupt ---")
            nodes_before_interrupt = []

            # Stream until interrupt (ask_user calls interrupt() internally)
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_before_interrupt.append(node_name)
                    print(f"  → {node_name}")

            # Verify we're paused with an interrupt payload
            snapshot = graph.get_state(config)
            interrupt_payload = None
            if snapshot.tasks:
                for task in snapshot.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        interrupt_payload = task.interrupts[0].value
                        break
            
            assert interrupt_payload is not None, \
                f"Should have interrupt payload, got snapshot.next={snapshot.next}"
            
            print(f"\n  Interrupt payload: {interrupt_payload}")
            print(f"  Trigger: {interrupt_payload.get('trigger')}")

            print("\n--- Phase 2: Resume with user response ---")
            
            # Resume with Command(resume=...) - this passes the value to interrupt()
            nodes_after_resume = []
            for event in graph.stream(Command(resume="APPROVE_PLAN"), config):
                for node_name, _ in event.items():
                    nodes_after_resume.append(node_name)
                    print(f"  → {node_name}")
                    
                    # Stop after supervisor to avoid running the whole graph
                    if node_name == "supervisor":
                        break
                # Break outer loop too
                if "supervisor" in nodes_after_resume:
                    break

            print(f"\n  Nodes after resume: {nodes_after_resume}")
            
            # CRITICAL ASSERTIONS:
            # 1. ask_user should complete (it was mid-interrupt, now resumes)
            assert "ask_user" in nodes_after_resume, \
                f"ask_user should complete after resume, got {nodes_after_resume}"
            
            # 2. supervisor should run (ask_user routes to supervisor)
            assert "supervisor" in nodes_after_resume, \
                f"supervisor should run after ask_user, got {nodes_after_resume}"
            
            # 3. adapt_prompts should NOT run (that would mean restart from START)
            assert "adapt_prompts" not in nodes_after_resume, \
                f"adapt_prompts should NOT run - graph should NOT restart! Got {nodes_after_resume}"
            
            # 4. planning should NOT run (supervisor should handle APPROVE_PLAN)
            assert "planning" not in nodes_after_resume, \
                f"planning should NOT run after APPROVE_PLAN, got {nodes_after_resume}"
            
            # Get final state and verify supervisor processed the response
            final_state = graph.get_state(config)
            
            # Supervisor should set verdict to ok_continue for APPROVE_PLAN
            supervisor_verdict = final_state.values.get("supervisor_verdict")
            print(f"  supervisor_verdict: {supervisor_verdict}")
            
            assert supervisor_verdict == "ok_continue", \
                f"APPROVE_PLAN should set supervisor_verdict='ok_continue', got '{supervisor_verdict}'"

            print("\n✅ Interrupt resume test passed!")

    def test_planning_with_empty_paper_text_escalates(self, initial_state):
        """Test: Missing/empty paper_text triggers user escalation.
        
        Verifies:
        - Plan node detects missing paper_text
        - Sets ask_user_trigger
        - Sets pending_user_questions with error message
        - Sets awaiting_user_input to True
        """
        # Create state with empty paper_text
        state_with_empty_text = {**initial_state, "paper_text": ""}

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            print(f"    [LLM] {agent}")
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Planning with empty paper_text")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("empty_text")}}

            print("\n--- Running graph ---")
            nodes_visited = []

            for event in graph.stream(state_with_empty_text, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")

            # Get state after interrupt
            state = graph.get_state(config)
            final_state = state.values
            
            # Verify planning node detected the issue
            assert "planning" in nodes_visited, "planning node should be visited"
            
            # Verify error handling
            assert final_state.get("ask_user_trigger") == "missing_paper_text", \
                "ask_user_trigger should be 'missing_paper_text'"
            assert final_state.get("awaiting_user_input") is True, \
                "awaiting_user_input should be True"
            assert len(final_state.get("pending_user_questions", [])) > 0, \
                "pending_user_questions should contain error message"
            
            # Verify error message content
            error_msg = final_state["pending_user_questions"][0]
            assert "paper text" in error_msg.lower() or "paper_text" in error_msg.lower(), \
                "Error message should mention paper text"

            print("\n✅ Empty paper_text test passed!")


class TestPlanReviewerStructuralValidation:
    """Test plan_reviewer_node structural validation logic.
    
    The plan_reviewer has complex validation that detects:
    - Plans with no stages
    - Stages missing required fields
    - Duplicate stage IDs
    - Missing dependencies
    - Circular dependencies
    - Self-dependencies
    
    NOTE: These structural validations happen BEFORE the LLM is called.
    The plan_reviewer returns needs_revision with blocking issues if structure is invalid.
    """

    def test_plan_with_no_stages_rejected(self, initial_state):
        """Test: Plan with empty stages list is rejected.
        
        Verifies:
        - Plan reviewer detects empty stages
        - Verdict is 'needs_revision'
        - Feedback mentions missing stages
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}")

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                # Return plan with no stages
                plan = MockLLMResponses.planner()
                plan["stages"] = []  # Empty stages
                return plan
            if agent == "plan_reviewer":
                # Should not be called with LLM since structural validation fails first
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with no stages")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("no_stages")}}

            print("\n--- Running graph ---")
            nodes_visited = []
            review_state = None

            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    # Capture state AFTER plan_review by waiting for next node
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            # Verify structural rejection
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Empty stages plan should be rejected, got {review_state.get('last_plan_review_verdict')}"
                            
                            feedback = review_state.get("planner_feedback", "")
                            assert "stage" in feedback.lower(), \
                                f"Feedback should mention stages issue: {feedback}"
                            print(f"    [Feedback: {feedback[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ No stages rejection test passed!")

    def test_plan_with_missing_stage_id_rejected(self, initial_state):
        """Test: Stage without stage_id is rejected.
        
        Verifies:
        - Plan reviewer detects missing stage_id
        - Verdict is 'needs_revision'
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                plan = MockLLMResponses.planner()
                # Remove stage_id from first stage
                del plan["stages"][0]["stage_id"]
                return plan
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with missing stage_id")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("no_stage_id")}}

            nodes_visited = []
            review_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Missing stage_id should cause rejection, got {review_state.get('last_plan_review_verdict')}"
                            print(f"    [Feedback: {review_state.get('planner_feedback', '')[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ Missing stage_id rejection test passed!")

    def test_plan_with_duplicate_stage_ids_rejected(self, initial_state):
        """Test: Duplicate stage IDs are rejected.
        
        Verifies:
        - Plan reviewer detects duplicate stage_id
        - Verdict is 'needs_revision'
        - Feedback mentions duplicate
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                plan = MockLLMResponses.planner()
                # Add duplicate stage with same ID
                plan["stages"].append({
                    "stage_id": "stage_0_materials",  # Duplicate!
                    "stage_type": "SINGLE_STRUCTURE",
                    "name": "Duplicate Stage",
                    "description": "This has duplicate ID",
                    "targets": ["Fig2"],
                    "dependencies": [],
                })
                return plan
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with duplicate stage IDs")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("dup_stage_id")}}

            nodes_visited = []
            review_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Duplicate stage_id should cause rejection, got {review_state.get('last_plan_review_verdict')}"
                            
                            feedback = review_state.get("planner_feedback", "")
                            assert "duplicate" in feedback.lower() or "stage_0_materials" in feedback, \
                                f"Feedback should mention duplicate: {feedback}"
                            print(f"    [Feedback: {feedback[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ Duplicate stage_id rejection test passed!")

    def test_plan_with_missing_targets_rejected(self, initial_state):
        """Test: Stage without targets is rejected.
        
        Verifies:
        - Plan reviewer detects missing targets
        - Verdict is 'needs_revision'
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                plan = MockLLMResponses.planner()
                # Remove targets from stage
                plan["stages"][0]["targets"] = []
                # Also clear target_details if present
                if "target_details" in plan["stages"][0]:
                    plan["stages"][0]["target_details"] = []
                return plan
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with missing targets")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("no_targets")}}

            nodes_visited = []
            review_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Missing targets should cause rejection, got {review_state.get('last_plan_review_verdict')}"
                            print(f"    [Feedback: {review_state.get('planner_feedback', '')[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ Missing targets rejection test passed!")

    def test_plan_with_circular_dependency_rejected(self, initial_state):
        """Test: Circular dependencies are rejected.
        
        Verifies:
        - Plan reviewer detects circular dependency (A → B → A)
        - Verdict is 'needs_revision'
        - Feedback mentions circular dependency
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                # Create plan with circular dependency
                return {
                    "paper_id": "test_circular",
                    "paper_domain": "plasmonics",
                    "title": "Test",
                    "summary": "Test",
                    "stages": [
                        {
                            "stage_id": "stage_A",
                            "stage_type": "MATERIAL_VALIDATION",
                            "name": "Stage A",
                            "description": "First stage",
                            "targets": ["Fig1"],
                            "dependencies": ["stage_B"],  # A depends on B
                        },
                        {
                            "stage_id": "stage_B",
                            "stage_type": "SINGLE_STRUCTURE",
                            "name": "Stage B",
                            "description": "Second stage",
                            "targets": ["Fig2"],
                            "dependencies": ["stage_A"],  # B depends on A - CIRCULAR!
                        },
                    ],
                    "targets": [],
                    "extracted_parameters": [],
                }
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with circular dependency")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("circular_dep")}}

            nodes_visited = []
            review_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Circular dependency should cause rejection, got {review_state.get('last_plan_review_verdict')}"
                            
                            feedback = review_state.get("planner_feedback", "")
                            assert "circular" in feedback.lower(), \
                                f"Feedback should mention circular dependency: {feedback}"
                            print(f"    [Feedback: {feedback[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ Circular dependency rejection test passed!")

    def test_plan_with_self_dependency_rejected(self, initial_state):
        """Test: Self-dependency (stage depends on itself) is rejected.
        
        Verifies:
        - Plan reviewer detects self-dependency
        - Verdict is 'needs_revision'
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                plan = MockLLMResponses.planner()
                # Make stage depend on itself
                plan["stages"][0]["dependencies"] = ["stage_0_materials"]
                return plan
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with self-dependency")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("self_dep")}}

            nodes_visited = []
            review_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Self-dependency should cause rejection, got {review_state.get('last_plan_review_verdict')}"
                            print(f"    [Feedback: {review_state.get('planner_feedback', '')[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ Self-dependency rejection test passed!")

    def test_plan_with_missing_dependency_rejected(self, initial_state):
        """Test: Dependency on non-existent stage is rejected.
        
        Verifies:
        - Plan reviewer detects missing dependency reference
        - Verdict is 'needs_revision'
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                plan = MockLLMResponses.planner()
                # Reference non-existent stage
                plan["stages"][0]["dependencies"] = ["nonexistent_stage"]
                return plan
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Plan with missing dependency")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("missing_dep")}}

            nodes_visited = []
            review_state = None
            
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}")
                    
                    if len(nodes_visited) > 1:
                        prev_node = nodes_visited[-2]
                        if prev_node == "plan_review" and review_state is None:
                            review_state = graph.get_state(config).values.copy()
                            
                            assert review_state.get("last_plan_review_verdict") == "needs_revision", \
                                f"Missing dependency should cause rejection, got {review_state.get('last_plan_review_verdict')}"
                            print(f"    [Feedback: {review_state.get('planner_feedback', '')[:100]}...]")
                            break
                else:
                    continue
                break

            assert review_state is not None, "Should have captured review state"
            print("\n✅ Missing dependency rejection test passed!")


class TestStageSelection:
    """Test planning through stage selection."""

    def test_stage_selection_picks_first_stage(self, initial_state):
        """After plan approval, select_stage should pick the first available stage.
        
        Verifies:
        - current_stage_id is set to first stage
        - current_stage_type matches stage type
        - progress stages are properly initialized
        - Counters are reset (design_revision_count, code_revision_count, etc.)
        """

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
            
            # Stream until we have state after select_stage
            final_state, nodes_visited, _ = stream_until_node(
                graph, initial_state, config, "select_stage", break_on_target=False
            )
            
            for node in nodes_visited:
                print(f"  → {node}")
            
            print(f"\n  Selected stage: {final_state.get('current_stage_id')}")
            print(f"  Stage type: {final_state.get('current_stage_type')}")

            # Verify stage selection
            assert final_state is not None, "Should complete stream"
            assert final_state.get("current_stage_id") == "stage_0_materials", \
                f"current_stage_id should be first stage, got {final_state.get('current_stage_id')}"
            assert final_state.get("current_stage_type") == "MATERIAL_VALIDATION", \
                f"current_stage_type should match, got {final_state.get('current_stage_type')}"

            # Verify progress structure
            progress = final_state.get("progress", {})
            stages = progress.get("stages", [])
            assert len(stages) == 1, f"Progress should have 1 stage, got {len(stages)}"
            assert stages[0]["stage_id"] == "stage_0_materials"
            assert stages[0]["status"] == "not_started"

            # Verify counters are reset/initialized
            assert final_state.get("design_revision_count", 0) == 0, \
                "design_revision_count should be 0 for new stage"
            assert final_state.get("code_revision_count", 0) == 0, \
                "code_revision_count should be 0 for new stage"
            assert final_state.get("execution_failure_count", 0) == 0, \
                "execution_failure_count should be 0 for new stage"

            print("\n✅ Stage selection test passed!")

    def test_stage_selection_with_dependencies(self, initial_state):
        """Test: Stage with unsatisfied dependencies is not selected.
        
        Verifies:
        - First stage (no dependencies) is selected
        - Second stage (with dependency) is NOT selected before first completes
        """

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")

            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                # Create plan with dependencies
                return {
                    "paper_id": "test_deps",
                    "paper_domain": "plasmonics",
                    "title": "Test Dependencies",
                    "summary": "Test",
                    "stages": [
                        {
                            "stage_id": "stage_0_materials",
                            "stage_type": "MATERIAL_VALIDATION",
                            "name": "Materials",
                            "description": "Validate materials",
                            "targets": ["Fig1"],
                            "dependencies": [],
                        },
                        {
                            "stage_id": "stage_1_structure",
                            "stage_type": "SINGLE_STRUCTURE",
                            "name": "Single Structure",
                            "description": "Run single structure sim",
                            "targets": ["Fig2"],
                            "dependencies": ["stage_0_materials"],  # Depends on stage 0
                        },
                    ],
                    "targets": [],
                    "extracted_parameters": [],
                }
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Stage Selection with Dependencies")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("stage_deps")}}

            # Stream until we have state after select_stage
            final_state, nodes_visited, _ = stream_until_node(
                graph, initial_state, config, "select_stage", break_on_target=False
            )
            
            for node in nodes_visited:
                print(f"  → {node}")

            # First selection should pick stage_0 (no dependencies)
            assert final_state is not None
            assert final_state.get("current_stage_id") == "stage_0_materials", \
                f"Should select stage with no dependencies first, got {final_state.get('current_stage_id')}"
            
            # Stage 1 should still be not_started (dependency not satisfied)
            progress = final_state.get("progress", {})
            stages = progress.get("stages", [])
            stage_1 = next((s for s in stages if s["stage_id"] == "stage_1_structure"), None)
            assert stage_1 is not None, "Stage 1 should exist in progress"
            assert stage_1["status"] == "not_started", \
                f"Stage 1 should be not_started (dependency not complete), got {stage_1['status']}"

            print("\n✅ Stage selection with dependencies test passed!")

    def test_stage_selection_no_stages_escalates(self, initial_state):
        """Test: Empty plan escalates to ask_user.
        
        Verifies:
        - When no stages exist, select_stage sets error fields
        - ask_user_trigger is set
        - awaiting_user_input is True
        """
        # Create state with plan but no stages in progress
        state_with_empty_progress = {
            **initial_state,
            "plan": {"stages": []},
            "progress": {"stages": []},
            "last_plan_review_verdict": "approve",  # Bypass plan review
        }

        # Directly test select_stage_node
        from src.agents.stage_selection import select_stage_node
        
        result = select_stage_node(state_with_empty_progress)
        
        print("\n" + "=" * 60)
        print("TEST: Stage Selection with no stages")
        print("=" * 60)
        print(f"  Result: {result}")
        
        # Verify error handling
        assert result.get("current_stage_id") is None, \
            "current_stage_id should be None when no stages"
        assert result.get("ask_user_trigger") == "no_stages_available", \
            f"ask_user_trigger should be 'no_stages_available', got {result.get('ask_user_trigger')}"
        assert result.get("awaiting_user_input") is True, \
            "awaiting_user_input should be True"
        assert len(result.get("pending_user_questions", [])) > 0, \
            "Should have pending user questions"

        print("\n✅ No stages escalation test passed!")

    def test_stage_selection_missing_stage_type_blocks_stage(self, initial_state):
        """Test: Stage without stage_type is blocked.
        
        Verifies:
        - Stage missing stage_type is marked as blocked
        - select_stage_node handles this gracefully
        """
        # Create state with stage missing stage_type
        state_with_bad_stage = {
            **initial_state,
            "plan": {
                "stages": [
                    {
                        "stage_id": "bad_stage",
                        # Missing stage_type!
                        "name": "Bad Stage",
                        "description": "Missing stage_type",
                        "targets": ["Fig1"],
                        "dependencies": [],
                    }
                ]
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "bad_stage",
                        "status": "not_started",
                        "summary": "",
                    }
                ]
            },
            "last_plan_review_verdict": "approve",
        }

        from src.agents.stage_selection import select_stage_node
        
        result = select_stage_node(state_with_bad_stage)
        
        print("\n" + "=" * 60)
        print("TEST: Stage Selection with missing stage_type")
        print("=" * 60)
        print(f"  Result: {result}")
        
        # Stage should not be selected (no valid stage_type)
        # It should either be blocked or return no stage
        assert result.get("current_stage_id") is None or \
               result.get("ask_user_trigger") is not None, \
            "Stage without stage_type should not be selectable"

        print("\n✅ Missing stage_type handling test passed!")


class TestAdaptPromptsNode:
    """Test adapt_prompts_node behavior."""

    def test_adapt_prompts_stores_adaptations(self, initial_state):
        """Test: adapt_prompts stores adaptations in state.
        
        Verifies:
        - prompt_adaptations is set in state
        - paper_domain can be updated by prompt_adaptor
        """

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            
            if agent == "prompt_adaptor":
                return {
                    "adaptations": [
                        {"type": "domain_specific", "content": "Focus on plasmonic resonance"},
                        {"type": "method_specific", "content": "Use FDTD for near-field"},
                    ],
                    "paper_domain": "plasmonics",
                }
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Adapt Prompts stores adaptations")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("adapt_prompts")}}

            # Stream until after adapt_prompts (wait for plan node)
            adapt_prompts_state, nodes_visited, _ = stream_until_node(
                graph, initial_state, config, "adapt_prompts", break_on_target=False
            )
            
            for node in nodes_visited:
                print(f"  → {node}")

            # Verify adaptations were stored
            assert adapt_prompts_state is not None
            assert "prompt_adaptations" in adapt_prompts_state, \
                "prompt_adaptations should be in state"
            
            adaptations = adapt_prompts_state["prompt_adaptations"]
            assert isinstance(adaptations, list), "adaptations should be a list"
            assert len(adaptations) == 2, f"Should have 2 adaptations, got {len(adaptations)}"
            
            # Verify paper_domain was updated
            assert adapt_prompts_state.get("paper_domain") == "plasmonics", \
                f"paper_domain should be updated to 'plasmonics', got {adapt_prompts_state.get('paper_domain')}"

            print("\n✅ Adapt prompts test passed!")

    def test_adapt_prompts_handles_llm_failure(self, initial_state):
        """Test: LLM failure in adapt_prompts is handled gracefully.
        
        Verifies:
        - Exception is caught
        - Default empty adaptations are used
        - Flow continues to plan node
        """

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            
            if agent == "prompt_adaptor":
                raise Exception("LLM API error")
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Adapt Prompts handles LLM failure")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("adapt_fail")}}

            # Stream until after adapt_prompts
            state_after_adapt, nodes_visited, _ = stream_until_node(
                graph, initial_state, config, "adapt_prompts", break_on_target=False
            )
            
            for node in nodes_visited:
                print(f"  → {node}")
            
            # Should have empty adaptations (fallback)
            assert state_after_adapt.get("prompt_adaptations") == [], \
                f"Should have empty adaptations on LLM failure, got {state_after_adapt.get('prompt_adaptations')}"

            # Should continue to planning despite failure
            assert "adapt_prompts" in nodes_visited
            assert "planning" in nodes_visited

            print("\n✅ LLM failure handling test passed!")


class TestVerdictNormalization:
    """Test verdict normalization in plan_reviewer_node."""

    def test_plan_reviewer_normalizes_pass_to_approve(self, initial_state):
        """Test: 'pass' verdict is normalized to 'approve'.
        
        Verifies:
        - LLM returning 'pass' is normalized to 'approve'
        - Flow continues to select_stage
        """

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                # Return 'pass' instead of 'approve'
                response = MockLLMResponses.plan_reviewer_approve()
                response["verdict"] = "pass"  # Non-standard but should normalize
                return response
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Verdict normalization (pass → approve)")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("normalize_pass")}}

            # Stream until after select_stage
            final_state, nodes_visited, _ = stream_until_node(
                graph, initial_state, config, "select_stage", break_on_target=False
            )
            
            for node in nodes_visited:
                print(f"  → {node}")

            # Should normalize 'pass' to 'approve'
            assert "select_stage" in nodes_visited, "Should reach select_stage"
            assert final_state.get("last_plan_review_verdict") == "approve", \
                f"'pass' should be normalized to 'approve', got {final_state.get('last_plan_review_verdict')}"

            print("\n✅ Verdict normalization test passed!")

    def test_plan_reviewer_normalizes_reject_to_needs_revision(self, initial_state):
        """Test: 'reject' verdict is normalized to 'needs_revision'.
        
        Verifies:
        - LLM returning 'reject' is normalized to 'needs_revision'
        - Plan is sent back for revision
        """
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            
            if agent == "prompt_adaptor":
                return MockLLMResponses.prompt_adaptor()
            if agent == "planner":
                return MockLLMResponses.planner()
            if agent == "plan_reviewer":
                review_count = sum(1 for v in visited if v == "plan_reviewer")
                if review_count <= 1:
                    # Return 'reject' instead of 'needs_revision'
                    response = MockLLMResponses.plan_reviewer_reject()
                    response["verdict"] = "reject"  # Non-standard
                    return response
                return MockLLMResponses.plan_reviewer_approve()
            return {}

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ):
            print("\n" + "=" * 60)
            print("TEST: Verdict normalization (reject → needs_revision)")
            print("=" * 60)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("normalize_reject")}}

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

            # Should normalize 'reject' to 'needs_revision' and trigger replan
            assert nodes_visited.count("planning") == 2, \
                f"'reject' should trigger replan (normalized to 'needs_revision'), got {nodes_visited.count('planning')} planning visits"

            print("\n✅ Reject normalization test passed!")
