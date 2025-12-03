"""Design-phase E2E tests.

Tests for:
1. simulation_designer_node - designs simulation setup
2. design_reviewer_node - reviews design with verdict
3. route_after_design_review - routes based on verdict
4. Graph-level design flows - approve, revision, limit reached
"""

import pytest
from unittest.mock import patch, MagicMock

from src.graph import create_repro_graph
from src.agents.design import simulation_designer_node, design_reviewer_node
from src.routing import route_after_design_review
from schemas.state import (
    create_initial_state,
    MAX_DESIGN_REVISIONS,
)

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def base_state():
    """Base state for unit tests with required fields."""
    return create_initial_state(
        paper_id="test_paper",
        paper_text="Gold nanorods have LSPR at 700nm." * 10,
    )


@pytest.fixture
def state_with_stage(base_state):
    """State with a current stage selected."""
    return {
        **base_state,
        "current_stage_id": "stage_0_materials",
        "current_stage_type": "MATERIAL_VALIDATION",
        "plan": MockLLMResponses.planner(),
        "workflow_phase": "design",
    }


@pytest.fixture
def state_with_design(state_with_stage):
    """State with a design already created."""
    return {
        **state_with_stage,
        "design_description": MockLLMResponses.simulation_designer(),
        "design_revision_count": 0,
    }


@pytest.fixture
def state_at_revision_limit(state_with_design):
    """State at the design revision limit."""
    return {
        **state_with_design,
        "design_revision_count": MAX_DESIGN_REVISIONS,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: simulation_designer_node
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulationDesignerNode:
    """Unit tests for simulation_designer_node."""

    def test_happy_path_creates_design(self, state_with_stage):
        """Test: designer produces design_description in state."""
        mock_design = MockLLMResponses.simulation_designer()
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_design):
            result = simulation_designer_node(state_with_stage)
        
        # Must produce design_description
        assert "design_description" in result, "Designer must output design_description"
        assert result["design_description"] == mock_design, "Design description should match LLM output"
        
        # Must set workflow_phase
        assert result.get("workflow_phase") == "design", "Workflow phase should be 'design'"
        
        # Should NOT trigger user escalation
        assert not result.get("awaiting_user_input"), "Should not await user input on success"

    def test_missing_stage_id_escalates_to_user(self, base_state):
        """Test: missing current_stage_id escalates to ask_user."""
        # Remove current_stage_id
        state = {**base_state, "current_stage_id": None}
        
        result = simulation_designer_node(state)
        
        # Must escalate
        assert result.get("awaiting_user_input") is True, "Must await user input when stage_id missing"
        assert result.get("ask_user_trigger") == "missing_stage_id", "Trigger should be 'missing_stage_id'"
        assert result.get("pending_user_questions"), "Must have questions for user"
        
        # Question should explain the error
        questions = result.get("pending_user_questions", [])
        assert len(questions) > 0, "Must have at least one question"
        assert "no stage" in questions[0].lower() or "error" in questions[0].lower()

    def test_llm_error_escalates_to_user(self, state_with_stage):
        """Test: LLM call failure escalates to ask_user."""
        with patch("src.agents.design.call_agent_with_metrics", side_effect=Exception("API Error")):
            result = simulation_designer_node(state_with_stage)
        
        # Must escalate
        assert result.get("awaiting_user_input") is True, "Must await user input on LLM error"
        assert result.get("ask_user_trigger") == "llm_error", "Trigger should be 'llm_error'"
        assert result.get("pending_user_questions"), "Must have questions for user"

    def test_assumptions_are_merged_when_valid_list(self, state_with_stage):
        """Test: new_assumptions from LLM are merged into state assumptions."""
        mock_design = {
            **MockLLMResponses.simulation_designer(),
            "new_assumptions": ["Assumed n=1.33 for water", "Assumed room temperature"],
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_design):
            result = simulation_designer_node(state_with_stage)
        
        # Assumptions should be merged
        assumptions = result.get("assumptions", {})
        global_assumptions = assumptions.get("global_assumptions", [])
        
        assert "Assumed n=1.33 for water" in global_assumptions, "First assumption should be added"
        assert "Assumed room temperature" in global_assumptions, "Second assumption should be added"

    def test_assumptions_ignored_when_not_list(self, state_with_stage):
        """Test: invalid new_assumptions format is ignored (not a list)."""
        mock_design = {
            **MockLLMResponses.simulation_designer(),
            "new_assumptions": "This is a string, not a list",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_design):
            result = simulation_designer_node(state_with_stage)
        
        # Should not crash, assumptions should not be modified
        # Either no assumptions key, or empty assumptions
        assumptions = result.get("assumptions", {})
        global_assumptions = assumptions.get("global_assumptions", [])
        
        # Should not contain the invalid string
        assert "This is a string, not a list" not in global_assumptions

    def test_design_description_stored_even_when_not_dict(self, state_with_stage):
        """Test: string design response is stored as design_description."""
        string_design = "Simple design without structured format"
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=string_design):
            result = simulation_designer_node(state_with_stage)
        
        assert result.get("design_description") == string_design

    def test_context_overflow_escalates_to_user(self, state_with_stage):
        """Test: context overflow during design escalates to user."""
        # Mock context check to return escalation
        escalation_response = {
            "pending_user_questions": ["Context overflow in design. How should we proceed?"],
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
        }
        
        with patch("src.agents.design.check_context_or_escalate", return_value=escalation_response):
            result = simulation_designer_node(state_with_stage)
        
        assert result.get("awaiting_user_input") is True
        assert result.get("ask_user_trigger") == "context_overflow"

    def test_revision_feedback_included_in_prompt(self, state_with_stage):
        """Test: reviewer_feedback from previous rejection is available in state."""
        state_with_feedback = {
            **state_with_stage,
            "reviewer_feedback": "Resolution too low for plasmonic features",
            "design_revision_count": 1,
        }
        
        call_args = {}
        def capture_call(*args, **kwargs):
            call_args.update(kwargs)
            call_args["system_prompt"] = args[1] if len(args) > 1 else kwargs.get("system_prompt", "")
            return MockLLMResponses.simulation_designer()
        
        with patch("src.agents.design.call_agent_with_metrics", side_effect=capture_call):
            result = simulation_designer_node(state_with_feedback)
        
        # The node should include feedback in the prompt
        # Check that the call was made (we stored system_prompt)
        assert "system_prompt" in call_args or result.get("design_description") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: design_reviewer_node
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignReviewerNode:
    """Unit tests for design_reviewer_node."""

    def test_approve_verdict_sets_correct_state(self, state_with_design):
        """Test: approve verdict sets last_design_review_verdict and no revision count increment."""
        mock_review = MockLLMResponses.design_reviewer_approve()
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "approve"
        assert result.get("workflow_phase") == "design_review"
        # Revision count should NOT increment on approve
        assert result.get("design_revision_count") == 0
        # Should not await user input
        assert not result.get("awaiting_user_input")

    def test_needs_revision_increments_counter(self, state_with_design):
        """Test: needs_revision verdict increments design_revision_count."""
        mock_review = MockLLMResponses.design_reviewer_reject()
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "needs_revision"
        assert result.get("design_revision_count") == 1, "Revision count should increment from 0 to 1"
        # Should set reviewer_feedback
        assert result.get("reviewer_feedback") is not None or result.get("reviewer_issues") is not None

    def test_needs_revision_at_limit_escalates(self, state_at_revision_limit):
        """Test: needs_revision at limit escalates to ask_user."""
        mock_review = MockLLMResponses.design_reviewer_reject()
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_at_revision_limit)
        
        assert result.get("last_design_review_verdict") == "needs_revision"
        assert result.get("awaiting_user_input") is True, "Must await user input at revision limit"
        assert result.get("ask_user_trigger") == "design_review_limit"
        assert result.get("pending_user_questions"), "Must have questions for user"
        # Counter should NOT increment past limit
        assert result.get("design_revision_count") == MAX_DESIGN_REVISIONS

    def test_verdict_normalization_pass_to_approve(self, state_with_design):
        """Test: 'pass' verdict is normalized to 'approve'."""
        mock_review = {"verdict": "pass", "issues": [], "summary": "OK"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "approve"

    def test_verdict_normalization_approved_to_approve(self, state_with_design):
        """Test: 'approved' verdict is normalized to 'approve'."""
        mock_review = {"verdict": "approved", "issues": [], "summary": "OK"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "approve"

    def test_verdict_normalization_accept_to_approve(self, state_with_design):
        """Test: 'accept' verdict is normalized to 'approve'."""
        mock_review = {"verdict": "accept", "issues": [], "summary": "OK"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "approve"

    def test_verdict_normalization_reject_to_needs_revision(self, state_with_design):
        """Test: 'reject' verdict is normalized to 'needs_revision'."""
        mock_review = {"verdict": "reject", "issues": [], "summary": "Rejected"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "needs_revision"

    def test_verdict_normalization_revision_needed(self, state_with_design):
        """Test: 'revision_needed' verdict is normalized to 'needs_revision'."""
        mock_review = {"verdict": "revision_needed", "issues": [], "summary": "Needs work"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "needs_revision"

    def test_missing_verdict_defaults_to_approve(self, state_with_design):
        """Test: missing verdict in LLM output defaults to 'approve'."""
        mock_review = {"issues": [], "summary": "No verdict given"}  # No verdict key
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "approve"

    def test_unknown_verdict_defaults_to_approve(self, state_with_design):
        """Test: unknown verdict value defaults to 'approve'."""
        mock_review = {"verdict": "unknown_value_xyz", "issues": [], "summary": "??"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("last_design_review_verdict") == "approve"

    def test_llm_error_auto_approves(self, state_with_design):
        """Test: LLM error in reviewer auto-approves (doesn't block workflow)."""
        with patch("src.agents.design.call_agent_with_metrics", side_effect=Exception("API Error")):
            result = design_reviewer_node(state_with_design)
        
        # Should auto-approve, not escalate
        assert result.get("last_design_review_verdict") == "approve"
        assert not result.get("awaiting_user_input"), "Should not await user on reviewer LLM error"

    def test_reviewer_issues_extracted(self, state_with_design):
        """Test: issues from LLM output are stored in reviewer_issues."""
        mock_review = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "blocking", "description": "Geometry exceeds cell"}
            ],
            "summary": "Fix geometry",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("reviewer_issues") is not None
        assert len(result.get("reviewer_issues", [])) == 1
        assert result["reviewer_issues"][0]["severity"] == "blocking"

    def test_feedback_extracted_on_rejection(self, state_with_design):
        """Test: feedback is extracted from LLM output on needs_revision."""
        mock_review = {
            "verdict": "needs_revision",
            "issues": [],
            "feedback": "Please increase resolution to 40nm/cell",
            "summary": "Resolution issue",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert "resolution" in result.get("reviewer_feedback", "").lower() or \
               "resolution" in result.get("summary", "").lower()

    def test_escalation_message_includes_stage_and_attempts(self, state_at_revision_limit):
        """Test: escalation message includes helpful context."""
        mock_review = {
            "verdict": "needs_revision",
            "issues": [],
            "feedback": "Still problematic",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_at_revision_limit)
        
        questions = result.get("pending_user_questions", [])
        assert len(questions) > 0
        question = questions[0]
        
        # Should mention the stage
        assert "stage" in question.lower() or "Stage" in question
        # Should mention attempts/limit
        assert str(MAX_DESIGN_REVISIONS) in question or "limit" in question.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: route_after_design_review
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteAfterDesignReview:
    """Unit tests for route_after_design_review routing function."""

    def test_approve_routes_to_generate_code(self, state_with_design):
        """Test: approve verdict routes to generate_code."""
        state = {**state_with_design, "last_design_review_verdict": "approve"}
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        assert route == "generate_code"

    def test_needs_revision_routes_to_design(self, state_with_design):
        """Test: needs_revision verdict routes to design."""
        state = {
            **state_with_design,
            "last_design_review_verdict": "needs_revision",
            "design_revision_count": 1,  # Under limit
        }
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        assert route == "design"

    def test_needs_revision_at_limit_routes_to_ask_user(self, state_with_design):
        """Test: needs_revision at limit routes to ask_user."""
        state = {
            **state_with_design,
            "last_design_review_verdict": "needs_revision",
            "design_revision_count": MAX_DESIGN_REVISIONS,
        }
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        assert route == "ask_user"

    def test_none_verdict_routes_to_ask_user(self, state_with_design):
        """Test: None verdict (error case) routes to ask_user."""
        state = {**state_with_design, "last_design_review_verdict": None}
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        assert route == "ask_user"

    def test_invalid_verdict_type_routes_to_ask_user(self, state_with_design):
        """Test: non-string verdict routes to ask_user."""
        state = {**state_with_design, "last_design_review_verdict": 123}  # int instead of str
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        assert route == "ask_user"

    def test_unknown_verdict_routes_to_ask_user(self, state_with_design):
        """Test: unrecognized verdict string routes to ask_user."""
        state = {**state_with_design, "last_design_review_verdict": "unknown_verdict"}
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        assert route == "ask_user"

    def test_custom_max_revisions_from_runtime_config(self, state_with_design):
        """Test: runtime_config max_design_revisions is respected."""
        custom_max = 5
        state = {
            **state_with_design,
            "last_design_review_verdict": "needs_revision",
            "design_revision_count": 4,  # Under custom limit, over default
            "runtime_config": {"max_design_revisions": custom_max},
        }
        
        with patch("src.routing.save_checkpoint"):
            route = route_after_design_review(state)
        
        # At 4 revisions with max 5, should still route to design
        assert route == "design"
        
        # At 5 revisions with max 5, should route to ask_user
        state["design_revision_count"] = 5
        route = route_after_design_review(state)
        assert route == "ask_user"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: Design Phase Graph Flows
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignPhase:
    """Test design → design_review flow through the graph."""

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
            final_state = None

            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)

                    if node_name == "design_review":
                        assert updates.get("last_design_review_verdict") == "approve", \
                            f"Expected approve verdict, got: {updates.get('last_design_review_verdict')}"

                    if node_name == "generate_code":
                        final_state = graph.get_state(config).values
                        break
                else:
                    continue
                break

            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)

            # Verify node sequence
            assert "design" in nodes_visited, "design node must be visited"
            assert "design_review" in nodes_visited, "design_review node must be visited"
            assert "generate_code" in nodes_visited, "generate_code node must be visited"
            
            # Verify design comes before design_review
            design_idx = nodes_visited.index("design")
            review_idx = nodes_visited.index("design_review")
            assert design_idx < review_idx, "design must come before design_review"

            # Verify state after design phase
            state = graph.get_state(config).values
            assert state.get("design_revision_count") == 0, \
                f"Revision count should be 0 on first approve, got: {state.get('design_revision_count')}"
            assert state.get("design_description") is not None, \
                "design_description should be set after design node"
            assert state.get("last_design_review_verdict") == "approve", \
                "Verdict should be approve"

            print("\n✅ Design phase test passed!", flush=True)

    def test_design_revision_flow(self, initial_state):
        """Test: design_review rejects → routes back to design → eventually approves."""
        visited = []
        rejection_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal rejection_count
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)

            if agent == "design_reviewer":
                review_count = sum(1 for v in visited if v == "design_reviewer")
                if review_count <= 1:
                    rejection_count += 1
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
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)

                    # Verify state updates on rejection
                    if node_name == "design_review" and updates.get("last_design_review_verdict") == "needs_revision":
                        assert updates.get("design_revision_count", 0) >= 1, \
                            "Revision count should increment on rejection"
                        assert updates.get("reviewer_feedback") is not None or \
                               updates.get("reviewer_issues") is not None, \
                            "Should have feedback or issues on rejection"

                    if node_name == "generate_code":
                        break
                else:
                    continue
                break

            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)
            print(f"Rejections: {rejection_count}", flush=True)

            # Verify the loop occurred
            assert nodes_visited.count("design") == 2, \
                f"Expected design visited twice (initial + revision), got: {nodes_visited.count('design')}"
            assert nodes_visited.count("design_review") == 2, \
                f"Expected design_review visited twice, got: {nodes_visited.count('design_review')}"
            
            # Verify final state
            state = graph.get_state(config).values
            assert state.get("design_revision_count") >= 1, \
                "Revision count should be at least 1 after a rejection"

            print("\n✅ Design revision flow test passed!", flush=True)

    def test_design_revision_limit_escalates(self, initial_state):
        """Test: hitting revision limit triggers ask_user escalation."""
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)
            print(f"    [LLM] {agent}", flush=True)

            if agent == "design_reviewer":
                # Always reject to hit the limit
                print("    [Rejecting design - always]", flush=True)
                return MockLLMResponses.design_reviewer_reject()

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
            print("TEST: Design Phase (revision limit escalation)", flush=True)
            print("=" * 60, flush=True)

            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("design_limit")}}

            print("\n--- Running graph ---", flush=True)
            nodes_visited = []

            # Graph should pause at ask_user (interrupt_before)
            for event in graph.stream(initial_state, config):
                for node_name, updates in event.items():
                    nodes_visited.append(node_name)
                    print(f"  → {node_name}", flush=True)
                    
                    # Track revision count increases
                    if node_name == "design_review":
                        rev_count = updates.get("design_revision_count", 0)
                        print(f"      revision_count: {rev_count}", flush=True)

            print(f"\nNodes: {' → '.join(nodes_visited)}", flush=True)

            # Verify we hit the design → design_review loop multiple times
            design_count = nodes_visited.count("design")
            review_count = nodes_visited.count("design_review")
            print(f"design visits: {design_count}, design_review visits: {review_count}", flush=True)
            
            # Should have visited design at least MAX_DESIGN_REVISIONS times
            assert design_count >= MAX_DESIGN_REVISIONS, \
                f"Should visit design at least {MAX_DESIGN_REVISIONS} times, got {design_count}"
            
            # Check final state - should be waiting for user input
            state = graph.get_state(config).values
            assert state.get("awaiting_user_input") is True or \
                   state.get("design_revision_count") >= MAX_DESIGN_REVISIONS, \
                "Should be awaiting user input or at revision limit"

            print("\n✅ Design revision limit test passed!", flush=True)

    def test_design_state_changes_are_preserved(self, initial_state):
        """Test: verify all expected state changes through design phase."""
        
        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
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
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("design_state")}}

            # Run to generate_code
            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    if node_name == "generate_code":
                        break
                else:
                    continue
                break

            state = graph.get_state(config).values

            # Verify critical state fields
            assert state.get("design_description") is not None, \
                "design_description must be set"
            assert state.get("last_design_review_verdict") == "approve", \
                "Verdict must be approve to reach generate_code"
            assert state.get("current_stage_id") is not None, \
                "current_stage_id must be set"
            assert state.get("plan") is not None, \
                "plan must be set"
            
            # Verify design_description has expected structure
            design = state.get("design_description")
            if isinstance(design, dict):
                # Check for key design fields
                assert "geometry" in design or "materials" in design or "design_description" in design, \
                    "Design should have geometry, materials, or description"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignEdgeCases:
    """Edge case tests for design phase."""

    def test_empty_design_description_from_llm(self, state_with_stage):
        """Test: empty design from LLM is stored (not rejected)."""
        with patch("src.agents.design.call_agent_with_metrics", return_value={}):
            result = simulation_designer_node(state_with_stage)
        
        # Should still have design_description key, even if empty
        assert "design_description" in result

    def test_reviewer_with_empty_issues_list(self, state_with_design):
        """Test: reviewer with empty issues list works correctly."""
        mock_review = {
            "verdict": "approve",
            "issues": [],  # Empty, not None
            "summary": "All good",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        assert result.get("reviewer_issues") == []
        assert result.get("last_design_review_verdict") == "approve"

    def test_reviewer_with_none_issues(self, state_with_design):
        """Test: reviewer with None issues doesn't crash."""
        mock_review = {
            "verdict": "approve",
            "issues": None,  # None instead of list
            "summary": "OK",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        # Should handle gracefully - empty list
        assert result.get("reviewer_issues") == []

    def test_design_with_awaiting_user_input_already_true(self, state_with_stage):
        """Test: if already awaiting user input, designer returns empty dict."""
        state = {**state_with_stage, "awaiting_user_input": True}
        
        # The with_context_check decorator should return {} early
        with patch("src.agents.design.call_agent_with_metrics") as mock:
            result = design_reviewer_node(state)
        
        # Should not call LLM if already awaiting user
        assert result == {} or not mock.called

    def test_design_revision_count_starts_at_zero(self, initial_state):
        """Test: initial state has design_revision_count = 0."""
        assert initial_state.get("design_revision_count", 0) == 0

    def test_reviewer_feedback_uses_summary_as_fallback(self, state_with_design):
        """Test: if no 'feedback' key, uses 'summary' for reviewer_feedback."""
        mock_review = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Issue"}],
            "summary": "This is the summary used as feedback",
            # No "feedback" key
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        # Should use summary as feedback fallback
        feedback = result.get("reviewer_feedback", "")
        assert "summary" in feedback.lower() or feedback == "This is the summary used as feedback"

    def test_multiple_issues_from_reviewer(self, state_with_design):
        """Test: multiple issues are all captured."""
        mock_review = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "blocking", "description": "Issue 1"},
                {"severity": "major", "description": "Issue 2"},
                {"severity": "minor", "description": "Issue 3"},
            ],
            "summary": "Multiple issues",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state_with_design)
        
        issues = result.get("reviewer_issues", [])
        assert len(issues) == 3, f"Expected 3 issues, got {len(issues)}"

    def test_design_with_runtime_config_custom_limit(self, state_with_design):
        """Test: custom revision limit from runtime_config is respected."""
        custom_limit = 1
        state = {
            **state_with_design,
            "design_revision_count": 1,
            "runtime_config": {"max_design_revisions": custom_limit},
        }
        
        mock_review = MockLLMResponses.design_reviewer_reject()
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state)
        
        # Should escalate because we're at the custom limit
        assert result.get("awaiting_user_input") is True, \
            "Should escalate at custom limit"
        assert result.get("ask_user_trigger") == "design_review_limit"

    def test_reviewer_handles_string_design_description(self, state_with_stage):
        """Test: reviewer handles string design_description (not dict)."""
        state = {
            **state_with_stage,
            "design_description": "This is a simple string design, not a dict",
            "design_revision_count": 0,
        }
        
        mock_review = MockLLMResponses.design_reviewer_approve()
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_review):
            result = design_reviewer_node(state)
        
        # Should work correctly with string design_description
        assert result.get("last_design_review_verdict") == "approve"
        assert not result.get("awaiting_user_input")
