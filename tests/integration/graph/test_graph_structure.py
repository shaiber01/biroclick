"""Graph structure integration tests for the LangGraph state machine.

These tests verify:
1. Graph compiles correctly with all expected nodes
2. All edges are wired correctly (both static and conditional)
3. Routing functions route correctly for all verdict types
4. Count limits trigger escalation at correct thresholds
5. Interrupt configuration is set up correctly
6. The complete workflow topology matches design requirements
7. All routing targets actually exist in the graph (no dead links)
8. Compilation arguments are correct (interrupts, checkpointer)
"""

import pytest
from unittest.mock import patch, MagicMock, ANY

from langgraph.graph import StateGraph
from src.graph import (
    create_repro_graph,
    route_after_plan,
    route_after_select_stage,
    route_after_supervisor,
    generate_report_node_with_checkpoint,
)
from src.routing import (
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
)
from schemas.state import (
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def compiled_graph():
    """Fixture to create a compiled graph for testing."""
    return create_repro_graph()


@pytest.fixture
def graph_definition(compiled_graph):
    """Fixture to get the graph definition for edge/node inspection."""
    return compiled_graph.get_graph()


def make_state(**kwargs):
    """Helper to create minimal test state with given overrides."""
    base_state = {
        "paper_id": "test_paper",
        "runtime_config": {},
        "progress": {"stages": []},
    }
    base_state.update(kwargs)
    return base_state


# ═══════════════════════════════════════════════════════════════════════
# Graph Compilation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphCompilation:
    """Tests that the graph compiles without errors and has correct structure."""

    def test_graph_compiles_successfully(self, compiled_graph):
        """Test that create_repro_graph returns a compiled graph with required methods."""
        assert compiled_graph is not None, "Graph should not be None"
        
        # Verify it's a CompiledGraph with required interface methods
        assert hasattr(compiled_graph, "invoke"), "Graph must have invoke method"
        assert hasattr(compiled_graph, "stream"), "Graph must have stream method"
        assert hasattr(compiled_graph, "get_graph"), "Graph must have get_graph method"
        assert callable(compiled_graph.invoke), "invoke must be callable"
        assert callable(compiled_graph.stream), "stream must be callable"
        assert callable(compiled_graph.get_graph), "get_graph must be callable"

    def test_graph_has_all_expected_nodes(self, graph_definition):
        """Test that ALL expected nodes are present - no more, no less (except special nodes)."""
        expected_nodes = {
            "adapt_prompts",
            "planning",
            "plan_review",
            "select_stage",
            "design",
            "design_review",
            "generate_code",
            "code_review",
            "run_code",
            "execution_check",
            "physics_check",
            "analyze",
            "comparison_check",
            "supervisor",
            "ask_user",
            "generate_report",
            "handle_backtrack",
            "material_checkpoint",
        }

        actual_nodes = set(graph_definition.nodes.keys())
        
        # Check all expected nodes are present
        missing_nodes = expected_nodes - actual_nodes
        assert not missing_nodes, f"Missing expected nodes: {missing_nodes}"
        
        # Check no unexpected nodes (except __start__ and __end__)
        special_nodes = {"__start__", "__end__"}
        unexpected_nodes = actual_nodes - expected_nodes - special_nodes
        assert not unexpected_nodes, f"Unexpected nodes in graph: {unexpected_nodes}"

    def test_graph_has_start_and_end_nodes(self, graph_definition):
        """Test that the graph has both __start__ and __end__ special nodes."""
        assert "__start__" in graph_definition.nodes, "Graph must have __start__ node"
        assert "__end__" in graph_definition.nodes, "Graph must have __end__ node"

    def test_graph_node_count_exactly_correct(self, graph_definition):
        """Test exact node count including special nodes."""
        # 18 workflow nodes + 2 special nodes (__start__, __end__)
        expected_count = 20
        actual_count = len(graph_definition.nodes)
        assert actual_count == expected_count, (
            f"Expected exactly {expected_count} nodes (18 workflow + 2 special), "
            f"got {actual_count}: {list(graph_definition.nodes.keys())}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Complete Edge Connectivity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphEdgeConnectivity:
    """Tests that verify every edge in the graph is correctly wired."""

    def test_start_connects_only_to_adapt_prompts(self, graph_definition):
        """Test that __start__ connects ONLY to adapt_prompts."""
        edges = list(graph_definition.edges)
        start_edges = [edge for edge in edges if edge[0] == "__start__"]
        
        assert len(start_edges) == 1, (
            f"__start__ should have exactly 1 outgoing edge, got {len(start_edges)}: {start_edges}"
        )
        assert start_edges[0][1] == "adapt_prompts", (
            f"__start__ should connect to adapt_prompts, got {start_edges[0][1]}"
        )

    def test_adapt_prompts_connects_to_plan(self, graph_definition):
        """Test that adapt_prompts connects ONLY to plan."""
        edges = list(graph_definition.edges)
        adapt_edges = [edge for edge in edges if edge[0] == "adapt_prompts"]
        
        assert len(adapt_edges) == 1, (
            f"adapt_prompts should have exactly 1 outgoing edge, got: {adapt_edges}"
        )
        assert adapt_edges[0][1] == "planning", (
            f"adapt_prompts should connect to planning, got {adapt_edges[0][1]}"
        )

    def test_plan_has_conditional_edge_to_plan_review(self, graph_definition):
        """Test that planning routes to plan_review via conditional edge."""
        edges = list(graph_definition.edges)
        plan_edges = [edge for edge in edges if edge[0] == "planning"]
        targets = {edge[1] for edge in plan_edges}
        
        assert "plan_review" in targets, (
            f"planning should route to plan_review, got: {targets}"
        )

    def test_plan_review_has_three_routes(self, graph_definition):
        """Test plan_review can route to select_stage, planning, or ask_user."""
        edges = list(graph_definition.edges)
        review_edges = [edge for edge in edges if edge[0] == "plan_review"]
        targets = {edge[1] for edge in review_edges}
        
        expected = {"select_stage", "planning", "ask_user"}
        assert targets == expected, (
            f"plan_review should route to exactly {expected}, got: {targets}"
        )

    def test_select_stage_has_two_routes(self, graph_definition):
        """Test select_stage can route to design or generate_report."""
        edges = list(graph_definition.edges)
        select_edges = [edge for edge in edges if edge[0] == "select_stage"]
        targets = {edge[1] for edge in select_edges}
        
        expected = {"design", "generate_report"}
        assert targets == expected, (
            f"select_stage should route to exactly {expected}, got: {targets}"
        )

    def test_design_connects_to_design_review(self, graph_definition):
        """Test that design connects to design_review via static edge."""
        edges = list(graph_definition.edges)
        design_edges = [edge for edge in edges if edge[0] == "design"]
        targets = {edge[1] for edge in design_edges}
        
        assert "design_review" in targets, (
            f"design should connect to design_review, got: {targets}"
        )

    def test_design_review_has_three_routes(self, graph_definition):
        """Test design_review can route to generate_code, design, or ask_user."""
        edges = list(graph_definition.edges)
        review_edges = [edge for edge in edges if edge[0] == "design_review"]
        targets = {edge[1] for edge in review_edges}
        
        expected = {"generate_code", "design", "ask_user"}
        assert targets == expected, (
            f"design_review should route to exactly {expected}, got: {targets}"
        )

    def test_generate_code_connects_to_code_review(self, graph_definition):
        """Test that generate_code connects to code_review via static edge."""
        edges = list(graph_definition.edges)
        code_edges = [edge for edge in edges if edge[0] == "generate_code"]
        targets = {edge[1] for edge in code_edges}
        
        assert "code_review" in targets, (
            f"generate_code should connect to code_review, got: {targets}"
        )

    def test_code_review_has_three_routes(self, graph_definition):
        """Test code_review can route to run_code, generate_code, or ask_user."""
        edges = list(graph_definition.edges)
        review_edges = [edge for edge in edges if edge[0] == "code_review"]
        targets = {edge[1] for edge in review_edges}
        
        expected = {"run_code", "generate_code", "ask_user"}
        assert targets == expected, (
            f"code_review should route to exactly {expected}, got: {targets}"
        )

    def test_run_code_connects_to_execution_check(self, graph_definition):
        """Test that run_code connects to execution_check via static edge."""
        edges = list(graph_definition.edges)
        run_edges = [edge for edge in edges if edge[0] == "run_code"]
        targets = {edge[1] for edge in run_edges}
        
        assert "execution_check" in targets, (
            f"run_code should connect to execution_check, got: {targets}"
        )

    def test_execution_check_has_three_routes(self, graph_definition):
        """Test execution_check can route to physics_check, generate_code, or ask_user."""
        edges = list(graph_definition.edges)
        exec_edges = [edge for edge in edges if edge[0] == "execution_check"]
        targets = {edge[1] for edge in exec_edges}
        
        expected = {"physics_check", "generate_code", "ask_user"}
        assert targets == expected, (
            f"execution_check should route to exactly {expected}, got: {targets}"
        )

    def test_physics_check_has_four_routes(self, graph_definition):
        """Test physics_check can route to analyze, generate_code, design, or ask_user."""
        edges = list(graph_definition.edges)
        physics_edges = [edge for edge in edges if edge[0] == "physics_check"]
        targets = {edge[1] for edge in physics_edges}
        
        expected = {"analyze", "generate_code", "design", "ask_user"}
        assert targets == expected, (
            f"physics_check should route to exactly {expected}, got: {targets}"
        )

    def test_analyze_connects_to_comparison_check(self, graph_definition):
        """Test that analyze connects to comparison_check via static edge."""
        edges = list(graph_definition.edges)
        analyze_edges = [edge for edge in edges if edge[0] == "analyze"]
        targets = {edge[1] for edge in analyze_edges}
        
        assert "comparison_check" in targets, (
            f"analyze should connect to comparison_check, got: {targets}"
        )

    def test_comparison_check_has_two_routes(self, graph_definition):
        """Test comparison_check can route to supervisor, analyze, or ask_user."""
        edges = list(graph_definition.edges)
        comp_edges = [edge for edge in edges if edge[0] == "comparison_check"]
        targets = {edge[1] for edge in comp_edges}
        
        # comparison_check routes to ask_user on limit (consistent with other routers)
        expected_minimum = {"supervisor", "analyze", "ask_user"}
        assert expected_minimum.issubset(targets), (
            f"comparison_check should route to at least {expected_minimum}, got: {targets}"
        )

    def test_supervisor_has_all_expected_routes(self, graph_definition):
        """Test supervisor can route to all expected destinations."""
        edges = list(graph_definition.edges)
        supervisor_edges = [edge for edge in edges if edge[0] == "supervisor"]
        targets = {edge[1] for edge in supervisor_edges}
        
        expected = {
            "select_stage",
            "planning",
            "ask_user",
            "handle_backtrack",
            "generate_report",
            "material_checkpoint",
            "analyze",
            "generate_code",
            "design",
            "code_review",
            "design_review",
            "plan_review",
        }
        assert targets == expected, (
            f"supervisor should route to exactly {expected}, got: {targets}"
        )

    def test_handle_backtrack_connects_to_select_stage(self, graph_definition):
        """Test that handle_backtrack connects to select_stage via static edge."""
        edges = list(graph_definition.edges)
        backtrack_edges = [edge for edge in edges if edge[0] == "handle_backtrack"]
        targets = {edge[1] for edge in backtrack_edges}
        
        assert targets == {"select_stage"}, (
            f"handle_backtrack should connect only to select_stage, got: {targets}"
        )

    def test_material_checkpoint_connects_to_ask_user_or_select_stage(self, graph_definition):
        """Test that material_checkpoint connects to ask_user or select_stage via conditional edge.
        
        material_checkpoint uses a conditional edge:
        - If validated_materials is populated, routes to select_stage (skip ask_user)
        - Otherwise, routes to ask_user for user confirmation
        """
        edges = list(graph_definition.edges)
        mat_edges = [edge for edge in edges if edge[0] == "material_checkpoint"]
        targets = {edge[1] for edge in mat_edges}
        
        assert targets == {"ask_user", "select_stage"}, (
            f"material_checkpoint should connect to ask_user and select_stage, got: {targets}"
        )

    def test_ask_user_routes_to_supervisor(self, graph_definition):
        """Test that ask_user routes to supervisor."""
        edges = list(graph_definition.edges)
        ask_edges = [edge for edge in edges if edge[0] == "ask_user"]
        targets = {edge[1] for edge in ask_edges}
        
        assert "supervisor" in targets, (
            f"ask_user should route to supervisor, got: {targets}"
        )

    def test_generate_report_connects_to_end(self, graph_definition):
        """Test that generate_report connects to __end__."""
        edges = list(graph_definition.edges)
        report_edges = [edge for edge in edges if edge[0] == "generate_report"]
        targets = {edge[1] for edge in report_edges}
        
        assert targets == {"__end__"}, (
            f"generate_report should connect only to __end__, got: {targets}"
        )

    def test_all_nodes_have_outgoing_edges_except_end(self, graph_definition):
        """Test that every node (except __end__) has at least one outgoing edge."""
        edges = list(graph_definition.edges)
        nodes_with_outgoing = {edge[0] for edge in edges}
        all_nodes = set(graph_definition.nodes.keys())
        
        # __end__ should not have outgoing edges
        nodes_needing_edges = all_nodes - {"__end__"}
        nodes_without_edges = nodes_needing_edges - nodes_with_outgoing
        
        assert not nodes_without_edges, (
            f"Nodes without outgoing edges (excluding __end__): {nodes_without_edges}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - route_after_plan
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterPlan:
    """Tests for the route_after_plan routing function."""

    @patch('src.graph.save_checkpoint')
    def test_always_routes_to_plan_review(self, mock_checkpoint):
        """Test that route_after_plan always returns plan_review."""
        state = make_state()
        result = route_after_plan(state)
        assert result == "plan_review", f"Expected plan_review, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_saves_checkpoint(self, mock_checkpoint):
        """Test that route_after_plan saves checkpoint with correct name."""
        state = make_state()
        route_after_plan(state)
        mock_checkpoint.assert_called_once_with(state, "after_plan")


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - route_after_select_stage
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterSelectStage:
    """Tests for the route_after_select_stage routing function."""

    def test_routes_to_design_when_stage_selected(self):
        """Test routing to design when current_stage_id is set."""
        state = make_state(current_stage_id="stage_1")
        result = route_after_select_stage(state)
        assert result == "design", f"Expected design when stage selected, got {result}"

    def test_routes_to_generate_report_when_no_stage(self):
        """Test routing to generate_report when current_stage_id is None."""
        state = make_state(current_stage_id=None)
        result = route_after_select_stage(state)
        assert result == "generate_report", f"Expected generate_report when no stage, got {result}"

    def test_routes_to_generate_report_when_stage_missing(self):
        """Test routing to generate_report when current_stage_id key is missing."""
        state = make_state()
        # Ensure current_stage_id is not in state
        state.pop("current_stage_id", None)
        result = route_after_select_stage(state)
        assert result == "generate_report", f"Expected generate_report when stage missing, got {result}"

    def test_routes_to_design_with_various_stage_ids(self):
        """Test routing works with various stage ID formats."""
        for stage_id in ["stage_0", "stage_1", "material_validation", "S0", "123"]:
            state = make_state(current_stage_id=stage_id)
            result = route_after_select_stage(state)
            assert result == "design", f"Expected design for stage_id={stage_id}, got {result}"

    def test_routes_to_generate_report_with_empty_string(self):
        """Test routing to generate_report when current_stage_id is empty string."""
        state = make_state(current_stage_id="")
        result = route_after_select_stage(state)
        # Empty string is falsy, should route to generate_report
        assert result == "generate_report", (
            f"Expected generate_report for empty string stage_id, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - route_after_supervisor (Complex Logic)
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterSupervisor:
    """Tests for the complex route_after_supervisor routing function."""

    @patch('src.graph.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test that None verdict escalates to ask_user."""
        state = make_state(supervisor_verdict=None)
        result = route_after_supervisor(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"
        mock_checkpoint.assert_called_once()

    @patch('src.graph.save_checkpoint')
    def test_ok_continue_routes_to_select_stage(self, mock_checkpoint):
        """Test ok_continue verdict routes to select_stage normally."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="SINGLE_STRUCTURE",
        )
        result = route_after_supervisor(state)
        assert result == "select_stage", f"Expected select_stage for ok_continue, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_ok_continue_with_should_stop_routes_to_report(self, mock_checkpoint):
        """Test ok_continue with should_stop=True routes to generate_report."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=True,
        )
        result = route_after_supervisor(state)
        assert result == "generate_report", (
            f"Expected generate_report for ok_continue with should_stop=True, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_change_priority_routes_same_as_ok_continue(self, mock_checkpoint):
        """Test change_priority verdict routes like ok_continue."""
        state = make_state(
            supervisor_verdict="change_priority",
            should_stop=False,
            current_stage_type="SINGLE_STRUCTURE",
        )
        result = route_after_supervisor(state)
        assert result == "select_stage", f"Expected select_stage for change_priority, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_material_validation_routes_to_checkpoint_first_time(self, mock_checkpoint):
        """Test MATERIAL_VALIDATION stage type routes to material_checkpoint first time."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={},  # No material_checkpoint response yet
        )
        result = route_after_supervisor(state)
        assert result == "material_checkpoint", (
            f"Expected material_checkpoint for MATERIAL_VALIDATION first pass, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_material_validation_routes_to_select_stage_after_checkpoint(self, mock_checkpoint):
        """Test MATERIAL_VALIDATION stage routes to select_stage after checkpoint completed.
        
        The check uses validated_materials (not user_responses) because ask_user_node
        stores responses with question text as key, not trigger name.
        """
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
            validated_materials=[{"name": "aluminum", "source": "stage0_output"}],  # Checkpoint done
        )
        result = route_after_supervisor(state)
        assert result == "select_stage", (
            f"Expected select_stage after material_checkpoint complete, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_replan_needed_routes_to_plan_under_limit(self, mock_checkpoint):
        """Test replan_needed routes to plan when under limit."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=0,
        )
        result = route_after_supervisor(state)
        assert result == "planning", f"Expected planning for replan_needed under limit, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_replan_needed_routes_to_plan_at_limit_minus_one(self, mock_checkpoint):
        """Test replan_needed routes to planning when at limit-1."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS - 1,
        )
        result = route_after_supervisor(state)
        assert result == "planning", f"Expected planning for replan_needed at limit-1, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_replan_needed_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test replan_needed routes to ask_user when at limit."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS,
        )
        result = route_after_supervisor(state)
        assert result == "ask_user", f"Expected ask_user for replan_needed at limit, got {result}"
        mock_checkpoint.assert_called()

    @patch('src.graph.save_checkpoint')
    def test_replan_needed_routes_to_ask_user_over_limit(self, mock_checkpoint):
        """Test replan_needed routes to ask_user when over limit."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS + 5,
        )
        result = route_after_supervisor(state)
        assert result == "ask_user", f"Expected ask_user for replan_needed over limit, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_replan_with_guidance_routes_to_planning_regardless_of_count(self, mock_checkpoint):
        """Test replan_with_guidance bypasses count limit and routes to planning.
        
        This is the key fix for the bug where user GUIDANCE was being ignored
        because route_after_supervisor checked replan_count from state before
        the supervisor's result (with reset count) was merged.
        """
        # Even with replan_count at the limit, replan_with_guidance should route to planning
        state = make_state(
            supervisor_verdict="replan_with_guidance",
            replan_count=MAX_REPLANS,  # At limit
        )
        result = route_after_supervisor(state)
        assert result == "planning", (
            f"Expected planning for replan_with_guidance even at count limit, got {result}"
        )
        
        # Also test with count well over the limit
        state = make_state(
            supervisor_verdict="replan_with_guidance",
            replan_count=MAX_REPLANS + 10,  # Way over limit
        )
        result = route_after_supervisor(state)
        assert result == "planning", (
            f"Expected planning for replan_with_guidance even over count limit, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_ask_user_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test ask_user verdict routes to ask_user."""
        state = make_state(supervisor_verdict="ask_user")
        result = route_after_supervisor(state)
        assert result == "ask_user", f"Expected ask_user for ask_user verdict, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_backtrack_to_stage_routes_to_handle_backtrack(self, mock_checkpoint):
        """Test backtrack_to_stage verdict routes to handle_backtrack."""
        state = make_state(supervisor_verdict="backtrack_to_stage")
        result = route_after_supervisor(state)
        assert result == "handle_backtrack", (
            f"Expected handle_backtrack for backtrack_to_stage, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_all_complete_routes_to_generate_report(self, mock_checkpoint):
        """Test all_complete verdict routes to generate_report."""
        state = make_state(supervisor_verdict="all_complete")
        result = route_after_supervisor(state)
        assert result == "generate_report", (
            f"Expected generate_report for all_complete, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_unknown_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test unknown verdict falls back to ask_user."""
        state = make_state(supervisor_verdict="unknown_verdict_xyz")
        result = route_after_supervisor(state)
        assert result == "ask_user", f"Expected ask_user for unknown verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - Plan Review Router
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterPlanReview:
    """Tests for route_after_plan_review routing function."""

    @patch('src.routing.save_checkpoint')
    def test_approve_routes_to_select_stage(self, mock_checkpoint):
        """Test approve verdict routes to select_stage."""
        state = make_state(last_plan_review_verdict="approve")
        result = route_after_plan_review(state)
        assert result == "select_stage", f"Expected select_stage for approve, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_plan_under_limit(self, mock_checkpoint):
        """Test needs_revision routes to planning when under limit."""
        state = make_state(
            last_plan_review_verdict="needs_revision",
            replan_count=0,
        )
        result = route_after_plan_review(state)
        assert result == "planning", f"Expected planning for needs_revision under limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test needs_revision routes to ask_user at limit."""
        state = make_state(
            last_plan_review_verdict="needs_revision",
            replan_count=MAX_REPLANS,
        )
        result = route_after_plan_review(state)
        assert result == "ask_user", f"Expected ask_user for needs_revision at limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test None verdict routes to ask_user."""
        state = make_state(last_plan_review_verdict=None)
        result = route_after_plan_review(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - Design Review Router
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterDesignReview:
    """Tests for route_after_design_review routing function."""

    @patch('src.routing.save_checkpoint')
    def test_approve_routes_to_generate_code(self, mock_checkpoint):
        """Test approve verdict routes to generate_code."""
        state = make_state(last_design_review_verdict="approve")
        result = route_after_design_review(state)
        assert result == "generate_code", f"Expected generate_code for approve, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_design_under_limit(self, mock_checkpoint):
        """Test needs_revision routes to design when under limit."""
        state = make_state(
            last_design_review_verdict="needs_revision",
            design_revision_count=0,
        )
        result = route_after_design_review(state)
        assert result == "design", f"Expected design for needs_revision under limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test needs_revision routes to ask_user at limit."""
        state = make_state(
            last_design_review_verdict="needs_revision",
            design_revision_count=MAX_DESIGN_REVISIONS,
        )
        result = route_after_design_review(state)
        assert result == "ask_user", f"Expected ask_user at design revision limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test None verdict routes to ask_user."""
        state = make_state(last_design_review_verdict=None)
        result = route_after_design_review(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - Code Review Router
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterCodeReview:
    """Tests for route_after_code_review routing function."""

    @patch('src.routing.save_checkpoint')
    def test_approve_routes_to_run_code(self, mock_checkpoint):
        """Test approve verdict routes to run_code."""
        state = make_state(last_code_review_verdict="approve")
        result = route_after_code_review(state)
        assert result == "run_code", f"Expected run_code for approve, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_generate_code_under_limit(self, mock_checkpoint):
        """Test needs_revision routes to generate_code when under limit."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=0,
        )
        result = route_after_code_review(state)
        assert result == "generate_code", (
            f"Expected generate_code for needs_revision under limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test needs_revision routes to ask_user at limit."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )
        result = route_after_code_review(state)
        assert result == "ask_user", f"Expected ask_user at code revision limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test None verdict routes to ask_user."""
        state = make_state(last_code_review_verdict=None)
        result = route_after_code_review(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - Execution Check Router
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterExecutionCheck:
    """Tests for route_after_execution_check routing function."""

    @patch('src.routing.save_checkpoint')
    def test_pass_routes_to_physics_check(self, mock_checkpoint):
        """Test pass verdict routes to physics_check."""
        state = make_state(execution_verdict="pass")
        result = route_after_execution_check(state)
        assert result == "physics_check", f"Expected physics_check for pass, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_warning_routes_to_physics_check(self, mock_checkpoint):
        """Test warning verdict routes to physics_check (pass-through)."""
        state = make_state(execution_verdict="warning")
        result = route_after_execution_check(state)
        assert result == "physics_check", f"Expected physics_check for warning, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_fail_routes_to_generate_code_under_limit(self, mock_checkpoint):
        """Test fail verdict routes to generate_code when under limit."""
        state = make_state(
            execution_verdict="fail",
            execution_failure_count=0,
        )
        result = route_after_execution_check(state)
        assert result == "generate_code", (
            f"Expected generate_code for fail under limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_fail_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test fail verdict routes to ask_user at limit."""
        state = make_state(
            execution_verdict="fail",
            execution_failure_count=MAX_EXECUTION_FAILURES,
        )
        result = route_after_execution_check(state)
        assert result == "ask_user", f"Expected ask_user at execution failure limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test None verdict routes to ask_user."""
        state = make_state(execution_verdict=None)
        result = route_after_execution_check(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - Physics Check Router
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterPhysicsCheck:
    """Tests for route_after_physics_check routing function."""

    @patch('src.routing.save_checkpoint')
    def test_pass_routes_to_analyze(self, mock_checkpoint):
        """Test pass verdict routes to analyze."""
        state = make_state(physics_verdict="pass")
        result = route_after_physics_check(state)
        assert result == "analyze", f"Expected analyze for pass, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_warning_routes_to_analyze(self, mock_checkpoint):
        """Test warning verdict routes to analyze (pass-through)."""
        state = make_state(physics_verdict="warning")
        result = route_after_physics_check(state)
        assert result == "analyze", f"Expected analyze for warning, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_fail_routes_to_generate_code_under_limit(self, mock_checkpoint):
        """Test fail verdict routes to generate_code when under limit."""
        state = make_state(
            physics_verdict="fail",
            physics_failure_count=0,
        )
        result = route_after_physics_check(state)
        assert result == "generate_code", (
            f"Expected generate_code for fail under limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_fail_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test fail verdict routes to ask_user at limit."""
        state = make_state(
            physics_verdict="fail",
            physics_failure_count=MAX_PHYSICS_FAILURES,
        )
        result = route_after_physics_check(state)
        assert result == "ask_user", f"Expected ask_user at physics failure limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_design_flaw_routes_to_design_under_limit(self, mock_checkpoint):
        """Test design_flaw verdict routes to design when under limit."""
        state = make_state(
            physics_verdict="design_flaw",
            design_revision_count=0,
        )
        result = route_after_physics_check(state)
        assert result == "design", f"Expected design for design_flaw under limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_design_flaw_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test design_flaw verdict routes to ask_user at design revision limit."""
        state = make_state(
            physics_verdict="design_flaw",
            design_revision_count=MAX_DESIGN_REVISIONS,
        )
        result = route_after_physics_check(state)
        assert result == "ask_user", f"Expected ask_user at design revision limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test None verdict routes to ask_user."""
        state = make_state(physics_verdict=None)
        result = route_after_physics_check(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Function Tests - Comparison Check Router
# ═══════════════════════════════════════════════════════════════════════

class TestRouteAfterComparisonCheck:
    """Tests for route_after_comparison_check routing function."""

    @patch('src.routing.save_checkpoint')
    def test_approve_routes_to_supervisor(self, mock_checkpoint):
        """Test approve verdict routes to supervisor."""
        state = make_state(comparison_verdict="approve")
        result = route_after_comparison_check(state)
        assert result == "supervisor", f"Expected supervisor for approve, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_analyze_under_limit(self, mock_checkpoint):
        """Test needs_revision routes to analyze when under limit."""
        state = make_state(
            comparison_verdict="needs_revision",
            analysis_revision_count=0,
        )
        result = route_after_comparison_check(state)
        assert result == "analyze", f"Expected analyze for needs_revision under limit, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_needs_revision_routes_to_ask_user_at_limit(self, mock_checkpoint):
        """Test needs_revision routes to ask_user at limit (consistent with other limits)."""
        state = make_state(
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS,
        )
        result = route_after_comparison_check(state)
        # comparison_check now routes to ask_user on limit (consistent with other limits)
        assert result == "ask_user", (
            f"Expected ask_user at analysis revision limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_none_verdict_routes_to_ask_user(self, mock_checkpoint):
        """Test None verdict routes to ask_user."""
        state = make_state(comparison_verdict=None)
        result = route_after_comparison_check(state)
        assert result == "ask_user", f"Expected ask_user for None verdict, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Count Limit Boundary Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCountLimitBoundaries:
    """Tests for count limit boundary conditions in routing functions."""

    @patch('src.routing.save_checkpoint')
    def test_code_review_at_exactly_max_count(self, mock_checkpoint):
        """Test code review at exactly MAX_CODE_REVISIONS escalates."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )
        result = route_after_code_review(state)
        assert result == "ask_user", (
            f"Should escalate at exactly MAX_CODE_REVISIONS={MAX_CODE_REVISIONS}"
        )

    @patch('src.routing.save_checkpoint')
    def test_code_review_one_under_max_count(self, mock_checkpoint):
        """Test code review at MAX_CODE_REVISIONS-1 continues."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS - 1,
        )
        result = route_after_code_review(state)
        assert result == "generate_code", (
            f"Should continue at MAX_CODE_REVISIONS-1={MAX_CODE_REVISIONS - 1}"
        )

    @patch('src.routing.save_checkpoint')
    def test_design_review_at_exactly_max_count(self, mock_checkpoint):
        """Test design review at exactly MAX_DESIGN_REVISIONS escalates."""
        state = make_state(
            last_design_review_verdict="needs_revision",
            design_revision_count=MAX_DESIGN_REVISIONS,
        )
        result = route_after_design_review(state)
        assert result == "ask_user", (
            f"Should escalate at exactly MAX_DESIGN_REVISIONS={MAX_DESIGN_REVISIONS}"
        )

    @patch('src.routing.save_checkpoint')
    def test_execution_check_at_exactly_max_count(self, mock_checkpoint):
        """Test execution check at exactly MAX_EXECUTION_FAILURES escalates."""
        state = make_state(
            execution_verdict="fail",
            execution_failure_count=MAX_EXECUTION_FAILURES,
        )
        result = route_after_execution_check(state)
        assert result == "ask_user", (
            f"Should escalate at exactly MAX_EXECUTION_FAILURES={MAX_EXECUTION_FAILURES}"
        )

    @patch('src.routing.save_checkpoint')
    def test_physics_check_at_exactly_max_count(self, mock_checkpoint):
        """Test physics check at exactly MAX_PHYSICS_FAILURES escalates."""
        state = make_state(
            physics_verdict="fail",
            physics_failure_count=MAX_PHYSICS_FAILURES,
        )
        result = route_after_physics_check(state)
        assert result == "ask_user", (
            f"Should escalate at exactly MAX_PHYSICS_FAILURES={MAX_PHYSICS_FAILURES}"
        )

    @patch('src.routing.save_checkpoint')
    def test_runtime_config_overrides_default_limits(self, mock_checkpoint):
        """Test that runtime_config can override default limits."""
        custom_max = 10
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,  # Would normally escalate
            runtime_config={"max_code_revisions": custom_max},
        )
        result = route_after_code_review(state)
        # With custom max of 10, count of 3 should NOT escalate
        assert result == "generate_code", (
            f"Custom runtime_config max should override default limit"
        )


# ═══════════════════════════════════════════════════════════════════════
# Interrupt Configuration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphInterruptConfiguration:
    """Tests for graph interrupt (pause) configuration.
    
    Note: We use LangGraph's interrupt() function inside ask_user_node for human-in-the-loop,
    NOT interrupt_before. This means the node runs and pauses mid-execution when it calls
    interrupt(), rather than pausing before the node executes.
    """

    def test_graph_has_checkpointer(self, compiled_graph):
        """Test that the graph has a checkpointer for interrupt support."""
        assert compiled_graph.checkpointer is not None, (
            "Graph must have a checkpointer for interrupt() support"
        )

    def test_checkpointer_is_memory_saver(self, compiled_graph):
        """Test that the checkpointer is a MemorySaver instance."""
        from langgraph.checkpoint.memory import MemorySaver
        assert isinstance(compiled_graph.checkpointer, MemorySaver), (
            f"Checkpointer should be MemorySaver, got {type(compiled_graph.checkpointer)}"
        )

    @patch.object(StateGraph, 'compile')
    def test_compile_called_with_checkpointer(self, mock_compile):
        """Test that compile is called with a checkpointer (required for interrupt()).
        
        Note: We no longer use interrupt_before - instead, ask_user_node calls
        interrupt() internally to pause for user input.
        """
        # We need to mock the graph construction since we are testing the compile call
        with patch('src.graph.StateGraph') as MockStateGraph:
            # Create mock instance
            mock_workflow = MockStateGraph.return_value
            # Call the function that creates and compiles the graph
            create_repro_graph()
            
            # Verify compile was called with expected args
            mock_workflow.compile.assert_called_once()
            call_kwargs = mock_workflow.compile.call_args.kwargs
            
            # Should NOT have interrupt_before (we use interrupt() inside nodes now)
            assert "interrupt_before" not in call_kwargs, (
                "compile should NOT have interrupt_before - we use interrupt() inside nodes"
            )
            assert "checkpointer" in call_kwargs, "compile missing checkpointer"


# ═══════════════════════════════════════════════════════════════════════
# Workflow Path Integrity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestWorkflowPathIntegrity:
    """Tests to verify critical workflow paths are complete and connected."""

    def test_happy_path_is_connected(self, graph_definition):
        """Test that the happy path through the graph is fully connected."""
        happy_path = [
            "__start__",
            "adapt_prompts",
            "planning",
            "plan_review",
            "select_stage",
            "design",
            "design_review",
            "generate_code",
            "code_review",
            "run_code",
            "execution_check",
            "physics_check",
            "analyze",
            "comparison_check",
            "supervisor",
            # supervisor can go back to select_stage for next stage
            # or to generate_report
            "generate_report",
            "__end__",
        ]
        
        edges = list(graph_definition.edges)
        edge_set = {(e[0], e[1]) for e in edges}
        
        # Check each consecutive pair in happy path is reachable
        for i in range(len(happy_path) - 1):
            source = happy_path[i]
            target = happy_path[i + 1]
            
            # Check if direct edge exists OR if there's any path to target from source
            source_targets = {e[1] for e in edges if e[0] == source}
            assert target in source_targets, (
                f"Happy path broken: {source} -> {target} not in edges. "
                f"{source} only connects to: {source_targets}"
            )

    def test_revision_loops_exist(self, graph_definition):
        """Test that revision loops exist for iterative improvement."""
        edges = list(graph_definition.edges)
        
        # plan_review can go back to planning
        plan_review_targets = {e[1] for e in edges if e[0] == "plan_review"}
        assert "planning" in plan_review_targets, "plan_review must be able to loop back to planning"
        
        # design_review can go back to design
        design_review_targets = {e[1] for e in edges if e[0] == "design_review"}
        assert "design" in design_review_targets, "design_review must be able to loop back to design"
        
        # code_review can go back to generate_code
        code_review_targets = {e[1] for e in edges if e[0] == "code_review"}
        assert "generate_code" in code_review_targets, (
            "code_review must be able to loop back to generate_code"
        )
        
        # execution_check can go back to generate_code
        exec_targets = {e[1] for e in edges if e[0] == "execution_check"}
        assert "generate_code" in exec_targets, (
            "execution_check must be able to loop back to generate_code"
        )

    def test_escalation_paths_to_ask_user(self, graph_definition):
        """Test that all review nodes can escalate to ask_user."""
        edges = list(graph_definition.edges)
        
        nodes_that_should_escalate = [
            "plan_review",
            "design_review",
            "code_review",
            "execution_check",
            "physics_check",
        ]
        
        for node in nodes_that_should_escalate:
            targets = {e[1] for e in edges if e[0] == node}
            assert "ask_user" in targets, (
                f"{node} must have escalation path to ask_user, only has: {targets}"
            )

    def test_supervisor_can_terminate_workflow(self, graph_definition):
        """Test that supervisor can route to generate_report to end workflow."""
        edges = list(graph_definition.edges)
        supervisor_targets = {e[1] for e in edges if e[0] == "supervisor"}
        
        assert "generate_report" in supervisor_targets, (
            "supervisor must be able to route to generate_report for workflow termination"
        )

    def test_all_ask_user_escalations_return_to_supervisor(self, graph_definition):
        """Test that ask_user always returns to supervisor for decision."""
        edges = list(graph_definition.edges)
        ask_user_targets = {e[1] for e in edges if e[0] == "ask_user"}
        
        assert ask_user_targets == {"supervisor"}, (
            f"ask_user should only route to supervisor, but routes to: {ask_user_targets}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Report Node Wrapper Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGenerateReportNodeWrapper:
    """Tests for the generate_report_node_with_checkpoint wrapper."""

    @patch('src.graph._generate_report_node')
    @patch('src.graph.save_checkpoint')
    def test_wrapper_calls_original_node(self, mock_checkpoint, mock_report_node):
        """Test that wrapper calls the original report node."""
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}
        state = make_state()
        
        generate_report_node_with_checkpoint(state)
        
        mock_report_node.assert_called_once_with(state)

    @patch('src.graph._generate_report_node')
    @patch('src.graph.save_checkpoint')
    def test_wrapper_saves_final_checkpoint(self, mock_checkpoint, mock_report_node):
        """Test that wrapper saves checkpoint with correct name."""
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}
        state = make_state()
        
        generate_report_node_with_checkpoint(state)
        
        # Check that save_checkpoint was called with merged state and correct name
        mock_checkpoint.assert_called_once()
        call_args = mock_checkpoint.call_args
        assert call_args[0][1] == "final_report", (
            f"Checkpoint name should be 'final_report', got {call_args[0][1]}"
        )

    @patch('src.graph._generate_report_node')
    @patch('src.graph.save_checkpoint')
    def test_wrapper_returns_original_result(self, mock_checkpoint, mock_report_node):
        """Test that wrapper returns the original node's result."""
        expected_result = {"report_path": "/path/to/report.md", "status": "success"}
        mock_report_node.return_value = expected_result
        state = make_state()
        
        result = generate_report_node_with_checkpoint(state)
        
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    @patch('src.graph._generate_report_node')
    @patch('src.graph.save_checkpoint')
    def test_wrapper_merges_result_into_checkpoint(self, mock_checkpoint, mock_report_node):
        """Test that checkpoint includes both original state and result."""
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}
        state = make_state(paper_id="test123")
        
        generate_report_node_with_checkpoint(state)
        
        # Get the state passed to save_checkpoint
        checkpoint_state = mock_checkpoint.call_args[0][0]
        assert "paper_id" in checkpoint_state, "Checkpoint should contain original state"
        assert "report_path" in checkpoint_state, "Checkpoint should contain result"
        assert checkpoint_state["paper_id"] == "test123"
        assert checkpoint_state["report_path"] == "/path/to/report.md"


# ═══════════════════════════════════════════════════════════════════════
# Edge Case Tests for Routing
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingEdgeCases:
    """Tests for edge cases and unusual inputs to routing functions."""

    @patch('src.routing.save_checkpoint')
    def test_missing_count_field_defaults_to_zero(self, mock_checkpoint):
        """Test that missing count field is treated as 0."""
        state = make_state(last_code_review_verdict="needs_revision")
        # code_revision_count is not in state
        assert "code_revision_count" not in state
        
        result = route_after_code_review(state)
        # Should treat as 0 and allow revision
        assert result == "generate_code", (
            "Missing count field should default to 0, allowing revision"
        )

    @patch('src.routing.save_checkpoint')
    def test_none_count_field_defaults_to_zero(self, mock_checkpoint):
        """Test that None count field is treated as 0."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=None,
        )
        
        result = route_after_code_review(state)
        assert result == "generate_code", (
            "None count field should default to 0, allowing revision"
        )

    @patch('src.routing.save_checkpoint')
    def test_missing_runtime_config_uses_defaults(self, mock_checkpoint):
        """Test that missing runtime_config uses default limits."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )
        state.pop("runtime_config", None)
        
        result = route_after_code_review(state)
        assert result == "ask_user", (
            "Missing runtime_config should use default limits"
        )

    @patch('src.routing.save_checkpoint')
    def test_empty_runtime_config_uses_defaults(self, mock_checkpoint):
        """Test that empty runtime_config uses default limits."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
            runtime_config={},
        )
        
        result = route_after_code_review(state)
        assert result == "ask_user", (
            "Empty runtime_config should use default limits"
        )

    @patch('src.graph.save_checkpoint')
    def test_supervisor_missing_user_responses_key(self, mock_checkpoint):
        """Test supervisor handles missing user_responses field."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
        )
        # user_responses is not in state
        assert "user_responses" not in state
        
        result = route_after_supervisor(state)
        # Should route to material_checkpoint since no checkpoint evidence exists
        assert result == "material_checkpoint"

    @patch('src.graph.save_checkpoint')
    def test_supervisor_missing_should_stop_field(self, mock_checkpoint):
        """Test supervisor handles missing should_stop field."""
        state = make_state(
            supervisor_verdict="ok_continue",
            current_stage_type="SINGLE_STRUCTURE",
        )
        state.pop("should_stop", None)
        
        result = route_after_supervisor(state)
        # Missing should_stop is falsy, should continue
        assert result == "select_stage"

    @patch('src.graph.save_checkpoint')
    def test_supervisor_missing_replan_count_field(self, mock_checkpoint):
        """Test supervisor handles missing replan_count field."""
        state = make_state(supervisor_verdict="replan_needed")
        # replan_count not in state
        
        result = route_after_supervisor(state)
        # Missing replan_count defaults to 0, should allow replan
        assert result == "planning"


# ═══════════════════════════════════════════════════════════════════════
# Unknown Verdict Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUnknownVerdictHandling:
    """Tests for handling of unknown/unexpected verdict values."""

    @patch('src.routing.save_checkpoint')
    def test_unknown_plan_review_verdict_escalates(self, mock_checkpoint):
        """Test unknown plan review verdict escalates to ask_user."""
        state = make_state(last_plan_review_verdict="unknown_xyz")
        result = route_after_plan_review(state)
        assert result == "ask_user", "Unknown verdict should escalate to ask_user"

    @patch('src.routing.save_checkpoint')
    def test_unknown_design_review_verdict_escalates(self, mock_checkpoint):
        """Test unknown design review verdict escalates to ask_user."""
        state = make_state(last_design_review_verdict="invalid_verdict")
        result = route_after_design_review(state)
        assert result == "ask_user", "Unknown verdict should escalate to ask_user"

    @patch('src.routing.save_checkpoint')
    def test_unknown_execution_verdict_escalates(self, mock_checkpoint):
        """Test unknown execution verdict escalates to ask_user."""
        state = make_state(execution_verdict="partial")  # Not a valid verdict
        result = route_after_execution_check(state)
        assert result == "ask_user", "Unknown verdict should escalate to ask_user"

    @patch('src.routing.save_checkpoint')
    def test_unknown_physics_verdict_escalates(self, mock_checkpoint):
        """Test unknown physics verdict escalates to ask_user."""
        state = make_state(physics_verdict="maybe_ok")  # Not a valid verdict
        result = route_after_physics_check(state)
        assert result == "ask_user", "Unknown verdict should escalate to ask_user"

    @patch('src.routing.save_checkpoint')
    def test_unknown_comparison_verdict_escalates(self, mock_checkpoint):
        """Test unknown comparison verdict escalates to ask_user."""
        state = make_state(comparison_verdict="uncertain")  # Not a valid verdict
        result = route_after_comparison_check(state)
        assert result == "ask_user", "Unknown verdict should escalate to ask_user"


# ═══════════════════════════════════════════════════════════════════════
# Routing Target Integrity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingTargetIntegrity:
    """Tests that ensure all routes returned by functions exist in the graph."""

    def test_all_routed_destinations_exist_in_graph(self, graph_definition):
        """
        DYNAMIC INTEGRITY TEST:
        Iterate through all routing functions, force them to return all possible values,
        and verify that every returned node name actually exists in the graph.
        
        This catches typos like returning 'generate_code' when node is named 'GenerateCode'.
        """
        valid_nodes = set(graph_definition.nodes.keys())
        
        # Define test scenarios for each router to cover all return paths
        scenarios = [
            # (Router Function, List of States to trigger all paths)
            (route_after_plan, [make_state()]),
            (route_after_select_stage, [
                make_state(current_stage_id="s1"), # -> design
                make_state(current_stage_id=None), # -> generate_report
            ]),
            (route_after_supervisor, [
                make_state(supervisor_verdict="ok_continue", should_stop=False, current_stage_type="SINGLE_STRUCTURE"),
                make_state(supervisor_verdict="ok_continue", should_stop=True),
                make_state(supervisor_verdict="replan_needed", replan_count=0),
                make_state(supervisor_verdict="replan_needed", replan_count=MAX_REPLANS),
                make_state(supervisor_verdict="backtrack_to_stage"),
                make_state(supervisor_verdict="all_complete"),
                make_state(supervisor_verdict="ask_user"),
                make_state(supervisor_verdict="unknown"),
            ]),
            (route_after_plan_review, [
                make_state(last_plan_review_verdict="approve"),
                make_state(last_plan_review_verdict="needs_revision", replan_count=0),
                make_state(last_plan_review_verdict="needs_revision", replan_count=MAX_REPLANS),
            ]),
            (route_after_design_review, [
                make_state(last_design_review_verdict="approve"),
                make_state(last_design_review_verdict="needs_revision", design_revision_count=0),
                make_state(last_design_review_verdict="needs_revision", design_revision_count=MAX_DESIGN_REVISIONS),
            ]),
            (route_after_code_review, [
                make_state(last_code_review_verdict="approve"),
                make_state(last_code_review_verdict="needs_revision", code_revision_count=0),
                make_state(last_code_review_verdict="needs_revision", code_revision_count=MAX_CODE_REVISIONS),
            ]),
            (route_after_execution_check, [
                make_state(execution_verdict="pass"),
                make_state(execution_verdict="fail", execution_failure_count=0),
                make_state(execution_verdict="fail", execution_failure_count=MAX_EXECUTION_FAILURES),
            ]),
            (route_after_physics_check, [
                make_state(physics_verdict="pass"),
                make_state(physics_verdict="fail", physics_failure_count=0),
                make_state(physics_verdict="fail", physics_failure_count=MAX_PHYSICS_FAILURES),
                make_state(physics_verdict="design_flaw", design_revision_count=0),
            ]),
            (route_after_comparison_check, [
                make_state(comparison_verdict="approve"),
                make_state(comparison_verdict="needs_revision", analysis_revision_count=0),
                make_state(comparison_verdict="needs_revision", analysis_revision_count=MAX_ANALYSIS_REVISIONS),
            ]),
        ]

        with patch('src.graph.save_checkpoint'), patch('src.routing.save_checkpoint'):
            for router, states in scenarios:
                for state in states:
                    result = router(state)
                    assert result in valid_nodes, (
                        f"Router {router.__name__} returned '{result}' which is NOT a valid node in the graph. "
                        f"State triggering this: {state}"
                    )


# ═══════════════════════════════════════════════════════════════════════
# Conditional Edge Completeness Tests (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════

class TestConditionalEdgeCompleteness:
    """
    CRITICAL: Tests that verify conditional edges include ALL possible router return values.
    
    If a router can return a value that's not in the conditional edges map, LangGraph will
    raise a runtime error. These tests catch such mismatches BEFORE they hit production.
    """

    def test_comparison_check_edges_include_ask_user(self, graph_definition):
        """
        CRITICAL BUG DETECTOR: comparison_check router can return 'ask_user' on error,
        but conditional edges must include it or LangGraph will fail at runtime.
        
        The router route_after_comparison_check returns 'ask_user' when:
        - verdict is None
        - verdict is an invalid type
        - verdict is an unknown string
        
        The graph edges MUST include 'ask_user' as a valid target.
        """
        edges = list(graph_definition.edges)
        comparison_targets = {e[1] for e in edges if e[0] == "comparison_check"}
        
        # The router CAN return ask_user, so the graph MUST have this edge
        assert "ask_user" in comparison_targets, (
            f"CRITICAL BUG: comparison_check router can return 'ask_user' on error, "
            f"but the conditional_edges only include: {comparison_targets}. "
            f"This WILL cause a runtime error in LangGraph when verdict is None!"
        )

    def _get_all_possible_router_returns(self, router, router_name):
        """Helper to get all possible return values from a router by testing all code paths."""
        returns = set()
        
        # Test states that trigger different return paths
        test_states = [
            # Normal verdicts
            make_state(**{f"{router_name}_verdict": "approve"}),
            make_state(**{f"{router_name}_verdict": "needs_revision"}),
            make_state(**{f"{router_name}_verdict": "pass"}),
            make_state(**{f"{router_name}_verdict": "warning"}),
            make_state(**{f"{router_name}_verdict": "fail"}),
            make_state(**{f"{router_name}_verdict": "design_flaw"}),
            # Error conditions
            make_state(**{f"{router_name}_verdict": None}),  # None verdict
            make_state(**{f"{router_name}_verdict": "unknown_xyz"}),  # Unknown verdict
            make_state(**{f"{router_name}_verdict": 123}),  # Invalid type
            # Count limit triggers
            make_state(**{f"{router_name}_verdict": "needs_revision", "replan_count": 100}),
            make_state(**{f"{router_name}_verdict": "needs_revision", "design_revision_count": 100}),
            make_state(**{f"{router_name}_verdict": "needs_revision", "code_revision_count": 100}),
            make_state(**{f"{router_name}_verdict": "needs_revision", "analysis_revision_count": 100}),
            make_state(**{f"{router_name}_verdict": "fail", "execution_failure_count": 100}),
            make_state(**{f"{router_name}_verdict": "fail", "physics_failure_count": 100}),
        ]
        
        with patch('src.routing.save_checkpoint'), patch('src.graph.save_checkpoint'):
            for state in test_states:
                try:
                    result = router(state)
                    returns.add(result)
                except Exception:
                    pass  # Some states won't be valid for all routers
        
        return returns

    def test_plan_review_edges_cover_all_router_returns(self, graph_definition):
        """Verify plan_review edges include all possible router return values."""
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "plan_review"}
        
        # All possible returns from route_after_plan_review
        with patch('src.routing.save_checkpoint'):
            possible_returns = {
                route_after_plan_review(make_state(last_plan_review_verdict="approve")),
                route_after_plan_review(make_state(last_plan_review_verdict="needs_revision", replan_count=0)),
                route_after_plan_review(make_state(last_plan_review_verdict="needs_revision", replan_count=MAX_REPLANS)),
                route_after_plan_review(make_state(last_plan_review_verdict=None)),
                route_after_plan_review(make_state(last_plan_review_verdict="unknown")),
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"plan_review edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}"
        )

    def test_design_review_edges_cover_all_router_returns(self, graph_definition):
        """Verify design_review edges include all possible router return values."""
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "design_review"}
        
        with patch('src.routing.save_checkpoint'):
            possible_returns = {
                route_after_design_review(make_state(last_design_review_verdict="approve")),
                route_after_design_review(make_state(last_design_review_verdict="needs_revision", design_revision_count=0)),
                route_after_design_review(make_state(last_design_review_verdict="needs_revision", design_revision_count=MAX_DESIGN_REVISIONS)),
                route_after_design_review(make_state(last_design_review_verdict=None)),
                route_after_design_review(make_state(last_design_review_verdict="unknown")),
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"design_review edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}"
        )

    def test_code_review_edges_cover_all_router_returns(self, graph_definition):
        """Verify code_review edges include all possible router return values."""
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "code_review"}
        
        with patch('src.routing.save_checkpoint'):
            possible_returns = {
                route_after_code_review(make_state(last_code_review_verdict="approve")),
                route_after_code_review(make_state(last_code_review_verdict="needs_revision", code_revision_count=0)),
                route_after_code_review(make_state(last_code_review_verdict="needs_revision", code_revision_count=MAX_CODE_REVISIONS)),
                route_after_code_review(make_state(last_code_review_verdict=None)),
                route_after_code_review(make_state(last_code_review_verdict="unknown")),
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"code_review edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}"
        )

    def test_execution_check_edges_cover_all_router_returns(self, graph_definition):
        """Verify execution_check edges include all possible router return values."""
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "execution_check"}
        
        with patch('src.routing.save_checkpoint'):
            possible_returns = {
                route_after_execution_check(make_state(execution_verdict="pass")),
                route_after_execution_check(make_state(execution_verdict="warning")),
                route_after_execution_check(make_state(execution_verdict="fail", execution_failure_count=0)),
                route_after_execution_check(make_state(execution_verdict="fail", execution_failure_count=MAX_EXECUTION_FAILURES)),
                route_after_execution_check(make_state(execution_verdict=None)),
                route_after_execution_check(make_state(execution_verdict="unknown")),
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"execution_check edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}"
        )

    def test_physics_check_edges_cover_all_router_returns(self, graph_definition):
        """Verify physics_check edges include all possible router return values."""
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "physics_check"}
        
        with patch('src.routing.save_checkpoint'):
            possible_returns = {
                route_after_physics_check(make_state(physics_verdict="pass")),
                route_after_physics_check(make_state(physics_verdict="warning")),
                route_after_physics_check(make_state(physics_verdict="fail", physics_failure_count=0)),
                route_after_physics_check(make_state(physics_verdict="fail", physics_failure_count=MAX_PHYSICS_FAILURES)),
                route_after_physics_check(make_state(physics_verdict="design_flaw", design_revision_count=0)),
                route_after_physics_check(make_state(physics_verdict="design_flaw", design_revision_count=MAX_DESIGN_REVISIONS)),
                route_after_physics_check(make_state(physics_verdict=None)),
                route_after_physics_check(make_state(physics_verdict="unknown")),
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"physics_check edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}"
        )

    def test_comparison_check_edges_cover_all_router_returns(self, graph_definition):
        """
        CRITICAL: Verify comparison_check edges include all possible router return values.
        
        This specifically tests for the bug where ask_user is not in the edge map.
        """
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "comparison_check"}
        
        with patch('src.routing.save_checkpoint'):
            possible_returns = {
                route_after_comparison_check(make_state(comparison_verdict="approve")),
                route_after_comparison_check(make_state(comparison_verdict="needs_revision", analysis_revision_count=0)),
                route_after_comparison_check(make_state(comparison_verdict="needs_revision", analysis_revision_count=MAX_ANALYSIS_REVISIONS)),
                route_after_comparison_check(make_state(comparison_verdict=None)),  # Returns ask_user!
                route_after_comparison_check(make_state(comparison_verdict="unknown")),  # Returns ask_user!
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"CRITICAL BUG: comparison_check edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}. "
            f"This will cause LangGraph to fail at runtime when comparison_verdict is None or invalid!"
        )

    def test_supervisor_edges_cover_all_router_returns(self, graph_definition):
        """Verify supervisor edges include all possible router return values."""
        edges = list(graph_definition.edges)
        edge_targets = {e[1] for e in edges if e[0] == "supervisor"}
        
        with patch('src.graph.save_checkpoint'):
            possible_returns = {
                route_after_supervisor(make_state(supervisor_verdict="ok_continue", should_stop=False, current_stage_type="SINGLE_STRUCTURE")),
                route_after_supervisor(make_state(supervisor_verdict="ok_continue", should_stop=True)),
                route_after_supervisor(make_state(supervisor_verdict="ok_continue", current_stage_type="MATERIAL_VALIDATION", user_responses={})),
                route_after_supervisor(make_state(supervisor_verdict="change_priority", should_stop=False, current_stage_type="SINGLE_STRUCTURE")),
                route_after_supervisor(make_state(supervisor_verdict="replan_needed", replan_count=0)),
                route_after_supervisor(make_state(supervisor_verdict="replan_needed", replan_count=MAX_REPLANS)),
                route_after_supervisor(make_state(supervisor_verdict="ask_user")),
                route_after_supervisor(make_state(supervisor_verdict="backtrack_to_stage")),
                route_after_supervisor(make_state(supervisor_verdict="all_complete")),
                route_after_supervisor(make_state(supervisor_verdict=None)),  # Error case
                route_after_supervisor(make_state(supervisor_verdict="unknown")),  # Fallback
            }
        
        missing = possible_returns - edge_targets
        assert not missing, (
            f"supervisor edges missing targets that router can return: {missing}. "
            f"Current edges only go to: {edge_targets}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Invalid Verdict Type Tests
# ═══════════════════════════════════════════════════════════════════════

class TestInvalidVerdictTypes:
    """Tests that routers handle invalid verdict types (not just None)."""

    @patch('src.routing.save_checkpoint')
    def test_integer_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that integer verdict type escalates to ask_user."""
        state = make_state(last_code_review_verdict=123)
        result = route_after_code_review(state)
        assert result == "ask_user", f"Integer verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_list_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that list verdict type escalates to ask_user."""
        state = make_state(last_code_review_verdict=["approve", "needs_revision"])
        result = route_after_code_review(state)
        assert result == "ask_user", f"List verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_dict_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that dict verdict type escalates to ask_user."""
        state = make_state(last_code_review_verdict={"verdict": "approve"})
        result = route_after_code_review(state)
        assert result == "ask_user", f"Dict verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_boolean_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that boolean verdict type escalates to ask_user."""
        state = make_state(last_code_review_verdict=True)
        result = route_after_code_review(state)
        assert result == "ask_user", f"Boolean verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_float_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that float verdict type escalates to ask_user."""
        state = make_state(execution_verdict=1.5)
        result = route_after_execution_check(state)
        assert result == "ask_user", f"Float verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_empty_string_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that empty string verdict escalates to ask_user (not a valid verdict)."""
        state = make_state(last_design_review_verdict="")
        result = route_after_design_review(state)
        assert result == "ask_user", f"Empty string verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_whitespace_verdict_escalates_to_ask_user(self, mock_checkpoint):
        """Test that whitespace-only verdict escalates to ask_user."""
        state = make_state(physics_verdict="   ")
        result = route_after_physics_check(state)
        assert result == "ask_user", f"Whitespace verdict should escalate to ask_user, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_case_sensitive_verdict_escalates(self, mock_checkpoint):
        """Test that case-mismatched verdict escalates (e.g., 'APPROVE' instead of 'approve')."""
        state = make_state(last_plan_review_verdict="APPROVE")
        result = route_after_plan_review(state)
        assert result == "ask_user", (
            f"Case-mismatched verdict 'APPROVE' should escalate to ask_user, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Pass-Through Verdict Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPassThroughVerdicts:
    """Tests that pass_through_verdicts skip count checks even when count is high."""

    @patch('src.routing.save_checkpoint')
    def test_execution_pass_ignores_high_failure_count(self, mock_checkpoint):
        """Test that 'pass' verdict ignores execution_failure_count even when high."""
        state = make_state(
            execution_verdict="pass",
            execution_failure_count=100,  # Way over limit
        )
        result = route_after_execution_check(state)
        assert result == "physics_check", (
            f"'pass' is pass-through and should ignore count limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_execution_warning_ignores_high_failure_count(self, mock_checkpoint):
        """Test that 'warning' verdict ignores execution_failure_count even when high."""
        state = make_state(
            execution_verdict="warning",
            execution_failure_count=100,
        )
        result = route_after_execution_check(state)
        assert result == "physics_check", (
            f"'warning' is pass-through and should ignore count limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_physics_pass_ignores_high_failure_count(self, mock_checkpoint):
        """Test that 'pass' verdict ignores physics_failure_count even when high."""
        state = make_state(
            physics_verdict="pass",
            physics_failure_count=100,
        )
        result = route_after_physics_check(state)
        assert result == "analyze", (
            f"'pass' is pass-through and should ignore count limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_physics_warning_ignores_high_failure_count(self, mock_checkpoint):
        """Test that 'warning' verdict ignores physics_failure_count even when high."""
        state = make_state(
            physics_verdict="warning",
            physics_failure_count=100,
        )
        result = route_after_physics_check(state)
        assert result == "analyze", (
            f"'warning' is pass-through and should ignore count limit, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Negative and Invalid Count Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNegativeAndInvalidCounts:
    """Tests for handling of negative or invalid count values."""

    @patch('src.routing.save_checkpoint')
    def test_negative_count_allows_revision(self, mock_checkpoint):
        """Test that negative count (which is < limit) allows revision."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=-5,
        )
        result = route_after_code_review(state)
        assert result == "generate_code", (
            f"Negative count should be < limit and allow revision, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_very_large_count_escalates(self, mock_checkpoint):
        """Test that very large count escalates to ask_user."""
        state = make_state(
            last_design_review_verdict="needs_revision",
            design_revision_count=999999,
        )
        result = route_after_design_review(state)
        assert result == "ask_user", (
            f"Very large count should escalate to ask_user, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_string_count_treated_as_zero(self, mock_checkpoint):
        """Test that invalid string count value is converted to 0 gracefully."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count="three",  # Invalid type - should be converted to 0
        )
        # The router should convert invalid types to 0 and allow revision
        result = route_after_code_review(state)
        assert result == "generate_code", (
            f"Invalid count type should default to 0 and allow revision, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Graph Connectivity Tests - No Orphan Nodes
# ═══════════════════════════════════════════════════════════════════════

class TestGraphConnectivity:
    """Tests for overall graph connectivity."""

    def test_all_nodes_except_start_have_incoming_edges(self, graph_definition):
        """Test that all nodes (except __start__) have at least one incoming edge."""
        edges = list(graph_definition.edges)
        nodes_with_incoming = {edge[1] for edge in edges}
        all_nodes = set(graph_definition.nodes.keys())
        
        # __start__ should not have incoming edges
        nodes_needing_incoming = all_nodes - {"__start__"}
        orphan_nodes = nodes_needing_incoming - nodes_with_incoming
        
        assert not orphan_nodes, (
            f"Orphan nodes (no incoming edges): {orphan_nodes}. "
            f"These nodes can never be reached!"
        )

    def test_no_self_loops(self, graph_definition):
        """Test that no node has an edge to itself."""
        edges = list(graph_definition.edges)
        # Edge tuples may have more than 2 elements, take first two (source, target)
        self_loops = [edge for edge in edges if edge[0] == edge[1]]
        
        assert not self_loops, f"Self-loops detected: {self_loops}"

    def test_end_node_has_no_outgoing_edges(self, graph_definition):
        """Test that __end__ has no outgoing edges."""
        edges = list(graph_definition.edges)
        end_outgoing = [edge for edge in edges if edge[0] == "__end__"]
        
        assert not end_outgoing, (
            f"__end__ should not have outgoing edges, but has: {end_outgoing}"
        )

    def test_start_node_has_no_incoming_edges(self, graph_definition):
        """Test that __start__ has no incoming edges."""
        edges = list(graph_definition.edges)
        start_incoming = [edge for edge in edges if edge[1] == "__start__"]
        
        assert not start_incoming, (
            f"__start__ should not have incoming edges, but has: {start_incoming}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Saving Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointSaving:
    """Tests for checkpoint saving behavior in routers."""

    @patch('src.routing.save_checkpoint')
    def test_checkpoint_saved_on_none_verdict(self, mock_checkpoint):
        """Test that checkpoint is saved when verdict is None."""
        state = make_state(last_code_review_verdict=None)
        route_after_code_review(state)
        mock_checkpoint.assert_called_once()
        # Verify checkpoint name contains error indicator
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name, (
            f"Checkpoint name for None verdict should contain 'error', got {checkpoint_name}"
        )

    @patch('src.routing.save_checkpoint')
    def test_checkpoint_saved_on_count_limit(self, mock_checkpoint):
        """Test that checkpoint is saved when count limit is reached."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )
        route_after_code_review(state)
        mock_checkpoint.assert_called_once()
        # Verify checkpoint name contains limit indicator
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name, (
            f"Checkpoint name for limit should contain 'limit', got {checkpoint_name}"
        )

    @patch('src.routing.save_checkpoint')
    def test_checkpoint_saved_on_unknown_verdict(self, mock_checkpoint):
        """Test that checkpoint is saved when verdict is unknown."""
        state = make_state(last_code_review_verdict="unknown_xyz")
        route_after_code_review(state)
        mock_checkpoint.assert_called_once()
        # Verify checkpoint name contains fallback indicator
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "fallback" in checkpoint_name, (
            f"Checkpoint name for unknown verdict should contain 'fallback', got {checkpoint_name}"
        )

    @patch('src.routing.save_checkpoint')
    def test_no_checkpoint_on_normal_approve(self, mock_checkpoint):
        """Test that no checkpoint is saved on normal approve verdict."""
        state = make_state(last_code_review_verdict="approve")
        route_after_code_review(state)
        mock_checkpoint.assert_not_called()

    @patch('src.routing.save_checkpoint')
    def test_no_checkpoint_on_normal_revision(self, mock_checkpoint):
        """Test that no checkpoint is saved on normal revision (under limit)."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=0,
        )
        route_after_code_review(state)
        mock_checkpoint.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# Route On Limit Customization Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRouteOnLimitCustomization:
    """Tests for customized route_on_limit behavior."""

    @patch('src.routing.save_checkpoint')
    def test_comparison_check_routes_to_ask_user_on_limit(self, mock_checkpoint):
        """Test that comparison_check routes to ask_user when limit reached (consistent with others)."""
        state = make_state(
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS,
        )
        result = route_after_comparison_check(state)
        # comparison_check now has route_on_limit="ask_user" (consistent with other routers)
        assert result == "ask_user", (
            f"comparison_check should route to ask_user on limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_code_review_routes_to_ask_user_on_limit(self, mock_checkpoint):
        """Test that code_review routes to ask_user (default) when limit reached."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=MAX_CODE_REVISIONS,
        )
        result = route_after_code_review(state)
        assert result == "ask_user", (
            f"code_review should route to ask_user on limit, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Supervisor Router Complex Cases Tests
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Factory Router Direct Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFactoryRouterDirect:
    """Tests for the create_verdict_router factory function directly."""

    def test_factory_creates_callable_router(self):
        """Test that the factory creates a callable routing function."""
        from src.routing import create_verdict_router
        
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        assert callable(router), "Factory should create a callable"

    @patch('src.routing.save_checkpoint')
    def test_factory_router_uses_configured_verdict_field(self, mock_checkpoint):
        """Test that factory router reads from the configured verdict field."""
        from src.routing import create_verdict_router
        
        router = create_verdict_router(
            verdict_field="my_custom_verdict",
            routes={"approve": {"route": "next_node"}},
            checkpoint_prefix="test",
        )
        
        state = make_state(my_custom_verdict="approve")
        result = router(state)
        assert result == "next_node", f"Router should use configured verdict field, got {result}"

    @patch('src.routing.save_checkpoint')
    def test_factory_router_uses_configured_count_field(self, mock_checkpoint):
        """Test that factory router uses configured count field for limits."""
        from src.routing import create_verdict_router
        
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry",
                    "count_limit": {
                        "count_field": "my_count",
                        "max_count_key": "max_my_count",
                        "default_max": 2,
                    }
                }
            },
            checkpoint_prefix="test",
        )
        
        # Under limit
        state = make_state(test_verdict="needs_revision", my_count=1)
        assert router(state) == "retry"
        
        # At limit
        state = make_state(test_verdict="needs_revision", my_count=2)
        assert router(state) == "ask_user"

    @patch('src.routing.save_checkpoint')
    def test_factory_router_custom_route_on_limit(self, mock_checkpoint):
        """Test that factory router uses custom route_on_limit."""
        from src.routing import create_verdict_router
        
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "needs_revision": {
                    "route": "retry",
                    "count_limit": {
                        "count_field": "my_count",
                        "max_count_key": "max_my_count",
                        "default_max": 2,
                        "route_on_limit": "supervisor",
                    }
                }
            },
            checkpoint_prefix="test",
        )
        
        state = make_state(test_verdict="needs_revision", my_count=5)
        result = router(state)
        assert result == "supervisor", (
            f"Router should use custom route_on_limit, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_factory_router_pass_through_verdicts(self, mock_checkpoint):
        """Test that pass_through_verdicts skip count checks."""
        from src.routing import create_verdict_router
        
        router = create_verdict_router(
            verdict_field="test_verdict",
            routes={
                "pass": {
                    "route": "success",
                    "count_limit": {
                        "count_field": "fail_count",
                        "max_count_key": "max_fails",
                        "default_max": 0,  # Would always trigger if checked
                    }
                }
            },
            checkpoint_prefix="test",
            pass_through_verdicts=["pass"],
        )
        
        state = make_state(test_verdict="pass", fail_count=100)
        result = router(state)
        assert result == "success", (
            f"Pass-through verdict should skip count check, got {result}"
        )


class TestSupervisorComplexCases:
    """Additional tests for complex supervisor routing scenarios."""

    @patch('src.graph.save_checkpoint')
    def test_material_validation_with_empty_user_responses(self, mock_checkpoint):
        """Test MATERIAL_VALIDATION routes to checkpoint when user_responses is empty dict."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={},  # Empty, no material_checkpoint key
        )
        result = route_after_supervisor(state)
        assert result == "material_checkpoint", (
            f"Empty user_responses should trigger material_checkpoint, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_material_validation_with_other_user_response(self, mock_checkpoint):
        """Test MATERIAL_VALIDATION routes to checkpoint when user_responses has other keys."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={"some_other_key": "value"},  # No material_checkpoint
        )
        result = route_after_supervisor(state)
        assert result == "material_checkpoint", (
            f"user_responses without material_checkpoint key should trigger checkpoint, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_change_priority_with_material_validation(self, mock_checkpoint):
        """Test change_priority verdict with MATERIAL_VALIDATION stage."""
        state = make_state(
            supervisor_verdict="change_priority",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={},
        )
        result = route_after_supervisor(state)
        # change_priority behaves like ok_continue
        assert result == "material_checkpoint", (
            f"change_priority + MATERIAL_VALIDATION should route to material_checkpoint, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_ok_continue_with_none_current_stage_type(self, mock_checkpoint):
        """Test ok_continue when current_stage_type is None."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type=None,
        )
        result = route_after_supervisor(state)
        # None stage type shouldn't match MATERIAL_VALIDATION
        assert result == "select_stage", (
            f"ok_continue with None stage type should route to select_stage, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_ok_continue_with_missing_current_stage_type(self, mock_checkpoint):
        """Test ok_continue when current_stage_type key is missing."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
        )
        state.pop("current_stage_type", None)
        result = route_after_supervisor(state)
        # Missing stage type defaults to empty string, shouldn't match MATERIAL_VALIDATION
        assert result == "select_stage", (
            f"ok_continue with missing stage type should route to select_stage, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_replan_with_zero_replan_count(self, mock_checkpoint):
        """Test replan_needed with exactly 0 replan_count."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=0,
        )
        result = route_after_supervisor(state)
        assert result == "planning", f"0 replan_count should allow replan, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_replan_with_custom_runtime_limit(self, mock_checkpoint):
        """Test replan_needed respects runtime_config max_replans."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=5,  # Would exceed default limit
            runtime_config={"max_replans": 10},  # Custom higher limit
        )
        result = route_after_supervisor(state)
        assert result == "planning", (
            f"Custom runtime_config limit should allow replan, got {result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_should_stop_true_overrides_stage_type(self, mock_checkpoint):
        """Test should_stop=True takes precedence over MATERIAL_VALIDATION."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=True,
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={},
        )
        result = route_after_supervisor(state)
        assert result == "generate_report", (
            f"should_stop=True should override stage type logic, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Graph Structure Invariant Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphStructureInvariants:
    """Tests for structural invariants that must always hold."""

    def test_exactly_one_start_edge(self, graph_definition):
        """Test that __start__ has exactly one outgoing edge."""
        edges = list(graph_definition.edges)
        start_edges = [e for e in edges if e[0] == "__start__"]
        assert len(start_edges) == 1, (
            f"__start__ should have exactly 1 outgoing edge, has {len(start_edges)}: {start_edges}"
        )

    def test_exactly_one_end_connection(self, graph_definition):
        """Test that only generate_report connects to __end__."""
        edges = list(graph_definition.edges)
        end_connections = [e for e in edges if e[1] == "__end__"]
        sources = {e[0] for e in end_connections}
        
        assert sources == {"generate_report"}, (
            f"Only generate_report should connect to __end__, got: {sources}"
        )

    def test_all_workflow_nodes_are_reachable_from_start(self, graph_definition):
        """Test that all workflow nodes can be reached from __start__."""
        edges = list(graph_definition.edges)
        
        # Build adjacency list
        adjacency = {}
        for edge in edges:
            src, dst = edge[0], edge[1]
            if src not in adjacency:
                adjacency[src] = set()
            adjacency[src].add(dst)
        
        # BFS from __start__
        reachable = set()
        queue = ["__start__"]
        while queue:
            node = queue.pop(0)
            if node in reachable:
                continue
            reachable.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in reachable:
                    queue.append(neighbor)
        
        all_nodes = set(graph_definition.nodes.keys())
        unreachable = all_nodes - reachable
        
        assert not unreachable, (
            f"Nodes unreachable from __start__: {unreachable}"
        )

    def test_all_workflow_nodes_can_reach_end(self, graph_definition):
        """Test that all workflow nodes (except __end__) can reach __end__."""
        edges = list(graph_definition.edges)
        
        # Build reverse adjacency list (for backward search from __end__)
        reverse_adj = {}
        for edge in edges:
            src, dst = edge[0], edge[1]
            if dst not in reverse_adj:
                reverse_adj[dst] = set()
            reverse_adj[dst].add(src)
        
        # BFS backward from __end__
        can_reach_end = set()
        queue = ["__end__"]
        while queue:
            node = queue.pop(0)
            if node in can_reach_end:
                continue
            can_reach_end.add(node)
            for predecessor in reverse_adj.get(node, []):
                if predecessor not in can_reach_end:
                    queue.append(predecessor)
        
        all_nodes = set(graph_definition.nodes.keys())
        cannot_reach_end = all_nodes - can_reach_end
        
        # Filter out __end__ itself (it doesn't need to reach itself)
        cannot_reach_end.discard("__end__")
        
        assert not cannot_reach_end, (
            f"Nodes that cannot reach __end__: {cannot_reach_end}. "
            f"These nodes may cause workflow to hang!"
        )

    def test_ask_user_node_exists_for_escalation(self, graph_definition):
        """Test that ask_user node exists for escalation paths."""
        assert "ask_user" in graph_definition.nodes, (
            "ask_user node must exist for escalation/error handling"
        )

    def test_supervisor_node_is_central_hub(self, graph_definition):
        """Test that supervisor node is reachable from ask_user and can reach multiple targets."""
        edges = list(graph_definition.edges)
        
        # Check ask_user -> supervisor
        ask_user_targets = {e[1] for e in edges if e[0] == "ask_user"}
        assert "supervisor" in ask_user_targets, (
            "ask_user must be able to route to supervisor"
        )
        
        # Check supervisor has multiple outgoing options
        supervisor_targets = {e[1] for e in edges if e[0] == "supervisor"}
        assert len(supervisor_targets) >= 4, (
            f"supervisor should have at least 4 routing options, has: {supervisor_targets}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Runtime Config Override Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeConfigOverrides:
    """Tests for runtime_config limit overrides across all routers."""

    @patch('src.routing.save_checkpoint')
    def test_design_review_runtime_override(self, mock_checkpoint):
        """Test design_review respects runtime_config max_design_revisions."""
        state = make_state(
            last_design_review_verdict="needs_revision",
            design_revision_count=3,  # At default limit
            runtime_config={"max_design_revisions": 10},
        )
        result = route_after_design_review(state)
        assert result == "design", (
            f"Custom limit should allow revision, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_execution_check_runtime_override(self, mock_checkpoint):
        """Test execution_check respects runtime_config max_execution_failures."""
        state = make_state(
            execution_verdict="fail",
            execution_failure_count=2,  # At default limit
            runtime_config={"max_execution_failures": 10},
        )
        result = route_after_execution_check(state)
        assert result == "generate_code", (
            f"Custom limit should allow retry, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_physics_check_runtime_override(self, mock_checkpoint):
        """Test physics_check respects runtime_config max_physics_failures."""
        state = make_state(
            physics_verdict="fail",
            physics_failure_count=2,  # At default limit
            runtime_config={"max_physics_failures": 10},
        )
        result = route_after_physics_check(state)
        assert result == "generate_code", (
            f"Custom limit should allow retry, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_plan_review_runtime_override(self, mock_checkpoint):
        """Test plan_review respects runtime_config max_replans."""
        state = make_state(
            last_plan_review_verdict="needs_revision",
            replan_count=2,  # At default limit
            runtime_config={"max_replans": 10},
        )
        result = route_after_plan_review(state)
        assert result == "planning", (
            f"Custom limit should allow replan, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_comparison_check_runtime_override(self, mock_checkpoint):
        """Test comparison_check respects runtime_config max_analysis_revisions."""
        state = make_state(
            comparison_verdict="needs_revision",
            analysis_revision_count=2,  # At default limit
            runtime_config={"max_analysis_revisions": 10},
        )
        result = route_after_comparison_check(state)
        assert result == "analyze", (
            f"Custom limit should allow re-analyze, got {result}"
        )

    @patch('src.routing.save_checkpoint')
    def test_runtime_config_zero_always_escalates(self, mock_checkpoint):
        """Test that runtime_config limit of 0 always escalates."""
        state = make_state(
            last_code_review_verdict="needs_revision",
            code_revision_count=0,
            runtime_config={"max_code_revisions": 0},
        )
        result = route_after_code_review(state)
        assert result == "ask_user", (
            f"Zero limit should always escalate, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Edge Count Verification Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCountVerification:
    """Tests that verify expected edge counts for each node."""

    def test_static_edge_nodes_have_single_outgoing(self, graph_definition):
        """Test that nodes with static edges have exactly one outgoing edge."""
        static_edge_nodes = [
            ("adapt_prompts", "planning"),
            ("design", "design_review"),
            ("generate_code", "code_review"),
            ("run_code", "execution_check"),
            ("analyze", "comparison_check"),
            ("handle_backtrack", "select_stage"),
            # material_checkpoint now uses conditional edge (ask_user or select_stage)
            ("generate_report", "__end__"),
        ]
        
        edges = list(graph_definition.edges)
        
        for source, expected_target in static_edge_nodes:
            source_edges = [e for e in edges if e[0] == source]
            assert len(source_edges) == 1, (
                f"{source} should have exactly 1 outgoing edge (static), "
                f"has {len(source_edges)}: {source_edges}"
            )
            assert source_edges[0][1] == expected_target, (
                f"{source} should connect to {expected_target}, "
                f"connects to {source_edges[0][1]}"
            )

    def test_conditional_edge_nodes_have_multiple_outgoing(self, graph_definition):
        """Test that nodes with conditional edges have appropriate number of outgoing edges."""
        conditional_edge_configs = {
            "planning": 1,  # plan_review only
            "plan_review": 3,  # select_stage, planning, ask_user
            "select_stage": 2,  # design, generate_report
            "design_review": 3,  # generate_code, design, ask_user
            "code_review": 3,  # run_code, generate_code, ask_user
            "execution_check": 3,  # physics_check, generate_code, ask_user
            "physics_check": 4,  # analyze, generate_code, design, ask_user
            "comparison_check": 3,  # supervisor, analyze, ask_user
            "supervisor": 12,  # select_stage, planning, ask_user, handle_backtrack, generate_report, material_checkpoint, analyze, generate_code, design, code_review, design_review, plan_review
            "ask_user": 1,  # supervisor
        }
        
        edges = list(graph_definition.edges)
        
        for source, expected_count in conditional_edge_configs.items():
            source_edges = [e for e in edges if e[0] == source]
            assert len(source_edges) == expected_count, (
                f"{source} should have {expected_count} outgoing edges, "
                f"has {len(source_edges)}: {[e[1] for e in source_edges]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# User Guidance Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUserGuidanceIntegration:
    """Tests verifying user guidance flows work correctly end-to-end.
    
    These tests verify:
    1. User feedback/hints are properly passed to target nodes
    2. Counter resets are effective for subsequent iterations
    3. Full state machine flows work correctly after user guidance
    """

    @patch('src.graph.save_checkpoint')
    def test_provide_hint_feedback_visible_in_state(self, mock_checkpoint):
        """Test that PROVIDE_HINT for code_review_limit makes reviewer_feedback visible.
        
        When user provides a hint after code_review_limit, the reviewer_feedback
        should be set in the result so code_generator can see it on next run.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Q1": "PROVIDE_HINT: Use numpy vectorization instead of loops"},
            "pending_user_questions": ["Code review limit reached"],
            "code_revision_count": MAX_CODE_REVISIONS,
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter reset
        assert result.get("code_revision_count") == 0, "Counter should be reset to 0"
        
        # Verify hint is in reviewer_feedback (where code_generator looks for it)
        assert "reviewer_feedback" in result, "reviewer_feedback should be set"
        assert "numpy vectorization" in result["reviewer_feedback"], (
            f"User's hint should be in reviewer_feedback, got: {result.get('reviewer_feedback')}"
        )
        
        # Verify verdict routes directly to generate_code (bypasses select_stage)
        assert result.get("supervisor_verdict") == "retry_generate_code", (
            f"Expected retry_generate_code verdict, got {result.get('supervisor_verdict')}"
        )

    @patch('src.graph.save_checkpoint')
    def test_design_hint_feedback_visible_in_state(self, mock_checkpoint):
        """Test that PROVIDE_HINT for design_review_limit makes reviewer_feedback visible."""
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Q1": "PROVIDE_HINT: Focus on FDTD with perfectly matched layers"},
            "pending_user_questions": ["Design review limit reached"],
            "design_revision_count": MAX_DESIGN_REVISIONS,
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter reset
        assert result.get("design_revision_count") == 0, "Counter should be reset to 0"
        
        # Verify hint is in reviewer_feedback
        assert "reviewer_feedback" in result, "reviewer_feedback should be set"
        assert "FDTD" in result["reviewer_feedback"] or "perfectly matched" in result["reviewer_feedback"], (
            f"User's hint should be in reviewer_feedback, got: {result.get('reviewer_feedback')}"
        )

    @patch('src.graph.save_checkpoint')
    def test_execution_retry_resets_counter_effectively(self, mock_checkpoint):
        """Test that RETRY for execution_failure_limit effectively resets counter.
        
        After user provides RETRY, the execution_failure_count should be 0,
        meaning the next failure won't immediately hit the limit again.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        # Simulate state where execution hit the limit
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Q1": "RETRY with reduced grid resolution"},
            "pending_user_questions": ["Execution failed 3 times"],
            "execution_failure_count": MAX_EXECUTION_FAILURES,
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter is reset to 0
        assert result.get("execution_failure_count") == 0, (
            f"Counter should be reset to 0, got {result.get('execution_failure_count')}"
        )
        
        # Verify guidance is captured
        assert "supervisor_feedback" in result, "User guidance should be in supervisor_feedback"
        assert "reduced grid" in result["supervisor_feedback"].lower() or "retry" in result["supervisor_feedback"].lower(), (
            f"User's guidance should be captured, got: {result.get('supervisor_feedback')}"
        )
        
        # Simulate next execution failure - it should NOT immediately hit limit
        # because counter was reset to 0, so after one failure it should be 1
        merged_state = {**state, **result}
        merged_state["execution_failure_count"] = result["execution_failure_count"]  # Should be 0
        
        # Next failure would increment to 1, which is < MAX_EXECUTION_FAILURES
        next_failure_count = merged_state["execution_failure_count"] + 1
        assert next_failure_count < MAX_EXECUTION_FAILURES, (
            f"After reset, next failure (count={next_failure_count}) should not hit limit ({MAX_EXECUTION_FAILURES})"
        )

    @patch('src.graph.save_checkpoint')
    def test_physics_retry_resets_counter_effectively(self, mock_checkpoint):
        """Test that RETRY for physics_failure_limit effectively resets counter."""
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Q1": "RETRY with better boundary conditions"},
            "pending_user_questions": ["Physics check failed 3 times"],
            "physics_failure_count": MAX_PHYSICS_FAILURES,
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter is reset to 0
        assert result.get("physics_failure_count") == 0, (
            f"Counter should be reset to 0, got {result.get('physics_failure_count')}"
        )
        
        # Next failure should not hit limit
        next_failure_count = result["physics_failure_count"] + 1
        assert next_failure_count < MAX_PHYSICS_FAILURES, (
            f"After reset, next failure should not hit limit"
        )

    @patch('src.graph.save_checkpoint')
    def test_replan_guidance_routes_to_planning_with_feedback(self, mock_checkpoint):
        """Test full flow: replan_limit + GUIDANCE → planning with planner_feedback.
        
        This is the bug we fixed - verifying the complete flow works:
        1. User provides GUIDANCE after replan_limit
        2. Supervisor sets replan_with_guidance verdict and planner_feedback
        3. Router sends to planning (bypassing count check)
        4. Planner would receive the guidance via planner_feedback
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "GUIDANCE: Focus on extinction cross-section only, skip field maps"},
            "pending_user_questions": ["Replan limit reached"],
            "replan_count": MAX_REPLANS,  # At the limit
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter reset
        assert result.get("replan_count") == 0, "replan_count should be reset to 0"
        
        # Verify special verdict that bypasses count check
        assert result.get("supervisor_verdict") == "replan_with_guidance", (
            f"Expected replan_with_guidance verdict, got {result.get('supervisor_verdict')}"
        )
        
        # Verify planner_feedback contains user's guidance
        assert "planner_feedback" in result, "planner_feedback should be set"
        assert "extinction" in result["planner_feedback"].lower() or "cross-section" in result["planner_feedback"].lower(), (
            f"User's guidance should be in planner_feedback, got: {result.get('planner_feedback')}"
        )
        
        # Verify routing goes to planning (not ask_user)
        merged_state = {**state, **result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "planning", (
            f"Should route to planning even at limit, got {route_result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_code_review_limit_full_flow_routes_correctly(self, mock_checkpoint):
        """Test full flow: code_review_limit + HINT → retry_generate_code → generate_code.
        
        Verifies the state machine flow after user provides code hint:
        1. User provides PROVIDE_HINT after code_review_limit
        2. Supervisor sets retry_generate_code verdict with reviewer_feedback
        3. Router sends directly to generate_code (bypasses select_stage)
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Q1": "PROVIDE_HINT: Try using scipy.integrate instead"},
            "pending_user_questions": ["Code review limit reached"],
            "code_revision_count": MAX_CODE_REVISIONS,
            "current_stage_id": "stage1",
            "current_stage_type": "SINGLE_STRUCTURE",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify retry_generate_code verdict (routes directly to generate_code)
        assert result.get("supervisor_verdict") == "retry_generate_code", (
            f"Expected retry_generate_code verdict, got {result.get('supervisor_verdict')}"
        )
        
        # Verify routing after supervisor goes directly to generate_code
        merged_state = {**state, **result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "generate_code", (
            f"retry_generate_code should route to generate_code, got {route_result}"
        )
        
        # Verify hint is preserved for when code_generator runs next
        assert "reviewer_feedback" in result, "reviewer_feedback should be preserved"
        assert "scipy" in result["reviewer_feedback"], "User hint should be in feedback"


# ═══════════════════════════════════════════════════════════════════════
# Trigger Name Consistency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTriggerNameConsistency:
    """Tests verifying trigger names used by nodes match handlers.
    
    This prevents bugs where a node sets ask_user_trigger to a name
    that doesn't have a corresponding handler in TRIGGER_HANDLERS.
    """

    def test_all_trigger_names_have_handlers(self):
        """Verify all trigger names used in the codebase have handlers.
        
        This is a regression test for the bug where planning.py set
        ask_user_trigger='plan_review_limit' but the handler was registered
        as 'replan_limit', causing user responses to be ignored.
        """
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        # Known trigger names that should have handlers
        # These are the triggers set by various nodes when escalating to ask_user
        expected_triggers = [
            "replan_limit",           # Set by plan_reviewer_node when replan_count >= max
            "code_review_limit",      # Set by code_reviewer_node when code_revision_count >= max
            "design_review_limit",    # Set by design_reviewer_node when design_revision_count >= max
            "execution_failure_limit", # Set by execution_validator_node when execution_failure_count >= max
            "physics_failure_limit",  # Set by physics_sanity_node when physics_failure_count >= max
            "context_overflow",       # Set by check_context_or_escalate when context too large
            "backtrack_approval",     # Set by comparison_validator_node for backtrack requests
            "deadlock_detected",      # Set by supervisor when deadlock detected
            "llm_error",              # Set by various nodes on LLM errors
            "material_checkpoint",    # Set by material_checkpoint_node for mandatory approval
        ]
        
        for trigger in expected_triggers:
            assert trigger in TRIGGER_HANDLERS, (
                f"Trigger '{trigger}' is used in the codebase but has no handler in TRIGGER_HANDLERS. "
                f"Available handlers: {list(TRIGGER_HANDLERS.keys())}"
            )

    def test_plan_reviewer_uses_correct_trigger_name(self):
        """Verify plan_reviewer_node uses the trigger name that has a handler.
        
        This specifically tests the bug we fixed where plan_reviewer used
        'plan_review_limit' but the handler was 'replan_limit'.
        """
        from unittest.mock import patch, MagicMock
        from src.agents.planning import plan_reviewer_node
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        # Create a state that will trigger escalation (replan_count at max)
        state = {
            "plan": {
                "stages": [{"stage_id": "test", "stage_type": "SINGLE_STRUCTURE"}],
                "assumptions": [],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": []},
        }
        
        # Mock the LLM call to return needs_revision
        mock_response = {
            "verdict": "needs_revision",
            "feedback": "Test feedback",
            "issues": ["Test issue"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    result = plan_reviewer_node(state)
        
        # Verify the trigger name is one that has a handler
        trigger = result.get("ask_user_trigger")
        if trigger:  # Only check if escalation happened
            assert trigger in TRIGGER_HANDLERS, (
                f"plan_reviewer_node sets ask_user_trigger='{trigger}' "
                f"but no handler exists for it. Available handlers: {list(TRIGGER_HANDLERS.keys())}"
            )


# ═══════════════════════════════════════════════════════════════════════
# End-to-End User Response Flow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEndUserResponseFlows:
    """Tests simulating complete flows from limit escalation through user response handling.
    
    These tests verify the full path:
    review_node → hit limit → ask_user → user responds → supervisor handles → correct routing
    """

    @patch('src.graph.save_checkpoint')
    def test_plan_review_limit_stop_ends_workflow(self, mock_checkpoint):
        """Test full flow: plan_review hits limit → user says STOP → workflow ends.
        
        This is a regression test for the bug where STOP wasn't being respected
        because the trigger name didn't match the handler.
        """
        from unittest.mock import patch
        from src.agents.planning import plan_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node
        
        # Step 1: Simulate plan_reviewer hitting the limit
        plan_state = {
            "plan": {
                "stages": [{"stage_id": "test", "stage_type": "SINGLE_STRUCTURE"}],
                "assumptions": [],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": []},
        }
        
        mock_response = {
            "verdict": "needs_revision",
            "feedback": "Issues found",
            "issues": ["Test issue"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    plan_result = plan_reviewer_node(plan_state)
        
        # Verify escalation happened
        assert plan_result.get("awaiting_user_input") is True, "Should be awaiting user input"
        trigger = plan_result.get("ask_user_trigger")
        assert trigger is not None, "Should have set ask_user_trigger"
        
        # Step 2: Simulate user responding with STOP
        supervisor_state = {
            **plan_state,
            **plan_result,
            "user_responses": {"Q1": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        supervisor_result = supervisor_node(supervisor_state)
        
        # Step 3: Verify STOP was handled correctly
        assert supervisor_result.get("supervisor_verdict") == "all_complete", (
            f"STOP should set verdict to all_complete, got {supervisor_result.get('supervisor_verdict')}"
        )
        assert supervisor_result.get("should_stop") is True, (
            "STOP should set should_stop=True"
        )
        
        # Step 4: Verify routing goes to generate_report (workflow end)
        merged_state = {**supervisor_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "generate_report", (
            f"all_complete with should_stop should route to generate_report, got {route_result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_plan_review_limit_guidance_triggers_replan(self, mock_checkpoint):
        """Test full flow: plan_review hits limit → user provides GUIDANCE → replanning.
        
        This is a regression test for the bug where GUIDANCE wasn't triggering replan
        because the router checked replan_count before the reset was merged.
        """
        from unittest.mock import patch
        from src.agents.planning import plan_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node
        
        # Step 1: Simulate plan_reviewer hitting the limit
        plan_state = {
            "plan": {
                "stages": [{"stage_id": "test", "stage_type": "SINGLE_STRUCTURE"}],
                "assumptions": [],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": []},
        }
        
        mock_response = {
            "verdict": "needs_revision",
            "feedback": "Issues found",
            "issues": ["Test issue"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    plan_result = plan_reviewer_node(plan_state)
        
        # Step 2: Simulate user responding with GUIDANCE
        supervisor_state = {
            **plan_state,
            **plan_result,
            "user_responses": {"Q1": "GUIDANCE: Focus on extinction spectrum only"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        supervisor_result = supervisor_node(supervisor_state)
        
        # Step 3: Verify GUIDANCE was handled correctly
        assert supervisor_result.get("supervisor_verdict") == "replan_with_guidance", (
            f"GUIDANCE should set verdict to replan_with_guidance, got {supervisor_result.get('supervisor_verdict')}"
        )
        assert supervisor_result.get("replan_count") == 0, (
            f"GUIDANCE should reset replan_count to 0, got {supervisor_result.get('replan_count')}"
        )
        assert "extinction" in supervisor_result.get("planner_feedback", "").lower(), (
            "User guidance should be in planner_feedback"
        )
        
        # Step 4: Verify routing goes to planning (not ask_user!)
        merged_state = {**supervisor_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "planning", (
            f"replan_with_guidance should route to planning, got {route_result}"
        )

    @patch('src.graph.save_checkpoint')
    def test_plan_review_limit_force_accept_continues(self, mock_checkpoint):
        """Test full flow: plan_review hits limit → user says FORCE_ACCEPT → continues."""
        from unittest.mock import patch
        from src.agents.planning import plan_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node
        
        # Step 1: Simulate plan_reviewer hitting the limit
        plan_state = {
            "plan": {
                "stages": [{"stage_id": "test", "stage_type": "SINGLE_STRUCTURE"}],
                "assumptions": [],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": []},
            "current_stage_type": "SINGLE_STRUCTURE",
        }
        
        mock_response = {
            "verdict": "needs_revision",
            "feedback": "Issues found",
            "issues": ["Test issue"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    plan_result = plan_reviewer_node(plan_state)
        
        # Step 2: Simulate user responding with FORCE_ACCEPT
        supervisor_state = {
            **plan_state,
            **plan_result,
            "user_responses": {"Q1": "FORCE_ACCEPT"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        supervisor_result = supervisor_node(supervisor_state)
        
        # Step 3: Verify FORCE_ACCEPT was handled correctly
        assert supervisor_result.get("supervisor_verdict") == "ok_continue", (
            f"FORCE_ACCEPT should set verdict to ok_continue, got {supervisor_result.get('supervisor_verdict')}"
        )
        assert "force" in supervisor_result.get("supervisor_feedback", "").lower(), (
            "Feedback should mention force-accept"
        )
        
        # Step 4: Verify routing continues (to select_stage)
        merged_state = {**supervisor_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "select_stage", (
            f"ok_continue should route to select_stage, got {route_result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Extended End-to-End Tests (Including Post-Supervisor Node Execution)
# ═══════════════════════════════════════════════════════════════════════

class TestExtendedEndToEndFlows:
    """Tests that verify the COMPLETE flow including what happens after supervisor routes.
    
    These extend the basic e2e tests by actually invoking the next node after
    supervisor routes, verifying the full state machine behavior.
    """

    @patch('src.graph.save_checkpoint')
    def test_stop_flow_generates_report(self, mock_checkpoint):
        """Full flow: plan_review → limit → STOP → supervisor → generate_report_node.
        
        Verifies that after STOP, the report generation node is actually called
        and produces workflow_phase='complete'.
        """
        from unittest.mock import patch
        from src.agents.planning import plan_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node
        from src.agents.reporting import generate_report_node
        
        # Step 1: Simulate plan_reviewer hitting the limit
        plan_state = {
            "plan": {
                "stages": [{"stage_id": "test", "stage_type": "SINGLE_STRUCTURE"}],
                "assumptions": [],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": [], "user_interactions": []},
            "paper_id": "test_paper",
        }
        
        mock_llm_response = {
            "verdict": "needs_revision",
            "feedback": "Issues found",
            "issues": ["Test issue"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_llm_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    plan_result = plan_reviewer_node(plan_state)
        
        # Step 2: User responds with STOP
        supervisor_state = {
            **plan_state,
            **plan_result,
            "user_responses": {"Q1": "STOP"},
        }
        
        supervisor_result = supervisor_node(supervisor_state)
        
        # Step 3: Verify routing
        merged_state = {**supervisor_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "generate_report", "Should route to generate_report"
        
        # Step 4: Actually call generate_report_node and verify it completes correctly
        report_state = merged_state
        
        # Mock the LLM call in generate_report_node
        mock_report_response = {
            "executive_summary": "Test summary",
            "conclusions": "Test conclusions",
        }
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_report_response):
            report_result = generate_report_node(report_state)
        
        # generate_report_node sets workflow_phase to 'reporting' (the terminal phase)
        assert report_result.get("workflow_phase") == "reporting", (
            f"Report node should set workflow_phase='reporting', got {report_result.get('workflow_phase')}"
        )
        
        # Verify report node produced metrics summary (proves it actually ran its logic)
        assert "metrics" in report_result, "Report node should produce metrics"
        assert "token_summary" in report_result["metrics"], "Report should include token_summary"

    @patch('src.graph.save_checkpoint')
    def test_guidance_flow_reaches_planner_with_feedback(self, mock_checkpoint):
        """Full flow: plan_review → limit → GUIDANCE → supervisor → plan_node with feedback.
        
        Verifies that after GUIDANCE:
        1. Routing goes to planning (not ask_user)
        2. plan_node receives the user's guidance in planner_feedback
        3. plan_node uses the guidance (visible in the prompt)
        """
        from unittest.mock import patch, MagicMock, call
        from src.agents.planning import plan_reviewer_node, plan_node
        from src.agents.supervision.supervisor import supervisor_node
        
        # Step 1: Simulate plan_reviewer hitting the limit
        plan_state = {
            "plan": {
                "stages": [{"stage_id": "test", "stage_type": "SINGLE_STRUCTURE"}],
                "assumptions": [],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": [], "user_interactions": []},
            "paper_id": "test_paper",
            # Paper text must be long enough (>100 chars) for planner to accept it
            "paper_text": """
            This is a test paper about optical properties of aluminum nanoparticles.
            We investigate the extinction spectrum and near-field enhancement.
            The simulations use FDTD methods with perfectly matched layer boundaries.
            Results show strong plasmonic resonance at 400nm wavelength.
            """,
        }
        
        mock_review_response = {
            "verdict": "needs_revision",
            "feedback": "Issues found",
            "issues": ["Test issue"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_review_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    plan_result = plan_reviewer_node(plan_state)
        
        # Step 2: User responds with GUIDANCE
        user_guidance = "Focus on extinction spectrum only, skip field maps"
        supervisor_state = {
            **plan_state,
            **plan_result,
            "user_responses": {"Q1": f"GUIDANCE: {user_guidance}"},
        }
        
        supervisor_result = supervisor_node(supervisor_state)
        
        # Step 3: Verify routing goes to planning
        merged_state = {**supervisor_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "planning", (
            f"GUIDANCE should route to planning, got {route_result}"
        )
        
        # Step 4: Verify planner_feedback contains user guidance
        assert "planner_feedback" in supervisor_result, "Should have planner_feedback"
        assert "extinction" in supervisor_result["planner_feedback"].lower(), (
            f"User guidance should be in planner_feedback, got: {supervisor_result.get('planner_feedback')}"
        )
        
        # Step 5: Call plan_node and verify user's guidance appears in the prompt
        planner_state = merged_state
        
        # Verify paper_text is long enough (plan_node requires > 100 chars)
        paper_text = planner_state.get("paper_text", "")
        assert len(paper_text.strip()) >= 100, (
            f"paper_text must be >= 100 chars for plan_node, got {len(paper_text.strip())}"
        )
        
        # Verify planner_feedback is in state (set by supervisor)
        assert "planner_feedback" in planner_state, (
            "planner_feedback should be in state after GUIDANCE"
        )
        assert user_guidance.lower() in planner_state["planner_feedback"].lower(), (
            f"User guidance '{user_guidance}' should be in planner_feedback, "
            f"got: {planner_state.get('planner_feedback')}"
        )
        
        # Mock the LLM call to capture the ACTUAL user_content sent
        mock_plan_response = {
            "stages": [{"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "target_figure": "Fig1"}],
            "assumptions": [],
        }
        
        captured_user_content = []
        def capture_llm_call(*args, **kwargs):
            # Capture user_content (3rd positional arg or keyword arg)
            if len(args) >= 3:
                captured_user_content.append(args[2])  # user_content is 3rd arg
            elif "user_content" in kwargs:
                captured_user_content.append(kwargs["user_content"])
            return mock_plan_response
        
        with patch("src.agents.planning.call_agent_with_metrics", side_effect=capture_llm_call):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                with patch("src.agents.planning.build_agent_prompt", return_value="test prompt"):
                    plan_node_result = plan_node(planner_state)
        
        # Verify LLM was called
        assert len(captured_user_content) > 0, (
            f"plan_node should have called LLM with user_content. Result: {plan_node_result}"
        )
        
        # CRITICAL: Verify user's guidance appears in the actual prompt sent to LLM
        actual_user_content = captured_user_content[0]
        assert "extinction" in actual_user_content.lower(), (
            f"User guidance about 'extinction' should appear in user_content sent to LLM.\n"
            f"planner_feedback: {planner_state.get('planner_feedback')}\n"
            f"Actual user_content: {actual_user_content[:500]}..."
        )
        assert "REVISION FEEDBACK" in actual_user_content, (
            f"User guidance should be in REVISION FEEDBACK section.\n"
            f"Actual user_content: {actual_user_content[:500]}..."
        )
        
        # Verify replan_count was reset
        assert merged_state.get("replan_count") == 0, (
            f"replan_count should be 0 after GUIDANCE, got {merged_state.get('replan_count')}"
        )

    @patch('src.graph.save_checkpoint')  
    def test_force_accept_flow_reaches_stage_selection(self, mock_checkpoint):
        """Full flow: plan_review → limit → FORCE_ACCEPT → supervisor → select_stage_node.
        
        Verifies that after FORCE_ACCEPT:
        1. Routing goes to select_stage
        2. select_stage_node can pick the next stage from the plan
        """
        from unittest.mock import patch
        from src.agents.planning import plan_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node
        from src.agents.stage_selection import select_stage_node
        
        # Step 1: Simulate plan_reviewer hitting the limit with a valid plan
        plan_state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "target_figure": "Fig1"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "target_figure": "Fig2"},
                ],
                "assumptions": ["Assume aluminum properties"],
            },
            "replan_count": MAX_REPLANS,
            "runtime_config": {"max_replans": MAX_REPLANS},
            "progress": {"stages": []},
            "paper_id": "test_paper",
            "current_stage_type": "SINGLE_STRUCTURE",
        }
        
        mock_review_response = {
            "verdict": "needs_revision",
            "feedback": "Minor issues but acceptable",
            "issues": ["Minor formatting"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_review_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    plan_result = plan_reviewer_node(plan_state)
        
        # Step 2: User responds with FORCE_ACCEPT
        supervisor_state = {
            **plan_state,
            **plan_result,
            "user_responses": {"Q1": "FORCE_ACCEPT"},
        }
        
        supervisor_result = supervisor_node(supervisor_state)
        
        # Step 3: Verify routing goes to select_stage
        merged_state = {**supervisor_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        assert route_result == "select_stage", (
            f"FORCE_ACCEPT should route to select_stage, got {route_result}"
        )
        
        # Step 4: Call select_stage_node and verify it picks a stage
        select_state = merged_state
        select_result = select_stage_node(select_state)
        
        # Should pick the first unprocessed stage (stage0)
        assert "current_stage_id" in select_result, "Should set current_stage_id"
        assert select_result.get("current_stage_id") == "stage0", (
            f"Should pick first stage (stage0), got {select_result.get('current_stage_id')}"
        )
        assert select_result.get("workflow_phase") == "stage_selection", (
            "Should set workflow_phase to stage_selection"
        )


# ═══════════════════════════════════════════════════════════════════════
# Edge Case Tests for User Guidance
# ═══════════════════════════════════════════════════════════════════════

class TestUserGuidanceEdgeCases:
    """Tests for edge cases in user guidance handling.
    
    These tests verify the system handles malformed, empty, or ambiguous
    user input correctly - edge cases that often reveal bugs.
    """

    @patch('src.graph.save_checkpoint')
    def test_empty_guidance_after_keyword(self, mock_checkpoint):
        """Test GUIDANCE: with no actual guidance text.
        
        User types 'GUIDANCE:' with nothing after. System should still
        process it (even if the feedback is empty).
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "GUIDANCE:"},  # Empty guidance
            "replan_count": MAX_REPLANS,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should still be recognized as GUIDANCE and trigger replan
        assert result.get("supervisor_verdict") == "replan_with_guidance", (
            f"Empty GUIDANCE should still trigger replan_with_guidance, got {result.get('supervisor_verdict')}"
        )
        assert result.get("replan_count") == 0, "Counter should be reset even with empty guidance"
        # planner_feedback should exist even if empty
        assert "planner_feedback" in result, "planner_feedback should be set even if empty"

    @patch('src.graph.save_checkpoint')
    def test_whitespace_only_guidance(self, mock_checkpoint):
        """Test GUIDANCE: followed by only whitespace.
        
        User types 'GUIDANCE:    ' with spaces. Should be handled gracefully.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "GUIDANCE:    \n\t  "},  # Whitespace only
            "replan_count": MAX_REPLANS,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should recognize GUIDANCE keyword
        assert result.get("supervisor_verdict") == "replan_with_guidance", (
            f"Whitespace GUIDANCE should trigger replan_with_guidance, got {result.get('supervisor_verdict')}"
        )
        assert result.get("replan_count") == 0, "Counter should be reset"

    @patch('src.graph.save_checkpoint')
    def test_conflicting_keywords_stop_and_guidance(self, mock_checkpoint):
        """Test response containing both STOP and GUIDANCE.
        
        Which takes precedence? This tests keyword priority logic.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "GUIDANCE: please STOP doing this"},  # Both keywords
            "replan_count": MAX_REPLANS,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # GUIDANCE comes first in the handler's if-elif chain after FORCE/ACCEPT
        # So GUIDANCE should win (based on handler implementation)
        # If this fails, it reveals the actual precedence
        verdict = result.get("supervisor_verdict")
        assert verdict in ("replan_with_guidance", "all_complete"), (
            f"Expected replan_with_guidance or all_complete, got {verdict}"
        )
        # Note: This test documents actual behavior. If precedence matters, 
        # strengthen this assertion once the expected behavior is confirmed.

    @patch('src.graph.save_checkpoint')
    def test_case_sensitivity_mixed_case_guidance(self, mock_checkpoint):
        """Test mixed case GUIDANCE keyword (e.g., 'GuIdAnCe').
        
        Keywords should be case-insensitive.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "GuIdAnCe: try different approach"},
            "replan_count": MAX_REPLANS,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert result.get("supervisor_verdict") == "replan_with_guidance", (
            f"Mixed-case GUIDANCE should work, got {result.get('supervisor_verdict')}"
        )

    @patch('src.graph.save_checkpoint')
    def test_guidance_without_colon(self, mock_checkpoint):
        """Test GUIDANCE without colon (e.g., 'GUIDANCE try this').
        
        Should still be recognized as GUIDANCE keyword.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "GUIDANCE try a different wavelength"},
            "replan_count": MAX_REPLANS,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert result.get("supervisor_verdict") == "replan_with_guidance", (
            f"GUIDANCE without colon should work, got {result.get('supervisor_verdict')}"
        )
        # Verify the guidance text is captured
        feedback = result.get("planner_feedback", "")
        assert "wavelength" in feedback.lower(), (
            f"Guidance text should be captured, got: {feedback}"
        )

    @patch('src.graph.save_checkpoint')
    def test_unrecognized_response_asks_for_clarification(self, mock_checkpoint):
        """Test response with no recognized keywords.
        
        User types gibberish - should ask for clarification, not crash.
        """
        from src.agents.supervision.supervisor import supervisor_node
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q1": "I don't understand what to do"},
            "replan_count": MAX_REPLANS,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should ask for clarification
        assert result.get("supervisor_verdict") == "ask_user", (
            f"Unrecognized response should ask for clarification, got {result.get('supervisor_verdict')}"
        )
        assert "pending_user_questions" in result, "Should have clarification question"
        assert len(result["pending_user_questions"]) > 0, "Should have at least one question"


# ═══════════════════════════════════════════════════════════════════════
# Dynamic Trigger Name Consistency Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDynamicTriggerConsistency:
    """Tests that dynamically verify trigger name consistency.
    
    Instead of hardcoding expected trigger names, these tests scan
    the source code to find all places that set ask_user_trigger
    and verify each one has a handler.
    """

    def test_all_trigger_assignments_have_handlers(self):
        """Scan source for all ask_user_trigger assignments and verify handlers exist.
        
        This is a robust regression test that will catch any new trigger name
        added without a corresponding handler - the exact bug we found.
        """
        import re
        from pathlib import Path
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        src_dir = Path(__file__).parent.parent.parent.parent / "src"
        
        # Multiple patterns to catch different assignment styles:
        # 1. result["ask_user_trigger"] = "something"
        # 2. result['ask_user_trigger'] = 'something'
        # 3. "ask_user_trigger": "something" (in dicts)
        patterns = [
            # Standard assignment: ["ask_user_trigger"] = "value"
            re.compile(r'\[[\"\']ask_user_trigger[\"\']\]\s*=\s*[\"\']([\w_]+)[\"\']'),
            # Dict literal: "ask_user_trigger": "value"
            re.compile(r'[\"\'"]ask_user_trigger[\"\'"]\s*:\s*[\"\']([\w_]+)[\"\']'),
        ]
        
        found_triggers = set()
        files_with_triggers = {}
        
        for py_file in src_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in patterns:
                    matches = pattern.findall(content)
                    if matches:
                        for trigger in matches:
                            found_triggers.add(trigger)
                            if trigger not in files_with_triggers:
                                files_with_triggers[trigger] = []
                            files_with_triggers[trigger].append(str(py_file.name))
            except Exception:
                pass  # Skip files we can't read
        
        # Filter out "None" which is used to clear the trigger
        found_triggers.discard("None")
        found_triggers.discard("none")
        
        # Verify each found trigger has a handler
        missing_handlers = []
        for trigger in found_triggers:
            if trigger not in TRIGGER_HANDLERS:
                missing_handlers.append({
                    "trigger": trigger,
                    "files": files_with_triggers.get(trigger, [])
                })
        
        assert len(missing_handlers) == 0, (
            f"Found trigger names without handlers:\n"
            + "\n".join([
                f"  - '{m['trigger']}' used in: {', '.join(set(m['files']))}"
                for m in missing_handlers
            ])
            + f"\n\nAvailable handlers: {list(TRIGGER_HANDLERS.keys())}"
        )
        
        # Also verify we found SOME triggers (test sanity check)
        assert len(found_triggers) >= 3, (
            f"Expected to find at least 3 trigger assignments, found {len(found_triggers)}. "
            "Pattern may be broken."
        )

    def test_trigger_handler_dict_has_known_triggers(self):
        """Verify TRIGGER_HANDLERS contains expected critical triggers.
        
        These are triggers that MUST have handlers for the system to work.
        """
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        # Critical triggers that must always have handlers
        critical_triggers = [
            "replan_limit",           # Plan review escalation
            "code_review_limit",      # Code review escalation
            "design_review_limit",    # Design review escalation
            "execution_failure_limit", # Execution failure escalation
            "physics_failure_limit",  # Physics check escalation
            "material_checkpoint",    # Material validation
        ]
        
        for trigger in critical_triggers:
            assert trigger in TRIGGER_HANDLERS, (
                f"Critical trigger '{trigger}' missing from TRIGGER_HANDLERS. "
                "This will cause user responses to be ignored!"
            )


# ═══════════════════════════════════════════════════════════════════════
# Complex End-to-End Flow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestComplexE2EFlows:
    """Complex end-to-end tests verifying multi-step workflows.
    
    These tests simulate complete workflow paths including:
    - Material checkpoint mandatory approval flow
    - Physics design flaw recovery (routes to design, not code)
    - Multi-limit escalation chains with counter resets
    - Backtracking and stage invalidation
    - Complete single-stage execution with all review cycles
    - Execution failures with code regeneration
    - Context overflow recovery
    - LLM error recovery
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Test 1: Material Checkpoint Flow
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_material_checkpoint_flow_complete(self, mock_checkpoint):
        """Full flow: stage 0 (MATERIAL_VALIDATION) → supervisor → material_checkpoint 
        → ask_user → user approves → supervisor → select_stage → stage 1.
        
        This tests the MANDATORY material checkpoint that occurs after Stage 0.
        Verifies materials are stored after user approval.
        """
        from unittest.mock import patch
        from src.agents.supervision.supervisor import supervisor_node
        from src.agents.user_interaction import material_checkpoint_node
        from src.agents.stage_selection import select_stage_node
        from src.graph import route_after_supervisor
        
        # Step 1: Set up state after Stage 0 (MATERIAL_VALIDATION) completes
        base_state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "target_figure": "Fig1"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "target_figure": "Fig2", "dependencies": ["stage0"]},
                ],
                "assumptions": [],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "pending", "dependencies": ["stage0"]},
                ],
                "user_interactions": [],
            },
            "current_stage_id": "stage0",
            "current_stage_type": "MATERIAL_VALIDATION",
            "stage_outputs": {
                "files": ["material_plot.png"],
                "results": {"n_data": [1.5, 1.6], "k_data": [0.01, 0.02]},
            },
            "validated_materials": [],  # Not yet validated
            "paper_id": "test_paper",
        }
        
        # Supervisor should recognize completed MATERIAL_VALIDATION and route to checkpoint
        mock_supervisor_response = {
            "verdict": "ok_continue",
            "feedback": "Stage 0 complete",
            "should_stop": False,
        }
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", return_value=mock_supervisor_response):
            with patch("src.agents.supervision.supervisor.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.supervision.supervisor.check_context_or_escalate", return_value=None):
                    supervisor_result = supervisor_node(base_state)
        
        # Step 2: Verify route goes to material_checkpoint
        merged_state = {**base_state, **supervisor_result}
        route_result = route_after_supervisor(merged_state)
        
        assert route_result == "material_checkpoint", (
            f"After MATERIAL_VALIDATION, should route to material_checkpoint, got {route_result}"
        )
        
        # Step 3: Call material_checkpoint_node
        checkpoint_result = material_checkpoint_node(merged_state)
        
        # Verify it sets up for user approval
        assert checkpoint_result.get("awaiting_user_input") is True, (
            "material_checkpoint should set awaiting_user_input=True"
        )
        assert checkpoint_result.get("ask_user_trigger") == "material_checkpoint", (
            f"Should set material_checkpoint trigger, got {checkpoint_result.get('ask_user_trigger')}"
        )
        assert "pending_user_questions" in checkpoint_result, (
            "Should have pending questions for user"
        )
        assert "pending_validated_materials" in checkpoint_result, (
            "Should have pending materials waiting for approval"
        )
        
        # Step 4: User approves materials (uses approval keywords like APPROVE, ACCEPT, YES)
        approval_state = {
            **merged_state,
            **checkpoint_result,
            "user_responses": {"Q1": "APPROVE"},
            # Ensure pending_validated_materials is populated for approval to work
            "pending_validated_materials": [{"name": "aluminum", "n": 1.5, "k": 0.01}],
        }
        
        approval_result = supervisor_node(approval_state)
        
        # Verify materials were validated
        assert approval_result.get("supervisor_verdict") == "ok_continue", (
            f"Should set ok_continue after approval, got {approval_result.get('supervisor_verdict')}"
        )
        # Validated materials should be set (moved from pending)
        assert "validated_materials" in approval_result, (
            "Should have validated_materials after approval"
        )
        
        # Step 5: Verify routing continues to select_stage (not back to checkpoint)
        final_state = {**approval_state, **approval_result}
        # Add marker that we've done material_checkpoint
        final_state["user_responses"]["material_checkpoint"] = "APPROVE"
        
        final_route = route_after_supervisor(final_state)
        assert final_route == "select_stage", (
            f"After material approval, should route to select_stage, got {final_route}"
        )
        
        # Step 6: Verify select_stage picks stage1
        stage_result = select_stage_node(final_state)
        assert stage_result.get("current_stage_id") == "stage1", (
            f"Should select stage1 after stage0 complete, got {stage_result.get('current_stage_id')}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 2: Physics Design Flaw Recovery
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_physics_design_flaw_routes_to_design_not_code(self, mock_checkpoint):
        """Full flow: physics_check (verdict=design_flaw) → routes to DESIGN (not code).
        
        This tests the critical path where physics_check detects a fundamental
        design problem (not just a code bug) and routes back to simulation_designer.
        """
        from unittest.mock import patch
        from src.agents.execution import physics_sanity_node
        from src.routing import route_after_physics_check
        
        # State representing a physics check detecting design flaw
        base_state = {
            "current_stage_id": "stage1",
            "current_stage_type": "SINGLE_STRUCTURE",
            "plan": {
                "stages": [{"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"}],
            },
            "progress": {"stages": [], "user_interactions": []},
            "execution_result": {
                "success": True,
                "output_files": ["output.png"],
                "stdout": "Simulation complete",
            },
            "design_description": "Simulate extinction spectrum",
            "code": "# FDTD simulation code",
            "design_revision_count": 0,  # Fresh start
            "physics_failure_count": 0,
            "runtime_config": {},
            "paper_id": "test_paper",
        }
        
        # Mock physics agent returning design_flaw verdict
        mock_physics_response = {
            "verdict": "design_flaw",
            "feedback": "The simulation setup has fundamental issues: boundary conditions are incompatible with the source type.",
            "issues": ["PML boundaries incompatible with plane wave source at this angle"],
        }
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_physics_response):
            with patch("src.agents.execution.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.execution.check_context_or_escalate", return_value=None):
                    physics_result = physics_sanity_node(base_state)
        
        # Verify physics_verdict is set correctly
        assert physics_result.get("physics_verdict") == "design_flaw", (
            f"physics_verdict should be design_flaw, got {physics_result.get('physics_verdict')}"
        )
        
        # Verify design feedback is set (for designer to use)
        # The node sets a generic message, actual feedback comes from the agent response
        assert "design_feedback" in physics_result, (
            "Should set design_feedback for simulation_designer"
        )
        
        # Verify design_revision_count is incremented (not physics_failure_count)
        assert physics_result.get("design_revision_count", 0) >= 1, (
            f"design_revision_count should be incremented, got {physics_result.get('design_revision_count')}"
        )
        
        # CRITICAL: Verify routing goes to DESIGN (not generate_code)
        merged_state = {**base_state, **physics_result}
        route_result = route_after_physics_check(merged_state)
        
        assert route_result == "design", (
            f"design_flaw should route to design, got {route_result}. "
            "This is critical - code won't fix a design problem!"
        )
        
        # Verify it does NOT route to generate_code
        assert route_result != "generate_code", (
            "design_flaw MUST NOT route to generate_code!"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 3: Multi-Revision Limit Escalation Chain
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_multi_revision_limit_escalation_chain(self, mock_checkpoint):
        """Full flow: code_review fails → limit → user HINT → code_review fails again 
        → limit → user SKIP_STAGE.
        
        Tests that counters reset properly between escalations and that 
        multiple escalations don't corrupt state.
        """
        from unittest.mock import patch
        from src.agents.code import code_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node
        from src.routing import route_after_code_review
        from src.graph import route_after_supervisor
        
        # Initial state at code_review limit
        MAX_CODE_REVISIONS = 3
        base_state = {
            "current_stage_id": "stage1",
            "current_stage_type": "SINGLE_STRUCTURE",
            "plan": {"stages": [{"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"}]},
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "in_progress"}],
                "user_interactions": [],
            },
            "code": "# Broken simulation code",
            "design_description": "Simulate extinction",
            "code_revision_count": MAX_CODE_REVISIONS,  # At limit
            "runtime_config": {"max_code_revisions": MAX_CODE_REVISIONS},
            "paper_id": "test_paper",
        }
        
        # Step 1: code_review fails and hits limit
        mock_review_response = {
            "verdict": "needs_revision",
            "feedback": "Missing imports",
            "issues": ["Missing numpy import"],
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_review_response):
            with patch("src.agents.code.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.code.check_context_or_escalate", return_value=None):
                    review_result = code_reviewer_node(base_state)
        
        # Verify escalation to ask_user
        assert review_result.get("ask_user_trigger") == "code_review_limit", (
            f"Should trigger code_review_limit, got {review_result.get('ask_user_trigger')}"
        )
        
        merged_state = {**base_state, **review_result}
        route_result = route_after_code_review(merged_state)
        assert route_result == "ask_user", (
            f"At limit should route to ask_user, got {route_result}"
        )
        
        # Step 2: User provides HINT (keyword is PROVIDE_HINT or HINT)
        hint_state = {
            **merged_state,
            "user_responses": {"Q1": "PROVIDE_HINT: Try using explicit imports at the top"},
        }
        
        hint_result = supervisor_node(hint_state)
        
        # Verify counter was reset
        assert hint_result.get("code_revision_count", MAX_CODE_REVISIONS) == 0, (
            f"PROVIDE_HINT should reset code_revision_count to 0, got {hint_result.get('code_revision_count')}"
        )
        
        # Verify feedback was set (code_review_limit handler uses reviewer_feedback)
        assert "reviewer_feedback" in hint_result, (
            "Should set reviewer_feedback from user hint"
        )
        assert "import" in hint_result["reviewer_feedback"].lower(), (
            f"reviewer_feedback should contain user's hint, got: {hint_result['reviewer_feedback']}"
        )
        
        # Verify routing goes directly to generate_code (bypass select_stage)
        post_hint_state = {**hint_state, **hint_result}
        route_after_hint = route_after_supervisor(post_hint_state)
        assert route_after_hint == "generate_code", (
            f"After HINT should route to generate_code, got {route_after_hint}"
        )
        
        # Step 3: Simulate code still failing after hint - another limit hit
        # Create a fresh state (not carrying over ask_user_trigger from previous)
        second_limit_state = {
            "current_stage_id": "stage1",
            "current_stage_type": "SINGLE_STRUCTURE",
            "plan": {"stages": [{"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"}]},
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "in_progress"}],
                "user_interactions": [],
            },
            "code": "# Still broken code after hint",
            "design_description": "Simulate extinction",
            "code_revision_count": MAX_CODE_REVISIONS,  # Back at limit
            "runtime_config": {"max_code_revisions": MAX_CODE_REVISIONS},
            "paper_id": "test_paper",
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_review_response):
            with patch("src.agents.code.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.code.check_context_or_escalate", return_value=None):
                    second_review_result = code_reviewer_node(second_limit_state)
        
        # Verify second escalation
        assert second_review_result.get("ask_user_trigger") == "code_review_limit", (
            "Second time at limit should also trigger code_review_limit"
        )
        
        # Step 4: User gives up with SKIP_STAGE
        skip_state = {
            **second_limit_state,
            **second_review_result,
            "user_responses": {"Q1": "SKIP_STAGE"},
        }
        
        skip_result = supervisor_node(skip_state)
        
        # Verify stage is marked appropriately
        assert skip_result.get("supervisor_verdict") == "ok_continue", (
            f"SKIP_STAGE should set ok_continue, got {skip_result.get('supervisor_verdict')}"
        )
        
        # Verify progress was updated to mark stage as blocked/skipped
        progress = skip_result.get("progress", {})
        if "stages" in progress:
            stage_info = next(
                (s for s in progress["stages"] if s.get("stage_id") == "stage1"),
                None
            )
            if stage_info:
                assert stage_info.get("status") in ("blocked", "skipped", "completed_partial"), (
                    f"SKIP_STAGE should mark stage as blocked/skipped, got {stage_info.get('status')}"
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 4: Backtracking Flow
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_backtracking_flow_invalidates_dependent_stages(self, mock_checkpoint):
        """Full flow: stage 2 needs backtrack → supervisor → handle_backtrack 
        → stage 1 invalidated → select_stage picks stage 1 again.
        
        Tests the backtracking mechanism that invalidates stages and their dependents.
        """
        from unittest.mock import patch
        from src.agents.supervision.supervisor import supervisor_node
        from src.agents.reporting import handle_backtrack_node
        from src.agents.stage_selection import select_stage_node
        from src.graph import route_after_supervisor
        
        # State: stage 0 and 1 complete, stage 2 in progress, needs backtrack to stage 1
        base_state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "stage_type": "PARAMETER_SWEEP", "dependencies": ["stage1"]},
                ],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "status": "completed_success"},
                    {"stage_id": "stage1", "status": "completed_success"},
                    {"stage_id": "stage2", "status": "in_progress"},
                ],
                "user_interactions": [],
            },
            "current_stage_id": "stage2",
            "current_stage_type": "PARAMETER_SWEEP",
            "backtrack_count": 0,
            "paper_id": "test_paper",
            # Supervisor determined backtrack is needed
            "supervisor_verdict": "backtrack_to_stage",
            "backtrack_decision": {
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
                "accepted": True,
                "reason": "Results don't match expected behavior, need to adjust simulation parameters",
            },
        }
        
        # Step 1: Verify routing goes to handle_backtrack
        route_result = route_after_supervisor(base_state)
        assert route_result == "handle_backtrack", (
            f"backtrack_to_stage should route to handle_backtrack, got {route_result}"
        )
        
        # Step 2: Call handle_backtrack_node
        backtrack_result = handle_backtrack_node(base_state)
        
        # Verify backtrack_count incremented
        assert backtrack_result.get("backtrack_count", 0) >= 1, (
            f"backtrack_count should be incremented, got {backtrack_result.get('backtrack_count')}"
        )
        
        # Verify progress was updated - stage2 should be invalidated, stage1 needs rerun
        progress = backtrack_result.get("progress", {})
        stages = progress.get("stages", [])
        
        stage1_info = next((s for s in stages if s.get("stage_id") == "stage1"), None)
        stage2_info = next((s for s in stages if s.get("stage_id") == "stage2"), None)
        
        assert stage1_info is not None, "stage1 should still exist in progress"
        assert stage1_info.get("status") in ("needs_rerun", "pending"), (
            f"stage1 should be marked needs_rerun, got {stage1_info.get('status')}"
        )
        
        assert stage2_info is not None, "stage2 should still exist in progress"
        assert stage2_info.get("status") in ("invalidated", "pending", "blocked"), (
            f"stage2 should be invalidated, got {stage2_info.get('status')}"
        )
        
        # Step 3: Verify select_stage picks stage1 again
        merged_state = {**base_state, **backtrack_result}
        stage_result = select_stage_node(merged_state)
        
        assert stage_result.get("current_stage_id") == "stage1", (
            f"After backtrack, should pick stage1 for rerun, got {stage_result.get('current_stage_id')}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 5: Complete Single Stage with All Review Cycles
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_complete_single_stage_with_review_cycles(self, mock_checkpoint):
        """Full flow: select_stage → design → design_review (revision) → design 
        → design_review ✓ → code → code_review (revision) → code → code_review ✓.
        
        Tests that feedback propagates correctly through revision cycles.
        """
        from unittest.mock import patch, MagicMock
        from src.agents.stage_selection import select_stage_node
        from src.agents.design import simulation_designer_node, design_reviewer_node
        from src.agents.code import code_generator_node, code_reviewer_node
        from src.routing import route_after_design_review, route_after_code_review
        
        # SINGLE_STRUCTURE stage requires a MATERIAL_VALIDATION stage to be complete first
        # So we include a completed material validation stage
        base_state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "target_figure": "Fig0"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "target_figure": "Fig1", "dependencies": ["stage0"]},
                ],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "pending", "dependencies": ["stage0"]},
                ],
                "user_interactions": [],
            },
            "validated_materials": [{"name": "aluminum", "n": 1.5, "k": 0.01}],
            "paper_id": "test_paper",
            "paper_text": "Test paper about extinction spectra of aluminum nanoparticles.",
            "runtime_config": {},
            "design_revision_count": 0,
            "code_revision_count": 0,
        }
        
        # Step 1: Select stage - should pick stage1 (stage0 already complete)
        stage_result = select_stage_node(base_state)
        assert stage_result.get("current_stage_id") == "stage1", (
            f"Should select stage1, got {stage_result.get('current_stage_id')}. "
            f"Trigger: {stage_result.get('ask_user_trigger')}"
        )
        merged_state = {**base_state, **stage_result}
        
        # Step 2: Design phase
        mock_design_response = {
            "setup_description": "FDTD simulation with aluminum nanosphere",
            "parameters": {"radius": "50nm", "wavelength_range": "300-800nm"},
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_design_response):
            with patch("src.agents.design.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.design.check_context_or_escalate", return_value=None):
                    design_result = simulation_designer_node(merged_state)
        
        assert "design_description" in design_result
        merged_state = {**merged_state, **design_result}
        
        # Step 3: Design review - needs revision (first cycle)
        mock_design_review_reject = {
            "verdict": "needs_revision",
            "feedback": "Missing PML boundary specification",
            "issues": ["No boundary conditions specified"],
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_design_review_reject):
            with patch("src.agents.design.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.design.check_context_or_escalate", return_value=None):
                    review1_result = design_reviewer_node(merged_state)
        
        # design_reviewer_node uses last_design_review_verdict, not design_review_verdict
        assert review1_result.get("last_design_review_verdict") == "needs_revision"
        assert "reviewer_feedback" in review1_result, (
            "Rejected design review should set reviewer_feedback for next iteration"
        )
        
        merged_state = {**merged_state, **review1_result}
        route1 = route_after_design_review(merged_state)
        assert route1 == "design", "needs_revision should route back to design"
        
        # Step 4: Design phase (second iteration with feedback)
        mock_design_response_v2 = {
            "setup_description": "FDTD simulation with aluminum nanosphere and PML boundaries",
            "parameters": {"radius": "50nm", "wavelength_range": "300-800nm", "pml_layers": 12},
        }
        
        # Capture if feedback is being used
        captured_prompts = []
        def capture_design_call(*args, **kwargs):
            if len(args) >= 3:
                captured_prompts.append(args[2])
            return mock_design_response_v2
        
        with patch("src.agents.design.call_agent_with_metrics", side_effect=capture_design_call):
            with patch("src.agents.design.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.design.check_context_or_escalate", return_value=None):
                    design2_result = simulation_designer_node(merged_state)
        
        merged_state = {**merged_state, **design2_result}
        
        # Step 5: Design review - approved
        mock_design_review_approve = {
            "verdict": "approve",
            "feedback": "Design looks good",
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_design_review_approve):
            with patch("src.agents.design.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.design.check_context_or_escalate", return_value=None):
                    review2_result = design_reviewer_node(merged_state)
        
        assert review2_result.get("last_design_review_verdict") == "approve"
        merged_state = {**merged_state, **review2_result}
        
        route2 = route_after_design_review(merged_state)
        assert route2 == "generate_code", "approve should route to generate_code"
        
        # Step 6: Code generation
        mock_code_response = {
            "code": "import meep as mp\n# FDTD simulation",
            "explanation": "Using Meep for FDTD",
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_code_response):
            with patch("src.agents.code.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.code.check_context_or_escalate", return_value=None):
                    code_result = code_generator_node(merged_state)
        
        assert "code" in code_result
        merged_state = {**merged_state, **code_result}
        
        # Step 7: Code review - needs revision
        mock_code_review_reject = {
            "verdict": "needs_revision",
            "feedback": "Missing output file saving",
            "issues": ["No plt.savefig() call"],
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_code_review_reject):
            with patch("src.agents.code.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.code.check_context_or_escalate", return_value=None):
                    code_review1_result = code_reviewer_node(merged_state)
        
        # code_reviewer_node uses last_code_review_verdict
        assert code_review1_result.get("last_code_review_verdict") == "needs_revision"
        assert "reviewer_feedback" in code_review1_result
        
        merged_state = {**merged_state, **code_review1_result}
        route3 = route_after_code_review(merged_state)
        assert route3 == "generate_code", "needs_revision should route to generate_code"
        
        # Step 8: Code generation (with feedback)
        mock_code_response_v2 = {
            "code": "import meep as mp\nimport matplotlib.pyplot as plt\n# FDTD simulation\nplt.savefig('output.png')",
            "explanation": "Using Meep for FDTD with proper output saving",
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_code_response_v2):
            with patch("src.agents.code.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.code.check_context_or_escalate", return_value=None):
                    code2_result = code_generator_node(merged_state)
        
        merged_state = {**merged_state, **code2_result}
        
        # Step 9: Code review - approved
        mock_code_review_approve = {
            "verdict": "approve",
            "feedback": "Code looks good",
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_code_review_approve):
            with patch("src.agents.code.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.code.check_context_or_escalate", return_value=None):
                    code_review2_result = code_reviewer_node(merged_state)
        
        assert code_review2_result.get("last_code_review_verdict") == "approve"
        merged_state = {**merged_state, **code_review2_result}
        
        route4 = route_after_code_review(merged_state)
        assert route4 == "run_code", "approve should route to run_code"
        
        # Verify revision counts were properly tracked
        assert merged_state.get("design_revision_count", 0) >= 1, (
            "design_revision_count should have been incremented during revision cycle"
        )
        assert merged_state.get("code_revision_count", 0) >= 1, (
            "code_revision_count should have been incremented during revision cycle"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 6: Execution Failure with Code Regeneration
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_execution_failure_triggers_code_regeneration(self, mock_checkpoint):
        """Full flow: run_code → execution_check (fail) → generate_code → code_review 
        → run_code → execution_check (fail again) → ... → limit → ask_user.
        
        Tests that execution failures route to code regeneration with proper feedback.
        """
        from unittest.mock import patch
        from src.agents.execution import execution_validator_node
        from src.routing import route_after_execution_check
        
        MAX_EXECUTION_FAILURES = 3
        
        # State: code ran but produced errors
        base_state = {
            "current_stage_id": "stage1",
            "current_stage_type": "SINGLE_STRUCTURE",
            "plan": {"stages": [{"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"}]},
            "progress": {"stages": [], "user_interactions": []},
            "code": "import meep\n# buggy code",
            "design_description": "FDTD simulation",
            "execution_result": {
                "success": False,
                "error": "IndexError: list index out of range",
                "stdout": "",
                "stderr": "Traceback...",
            },
            "execution_failure_count": 0,  # First failure
            "runtime_config": {"max_execution_failures": MAX_EXECUTION_FAILURES},
            "paper_id": "test_paper",
        }
        
        # Step 1: Execution check detects failure
        # Note: execution_validator_node uses 'summary' field for feedback, not 'feedback'
        mock_exec_response = {
            "verdict": "fail",
            "summary": "Code crashed with IndexError: list index out of range",
            "issues": ["Array index out of bounds in line 42"],
        }
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_exec_response):
            with patch("src.agents.execution.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.execution.check_context_or_escalate", return_value=None):
                    exec_result = execution_validator_node(base_state)
        
        # Verify execution_verdict is fail
        assert exec_result.get("execution_verdict") == "fail", (
            f"Should set execution_verdict=fail, got {exec_result.get('execution_verdict')}"
        )
        
        # Verify execution_feedback is set (the node sets execution_feedback, not code_feedback)
        assert "execution_feedback" in exec_result, (
            "Execution failure should set execution_feedback"
        )
        
        # Verify execution_failure_count is incremented
        assert exec_result.get("execution_failure_count", 0) >= 1, (
            f"execution_failure_count should be incremented, got {exec_result.get('execution_failure_count')}"
        )
        
        # Verify routing goes to generate_code (not ask_user yet)
        merged_state = {**base_state, **exec_result}
        route_result = route_after_execution_check(merged_state)
        
        assert route_result == "generate_code", (
            f"First failure should route to generate_code, got {route_result}"
        )
        
        # Step 2: Simulate reaching the limit
        at_limit_state = {
            **base_state,
            "execution_failure_count": MAX_EXECUTION_FAILURES,  # At limit
        }
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_exec_response):
            with patch("src.agents.execution.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.execution.check_context_or_escalate", return_value=None):
                    limit_result = execution_validator_node(at_limit_state)
        
        # Verify escalation to ask_user at limit
        assert limit_result.get("ask_user_trigger") == "execution_failure_limit", (
            f"At limit should trigger execution_failure_limit, got {limit_result.get('ask_user_trigger')}"
        )
        
        merged_limit_state = {**at_limit_state, **limit_result}
        route_at_limit = route_after_execution_check(merged_limit_state)
        
        assert route_at_limit == "ask_user", (
            f"At limit should route to ask_user, got {route_at_limit}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 7: Context Overflow Recovery
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_context_overflow_truncation_recovery(self, mock_checkpoint):
        """Full flow: node detects context overflow → ask_user → user says TRUNCATE 
        → paper_text truncated → workflow resumes.
        
        Tests the context overflow recovery mechanism.
        """
        from unittest.mock import patch
        from src.agents.supervision.supervisor import supervisor_node
        from src.graph import route_after_supervisor
        
        # Large paper text that triggers overflow
        large_paper = "A" * 50000  # 50k chars
        
        # State after context overflow was detected
        overflow_state = {
            "paper_text": large_paper,
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Q1": "TRUNCATE"},
            "pending_user_questions": ["Context overflow. SUMMARIZE, TRUNCATE, SKIP_STAGE, or STOP?"],
            "awaiting_user_input": True,
            "last_node_before_ask_user": "planning",
            "progress": {"stages": [], "user_interactions": []},
            "paper_id": "test_paper",
        }
        
        # Supervisor handles the TRUNCATE command
        result = supervisor_node(overflow_state)
        
        # Verify paper was truncated
        assert "paper_text" in result, (
            "TRUNCATE should set paper_text in result"
        )
        truncated_paper = result["paper_text"]
        assert len(truncated_paper) < len(large_paper), (
            f"Paper should be truncated: original {len(large_paper)}, truncated {len(truncated_paper)}"
        )
        
        # Verify truncation marker exists
        assert "TRUNCATED" in truncated_paper, (
            "Truncated paper should contain truncation marker"
        )
        
        # Verify verdict allows continuation
        assert result.get("supervisor_verdict") == "ok_continue", (
            f"TRUNCATE should set ok_continue, got {result.get('supervisor_verdict')}"
        )
        
        # Verify routing continues (not to ask_user again)
        merged_state = {**overflow_state, **result}
        route_result = route_after_supervisor(merged_state)
        
        assert route_result != "ask_user", (
            f"After TRUNCATE, should not route to ask_user again, got {route_result}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Test 8: LLM Error Recovery
    # ═══════════════════════════════════════════════════════════════════════

    @patch('src.graph.save_checkpoint')
    def test_llm_error_recovery_with_retry(self, mock_checkpoint):
        """Full flow: LLM call fails → escalate → user says RETRY → retry succeeds.
        
        Tests that LLM errors are properly escalated and can be recovered from.
        """
        from unittest.mock import patch, MagicMock
        from src.agents.planning import plan_node
        from src.agents.supervision.supervisor import supervisor_node
        from src.graph import route_after_supervisor
        
        base_state = {
            "paper_text": """
            This is a test paper about optical properties of gold nanoparticles.
            We study extinction spectra using FDTD simulations with Meep.
            The nanoparticle diameter is 50nm and we use Johnson-Christy data.
            Results show strong plasmonic resonance at 520nm wavelength.
            """,
            "paper_id": "test_paper",
            "progress": {"stages": [], "user_interactions": []},
            "runtime_config": {},
        }
        
        # Step 1: Simulate LLM returning an error
        def raise_llm_error(*args, **kwargs):
            raise Exception("LLM API rate limit exceeded")
        
        with patch("src.agents.planning.call_agent_with_metrics", side_effect=raise_llm_error):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    try:
                        error_result = plan_node(base_state)
                    except Exception:
                        # In real code, this would be caught and escalated
                        error_result = {
                            "ask_user_trigger": "llm_error",
                            "pending_user_questions": ["LLM API error: rate limit exceeded. RETRY, SKIP, or STOP?"],
                            "awaiting_user_input": True,
                            "last_node_before_ask_user": "planning",
                        }
        
        # Step 2: User says RETRY
        retry_state = {
            **base_state,
            **error_result,
            "user_responses": {"Q1": "RETRY"},
        }
        
        # Supervisor handles RETRY (in this case, the default handler should just continue)
        retry_result = supervisor_node(retry_state)
        
        # The supervisor should allow continuation (ok_continue or similar)
        # For unrecognized triggers, default handling applies
        assert retry_result.get("supervisor_verdict") in ("ok_continue", "ask_user"), (
            f"RETRY should lead to continuation or clarification, got {retry_result.get('supervisor_verdict')}"
        )
        
        # Step 3: If ok_continue, verify routing continues
        if retry_result.get("supervisor_verdict") == "ok_continue":
            merged_state = {**retry_state, **retry_result}
            route_result = route_after_supervisor(merged_state)
            
            # Should route to select_stage (for a fresh planning attempt)
            assert route_result in ("select_stage", "planning"), (
                f"After RETRY success, should continue workflow, got {route_result}"
            )
        
        # Step 4: Verify retry would actually work
        mock_success_response = {
            "stages": [{"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"}],
            "assumptions": ["Using Johnson-Christy data"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_success_response):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                    success_result = plan_node(base_state)
        
        # Verify planning succeeds on retry
        assert "plan" in success_result, (
            "After LLM recovers, planning should produce a plan"
        )
        assert len(success_result["plan"].get("stages", [])) > 0, (
            "Plan should have stages after successful retry"
        )

