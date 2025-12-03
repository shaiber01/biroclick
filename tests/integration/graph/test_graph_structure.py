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
            "plan",
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
        assert adapt_edges[0][1] == "plan", (
            f"adapt_prompts should connect to plan, got {adapt_edges[0][1]}"
        )

    def test_plan_has_conditional_edge_to_plan_review(self, graph_definition):
        """Test that plan routes to plan_review via conditional edge."""
        edges = list(graph_definition.edges)
        plan_edges = [edge for edge in edges if edge[0] == "plan"]
        targets = {edge[1] for edge in plan_edges}
        
        assert "plan_review" in targets, (
            f"plan should route to plan_review, got: {targets}"
        )

    def test_plan_review_has_three_routes(self, graph_definition):
        """Test plan_review can route to select_stage, plan, or ask_user."""
        edges = list(graph_definition.edges)
        review_edges = [edge for edge in edges if edge[0] == "plan_review"]
        targets = {edge[1] for edge in review_edges}
        
        expected = {"select_stage", "plan", "ask_user"}
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
        """Test comparison_check can route to supervisor or analyze."""
        edges = list(graph_definition.edges)
        comp_edges = [edge for edge in edges if edge[0] == "comparison_check"]
        targets = {edge[1] for edge in comp_edges}
        
        # Note: ask_user is registered as a route but comparison_check routes to supervisor on limit
        # The route map includes ask_user but the router logic routes to supervisor
        expected_minimum = {"supervisor", "analyze"}
        assert expected_minimum.issubset(targets), (
            f"comparison_check should route to at least {expected_minimum}, got: {targets}"
        )

    def test_supervisor_has_six_routes(self, graph_definition):
        """Test supervisor can route to all expected destinations."""
        edges = list(graph_definition.edges)
        supervisor_edges = [edge for edge in edges if edge[0] == "supervisor"]
        targets = {edge[1] for edge in supervisor_edges}
        
        expected = {
            "select_stage",
            "plan",
            "ask_user",
            "handle_backtrack",
            "generate_report",
            "material_checkpoint",
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

    def test_material_checkpoint_connects_to_ask_user(self, graph_definition):
        """Test that material_checkpoint connects to ask_user via static edge."""
        edges = list(graph_definition.edges)
        mat_edges = [edge for edge in edges if edge[0] == "material_checkpoint"]
        targets = {edge[1] for edge in mat_edges}
        
        assert targets == {"ask_user"}, (
            f"material_checkpoint should connect only to ask_user, got: {targets}"
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
        """Test MATERIAL_VALIDATION stage routes to select_stage after checkpoint completed."""
        state = make_state(
            supervisor_verdict="ok_continue",
            should_stop=False,
            current_stage_type="MATERIAL_VALIDATION",
            user_responses={"material_checkpoint": "approved"},  # Checkpoint done
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
        assert result == "plan", f"Expected plan for replan_needed under limit, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_replan_needed_routes_to_plan_at_limit_minus_one(self, mock_checkpoint):
        """Test replan_needed routes to plan when at limit-1."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=MAX_REPLANS - 1,
        )
        result = route_after_supervisor(state)
        assert result == "plan", f"Expected plan for replan_needed at limit-1, got {result}"

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
        """Test needs_revision routes to plan when under limit."""
        state = make_state(
            last_plan_review_verdict="needs_revision",
            replan_count=0,
        )
        result = route_after_plan_review(state)
        assert result == "plan", f"Expected plan for needs_revision under limit, got {result}"

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
    def test_needs_revision_routes_to_supervisor_at_limit(self, mock_checkpoint):
        """Test needs_revision routes to supervisor at limit (not ask_user)."""
        state = make_state(
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS,
        )
        result = route_after_comparison_check(state)
        # Note: comparison_check routes to supervisor on limit, not ask_user
        assert result == "supervisor", (
            f"Expected supervisor at analysis revision limit, got {result}"
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
    """Tests for graph interrupt (pause) configuration."""

    def test_graph_has_checkpointer(self, compiled_graph):
        """Test that the graph has a checkpointer for interrupt support."""
        assert compiled_graph.checkpointer is not None, (
            "Graph must have a checkpointer for interrupt_before support"
        )

    def test_checkpointer_is_memory_saver(self, compiled_graph):
        """Test that the checkpointer is a MemorySaver instance."""
        from langgraph.checkpoint.memory import MemorySaver
        assert isinstance(compiled_graph.checkpointer, MemorySaver), (
            f"Checkpointer should be MemorySaver, got {type(compiled_graph.checkpointer)}"
        )

    @patch.object(StateGraph, 'compile')
    def test_compile_called_with_interrupt_before(self, mock_compile):
        """Test that compile is called with the correct interrupt_before arguments."""
        # We need to mock the graph construction since we are testing the compile call
        with patch('src.graph.StateGraph') as MockStateGraph:
            # Create mock instance
            mock_workflow = MockStateGraph.return_value
            # Call the function that creates and compiles the graph
            create_repro_graph()
            
            # Verify compile was called with expected args
            mock_workflow.compile.assert_called_once()
            call_kwargs = mock_workflow.compile.call_args.kwargs
            
            assert "interrupt_before" in call_kwargs, "compile missing interrupt_before"
            assert call_kwargs["interrupt_before"] == ["ask_user"], (
                f"interrupt_before should be ['ask_user'], got {call_kwargs['interrupt_before']}"
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
            "plan",
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
        
        # plan_review can go back to plan
        plan_review_targets = {e[1] for e in edges if e[0] == "plan_review"}
        assert "plan" in plan_review_targets, "plan_review must be able to loop back to plan"
        
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
        assert result == "plan"


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
    def test_comparison_check_routes_to_supervisor_on_limit(self, mock_checkpoint):
        """Test that comparison_check routes to supervisor (not ask_user) when limit reached."""
        state = make_state(
            comparison_verdict="needs_revision",
            analysis_revision_count=MAX_ANALYSIS_REVISIONS,
        )
        result = route_after_comparison_check(state)
        # comparison_check has route_on_limit="supervisor", not the default "ask_user"
        assert result == "supervisor", (
            f"comparison_check should route to supervisor on limit, got {result}"
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
        assert result == "plan", f"0 replan_count should allow replan, got {result}"

    @patch('src.graph.save_checkpoint')
    def test_replan_with_custom_runtime_limit(self, mock_checkpoint):
        """Test replan_needed respects runtime_config max_replans."""
        state = make_state(
            supervisor_verdict="replan_needed",
            replan_count=5,  # Would exceed default limit
            runtime_config={"max_replans": 10},  # Custom higher limit
        )
        result = route_after_supervisor(state)
        assert result == "plan", (
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
        assert result == "plan", (
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
            ("adapt_prompts", "plan"),
            ("design", "design_review"),
            ("generate_code", "code_review"),
            ("run_code", "execution_check"),
            ("analyze", "comparison_check"),
            ("handle_backtrack", "select_stage"),
            ("material_checkpoint", "ask_user"),
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
            "plan": 1,  # plan_review only
            "plan_review": 3,  # select_stage, plan, ask_user
            "select_stage": 2,  # design, generate_report
            "design_review": 3,  # generate_code, design, ask_user
            "code_review": 3,  # run_code, generate_code, ask_user
            "execution_check": 3,  # physics_check, generate_code, ask_user
            "physics_check": 4,  # analyze, generate_code, design, ask_user
            "comparison_check": 3,  # supervisor, analyze, ask_user
            "supervisor": 6,  # select_stage, plan, ask_user, handle_backtrack, generate_report, material_checkpoint
            "ask_user": 1,  # supervisor
        }
        
        edges = list(graph_definition.edges)
        
        for source, expected_count in conditional_edge_configs.items():
            source_edges = [e for e in edges if e[0] == source]
            assert len(source_edges) == expected_count, (
                f"{source} should have {expected_count} outgoing edges, "
                f"has {len(source_edges)}: {[e[1] for e in source_edges]}"
            )

