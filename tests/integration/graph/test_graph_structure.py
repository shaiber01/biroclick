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

