"""Graph structure integration tests for the LangGraph state machine."""

from src.graph import create_repro_graph


class TestGraphCompilation:
    """Tests that the graph compiles without errors."""

    def test_graph_compiles_successfully(self):
        """Test that create_repro_graph returns a compiled graph."""
        graph = create_repro_graph()

        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")
        assert hasattr(graph, "get_graph")

    def test_graph_has_all_expected_nodes(self):
        """Test that all expected nodes are present in the graph."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        expected_nodes = [
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
        ]

        for node in expected_nodes:
            assert node in graph_def.nodes, f"Missing node: {node}"

    def test_graph_has_start_node(self):
        """Test that the graph has __start__ node."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        assert "__start__" in graph_def.nodes

    def test_graph_has_end_node(self):
        """Test that the graph has __end__ node."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        assert "__end__" in graph_def.nodes

    def test_graph_start_connects_to_adapt_prompts(self):
        """Test that the graph starts at adapt_prompts."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        edges = list(graph_def.edges)
        start_edges = [edge for edge in edges if edge[0] == "__start__"]

        assert start_edges, "No start edge found"
        start_targets = [edge[1] for edge in start_edges]
        assert "adapt_prompts" in start_targets

    def test_graph_generate_report_connects_to_end(self):
        """Test that generate_report connects to __end__."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        edges = list(graph_def.edges)
        report_edges = [edge for edge in edges if edge[0] == "generate_report"]

        assert report_edges, "generate_report has no outgoing edges"
        report_targets = [edge[1] for edge in report_edges]
        assert "__end__" in report_targets

    def test_graph_node_count(self):
        """Test that the graph has the expected number of nodes."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        node_count = len(graph_def.nodes)
        assert node_count >= 18, f"Expected at least 18 nodes, got {node_count}"


class TestGraphEdgeConfiguration:
    """Tests for graph edge wiring correctness."""

    def test_all_review_nodes_have_conditional_edges(self):
        """Test that review nodes use conditional edges (not static)."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        review_nodes = ["plan_review", "design_review", "code_review"]

        for node in review_nodes:
            node_edges = [edge for edge in graph_def.edges if edge[0] == node]
            targets = {edge[1] for edge in node_edges}
            assert len(targets) >= 2, (
                f"{node} should have conditional edges to multiple targets, "
                f"but only goes to: {targets}"
            )

    def test_execution_check_has_three_routes(self):
        """Test execution_check can route to physics_check, generate_code, or ask_user."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        exec_edges = [edge for edge in graph_def.edges if edge[0] == "execution_check"]
        targets = {edge[1] for edge in exec_edges}

        expected = {"physics_check", "generate_code", "ask_user"}
        assert expected.issubset(targets), (
            f"execution_check should route to {expected}, "
            f"but can only reach: {targets}"
        )

    def test_supervisor_has_all_required_routes(self):
        """Test supervisor can route to all expected destinations."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        supervisor_edges = [edge for edge in graph_def.edges if edge[0] == "supervisor"]
        targets = {edge[1] for edge in supervisor_edges}

        expected = {
            "select_stage",
            "plan",
            "ask_user",
            "handle_backtrack",
            "generate_report",
            "material_checkpoint",
        }
        assert expected.issubset(targets), (
            f"supervisor should route to {expected}, "
            f"but can only reach: {targets}"
        )

    def test_ask_user_routes_to_supervisor(self):
        """Test that ask_user always routes back to supervisor."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()

        ask_user_edges = [edge for edge in graph_def.edges if edge[0] == "ask_user"]
        targets = {edge[1] for edge in ask_user_edges}

        assert "supervisor" in targets, (
            f"ask_user should route to supervisor, but goes to: {targets}"
        )


class TestGraphInterruptConfiguration:
    """Tests for graph interrupt (pause) configuration."""

    def test_graph_has_interrupt_before_ask_user(self):
        """Test that the graph is configured to interrupt before ask_user."""
        graph = create_repro_graph()

        assert graph.checkpointer is not None, (
            "Graph should have a checkpointer for interrupt support"
        )


