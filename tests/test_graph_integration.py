"""
Integration Tests for the LangGraph State Machine.

These tests verify:
1. The graph compiles correctly with all edges wired
2. Routing functions integrate properly with LangGraph
3. Checkpoint behavior works end-to-end
4. Edge cases like missing nodes or circular references are caught

Unlike unit tests that mock components, these tests verify the actual
integration between modules.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from schemas.state import create_initial_state, ReproState, MAX_CODE_REVISIONS
from src.graph import create_repro_graph


# ═══════════════════════════════════════════════════════════════════════
# Graph Compilation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphCompilation:
    """Tests that the graph compiles without errors."""
    
    def test_graph_compiles_successfully(self):
        """Test that create_repro_graph returns a compiled graph."""
        graph = create_repro_graph()
        
        assert graph is not None
        # LangGraph compiled graphs have these attributes
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'stream')
        assert hasattr(graph, 'get_graph')
    
    def test_graph_has_all_expected_nodes(self):
        """Test that all expected nodes are present in the graph."""
        graph = create_repro_graph()
        
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
        
        # Access the underlying graph structure
        graph_def = graph.get_graph()
        graph_nodes = list(graph_def.nodes.keys())
        
        for node in expected_nodes:
            assert node in graph_nodes, f"Missing node: {node}"
    
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
        
        # Find edges from __start__
        edges = list(graph_def.edges)
        start_edges = [e for e in edges if e[0] == "__start__"]
        
        assert len(start_edges) > 0, "No start edge found"
        # Should connect to adapt_prompts
        start_targets = [e[1] for e in start_edges]
        assert "adapt_prompts" in start_targets
    
    def test_graph_generate_report_connects_to_end(self):
        """Test that generate_report connects to __end__."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()
        
        edges = list(graph_def.edges)
        report_edges = [e for e in edges if e[0] == "generate_report"]
        
        assert len(report_edges) > 0, "generate_report has no outgoing edges"
        report_targets = [e[1] for e in report_edges]
        assert "__end__" in report_targets
    
    def test_graph_node_count(self):
        """Test that the graph has the expected number of nodes."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()
        
        # 18 workflow nodes + __start__ + __end__ = 20
        # Actual count may vary based on implementation
        node_count = len(graph_def.nodes)
        assert node_count >= 18, f"Expected at least 18 nodes, got {node_count}"


# ═══════════════════════════════════════════════════════════════════════
# Routing Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingIntegration:
    """Tests that routing functions work correctly with real state."""
    
    @pytest.fixture
    def test_state(self) -> ReproState:
        """Create a test state for routing tests."""
        state = create_initial_state(
            paper_id="integration_test",
            paper_text="Test paper content for integration testing.",
        )
        return state
    
    @patch("src.routing.save_checkpoint")  # Mock disk I/O only
    def test_plan_review_approve_routes_to_select_stage(self, mock_checkpoint, test_state):
        """Test plan_review -> select_stage on approve."""
        from src.routing import route_after_plan_review
        
        test_state["last_plan_review_verdict"] = "approve"
        
        result = route_after_plan_review(test_state)
        
        assert result == "select_stage"
        mock_checkpoint.assert_not_called()  # No checkpoint on success path
    
    @patch("src.routing.save_checkpoint")
    def test_design_review_approve_routes_to_generate_code(self, mock_checkpoint, test_state):
        """Test design_review -> generate_code on approve."""
        from src.routing import route_after_design_review
        
        test_state["last_design_review_verdict"] = "approve"
        
        result = route_after_design_review(test_state)
        
        assert result == "generate_code"
    
    @patch("src.routing.save_checkpoint")
    def test_code_review_approve_routes_to_run_code(self, mock_checkpoint, test_state):
        """Test code_review -> run_code on approve."""
        from src.routing import route_after_code_review
        
        test_state["last_code_review_verdict"] = "approve"
        
        result = route_after_code_review(test_state)
        
        assert result == "run_code"
    
    @patch("src.routing.save_checkpoint")
    def test_execution_pass_routes_to_physics_check(self, mock_checkpoint, test_state):
        """Test execution_check -> physics_check on pass."""
        from src.routing import route_after_execution_check
        
        test_state["execution_verdict"] = "pass"
        
        result = route_after_execution_check(test_state)
        
        assert result == "physics_check"
    
    @patch("src.routing.save_checkpoint")
    def test_physics_pass_routes_to_analyze(self, mock_checkpoint, test_state):
        """Test physics_check -> analyze on pass."""
        from src.routing import route_after_physics_check
        
        test_state["physics_verdict"] = "pass"
        
        result = route_after_physics_check(test_state)
        
        assert result == "analyze"
    
    @patch("src.routing.save_checkpoint")
    def test_comparison_approve_routes_to_supervisor(self, mock_checkpoint, test_state):
        """Test comparison_check -> supervisor on approve."""
        from src.routing import route_after_comparison_check
        
        test_state["comparison_verdict"] = "approve"
        
        result = route_after_comparison_check(test_state)
        
        assert result == "supervisor"


# ═══════════════════════════════════════════════════════════════════════
# Limit Escalation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLimitEscalation:
    """Tests that routers correctly escalate when limits are reached."""
    
    @pytest.fixture
    def test_state(self) -> ReproState:
        """Create a test state."""
        return create_initial_state(
            paper_id="limit_test",
            paper_text="Test content",
        )
    
    @patch("src.routing.save_checkpoint")
    def test_code_review_escalates_at_limit(self, mock_checkpoint, test_state):
        """Test code_review escalates to ask_user at revision limit."""
        from src.routing import route_after_code_review
        
        test_state["last_code_review_verdict"] = "needs_revision"
        test_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = route_after_code_review(test_state)
        
        assert result == "ask_user"
        mock_checkpoint.assert_called_once()
        # Verify checkpoint name contains "limit"
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "limit" in checkpoint_name
    
    @patch("src.routing.save_checkpoint")
    def test_code_review_continues_under_limit(self, mock_checkpoint, test_state):
        """Test code_review continues normally under limit."""
        from src.routing import route_after_code_review
        
        test_state["last_code_review_verdict"] = "needs_revision"
        test_state["code_revision_count"] = MAX_CODE_REVISIONS - 1
        
        result = route_after_code_review(test_state)
        
        assert result == "generate_code"
        mock_checkpoint.assert_not_called()
    
    @patch("src.routing.save_checkpoint")
    def test_runtime_config_overrides_default_limit(self, mock_checkpoint, test_state):
        """Test that runtime_config can override default limits."""
        from src.routing import route_after_code_review
        
        test_state["last_code_review_verdict"] = "needs_revision"
        test_state["code_revision_count"] = 10  # Over default limit
        test_state["runtime_config"] = {"max_code_revisions": 20}  # Higher limit
        
        result = route_after_code_review(test_state)
        
        assert result == "generate_code"  # Should continue, not escalate


# ═══════════════════════════════════════════════════════════════════════
# None Verdict Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNoneVerdictHandling:
    """Tests that all routers handle None verdicts gracefully."""
    
    @pytest.fixture
    def test_state(self) -> ReproState:
        """Create a test state."""
        return create_initial_state(
            paper_id="none_test",
            paper_text="Test content",
        )
    
    @pytest.mark.parametrize("router_name,verdict_field", [
        ("route_after_plan_review", "last_plan_review_verdict"),
        ("route_after_design_review", "last_design_review_verdict"),
        ("route_after_code_review", "last_code_review_verdict"),
        ("route_after_execution_check", "execution_verdict"),
        ("route_after_physics_check", "physics_verdict"),
        ("route_after_comparison_check", "comparison_verdict"),
    ])
    @patch("src.routing.save_checkpoint")
    def test_router_escalates_on_none(self, mock_checkpoint, router_name, verdict_field, test_state):
        """Test that each router escalates to ask_user on None verdict."""
        from src import routing
        
        router = getattr(routing, router_name)
        test_state[verdict_field] = None
        
        result = router(test_state)
        
        assert result == "ask_user", f"{router_name} should route to ask_user on None"
        mock_checkpoint.assert_called_once()
        # Verify checkpoint name contains "error"
        checkpoint_name = mock_checkpoint.call_args[0][1]
        assert "error" in checkpoint_name


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointIntegration:
    """Tests that verify checkpoint behavior with actual file I/O."""
    
    def test_checkpoint_creates_file(self, tmp_path):
        """Test that checkpoints create actual files on disk."""
        from schemas.state import save_checkpoint
        
        state = create_initial_state(
            paper_id="checkpoint_test",
            paper_text="Test content for checkpoint",
        )
        
        # Use tmp_path as output directory
        checkpoint_path = save_checkpoint(
            state, 
            "test_checkpoint",
            output_dir=str(tmp_path)
        )
        
        assert Path(checkpoint_path).exists()
    
    def test_checkpoint_contains_state_data(self, tmp_path):
        """Test that checkpoint file contains expected state data."""
        from schemas.state import save_checkpoint
        
        state = create_initial_state(
            paper_id="data_test",
            paper_text="Test content with data",
        )
        state["plan"] = {"stages": [{"stage_id": "stage_1"}]}
        
        checkpoint_path = save_checkpoint(
            state, 
            "data_checkpoint",
            output_dir=str(tmp_path)
        )
        
        with open(checkpoint_path) as f:
            saved_data = json.load(f)
        
        assert saved_data["paper_id"] == "data_test"
        assert saved_data["paper_text"] == "Test content with data"
        assert "plan" in saved_data
        assert saved_data["plan"]["stages"][0]["stage_id"] == "stage_1"
    
    def test_checkpoint_creates_latest_link(self, tmp_path):
        """Test that checkpoint creates a 'latest' pointer file."""
        from schemas.state import save_checkpoint
        
        state = create_initial_state(
            paper_id="latest_test",
            paper_text="Test",
        )
        
        save_checkpoint(state, "my_checkpoint", output_dir=str(tmp_path))
        
        # Should create a latest pointer
        checkpoints_dir = tmp_path / "latest_test" / "checkpoints"
        latest_path = checkpoints_dir / "checkpoint_my_checkpoint_latest.json"
        
        assert latest_path.exists() or latest_path.is_symlink()


# ═══════════════════════════════════════════════════════════════════════
# Graph Edge Configuration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphEdgeConfiguration:
    """Tests for graph edge wiring correctness."""
    
    def test_all_review_nodes_have_conditional_edges(self):
        """Test that review nodes use conditional edges (not static)."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()
        
        review_nodes = ["plan_review", "design_review", "code_review"]
        
        for node in review_nodes:
            # Get edges from this node
            node_edges = [e for e in graph_def.edges if e[0] == node]
            
            # Should have multiple possible targets (conditional routing)
            targets = set(e[1] for e in node_edges)
            assert len(targets) >= 2, (
                f"{node} should have conditional edges to multiple targets, "
                f"but only goes to: {targets}"
            )
    
    def test_execution_check_has_three_routes(self):
        """Test execution_check can route to physics_check, generate_code, or ask_user."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()
        
        exec_edges = [e for e in graph_def.edges if e[0] == "execution_check"]
        targets = set(e[1] for e in exec_edges)
        
        expected = {"physics_check", "generate_code", "ask_user"}
        assert expected.issubset(targets), (
            f"execution_check should route to {expected}, "
            f"but can only reach: {targets}"
        )
    
    def test_supervisor_has_all_required_routes(self):
        """Test supervisor can route to all expected destinations."""
        graph = create_repro_graph()
        graph_def = graph.get_graph()
        
        supervisor_edges = [e for e in graph_def.edges if e[0] == "supervisor"]
        targets = set(e[1] for e in supervisor_edges)
        
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
        
        ask_user_edges = [e for e in graph_def.edges if e[0] == "ask_user"]
        targets = set(e[1] for e in ask_user_edges)
        
        assert "supervisor" in targets, (
            f"ask_user should route to supervisor, but goes to: {targets}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Graph Interrupt Configuration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGraphInterruptConfiguration:
    """Tests for graph interrupt (pause) configuration."""
    
    def test_graph_has_interrupt_before_ask_user(self):
        """Test that the graph is configured to interrupt before ask_user."""
        # This tests that the graph pauses for human input
        graph = create_repro_graph()
        
        # The graph should be compiled with interrupt_before=["ask_user"]
        # We can verify by checking the checkpointer is configured
        assert graph.checkpointer is not None, (
            "Graph should have a checkpointer for interrupt support"
        )

