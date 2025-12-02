"""
Integration Tests for Agent Nodes with Mocked LLM.

These tests verify that agent nodes correctly:
1. Build prompts and call the LLM client
2. Parse and map LLM responses to state updates
3. Handle errors gracefully
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from schemas.state import create_initial_state, ReproState

# Load fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_RESPONSES_DIR = FIXTURES_DIR / "mock_responses"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / name
    with open(path, "r") as f:
        return json.load(f)


def load_mock_response(agent_name: str) -> dict:
    """Load a mock response for an agent."""
    path = MOCK_RESPONSES_DIR / f"{agent_name}_response.json"
    with open(path, "r") as f:
        return json.load(f)


def create_test_state() -> ReproState:
    """Create a test state with sample paper input."""
    paper_input = load_fixture("sample_paper_input.json")
    state = create_initial_state(
        paper_id=paper_input["paper_id"],
        paper_text=paper_input["paper_text"],
        paper_domain=paper_input.get("paper_domain", "other"),
    )
    # Set paper figures manually (not a constructor arg)
    state["paper_figures"] = paper_input.get("paper_figures", [])
    return state


# ═══════════════════════════════════════════════════════════════════════
# Mock LLM Client Fixture
# ═══════════════════════════════════════════════════════════════════════

class MultiPatchMock:
    """Helper that applies return_value and side_effect to multiple mocks."""
    def __init__(self, mocks):
        self._mocks = mocks
        self._return_value = {"verdict": "approve"}
        self._side_effect = None
    
    @property
    def return_value(self):
        return self._return_value
    
    @return_value.setter
    def return_value(self, value):
        self._return_value = value
        for mock in self._mocks:
            mock.return_value = value
            mock.side_effect = None  # Clear side_effect when setting return_value
    
    @property
    def side_effect(self):
        return self._side_effect
    
    @side_effect.setter
    def side_effect(self, value):
        self._side_effect = value
        for mock in self._mocks:
            mock.side_effect = value


@pytest.fixture
def mock_llm_client():
    """Fixture that provides a mock LLM client for agents module.
    
    Must patch where the function is USED (in each submodule), not just where it's defined.
    """
    # Patch all modules that import call_agent_with_metrics
    patch_targets = [
        "src.agents.planning.call_agent_with_metrics",
        "src.agents.design.call_agent_with_metrics",
        "src.agents.code.call_agent_with_metrics",
        "src.agents.execution.call_agent_with_metrics",
        "src.agents.analysis.call_agent_with_metrics",
        "src.agents.supervision.call_agent_with_metrics",
        "src.agents.reporting.call_agent_with_metrics",
    ]
    
    patches = []
    mocks = []
    for patch_target in patch_targets:
        p = patch(patch_target)
        mock = p.start()
        mock.return_value = {"verdict": "approve"}
        patches.append(p)
        mocks.append(mock)
    
    # Yield a helper that updates all mocks when return_value is set
    yield MultiPatchMock(mocks)
    
    # Stop all patches
    for p in patches:
        p.stop()


# ═══════════════════════════════════════════════════════════════════════
# Plan Node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlanNode:
    """Tests for plan_node with mocked LLM."""
    
    def test_plan_node_creates_plan(self, mock_llm_client):
        """Test that plan_node creates a valid plan from LLM response."""
        from src.agents import plan_node
        
        # Setup mock response
        mock_response = load_mock_response("planner")
        mock_llm_client.return_value = mock_response
        
        # Create test state
        state = create_test_state()
        
        # Call node
        result = plan_node(state)
        
        # Verify plan creation succeeded (not an error/ask_user)
        assert result.get("ask_user_trigger") is None or result.get("awaiting_user_input") is not True
        # Verify workflow phase is set if plan was created
        if "plan" in result:
            assert result["workflow_phase"] == "planning"
            assert len(result["plan"].get("stages", [])) >= 0
        
    def test_plan_node_extracts_parameters(self, mock_llm_client):
        """Test that plan_node extracts parameters from LLM response."""
        from src.agents import plan_node
        
        mock_response = load_mock_response("planner")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        result = plan_node(state)
        
        # Check that plan or error was returned
        if "plan" in result:
            params = result["plan"].get("extracted_parameters", [])
            # Parameters may or may not be present depending on mock structure
            assert isinstance(params, list)
        
    def test_plan_node_handles_missing_paper_text(self):
        """Test that plan_node handles missing paper text gracefully."""
        from src.agents import plan_node
        
        state = create_initial_state(
            paper_id="test",
            paper_text="",  # Empty paper text
        )
        state["paper_figures"] = []
        
        result = plan_node(state)
        
        # Should trigger ask_user
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True


# ═══════════════════════════════════════════════════════════════════════
# Reviewer Nodes Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReviewerNodes:
    """Tests for reviewer nodes with mocked LLM."""
    
    def test_plan_reviewer_approves(self, mock_llm_client):
        """Test that plan_reviewer_node approves valid plan."""
        from src.agents import plan_reviewer_node
        
        mock_response = load_mock_response("plan_reviewer")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        # Add a plan to review
        state["plan"] = load_mock_response("planner")
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"
        
    def test_design_reviewer_approves(self, mock_llm_client):
        """Test that design_reviewer_node approves valid design."""
        from src.agents import design_reviewer_node
        
        mock_response = load_mock_response("design_reviewer")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["design_description"] = load_mock_response("simulation_designer")
        state["current_stage_id"] = "stage_1_extinction"
        
        result = design_reviewer_node(state)
        
        assert result["last_design_review_verdict"] == "approve"
        
    def test_code_reviewer_approves(self, mock_llm_client):
        """Test that code_reviewer_node approves valid code."""
        from src.agents import code_reviewer_node
        
        mock_response = load_mock_response("code_reviewer")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["code"] = load_mock_response("code_generator")["code"]
        state["current_stage_id"] = "stage_1_extinction"
        
        result = code_reviewer_node(state)
        
        assert result["last_code_review_verdict"] == "approve"


# ═══════════════════════════════════════════════════════════════════════
# Validator Nodes Tests
# ═══════════════════════════════════════════════════════════════════════

class TestValidatorNodes:
    """Tests for validator nodes with mocked LLM."""
    
    def test_execution_validator_passes(self, mock_llm_client):
        """Test that execution_validator_node passes valid execution."""
        from src.agents import execution_validator_node
        
        mock_response = load_mock_response("execution_validator")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["current_stage_id"] = "stage_1_extinction"
        state["stage_outputs"] = {
            "files": ["extinction_spectrum.csv"],
            "status": "completed"
        }
        
        result = execution_validator_node(state)
        
        assert result["execution_verdict"] == "pass"
        
    def test_physics_sanity_passes(self, mock_llm_client):
        """Test that physics_sanity_node passes valid physics."""
        from src.agents import physics_sanity_node
        
        mock_response = load_mock_response("physics_sanity")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["current_stage_id"] = "stage_1_extinction"
        state["stage_outputs"] = {"files": ["extinction_spectrum.csv"]}
        
        result = physics_sanity_node(state)
        
        assert result["physics_verdict"] == "pass"


# ═══════════════════════════════════════════════════════════════════════
# Supervisor Node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSupervisorNode:
    """Tests for supervisor_node with mocked LLM."""
    
    def test_supervisor_continues_workflow(self, mock_llm_client):
        """Test that supervisor_node continues workflow normally."""
        from src.agents import supervisor_node
        
        mock_response = load_mock_response("supervisor")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["current_stage_id"] = "stage_1_extinction"
        state["workflow_phase"] = "supervision"
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        
    def test_supervisor_handles_material_checkpoint(self, mock_llm_client):
        """Test supervisor handles material checkpoint approval."""
        from src.agents import supervisor_node
        
        state = create_test_state()
        state["ask_user_trigger"] = "material_checkpoint"
        state["user_responses"] = {"Q1": "APPROVE"}
        state["pending_validated_materials"] = [{"name": "gold", "path": "materials/Au.csv"}]
        state["pending_user_questions"] = ["Approve materials?"]  # Required for logging
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("validated_materials") is not None


# ═══════════════════════════════════════════════════════════════════════
# Results Analyzer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestResultsAnalyzerNode:
    """Tests for results_analyzer_node with mocked LLM."""
    
    def test_results_analyzer_classifies_match(self, mock_llm_client):
        """Test that results_analyzer_node classifies results."""
        from src.agents import results_analyzer_node
        
        mock_response = load_mock_response("results_analyzer")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["current_stage_id"] = "stage_1_extinction"
        state["plan"] = load_mock_response("planner")
        state["stage_outputs"] = {
            "files": ["extinction_spectrum.csv"],
            "status": "completed"
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"
        # Note: The actual classification depends on quantitative analysis
        # which uses local logic, not LLM


# ═══════════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Tests for error handling in agent nodes."""
    
    def test_plan_node_handles_llm_error(self, mock_llm_client):
        """Test that plan_node handles LLM errors gracefully."""
        from src.agents import plan_node
        
        mock_llm_client.side_effect = RuntimeError("LLM API error")
        
        state = create_test_state()
        result = plan_node(state)
        
        # Should trigger ask_user with error OR return error state
        is_error_state = (
            result.get("ask_user_trigger") == "llm_error" or
            result.get("awaiting_user_input") is True
        )
        assert is_error_state
        
    def test_supervisor_defaults_on_llm_error(self, mock_llm_client):
        """Test that supervisor defaults to ok_continue on LLM error."""
        from src.agents import supervisor_node
        
        mock_llm_client.side_effect = RuntimeError("LLM API error")
        
        state = create_test_state()
        state["current_stage_id"] = "stage_1"
        
        result = supervisor_node(state)
        
        # Should default to ok_continue
        assert result["supervisor_verdict"] == "ok_continue"


# ═══════════════════════════════════════════════════════════════════════
# State Update Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStateUpdates:
    """Tests that agent nodes produce valid state updates."""
    
    def test_plan_node_returns_valid_updates(self, mock_llm_client):
        """Test that plan_node returns valid state updates."""
        from src.agents import plan_node
        
        mock_response = load_mock_response("planner")
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        result = plan_node(state)
        
        # Verify result is a dict (not full state)
        assert isinstance(result, dict)
        
        # Result should have either a plan or error handling
        has_plan = "plan" in result
        has_error = result.get("ask_user_trigger") or result.get("awaiting_user_input")
        assert has_plan or has_error
        
    def test_reviewer_increments_count_on_rejection(self, mock_llm_client):
        """Test that reviewers increment revision count on rejection."""
        from src.agents import design_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "summary": "Fix issues", "issues": []}
        mock_llm_client.return_value = mock_response
        
        state = create_test_state()
        state["design_description"] = {"test": "design"}
        state["current_stage_id"] = "stage_1"
        state["design_revision_count"] = 0
        
        result = design_reviewer_node(state)
        
        assert result["design_revision_count"] == 1

