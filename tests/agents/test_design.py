"""
Tests for Design Agents (SimulationDesignerAgent, DesignReviewerAgent).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.design import simulation_designer_node, design_reviewer_node

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Base state for design tests."""
    return {
        "paper_id": "test_paper",
        "current_stage_id": "stage_1_sim",
        "plan": {
            "stages": [
                {"stage_id": "stage_1_sim", "targets": ["Fig1"]}
            ]
        },
        "design_revision_count": 0,
        "runtime_config": {
            "max_design_revisions": 3
        },
        "assumptions": {"global_assumptions": []}
    }

# ═══════════════════════════════════════════════════════════════════════
# simulation_designer_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSimulationDesignerNode:
    """Tests for simulation_designer_node."""

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_success(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test successful design generation."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "design_description": "FDTD simulation setup...",
            "new_assumptions": [{"id": "A1", "description": "Use PML"}]
        }
        
        result = simulation_designer_node(base_state)
        
        assert result["workflow_phase"] == "design"
        assert result["design_description"] == mock_llm.return_value
        assert len(result["assumptions"]["global_assumptions"]) == 1

    def test_designer_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = simulation_designer_node(base_state)
        
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_handles_llm_failure(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = simulation_designer_node(base_state)
        
        # Should escalate to user
        assert result["ask_user_trigger"] == "llm_error"
        assert result["awaiting_user_input"] is True

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_injects_feedback(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test feedback injection into prompt."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt" # Ensure it returns a string
        mock_llm.return_value = {}
        base_state["reviewer_feedback"] = "Fix mesh size"
        
        simulation_designer_node(base_state)
        
        # Verify prompt contains feedback
        call_kwargs = mock_llm.call_args[1]
        assert "Fix mesh size" in call_kwargs["system_prompt"]


# ═══════════════════════════════════════════════════════════════════════
# design_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDesignReviewerNode:
    """Tests for design_reviewer_node."""

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer approving design."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        assert result["design_revision_count"] == 0

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, base_state):
        """Test reviewer requesting revision."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Add more details"
        }
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1
        assert "Add more details" in result["reviewer_feedback"]

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, base_state):
        """Test reviewer hitting max revisions."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision"}
        
        base_state["design_revision_count"] = 3 # Already at max
        
        result = design_reviewer_node(base_state)
        
        # Should not increment past max
        assert result["design_revision_count"] == 3
        assert result["last_design_review_verdict"] == "needs_revision"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer defaults to auto-approve on LLM failure."""
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
