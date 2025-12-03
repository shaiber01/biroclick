"""Tests for plan_node."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.planning import plan_node
from schemas.state import MAX_REPLANS, ReproState


@pytest.fixture(name="mock_state")
def plan_state_alias(plan_state):
    return plan_state


@pytest.fixture(name="mock_llm_output")
def planner_output_alias(planner_llm_output):
    return planner_llm_output


class TestPlanNode:
    """Tests for plan_node function."""

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_basic_success(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test successful plan generation and all output fields."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        # Mock progress and sync helpers
        mock_init.return_value = {**mock_state, "progress": ["stage1"]}
        mock_sync.return_value = {**mock_state, "progress": ["stage1"], "extracted_parameters": [{"p": 1}]}

        result = plan_node(mock_state)
        
        # Assert specific output fields
        assert result["workflow_phase"] == "planning"
        assert result["plan"]["paper_id"] == "test_paper"
        assert result["plan"]["stages"] == mock_llm_output["stages"]
        assert result["planned_materials"] == ["Gold"]
        assert result["assumptions"] == {"assumption1": "true"}
        assert result["paper_domain"] == "plasmonics"
        
        # Verify helpers were used and their output propagated
        mock_init.assert_called_once()
        mock_sync.assert_called_once()
        assert result["progress"] == ["stage1"]
        assert result["extracted_parameters"] == [{"p": 1}]

    def test_plan_node_missing_text(self, mock_state):
        """Test error when paper text is empty."""
        mock_state["paper_text"] = ""
        result = plan_node(mock_state)
        
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result["awaiting_user_input"] is True
        assert "missing or too short" in result["pending_user_questions"][0]

    def test_plan_node_short_text(self, mock_state):
        """Test error when paper text is too short (< 100 chars)."""
        mock_state["paper_text"] = "Too short"
        result = plan_node(mock_state)
        
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result["awaiting_user_input"] is True
        assert "too short" in result["pending_user_questions"][0]

    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_context_escalation(self, mock_check, mock_state):
        """Test immediate return on context check escalation."""
        escalation = {"awaiting_user_input": True, "reason": "context"}
        mock_check.return_value = escalation
        
        result = plan_node(mock_state)
        assert result == escalation

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_llm_failure(self, mock_check, mock_llm, mock_state):
        """Test handling of LLM failure."""
        mock_check.return_value = None
        mock_llm.side_effect = Exception("LLM Error")
        
        result = plan_node(mock_state)
        
        assert result["workflow_phase"] == "planning"
        assert result["ask_user_trigger"] == "llm_error"
        assert "LLM Error" in result["pending_user_questions"][0]

    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_progress_initialization_failure(self, mock_check, mock_llm, mock_init, mock_llm_output, mock_state):
        """Test handling of valid LLM output but invalid structure for progress init."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_init.side_effect = Exception("Bad Structure")
        
        result = plan_node(mock_state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Progress initialization failed" in result["planner_feedback"]
        assert "Bad Structure" in result["planner_feedback"]
        assert result["replan_count"] == 1  # Incremented from 0

    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_progress_init_failure_max_replans(self, mock_check, mock_llm, mock_init, mock_llm_output, mock_state):
        """Test replan count cap on structure failure."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_init.side_effect = Exception("Bad Structure")
        
        mock_state["replan_count"] = 3
        mock_state["runtime_config"] = {"max_replans": 3}
        
        result = plan_node(mock_state)
        
        assert result["replan_count"] == 3  # Should not exceed max

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_replan_prompt_injection(self, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that replan count triggers prompt modification."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_state["replan_count"] = 2
        
        plan_node(mock_state)
        
        call_kwargs = mock_llm.call_args[1]
        assert "Replan Attempt #2" in call_kwargs["system_prompt"]

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_preserves_paper_id(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that paper_id in state takes precedence over LLM output."""
        mock_check.return_value = None
        # LLM returns different ID
        mock_output = mock_llm_output.copy()
        mock_output["paper_id"] = "wrong_id"
        mock_llm.return_value = mock_output
        
        mock_init.return_value = mock_state
        mock_sync.return_value = mock_state
        
        mock_state["paper_id"] = "correct_id"
        result = plan_node(mock_state)
        
        assert result["plan"]["paper_id"] == "correct_id"


# ═══════════════════════════════════════════════════════════════════════
# plan_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════
