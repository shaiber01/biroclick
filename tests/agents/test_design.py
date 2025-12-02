"""Unit tests for src/agents/design.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.design import (
    simulation_designer_node,
    design_reviewer_node,
)


class TestSimulationDesignerNode:
    """Tests for simulation_designer_node function."""

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.build_user_content_for_designer")
    @patch("src.agents.design.get_stage_design_spec")
    def test_creates_design_on_success(
        self, mock_spec, mock_user, mock_prompt, mock_context, mock_call
    ):
        """Should create design from LLM output."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        mock_spec.return_value = "2D_light"
        mock_call.return_value = {
            "geometry": {"type": "sphere", "radius": "50nm"},
            "sources": [{"type": "plane_wave"}],
            "monitors": [{"type": "flux"}],
        }
        
        state = {"current_stage_id": "stage1"}
        
        result = simulation_designer_node(state)
        
        assert result["workflow_phase"] == "design"
        assert "design_description" in result
        assert result["design_description"]["geometry"]["type"] == "sphere"

    @patch("src.agents.design.check_context_or_escalate")
    def test_errors_on_missing_stage_id(self, mock_context):
        """Should error when current_stage_id is missing."""
        mock_context.return_value = None
        state = {"current_stage_id": None}
        
        result = simulation_designer_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"

    @patch("src.agents.design.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1"}
        
        result = simulation_designer_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.build_user_content_for_designer")
    @patch("src.agents.design.get_stage_design_spec")
    def test_handles_llm_error(
        self, mock_spec, mock_user, mock_prompt, mock_context, mock_call
    ):
        """Should handle LLM call failure gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_user.return_value = "content"
        mock_spec.return_value = "unknown"
        mock_call.side_effect = Exception("API error")
        
        state = {"current_stage_id": "stage1"}
        
        result = simulation_designer_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "llm_error"

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.build_user_content_for_designer")
    @patch("src.agents.design.get_stage_design_spec")
    def test_includes_new_assumptions(
        self, mock_spec, mock_user, mock_prompt, mock_context, mock_call
    ):
        """Should include new assumptions from design."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_user.return_value = "content"
        mock_spec.return_value = "2D"
        mock_call.return_value = {
            "geometry": {},
            "new_assumptions": [{"type": "boundary", "value": "PML"}],
        }
        
        state = {"current_stage_id": "stage1", "assumptions": {}}
        
        result = simulation_designer_node(state)
        
        assert "assumptions" in result
        assert len(result["assumptions"]["global_assumptions"]) == 1


class TestDesignReviewerNode:
    """Tests for design_reviewer_node function."""

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.get_plan_stage")
    def test_approves_valid_design(self, mock_stage, mock_prompt, mock_context, mock_call):
        """Should approve a valid design."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_stage.return_value = {"stage_id": "stage1"}
        mock_call.return_value = {
            "verdict": "approve",
            "issues": [],
            "summary": "Design looks good",
        }
        
        state = {
            "current_stage_id": "stage1",
            "design_description": {"geometry": {"type": "sphere"}},
        }
        
        result = design_reviewer_node(state)
        
        assert result["workflow_phase"] == "design_review"
        assert result["last_design_review_verdict"] == "approve"

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.get_plan_stage")
    def test_rejects_with_feedback(self, mock_stage, mock_prompt, mock_context, mock_call):
        """Should reject design and provide feedback."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stage.return_value = {"stage_id": "stage1"}
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Missing boundary conditions"}],
            "summary": "Design incomplete",
            "feedback": "Add PML boundaries",
        }
        
        state = {
            "current_stage_id": "stage1",
            "design_description": {"geometry": {}},
            "design_revision_count": 0,
        }
        
        result = design_reviewer_node(state)
        
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1
        assert "reviewer_feedback" in result

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.get_plan_stage")
    def test_auto_approves_on_llm_error(
        self, mock_stage, mock_prompt, mock_context, mock_call
    ):
        """Should auto-approve when LLM call fails."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stage.return_value = {"stage_id": "stage1"}
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "design_description": {},
        }
        
        result = design_reviewer_node(state)
        
        assert result["last_design_review_verdict"] == "approve"

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow.
        
        Note: Patches base.py because design_reviewer_node uses @with_context_check decorator.
        """
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "design_description": {}}
        
        result = design_reviewer_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.get_plan_stage")
    def test_respects_max_revisions(self, mock_stage, mock_prompt, mock_context, mock_call):
        """Should not exceed max revisions."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_stage.return_value = {"stage_id": "stage1"}
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": [],
            "summary": "Needs more work",
        }
        
        state = {
            "current_stage_id": "stage1",
            "design_description": {},
            "design_revision_count": 10,  # Already at max
            "runtime_config": {"max_design_revisions": 10},
        }
        
        result = design_reviewer_node(state)
        
        # Should not increment past max
        assert result["design_revision_count"] == 10

