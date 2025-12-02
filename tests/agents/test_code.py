"""Unit tests for src/agents/code.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.code import (
    code_generator_node,
    code_reviewer_node,
)


class TestCodeGeneratorNode:
    """Tests for code_generator_node function."""

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generates_code_on_success(
        self, mock_user, mock_prompt, mock_context, mock_call
    ):
        """Should generate code from LLM output."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        # Code must be >50 chars and not contain stub markers
        mock_call.return_value = {
            "code": "import meep as mp\nimport numpy as np\n\nsim = mp.Simulation(cell_size=mp.Vector3(2,2,0), resolution=10)\nsim.run(until=200)",
            "explanation": "Basic simulation setup",
            "expected_runtime_minutes": 5,
        }
        
        # Design must be >50 chars and not contain stub markers
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",  # Avoids validated_materials check
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation.",
        }
        
        result = code_generator_node(state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "code" in result
        assert "meep" in result["code"]

    @patch("src.agents.code.check_context_or_escalate")
    def test_errors_on_stub_design(self, mock_context):
        """Should error when design_description is a stub."""
        mock_context.return_value = None
        
        state = {
            "current_stage_id": "stage1",
            "design_description": "STUB design TODO replace",  # Contains stub marker
        }
        
        result = code_generator_node(state)
        
        # Returns supervisor_verdict to continue and tries again
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_handles_llm_error(self, mock_user, mock_prompt, mock_context, mock_call):
        """Should handle LLM call failure gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_user.return_value = "content"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation.",
        }
        
        result = code_generator_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "llm_error"

    @patch("src.agents.code.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "design_description": {}}
        
        result = code_generator_node(state)
        
        assert result["awaiting_user_input"] is True


class TestCodeReviewerNode:
    """Tests for code_reviewer_node function."""

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_approves_valid_code(self, mock_prompt, mock_context, mock_call):
        """Should approve valid code."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good",
        }
        
        state = {
            "current_stage_id": "stage1",
            "generated_code": "import meep as mp\nsim = mp.Simulation()",
        }
        
        result = code_reviewer_node(state)
        
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_rejects_with_feedback(self, mock_prompt, mock_context, mock_call):
        """Should reject code and provide feedback."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Missing output saving"}],
            "summary": "Code incomplete",
            "feedback": "Add np.savetxt to save results",
        }
        
        state = {
            "current_stage_id": "stage1",
            "code": "import meep as mp",  # Uses 'code' not 'generated_code'
            "code_revision_count": 0,
        }
        
        result = code_reviewer_node(state)
        
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1
        assert "reviewer_feedback" in result  # Uses 'reviewer_feedback' not 'code_reviewer_feedback'

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_auto_approves_on_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should auto-approve when LLM call fails."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "generated_code": "import meep",
        }
        
        result = code_reviewer_node(state)
        
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "generated_code": "code"}
        
        result = code_reviewer_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_respects_max_revisions(self, mock_prompt, mock_context, mock_call):
        """Should not exceed max revisions."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "needs_revision",
            "issues": [],
            "summary": "Needs more work",
        }
        
        state = {
            "current_stage_id": "stage1",
            "generated_code": "code",
            "code_revision_count": 5,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        result = code_reviewer_node(state)
        
        # Should not increment past max
        assert result["code_revision_count"] == 5

