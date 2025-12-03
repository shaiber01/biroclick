"""Tests for design_reviewer_node."""

from unittest.mock import patch

import pytest

from schemas.state import MAX_DESIGN_REVISIONS

from src.agents.design import design_reviewer_node


@pytest.fixture(name="base_state")
def design_base_state(design_state):
    return design_state


class TestDesignReviewerNode:
    """Tests for design_reviewer_node."""

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_approve(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer approving design."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        base_state["design_description"] = {"some": "design"}
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        assert result["design_revision_count"] == 0
        assert result["reviewer_issues"] == []
        assert result["workflow_phase"] == "design_review"
        assert "reviewer_feedback" not in result
        mock_prompt.assert_called_once_with("design_reviewer", base_state)
        mock_llm.assert_called_once()

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_plan_stage")
    def test_reviewer_constructs_prompt_correctly(self, mock_get_plan_stage, mock_llm, mock_prompt, mock_check, base_state):
        """Test that user content includes design, stage spec, and feedback."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        
        # Setup state
        base_state["design_description"] = {"method": "FDTD"}
        base_state["reviewer_feedback"] = "Previous feedback"
        mock_get_plan_stage.return_value = {"stage_id": "stage_1", "name": "Test Stage"}
        
        design_reviewer_node(base_state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Verify Design inclusion
        assert "DESIGN TO REVIEW" in user_content
        assert '"method": "FDTD"' in user_content
        
        # Verify Stage Spec inclusion
        assert "PLAN STAGE SPEC" in user_content
        assert '"name": "Test Stage"' in user_content
        
        # Verify Feedback inclusion
        assert "REVISION FEEDBACK" in user_content
        assert "Previous feedback" in user_content
        mock_prompt.assert_called_once_with("design_reviewer", base_state)
        mock_llm.assert_called_once()

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer requesting revision."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Add more details",
            "issues": ["Missing parameters"]
        }
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1
        assert result["reviewer_feedback"] == "Add more details"
        assert result["reviewer_issues"] == ["Missing parameters"]
        assert result["workflow_phase"] == "design_review"
        mock_prompt.assert_called_once_with("design_reviewer", base_state)
        mock_llm.assert_called_once()

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer hitting max revisions."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "more"}
        
        base_state["design_revision_count"] = 3 # Already at max
        
        result = design_reviewer_node(base_state)
        
        # Should not increment count, but should capture feedback
        assert result["design_revision_count"] == 3
        assert result["last_design_review_verdict"] == "needs_revision"
        assert "more" in result["reviewer_feedback"]
        mock_llm.assert_called_once()

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer defaults to auto-approve on LLM failure."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        # Verify issues contain error info
        assert result["reviewer_issues"] is not None
        assert any("LLM review unavailable" in issue.get("description", "") 
                  for issue in result["reviewer_issues"])
        assert result["design_revision_count"] == 0

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_missing_verdict(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test handling when LLM returns JSON without 'verdict'."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"feedback": "Some feedback but no verdict"}
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        assert result["reviewer_issues"] == []
        assert result["design_revision_count"] == 0

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.base.check_context_or_escalate")
    def test_reviewer_respects_context_escalation(self, mock_context, mock_prompt, mock_llm, base_state):
        """Ensure decorator short-circuits when user input is required."""
        escalation = {
            "workflow_phase": "design_review",
            "awaiting_user_input": True,
            "ask_user_trigger": "context_missing",
        }
        mock_context.return_value = escalation

        result = design_reviewer_node(base_state)

        assert result == escalation
        mock_prompt.assert_not_called()
        mock_llm.assert_not_called()

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.base.check_context_or_escalate")
    def test_reviewer_merges_context_updates_into_state(self, mock_context, mock_prompt, mock_llm, base_state):
        """Context metadata returned from the decorator must be propagated to the LLM call."""
        mock_context.return_value = {"context_refresh": True}
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}

        design_reviewer_node(base_state)

        merged_state = mock_llm.call_args[1]["state"]
        assert merged_state["context_refresh"] is True
        mock_prompt.assert_called_once_with("design_reviewer", merged_state)

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_uses_summary_when_feedback_missing(self, mock_llm, mock_prompt, mock_check, base_state):
        """Reviewer should fall back to summary text when explicit feedback is absent."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "summary": "Need tighter boundary conditions",
            "issues": [],
        }

        result = design_reviewer_node(base_state)

        assert result["reviewer_feedback"] == "Need tighter boundary conditions"
        assert result["design_revision_count"] == 1
        mock_llm.assert_called_once()

    @patch("src.agents.design.increment_counter_with_max")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_respects_runtime_revision_limit(
        self,
        mock_llm,
        mock_prompt,
        mock_check,
        mock_increment,
        base_state,
    ):
        """Increment helper must be invoked with the configured max."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "expand grid"}
        mock_increment.return_value = (5, True)

        result = design_reviewer_node(base_state)

        mock_increment.assert_called_once_with(
            base_state,
            "design_revision_count",
            "max_design_revisions",
            MAX_DESIGN_REVISIONS,
        )
        assert result["design_revision_count"] == 5

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_none_design(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when design_description is None."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        base_state["design_description"] = None
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        mock_llm.assert_called_once()
