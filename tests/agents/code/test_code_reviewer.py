"""Tests for code_reviewer_node."""

from unittest.mock import ANY, patch

import pytest

from src.agents.code import code_reviewer_node
from schemas.state import MAX_CODE_REVISIONS


@pytest.fixture(name="base_state")
def code_base_state(code_state):
    return code_state


class TestCodeReviewerNode:
    """Tests for code_reviewer_node."""

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer approving code."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "approve"
        assert result["code_revision_count"] == 0
        assert result["reviewer_issues"] == []
        assert "reviewer_feedback" not in result # Should not set feedback on approval
        
        # Verify call args with loose match for state since it might have metrics added
        mock_prompt.assert_called_once_with("code_reviewer", ANY)
        # Verify key state elements are present
        called_state = mock_prompt.call_args[0][1]
        assert called_state["paper_id"] == base_state["paper_id"]
        assert called_state["code"] == base_state["code"]
        mock_llm.assert_called_once()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, base_state):
        """Test reviewer requesting revision."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix boundary conditions",
            "issues": ["Boundary issue"]
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1
        assert result["reviewer_feedback"] == "Fix boundary conditions"
        assert result["reviewer_issues"] == ["Boundary issue"]
        
        # Verify call args with loose match for state
        mock_prompt.assert_called_once_with("code_reviewer", ANY)
        mock_llm.assert_called_once()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_uses_summary_fallback(self, mock_llm, mock_prompt, base_state):
        """Test reviewer uses summary as feedback if feedback is missing."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "summary": "Summary of issues",
            # No feedback key
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["reviewer_feedback"] == "Summary of issues"
        assert result["code_revision_count"] == 1
        assert result["reviewer_issues"] == []

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, base_state):
        """Test reviewer hitting max revisions."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = code_reviewer_node(base_state)
        
        # Should not increment past max
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["reviewer_feedback"] == "Fix it"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer defaults to auto-approve on LLM failure (non-critical)."""
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = code_reviewer_node(base_state)
        
        # Auto-approve logic for reviewers
        assert result["last_code_review_verdict"] == "approve" 
        # Should usually log error but continue
        assert result["code_revision_count"] == 0
        assert result["reviewer_issues"][0]["severity"] == "minor"
        assert "LLM review unavailable" in result["reviewer_issues"][0]["description"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_dict_design(self, mock_llm, mock_prompt, base_state):
        """Test that user content is constructed correctly with dict design."""
        base_state["design_description"] = {"key": "value"}
        base_state["reviewer_feedback"] = "Previous feedback"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        assert "CODE TO REVIEW" in user_content
        assert "DESIGN SPEC" in user_content
        assert '"key": "value"' in user_content # JSON dump of dict
        assert "REVISION FEEDBACK" in user_content
        assert "Previous feedback" in user_content
        assert f"Stage: {base_state['current_stage_id']}" in user_content
        assert "```python" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_str_design(self, mock_llm, mock_prompt, base_state):
        """Test that user content is constructed correctly with string design."""
        base_state["design_description"] = "String design"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        assert "DESIGN SPEC" in user_content
        assert "String design" in user_content
        assert f"Stage: {base_state['current_stage_id']}" in user_content
        assert "```python" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    def test_reviewer_context_escalation_short_circuits(self, mock_check, mock_llm, mock_prompt, base_state):
        """Ensure reviewer node respects context escalations before prompting LLM."""
        escalation = {
            "workflow_phase": "code_review",
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context too large"],
            "awaiting_user_input": True,
        }
        mock_check.return_value = escalation
        
        result = code_reviewer_node(base_state)
        
        assert result == escalation
        mock_llm.assert_not_called()
        mock_prompt.assert_not_called()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    def test_reviewer_context_state_updates_passed_to_llm(self, mock_check, mock_llm, mock_prompt, base_state):
        """Ensure reviewer LLM call sees any non-blocking context updates."""
        mock_check.return_value = {"context_trimmed": True}
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        
        code_reviewer_node(base_state)
        
        mock_llm.assert_called_once()
        llm_kwargs = mock_llm.call_args[1]
        assert llm_kwargs["state"]["context_trimmed"] is True
        assert llm_kwargs["state"]["paper_id"] == base_state["paper_id"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_missing_verdict_key(self, mock_llm, mock_prompt, base_state):
        """Test handling when LLM response is missing 'verdict' key."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "issues": [],
            "feedback": "Missing verdict"
        }
        
        # This should NOT raise KeyError
        # It should probably default to something or raise a handled error
        # For now, we assert that it raises KeyError to confirm the bug
        # OR if we are fixing the code, we assert the desired behavior.
        # User: "If a test reveals a bug, KEEP THE TEST FAILING"
        # But also "We will later fix the component under test to make the test pass."
        # So I will write the assertion for the DESIRED behavior (safe handling), which will fail.
        
        result = code_reviewer_node(base_state)
        
        # Desired behavior: treat as failure or default to needs_revision?
        # If we can't parse verdict, we probably shouldn't blindly approve.
        # Let's say we expect it to fallback to "needs_revision" or log error.
        # Checking the code: it calls create_llm_error_auto_approve on exception, 
        # but here no exception is raised during call_agent_with_metrics.
        
        assert result["last_code_review_verdict"] in ["approve", "needs_revision"]
        # If it crashes with KeyError, this line is never reached and test fails (Error).

