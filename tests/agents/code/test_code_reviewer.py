"""Tests for code_reviewer_node."""

from unittest.mock import ANY, MagicMock, patch

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
        
        initial_revision_count = base_state.get("code_revision_count", 0)
        result = code_reviewer_node(base_state)
        
        # Verify all required fields are present
        assert "workflow_phase" in result
        assert "last_code_review_verdict" in result
        assert "code_revision_count" in result
        assert "reviewer_issues" in result
        
        # Verify exact values
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "approve"
        assert result["code_revision_count"] == initial_revision_count
        assert result["reviewer_issues"] == []
        
        # Critical: feedback should NOT be set on approval
        assert "reviewer_feedback" not in result
        
        # Verify LLM was called with correct parameters
        mock_prompt.assert_called_once_with("code_reviewer", ANY)
        mock_llm.assert_called_once()
        
        # Verify user_content structure
        call_args = mock_llm.call_args[1]
        assert "user_content" in call_args
        assert "system_prompt" in call_args
        assert call_args["system_prompt"] == "Prompt"
        assert call_args["agent_name"] == "code_reviewer"
        
        # Verify state passed to LLM contains original state keys
        llm_state = call_args["state"]
        assert llm_state["paper_id"] == base_state["paper_id"]
        assert llm_state["code"] == base_state["code"]
        assert llm_state["current_stage_id"] == base_state["current_stage_id"]

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
        
        initial_revision_count = base_state.get("code_revision_count", 0)
        result = code_reviewer_node(base_state)
        
        # Verify all required fields
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == initial_revision_count + 1
        assert result["reviewer_feedback"] == "Fix boundary conditions"
        assert result["reviewer_issues"] == ["Boundary issue"]
        
        # Verify escalation fields are NOT set when under max
        assert "ask_user_trigger" not in result
        assert "awaiting_user_input" not in result
        
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
        
        # Should use summary when feedback is missing
        assert result["reviewer_feedback"] == "Summary of issues"
        assert result["code_revision_count"] == 1
        assert result["reviewer_issues"] == []
        assert result["last_code_review_verdict"] == "needs_revision"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_neither_feedback_nor_summary(self, mock_llm, mock_prompt, base_state):
        """Test reviewer when both feedback and summary are missing."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            # No feedback, no summary
        }
        
        result = code_reviewer_node(base_state)
        
        # Should use fallback message
        assert "reviewer_feedback" in result
        assert result["reviewer_feedback"] == "Missing verdict or feedback in review"
        assert result["code_revision_count"] == 1
        assert result["last_code_review_verdict"] == "needs_revision"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, base_state):
        """Test reviewer hitting max revisions triggers escalation."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = code_reviewer_node(base_state)
        
        # Should not increment past max
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["reviewer_feedback"] == "Fix it"
        
        # Critical: Should trigger user escalation
        assert result["ask_user_trigger"] == "code_review_limit"
        assert result["awaiting_user_input"] is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1
        assert "Code review limit reached" in result["pending_user_questions"][0]
        assert f"Stage: {base_state['current_stage_id']}" in result["pending_user_questions"][0]
        assert f"Attempts: {MAX_CODE_REVISIONS}" in result["pending_user_questions"][0]
        assert result["last_node_before_ask_user"] == "code_review"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_max_revisions_custom_config(self, mock_llm, mock_prompt, base_state):
        """Test reviewer respects custom max_revisions from runtime_config."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        custom_max = 5
        base_state["code_revision_count"] = custom_max
        base_state["runtime_config"] = {"max_code_revisions": custom_max}
        
        result = code_reviewer_node(base_state)
        
        assert result["code_revision_count"] == custom_max
        assert result["ask_user_trigger"] == "code_review_limit"
        assert f"Attempts: {custom_max}/{custom_max}" in result["pending_user_questions"][0]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_llm_failure_defaults_to_needs_revision(self, mock_llm, mock_prompt, base_state):
        """Test reviewer defaults to needs_revision on LLM failure (fail-closed safety)."""
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        initial_count = base_state.get("code_revision_count", 0)
        
        result = code_reviewer_node(base_state)
        
        # Fail-closed: LLM failure should trigger needs_revision (safer than auto-approve)
        assert result["last_code_review_verdict"] == "needs_revision"
        # Revision count should be incremented
        assert result["code_revision_count"] == initial_count + 1
        
        # Verify issues structure
        assert "reviewer_issues" in result
        assert len(result["reviewer_issues"]) == 1
        assert result["reviewer_issues"][0]["severity"] == "minor"
        assert "LLM review unavailable" in result["reviewer_issues"][0]["description"]
        assert "API Error" in result["reviewer_issues"][0]["description"]
        
        # Feedback should be set for needs_revision
        assert "reviewer_feedback" in result

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_llm_failure_empty_error_message(self, mock_llm, mock_prompt, base_state):
        """Test reviewer handles LLM failure with empty error message."""
        mock_prompt.return_value = "Prompt"
        error = Exception()
        error.__str__ = MagicMock(return_value=None)
        mock_llm.side_effect = error
        
        result = code_reviewer_node(base_state)
        
        # Fail-closed: LLM failure should trigger needs_revision
        assert result["last_code_review_verdict"] == "needs_revision"
        assert len(result["reviewer_issues"]) == 1
        assert "LLM review unavailable" in result["reviewer_issues"][0]["description"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_dict_design(self, mock_llm, mock_prompt, base_state):
        """Test that user content is constructed correctly with dict design."""
        base_state["design_description"] = {"key": "value", "nested": {"a": 1}}
        base_state["reviewer_feedback"] = "Previous feedback"
        base_state["code"] = "import meep\nprint('test')"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Verify all sections are present
        assert "CODE TO REVIEW" in user_content
        assert "DESIGN SPEC" in user_content
        assert "REVISION FEEDBACK" in user_content
        
        # Verify content
        assert f"Stage: {base_state['current_stage_id']}" in user_content
        assert "```python" in user_content
        assert base_state["code"] in user_content
        assert '"key": "value"' in user_content  # JSON dump of dict
        assert '"nested"' in user_content
        assert "Previous feedback" in user_content
        
        # Verify order: code should come before design
        code_section = user_content.find("CODE TO REVIEW")
        design_section = user_content.find("DESIGN SPEC")
        feedback_section = user_content.find("REVISION FEEDBACK")
        assert code_section < design_section < feedback_section

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_str_design(self, mock_llm, mock_prompt, base_state):
        """Test that user content is constructed correctly with string design."""
        base_state["design_description"] = "String design"
        base_state["code"] = "import meep\nprint('test')"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        assert "DESIGN SPEC" in user_content
        assert "String design" in user_content
        assert f"Stage: {base_state['current_stage_id']}" in user_content
        assert "```python" in user_content
        assert base_state["code"] in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_no_design(self, mock_llm, mock_prompt, base_state):
        """Test user content construction when design_description is missing."""
        base_state.pop("design_description", None)
        base_state["code"] = "import meep\nprint('test')"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Should still have code section
        assert "CODE TO REVIEW" in user_content
        assert base_state["code"] in user_content
        # Should not have design section
        assert "DESIGN SPEC" not in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_empty_design(self, mock_llm, mock_prompt, base_state):
        """Test user content construction when design_description is empty."""
        base_state["design_description"] = ""
        base_state["code"] = "import meep\nprint('test')"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Empty design should not add design section
        assert "DESIGN SPEC" not in user_content
        assert "CODE TO REVIEW" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_no_feedback(self, mock_llm, mock_prompt, base_state):
        """Test user content construction when reviewer_feedback is missing."""
        base_state.pop("reviewer_feedback", None)
        base_state["code"] = "import meep\nprint('test')"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Should not have revision feedback section
        assert "REVISION FEEDBACK" not in user_content
        assert "CODE TO REVIEW" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_empty_code(self, mock_llm, mock_prompt, base_state):
        """Test user content construction when code is empty."""
        base_state["code"] = ""
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Should still have code section even if empty
        assert "CODE TO REVIEW" in user_content
        assert "```python" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_missing_code(self, mock_llm, mock_prompt, base_state):
        """Test user content construction when code key is missing."""
        base_state.pop("code", None)
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Should handle missing code gracefully
        assert "CODE TO REVIEW" in user_content
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
        
        # Should return escalation exactly as-is
        assert result == escalation
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "context_overflow"
        
        # Should NOT call LLM or prompt builder
        mock_llm.assert_not_called()
        mock_prompt.assert_not_called()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    def test_reviewer_context_state_updates_passed_to_llm(self, mock_check, mock_llm, mock_prompt, base_state):
        """Ensure reviewer LLM call sees any non-blocking context updates."""
        mock_check.return_value = {"context_trimmed": True, "metrics": {"tokens": 1000}}
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        
        code_reviewer_node(base_state)
        
        mock_llm.assert_called_once()
        llm_kwargs = mock_llm.call_args[1]
        assert llm_kwargs["state"]["context_trimmed"] is True
        assert llm_kwargs["state"]["paper_id"] == base_state["paper_id"]
        assert llm_kwargs["state"]["metrics"]["tokens"] == 1000

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_missing_verdict_key(self, mock_llm, mock_prompt, base_state):
        """Test handling when LLM response is missing 'verdict' key."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "issues": [],
            "feedback": "Missing verdict"
        }
        
        # Code uses .get("verdict", "needs_revision") so should default to needs_revision
        result = code_reviewer_node(base_state)
        
        # Should default to needs_revision when verdict is missing
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1
        assert result["reviewer_feedback"] == "Missing verdict"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_invalid_verdict_value(self, mock_llm, mock_prompt, base_state):
        """Test handling when LLM returns invalid verdict value.
        
        Invalid verdicts should be normalized to 'needs_revision' (safer default)
        to avoid passing invalid values through the workflow.
        """
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "invalid_verdict",
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        # Invalid verdicts should be normalized to 'needs_revision' (safer for code)
        assert result["last_code_review_verdict"] == "needs_revision"
        # Should increment counter since normalized to needs_revision
        assert result["code_revision_count"] == base_state.get("code_revision_count", 0) + 1

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_missing_issues_key(self, mock_llm, mock_prompt, base_state):
        """Test handling when LLM response is missing 'issues' key."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            # No issues key
        }
        
        result = code_reviewer_node(base_state)
        
        # Should default to empty list
        assert result["reviewer_issues"] == []
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_issues_not_list(self, mock_llm, mock_prompt, base_state):
        """Test handling when issues is not a list."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": "not a list"  # Wrong type
        }
        
        result = code_reviewer_node(base_state)
        
        # Should accept whatever is returned (type checking happens elsewhere)
        assert result["reviewer_issues"] == "not a list"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_code_revision_count_none(self, mock_llm, mock_prompt, base_state):
        """Test handling when code_revision_count is None."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state["code_revision_count"] = None
        
        result = code_reviewer_node(base_state)
        
        # Should handle None gracefully (increment_counter_with_max handles this)
        assert result["code_revision_count"] == 1

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_code_revision_count_missing(self, mock_llm, mock_prompt, base_state):
        """Test handling when code_revision_count key is missing."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state.pop("code_revision_count", None)
        
        result = code_reviewer_node(base_state)
        
        # Should default to 0 and increment to 1
        assert result["code_revision_count"] == 1

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_code_revision_count_negative(self, mock_llm, mock_prompt, base_state):
        """Test handling when code_revision_count is negative."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state["code_revision_count"] = -5
        
        result = code_reviewer_node(base_state)
        
        # Should increment from negative value
        assert result["code_revision_count"] == -4

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_runtime_config_missing(self, mock_llm, mock_prompt, base_state):
        """Test handling when runtime_config is missing."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state.pop("runtime_config", None)
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = code_reviewer_node(base_state)
        
        # Should use default MAX_CODE_REVISIONS
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result["ask_user_trigger"] == "code_review_limit"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_runtime_config_empty(self, mock_llm, mock_prompt, base_state):
        """Test handling when runtime_config is empty dict."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state["runtime_config"] = {}
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = code_reviewer_node(base_state)
        
        # Should use default MAX_CODE_REVISIONS
        assert result["code_revision_count"] == MAX_CODE_REVISIONS

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_state_not_mutated(self, mock_llm, mock_prompt, base_state):
        """Test that original state is not mutated."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        
        original_state = base_state.copy()
        original_revision_count = base_state.get("code_revision_count", 0)
        
        result = code_reviewer_node(base_state)
        
        # Original state should be unchanged
        assert base_state["code_revision_count"] == original_revision_count
        assert base_state["paper_id"] == original_state["paper_id"]
        assert base_state["code"] == original_state["code"]
        
        # Result should be separate dict
        assert result is not base_state

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_approve_with_issues(self, mock_llm, mock_prompt, base_state):
        """Test approval can include non-blocking issues."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": [
                {"severity": "minor", "description": "Style issue"},
                {"severity": "minor", "description": "Documentation issue"}
            ]
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["last_code_review_verdict"] == "approve"
        assert len(result["reviewer_issues"]) == 2
        assert result["reviewer_issues"][0]["severity"] == "minor"
        assert "reviewer_feedback" not in result

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_needs_revision_increments_from_non_zero(self, mock_llm, mock_prompt, base_state):
        """Test revision count increments correctly from non-zero."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix it"}
        
        base_state["code_revision_count"] = 2
        
        result = code_reviewer_node(base_state)
        
        assert result["code_revision_count"] == 3
        assert result["last_code_review_verdict"] == "needs_revision"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_user_content_contains_exact_code(self, mock_llm, mock_prompt, base_state):
        """Test that user_content contains the exact code from state."""
        mock_prompt.return_value = "Prompt"
        test_code = "import meep as mp\n\nsim = mp.Simulation()\nprint('test')"
        base_state["code"] = test_code
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Code should appear exactly as provided
        assert test_code in user_content
        # Should be in code block
        code_start = user_content.find("```python")
        code_end = user_content.find("```", code_start + 9)
        code_block = user_content[code_start + 9:code_end]
        assert test_code.strip() in code_block.strip()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_stage_id_in_user_content(self, mock_llm, mock_prompt, base_state):
        """Test that current_stage_id appears in user_content."""
        mock_prompt.return_value = "Prompt"
        base_state["current_stage_id"] = "custom_stage_123"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        assert "Stage: custom_stage_123" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_stage_id_missing(self, mock_llm, mock_prompt, base_state):
        """Test handling when current_stage_id is missing."""
        mock_prompt.return_value = "Prompt"
        base_state.pop("current_stage_id", None)
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        # Should use "unknown" as fallback
        assert "Stage: unknown" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_escalation_message_format(self, mock_llm, mock_prompt, base_state):
        """Test that escalation message has correct format when max revisions hit."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Complex feedback with\nmultiple lines\nand details"
        }
        
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        base_state["current_stage_id"] = "stage_5"
        
        result = code_reviewer_node(base_state)
        
        question = result["pending_user_questions"][0]
        
        # Verify all required elements
        assert "Code review limit reached" in question
        assert "Stage: stage_5" in question
        assert f"Attempts: {MAX_CODE_REVISIONS}" in question
        assert "Complex feedback with" in question
        assert "PROVIDE_HINT" in question
        assert "SKIP" in question
        assert "STOP" in question

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_escalate_to_user_string_triggers_ask_user(self, mock_llm, mock_prompt, base_state):
        """Test that string escalate_to_user triggers user escalation."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",  # Even with approve, escalation should take priority
            "escalate_to_user": "Which material model should I use: Drude or tabulated data?",
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        # Should trigger escalation
        assert result["ask_user_trigger"] == "reviewer_escalation"
        assert result["awaiting_user_input"] is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1
        assert "material model" in result["pending_user_questions"][0]
        assert result["last_node_before_ask_user"] == "code_review"
        assert result["reviewer_escalation_source"] == "code_reviewer"
        
        # Should NOT have verdict-related fields (escalation short-circuits)
        assert "last_code_review_verdict" not in result
        assert "code_revision_count" not in result

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_escalate_to_user_false_continues_normally(self, mock_llm, mock_prompt, base_state):
        """Test that boolean false escalate_to_user is ignored."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "escalate_to_user": False,
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        # Should NOT trigger escalation
        assert result.get("ask_user_trigger") != "reviewer_escalation"
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_escalate_to_user_empty_string_continues_normally(self, mock_llm, mock_prompt, base_state):
        """Test that empty string escalate_to_user is ignored."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "escalate_to_user": "",
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        # Should NOT trigger escalation
        assert result.get("ask_user_trigger") != "reviewer_escalation"
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_escalate_to_user_whitespace_only_continues_normally(self, mock_llm, mock_prompt, base_state):
        """Test that whitespace-only escalate_to_user is ignored."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "escalate_to_user": "   \n\t  ",
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        # Should NOT trigger escalation (strip() makes it empty)
        assert result.get("ask_user_trigger") != "reviewer_escalation"
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_escalate_to_user_takes_priority_over_verdict(self, mock_llm, mock_prompt, base_state):
        """Test that escalation takes priority even with needs_revision verdict."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "escalate_to_user": "Should I use 2D or 3D simulation?",
            "feedback": "Some feedback",
            "issues": [{"severity": "major", "description": "Issue"}]
        }
        
        result = code_reviewer_node(base_state)
        
        # Should trigger escalation instead of normal verdict handling
        assert result["ask_user_trigger"] == "reviewer_escalation"
        assert "Should I use 2D or 3D" in result["pending_user_questions"][0]
        
        # Should NOT increment revision count (escalation happens before verdict processing)
        assert "code_revision_count" not in result
        assert "last_code_review_verdict" not in result
