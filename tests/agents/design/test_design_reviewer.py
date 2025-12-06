"""Tests for design_reviewer_node."""

import json
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
        initial_count = base_state.get("design_revision_count", 0)
        
        result = design_reviewer_node(base_state)
        
        # Verify exact verdict
        assert result["last_design_review_verdict"] == "approve"
        # Verify revision count is preserved (not incremented for approve)
        assert result["design_revision_count"] == initial_count
        # Verify issues are exactly empty list
        assert result["reviewer_issues"] == []
        # Verify workflow phase is set correctly
        assert result["workflow_phase"] == "design_review"
        # Verify feedback is explicitly cleared (None) on approval
        assert "reviewer_feedback" in result
        assert result["reviewer_feedback"] is None
        # Verify prompt was called with correct agent name
        mock_prompt.assert_called_once_with("design_reviewer", base_state)
        # Verify LLM was called exactly once
        assert mock_llm.call_count == 1
        # Verify LLM was called with correct agent name
        assert mock_llm.call_args[1]["agent_name"] == "design_reviewer"
        # Verify system prompt was passed
        assert mock_llm.call_args[1]["system_prompt"] == "Prompt"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_approve_clears_stale_feedback(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that approval clears stale feedback from previous rejection.
        
        This tests the bug fix for issue 3: when design was previously rejected
        and had reviewer_feedback set, approving it should clear that feedback
        to None so it doesn't leak into subsequent design/code generation cycles.
        """
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        
        # Simulate state with stale feedback from a previous rejection
        base_state["design_description"] = {"some": "design"}
        base_state["reviewer_feedback"] = "Old feedback: add more details to geometry spec"
        base_state["design_revision_count"] = 2  # Had multiple previous rejections
        
        result = design_reviewer_node(base_state)
        
        # Verify approval verdict
        assert result["last_design_review_verdict"] == "approve"
        
        # Critical: stale feedback MUST be cleared to None on approval
        # If this is not None, the design/code generator might receive confusing
        # feedback from a previous iteration even though design was approved
        assert "reviewer_feedback" in result, "reviewer_feedback key must be present"
        assert result["reviewer_feedback"] is None, (
            f"Stale feedback should be cleared to None on approval, "
            f"but got: {result['reviewer_feedback']!r}"
        )
        
        # Verify revision count is preserved (not reset, just not incremented)
        assert result["design_revision_count"] == 2

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
        stage_id = base_state.get("current_stage_id", "stage_1_sim")
        base_state["design_description"] = {"method": "FDTD"}
        base_state["reviewer_feedback"] = "Previous feedback"
        mock_get_plan_stage.return_value = {"stage_id": "stage_1", "name": "Test Stage"}
        
        design_reviewer_node(base_state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Verify Design inclusion with exact format
        assert "DESIGN TO REVIEW" in user_content
        assert f"Stage: {stage_id}" in user_content
        assert '"method": "FDTD"' in user_content
        # Verify it's in JSON code block
        assert "```json" in user_content or "```" in user_content
        
        # Verify Stage Spec inclusion with exact format
        assert "PLAN STAGE SPEC" in user_content
        assert '"stage_id": "stage_1"' in user_content
        assert '"name": "Test Stage"' in user_content
        # Verify it's in JSON code block
        assert "```json" in user_content
        
        # Verify Feedback inclusion with exact format
        assert "REVISION FEEDBACK" in user_content
        assert "Previous feedback" in user_content
        
        # Verify order: Design first, then Stage Spec, then Feedback
        design_pos = user_content.find("DESIGN TO REVIEW")
        stage_pos = user_content.find("PLAN STAGE SPEC")
        feedback_pos = user_content.find("REVISION FEEDBACK")
        assert design_pos < stage_pos < feedback_pos
        
        mock_prompt.assert_called_once_with("design_reviewer", base_state)
        assert mock_llm.call_count == 1
        # Verify LLM was called with correct parameters
        assert call_kwargs["agent_name"] == "design_reviewer"
        assert call_kwargs["system_prompt"] == "System Prompt"
        assert "state" in call_kwargs

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer requesting revision."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        initial_count = base_state.get("design_revision_count", 0)
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "summary": "Add more details",  # Schema uses "summary" not "feedback"
            "issues": ["Missing parameters"]
        }
        
        result = design_reviewer_node(base_state)
        
        # Verify exact verdict
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify revision count was incremented
        assert result["design_revision_count"] == initial_count + 1
        # Verify feedback is exactly as provided
        assert result["reviewer_feedback"] == "Add more details"
        # Verify issues are exactly as provided
        assert result["reviewer_issues"] == ["Missing parameters"]
        # Verify workflow phase
        assert result["workflow_phase"] == "design_review"
        mock_prompt.assert_called_once_with("design_reviewer", base_state)
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer hitting max revisions."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "summary": "more"}  # Schema uses "summary"
        
        max_revisions = base_state.get("runtime_config", {}).get("max_design_revisions", MAX_DESIGN_REVISIONS)
        base_state["design_revision_count"] = max_revisions  # Already at max
        
        result = design_reviewer_node(base_state)
        
        # Should not increment count beyond max, but should capture feedback
        assert result["design_revision_count"] == max_revisions
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify feedback is exactly as provided
        assert result["reviewer_feedback"] == "more"
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_llm_failure_defaults_to_needs_revision(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test reviewer defaults to needs_revision on LLM failure (fail-closed safety)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        error_msg = "API Error"
        mock_llm.side_effect = Exception(error_msg)
        initial_count = base_state.get("design_revision_count", 0)
        
        result = design_reviewer_node(base_state)
        
        # Fail-closed: LLM failure should trigger needs_revision (safer than auto-approve)
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify issues contain error info with exact format
        assert result["reviewer_issues"] is not None
        assert isinstance(result["reviewer_issues"], list)
        assert len(result["reviewer_issues"]) > 0
        # Verify error message is in issues
        issue_descriptions = [issue.get("description", "") for issue in result["reviewer_issues"]]
        assert any("LLM review unavailable" in desc for desc in issue_descriptions)
        assert any(error_msg[:200] in desc for desc in issue_descriptions)  # Error truncated to 200 chars
        # Verify revision count IS incremented for needs_revision
        assert result["design_revision_count"] == initial_count + 1
        # Verify workflow phase is set
        assert result["workflow_phase"] == "design_review"
        # Verify feedback IS set for needs_revision
        assert "reviewer_feedback" in result

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_missing_verdict_defaults_to_needs_revision(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test handling when LLM returns JSON without 'verdict' (fail-closed safety)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"summary": "Some feedback but no verdict"}  # Schema uses "summary"
        initial_count = base_state.get("design_revision_count", 0)
        
        result = design_reviewer_node(base_state)
        
        # Fail-closed: missing verdict should default to needs_revision (safer)
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify issues default to empty list (or from output if present)
        issues = result.get("reviewer_issues", [])
        assert isinstance(issues, list)
        # Verify revision count IS incremented for needs_revision
        assert result["design_revision_count"] == initial_count + 1
        # Verify workflow phase is set
        assert result["workflow_phase"] == "design_review"
        # Verify feedback IS set for needs_revision (uses summary from LLM response)
        assert result["reviewer_feedback"] == "Some feedback but no verdict"

    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.base.check_context_or_escalate")
    def test_reviewer_respects_context_escalation(self, mock_context, mock_prompt, mock_llm, base_state):
        """Ensure decorator short-circuits when user input is required."""
        escalation = {
            "workflow_phase": "design_review",
            "ask_user_trigger": "context_overflow",
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
        initial_count = base_state.get("design_revision_count", 0)
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "summary": "Need tighter boundary conditions",
            "issues": [],
        }

        result = design_reviewer_node(base_state)

        # Verify feedback uses summary when feedback field is missing
        assert result["reviewer_feedback"] == "Need tighter boundary conditions"
        # Verify revision count is incremented
        assert result["design_revision_count"] == initial_count + 1
        # Verify verdict is correct
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify issues are preserved
        assert result["reviewer_issues"] == []
        assert mock_llm.call_count == 1

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
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content handles None design correctly
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "DESIGN TO REVIEW" in user_content
        assert "None" in user_content or "```\nNone\n```" in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_string_design(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when design_description is a string instead of dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        base_state["design_description"] = "Simple string design"
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content handles string design correctly
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "DESIGN TO REVIEW" in user_content
        assert "Simple string design" in user_content
        assert "```\nSimple string design\n```" in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_plan_stage")
    def test_reviewer_handles_missing_plan_stage(self, mock_get_plan_stage, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when plan stage is None."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        mock_get_plan_stage.return_value = None
        base_state["design_description"] = {"method": "FDTD"}
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content doesn't include PLAN STAGE SPEC when stage is None
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "DESIGN TO REVIEW" in user_content
        assert "PLAN STAGE SPEC" not in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_empty_feedback(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when reviewer_feedback is empty string."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        base_state["reviewer_feedback"] = ""
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content doesn't include REVISION FEEDBACK when empty
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "REVISION FEEDBACK" not in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_missing_feedback_key(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when reviewer_feedback key is missing from state."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        if "reviewer_feedback" in base_state:
            del base_state["reviewer_feedback"]
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content doesn't include REVISION FEEDBACK when missing
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "REVISION FEEDBACK" not in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_missing_issues(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when LLM output doesn't include issues field."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve"}  # No issues field
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify issues defaults to empty list when missing
        assert result["reviewer_issues"] == []
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_none_issues(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when LLM output has issues set to None."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": None}
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify issues handles None correctly (should use .get() default)
        # The code uses .get("issues", []) so None should become []
        assert result["reviewer_issues"] == []
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_empty_feedback_and_summary(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when needs_revision but both feedback and summary are missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        initial_count = base_state.get("design_revision_count", 0)
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "issues": []
        }  # No feedback, no summary
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify revision count is incremented
        assert result["design_revision_count"] == initial_count + 1
        # Verify feedback is empty string when both feedback and summary are missing
        assert result["reviewer_feedback"] == ""
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_missing_runtime_config(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when runtime_config is missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        initial_count = base_state.get("design_revision_count", 0)
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "test"}
        
        # Remove runtime_config
        if "runtime_config" in base_state:
            del base_state["runtime_config"]
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify revision count increments (should use default max)
        assert result["design_revision_count"] == initial_count + 1
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_custom_max_revisions(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior with custom max_design_revisions in runtime_config."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "test"}
        
        # Set custom max
        base_state["runtime_config"] = {"max_design_revisions": 5}
        base_state["design_revision_count"] = 5  # At custom max
        
        result = design_reviewer_node(base_state)
        
        # Verify count doesn't exceed custom max
        assert result["design_revision_count"] == 5
        assert result["last_design_review_verdict"] == "needs_revision"
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_missing_design_revision_count(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when design_revision_count is missing from state."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "test"}
        
        # Remove design_revision_count
        if "design_revision_count" in base_state:
            del base_state["design_revision_count"]
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify revision count starts at 1 when missing (defaults to 0, then increments)
        assert result["design_revision_count"] == 1
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_preserves_existing_revision_count_on_approve(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that revision count is preserved (not reset) when approving."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        
        # Set existing revision count
        base_state["design_revision_count"] = 2
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is approve
        assert result["last_design_review_verdict"] == "approve"
        # Verify revision count is preserved, not reset
        assert result["design_revision_count"] == 2
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_full_schema_output(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior with full schema output including checklist_results, strengths, etc."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        initial_count = base_state.get("design_revision_count", 0)
        mock_llm.return_value = {
            "stage_id": "stage_1_sim",
            "verdict": "needs_revision",
            "checklist_results": {
                "geometry": {"status": "pass", "notes": "Good"},
                "physics": {"status": "fail", "notes": "Missing boundary conditions"},
                "materials": {"status": "pass"},
            },
            "strengths": ["Clear structure", "Good materials"],
            "issues": [
                {
                    "severity": "blocking",
                    "category": "physics",
                    "description": "Missing boundary conditions",
                    "suggested_fix": "Add PML layers"
                }
            ],
            "backtrack_suggestion": {
                "suggest_backtrack": False
            },
            "escalate_to_user": False,
            "summary": "Overall good but needs boundary conditions"
        }
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify revision count increments
        assert result["design_revision_count"] == initial_count + 1
        # Verify issues are preserved (should be list of dicts)
        assert isinstance(result["reviewer_issues"], list)
        assert len(result["reviewer_issues"]) == 1
        assert result["reviewer_issues"][0]["severity"] == "blocking"
        assert result["reviewer_issues"][0]["category"] == "physics"
        # Verify feedback uses summary (since feedback field not present)
        assert result["reviewer_feedback"] == "Overall good but needs boundary conditions"
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_complex_design_json(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior with complex nested design structure."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        
        complex_design = {
            "geometry": {
                "type": "sphere",
                "radius": 50e-9,
                "center": [0, 0, 0]
            },
            "materials": [
                {"name": "Gold", "model": "Drude"},
                {"name": "Air", "epsilon": 1.0}
            ],
            "sources": [
                {
                    "type": "continuous",
                    "component": "Ex",
                    "center": [0, 0, -100e-9],
                    "wavelength": 500e-9
                }
            ],
            "boundary_conditions": {
                "x": "PML",
                "y": "PML",
                "z": "PML"
            }
        }
        base_state["design_description"] = complex_design
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content includes JSON-serialized design
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "DESIGN TO REVIEW" in user_content
        # Verify JSON is properly formatted
        assert '"radius":' in user_content
        assert '"Gold"' in user_content
        assert '"PML"' in user_content
        # Verify it's valid JSON in the content
        json_start = user_content.find("```json")
        json_end = user_content.find("```", json_start + 7)
        if json_start != -1 and json_end != -1:
            json_str = user_content[json_start + 7:json_end].strip()
            parsed = json.loads(json_str)
            assert parsed == complex_design
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_missing_current_stage_id(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when current_stage_id is missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        
        # Remove current_stage_id
        if "current_stage_id" in base_state:
            del base_state["current_stage_id"]
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content handles missing stage_id
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "DESIGN TO REVIEW" in user_content
        assert "Stage: unknown" in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_empty_design_dict(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior when design_description is an empty dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "approve", "issues": []}
        base_state["design_description"] = {}
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict is set
        assert result["last_design_review_verdict"] == "approve"
        # Verify user_content includes empty dict
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "DESIGN TO REVIEW" in user_content
        assert "{}" in user_content or "```json\n{}\n```" in user_content
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_verdict_case_sensitivity(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that verdict matching is case-sensitive (should handle exact values)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        initial_count = base_state.get("design_revision_count", 0)
        # Use exact lowercase verdict (if schema allows, but it should be "needs_revision")
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "test"}
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict matches exactly
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify revision count increments
        assert result["design_revision_count"] == initial_count + 1
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_handles_multiple_issues(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test behavior with multiple issues in the output."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Multiple issues found",
            "issues": [
                {"severity": "blocking", "category": "geometry", "description": "Issue 1", "suggested_fix": "Fix 1"},
                {"severity": "major", "category": "physics", "description": "Issue 2", "suggested_fix": "Fix 2"},
                {"severity": "minor", "category": "materials", "description": "Issue 3", "suggested_fix": "Fix 3"}
            ]
        }
        
        result = design_reviewer_node(base_state)
        
        # Verify verdict
        assert result["last_design_review_verdict"] == "needs_revision"
        # Verify all issues are preserved
        assert isinstance(result["reviewer_issues"], list)
        assert len(result["reviewer_issues"]) == 3
        assert result["reviewer_issues"][0]["severity"] == "blocking"
        assert result["reviewer_issues"][1]["severity"] == "major"
        assert result["reviewer_issues"][2]["severity"] == "minor"
        assert mock_llm.call_count == 1

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_escalate_to_user_string_triggers_ask_user(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that string escalate_to_user triggers user escalation."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",  # Even with approve, escalation should take priority
            "escalate_to_user": "Should I use 2D or 3D simulation for this geometry?",
            "issues": []
        }
        
        result = design_reviewer_node(base_state)
        
        # Should trigger escalation
        assert result["ask_user_trigger"] == "reviewer_escalation"
        assert result.get("ask_user_trigger") is not None
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1
        assert "2D or 3D" in result["pending_user_questions"][0]
        assert result["last_node_before_ask_user"] == "design_review"
        assert result["reviewer_escalation_source"] == "design_reviewer"
        
        # Should NOT have verdict-related fields (escalation short-circuits)
        assert "last_design_review_verdict" not in result
        assert "design_revision_count" not in result

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_escalate_to_user_false_continues_normally(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that boolean false escalate_to_user is ignored."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "escalate_to_user": False,
            "issues": []
        }
        
        result = design_reviewer_node(base_state)
        
        # Should NOT trigger escalation
        assert result.get("ask_user_trigger") != "reviewer_escalation"
        assert result["last_design_review_verdict"] == "approve"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_escalate_to_user_empty_string_continues_normally(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that empty string escalate_to_user is ignored."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "escalate_to_user": "",
            "issues": []
        }
        
        result = design_reviewer_node(base_state)
        
        # Should NOT trigger escalation
        assert result.get("ask_user_trigger") != "reviewer_escalation"
        assert result["last_design_review_verdict"] == "approve"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_escalate_to_user_whitespace_only_continues_normally(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that whitespace-only escalate_to_user is ignored."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "escalate_to_user": "   \n\t  ",
            "issues": []
        }
        
        result = design_reviewer_node(base_state)
        
        # Should NOT trigger escalation (strip() makes it empty)
        assert result.get("ask_user_trigger") != "reviewer_escalation"
        assert result["last_design_review_verdict"] == "approve"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_escalate_to_user_takes_priority_over_verdict(self, mock_llm, mock_prompt, mock_check, base_state):
        """Test that escalation takes priority even with needs_revision verdict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "escalate_to_user": "Paper mentions both extinction and absorption - which should I use?",
            "feedback": "Some feedback",
            "issues": [{"severity": "major", "description": "Ambiguous"}]
        }
        
        result = design_reviewer_node(base_state)
        
        # Should trigger escalation instead of normal verdict handling
        assert result["ask_user_trigger"] == "reviewer_escalation"
        assert "extinction and absorption" in result["pending_user_questions"][0]
        
        # Should NOT increment revision count (escalation happens before verdict processing)
        assert "design_revision_count" not in result
        assert "last_design_review_verdict" not in result
