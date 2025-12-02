"""Unit tests for src/agents/planning.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.planning import (
    adapt_prompts_node,
    plan_node,
    plan_reviewer_node,
)


class TestAdaptPromptsNode:
    """Tests for adapt_prompts_node function."""

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.build_agent_prompt")
    def test_returns_adaptations_on_success(self, mock_prompt, mock_context, mock_call):
        """Should return adaptations from LLM output.
        
        Note: Patches base.py because adapt_prompts_node uses @with_context_check decorator.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "adaptations": [{"agent": "planner", "adjustment": "focus on plasmonics"}],
            "paper_domain": "plasmonics",
        }
        
        state = {"paper_text": "test paper about gold nanoparticles", "paper_domain": ""}
        
        result = adapt_prompts_node(state)
        
        assert result["workflow_phase"] == "adapting_prompts"
        assert len(result["prompt_adaptations"]) == 1
        assert result["paper_domain"] == "plasmonics"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.build_agent_prompt")
    def test_returns_empty_on_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should return empty adaptations on LLM error.
        
        Note: Patches base.py because adapt_prompts_node uses @with_context_check decorator.
        """
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {"paper_text": "test", "paper_domain": ""}
        
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow.
        
        Note: Patches base.py because adapt_prompts_node uses @with_context_check decorator.
        """
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"paper_text": "test"}
        
        result = adapt_prompts_node(state)
        
        assert result["awaiting_user_input"] is True


class TestPlanNode:
    """Tests for plan_node function."""

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.build_agent_prompt")
    @patch("src.agents.planning.build_user_content_for_planner")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_creates_plan_on_success(
        self, mock_sync, mock_progress, mock_user, mock_prompt, mock_context, mock_call, validated_planner_response
    ):
        """Should create plan from LLM output (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        
        # Use the validated mock response
        mock_call.return_value = validated_planner_response
        
        mock_progress.return_value = {"progress": {"stages": []}}
        mock_sync.return_value = {"extracted_parameters": []}
        
        state = {"paper_text": "x" * 200, "paper_id": "test_paper"}
        
        result = plan_node(state)
        
        assert result["workflow_phase"] == "planning"
        assert result["plan"]["title"] == validated_planner_response["title"]
        assert result["paper_domain"] == validated_planner_response["paper_domain"]
        assert len(result["plan"]["stages"]) > 0

    def test_errors_on_missing_paper_text(self):
        """Should error when paper_text is missing."""
        state = {"paper_text": ""}
        
        result = plan_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_paper_text"

    def test_errors_on_short_paper_text(self):
        """Should error when paper_text is too short."""
        state = {"paper_text": "short"}
        
        result = plan_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.build_agent_prompt")
    @patch("src.agents.planning.build_user_content_for_planner")
    def test_handles_llm_error(self, mock_user, mock_prompt, mock_context, mock_call):
        """Should handle LLM call failure gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_user.return_value = "content"
        mock_call.side_effect = Exception("API error")
        
        state = {"paper_text": "x" * 200}
        
        result = plan_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "llm_error"

    @patch("src.agents.planning.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"paper_text": "x" * 200}
        
        result = plan_node(state)
        
        assert result["awaiting_user_input"] is True


class TestPlanReviewerNode:
    """Tests for plan_reviewer_node function."""

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.build_agent_prompt")
    def test_approves_valid_plan(self, mock_prompt, mock_validate, mock_context, mock_call, validated_plan_reviewer_response):
        """Should approve a valid plan (using validated mock)."""
        mock_context.return_value = None
        mock_validate.return_value = []
        mock_prompt.return_value = "system prompt"
        
        # Ensure the validated mock is an "approve" case
        mock_response = validated_plan_reviewer_response.copy()
        mock_response["verdict"] = "approve"
        mock_response["issues"] = []
        
        mock_call.return_value = mock_response
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["Fig1"], "dependencies": []}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["workflow_phase"] == "plan_review"
        assert result["last_plan_review_verdict"] == "approve"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    def test_rejects_empty_stages(self, mock_validate, mock_context):
        """Should reject plan with no stages.
        
        Note: Patches base.py because plan_reviewer_node uses @with_context_check decorator.
        """
        mock_context.return_value = None
        mock_validate.return_value = []
        
        state = {"plan": {"stages": []}}
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    def test_rejects_stage_without_targets(self, mock_validate, mock_context):
        """Should reject stage without targets.
        
        Note: Patches base.py because plan_reviewer_node uses @with_context_check decorator.
        """
        mock_context.return_value = None
        mock_validate.return_value = []
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": [], "dependencies": []}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    def test_detects_self_dependency(self, mock_validate, mock_context):
        """Should detect self-dependency in stages.
        
        Note: Patches base.py because plan_reviewer_node uses @with_context_check decorator.
        """
        mock_context.return_value = None
        mock_validate.return_value = []
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["Fig1"], "dependencies": ["stage1"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"

    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    def test_detects_circular_dependencies(self, mock_validate, mock_context):
        """Should detect circular dependencies.
        
        Note: Patches base.py because plan_reviewer_node uses @with_context_check decorator.
        """
        mock_context.return_value = None
        mock_validate.return_value = []
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["Fig1"], "dependencies": ["stage2"]},
                    {"stage_id": "stage2", "targets": ["Fig2"], "dependencies": ["stage1"]},
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.build_agent_prompt")
    def test_increments_replan_count_on_rejection(
        self, mock_prompt, mock_validate, mock_context, mock_call, validated_plan_reviewer_response
    ):
        """Should increment replan_count when verdict is needs_revision."""
        mock_context.return_value = None
        mock_validate.return_value = []
        mock_prompt.return_value = "prompt"
        
        # Create a validated needs_revision response
        # We can reuse the schema-validated structure but enforce the verdict we want
        mock_response = validated_plan_reviewer_response.copy()
        mock_response["verdict"] = "needs_revision"
        # Ensure it has issues as per strict contract
        mock_response["issues"] = [{"severity": "major", "description": "Issue"}]
        mock_response["feedback"] = "Please fix X"
        
        mock_call.return_value = mock_response
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["Fig1"], "dependencies": []}
                ]
            },
            "replan_count": 0,
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["replan_count"] == 1
        assert "planner_feedback" in result

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.build_agent_prompt")
    def test_auto_approves_on_llm_error(
        self, mock_prompt, mock_validate, mock_context, mock_call
    ):
        """Should auto-approve when LLM call fails."""
        mock_context.return_value = None
        mock_validate.return_value = []
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["Fig1"], "dependencies": []}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"
