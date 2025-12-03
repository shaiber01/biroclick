"""Tests for plan_reviewer_node."""

from unittest.mock import patch

import pytest

from src.agents.planning import plan_reviewer_node


@pytest.fixture(name="state")
def planning_state_alias(plan_state):
    return plan_state


class TestPlanReviewerNode:
    """Tests for plan_reviewer_node function."""

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_passes_valid_plan(self, mock_llm, mock_validate):
        """Test valid plan structure is sent to LLM."""
        mock_validate.return_value = []
        mock_llm.return_value = {"verdict": "approve"}
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "dependencies": [], "targets": ["t1"]}
                ]
            },
            "assumptions": {"a": 1}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"
        # Verify user content includes plan and assumptions
        call_kwargs = mock_llm.call_args[1]
        assert "REPRODUCTION PLAN TO REVIEW" in call_kwargs["user_content"]
        assert "ASSUMPTIONS" in call_kwargs["user_content"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_empty_stages(self, mock_validate):
        """Test rejection of plan with no stages."""
        mock_validate.return_value = []
        state = {"plan": {"stages": []}}
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_missing_stage_id(self, mock_validate):
        """Test rejection of stage without ID."""
        mock_validate.return_value = []
        state = {"plan": {"stages": [{"targets": ["t1"]}]}} # Missing stage_id
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "missing 'stage_id'" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_duplicate_stage_id(self, mock_validate):
        """Test rejection of duplicate stage IDs."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": ["t1"]},
                    {"stage_id": "s1", "targets": ["t2"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Duplicate stage ID 's1'" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_no_targets(self, mock_validate):
        """Test rejection of stage with no targets."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": [], "target_details": []}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no targets defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_missing_dependency(self, mock_validate):
        """Test rejection of dependency on non-existent stage."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": ["t1"], "dependencies": ["s2"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "depends on missing stage 's2'" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_self_dependency(self, mock_validate):
        """Test rejection of self-dependency."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": ["t1"], "dependencies": ["s1"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "depends on itself" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_circular_dependency_complex(self, mock_validate):
        """Test rejection of complex circular dependency (A->B->C->A)."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "A", "targets": ["t"], "dependencies": ["B"]},
                    {"stage_id": "B", "targets": ["t"], "dependencies": ["C"]},
                    {"stage_id": "C", "targets": ["t"], "dependencies": ["A"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Circular dependencies" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection(self, mock_llm, mock_validate):
        """Test handling of LLM rejection."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Please improve X"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 0
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["planner_feedback"] == "Please improve X"
        assert result["replan_count"] == 1

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_validate):
        """Test that LLM failure results in auto-approval (as per current policy)."""
        mock_validate.return_value = []
        mock_llm.side_effect = Exception("LLM fail")
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["workflow_phase"] == "plan_review"
        # create_llm_error_auto_approve returns "approved" usually
        assert result["last_plan_review_verdict"] == "approve" 

# ═══════════════════════════════════════════════════════════════════════
# adapt_prompts_node Tests
# ═══════════════════════════════════════════════════════════════════════
