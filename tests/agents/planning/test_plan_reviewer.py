"""Tests for plan_reviewer_node."""

from unittest.mock import patch, MagicMock

import pytest

from src.agents.planning import plan_reviewer_node
from schemas.state import MAX_REPLANS


@pytest.fixture(name="state")
def planning_state_alias(plan_state):
    return plan_state


class TestPlanReviewerNode:
    """Tests for plan_reviewer_node function."""

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_reviewer_passes_valid_plan(self, mock_prompt, mock_llm, mock_validate):
        """Test valid plan structure is sent to LLM with all required fields."""
        mock_validate.return_value = []
        mock_prompt.return_value = "System prompt"
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
        
        # Verify all return fields
        assert result["workflow_phase"] == "plan_review"
        assert result["last_plan_review_verdict"] == "approve"
        assert "planner_feedback" not in result  # Should not be set for approve
        assert "replan_count" not in result  # Should not increment for approve
        
        # Verify LLM was called correctly
        assert mock_llm.called
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["agent_name"] == "plan_reviewer"
        assert call_kwargs["system_prompt"] == "System prompt"
        assert "REPRODUCTION PLAN TO REVIEW" in call_kwargs["user_content"]
        assert "ASSUMPTIONS" in call_kwargs["user_content"]
        assert '"a": 1' in call_kwargs["user_content"]  # Verify assumptions JSON included
        
        # Verify prompt builder was called
        assert mock_prompt.called
        assert mock_prompt.call_args[0][0] == "plan_reviewer"

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_reviewer_passes_valid_plan_no_assumptions(self, mock_prompt, mock_llm, mock_validate):
        """Test valid plan without assumptions field."""
        mock_validate.return_value = []
        mock_prompt.return_value = "System prompt"
        mock_llm.return_value = {"verdict": "approve"}
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "dependencies": [], "targets": ["t1"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"
        call_kwargs = mock_llm.call_args[1]
        assert "REPRODUCTION PLAN TO REVIEW" in call_kwargs["user_content"]
        # Assumptions section should not appear if assumptions is missing
        assert "ASSUMPTIONS" not in call_kwargs["user_content"]

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_reviewer_passes_valid_plan_empty_assumptions(self, mock_prompt, mock_llm, mock_validate):
        """Test valid plan with empty assumptions dict."""
        mock_validate.return_value = []
        mock_prompt.return_value = "System prompt"
        mock_llm.return_value = {"verdict": "approve"}
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "dependencies": [], "targets": ["t1"]}
                ]
            },
            "assumptions": {}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"
        call_kwargs = mock_llm.call_args[1]
        # Empty assumptions should still be included
        assert "ASSUMPTIONS" in call_kwargs["user_content"]
        assert "{}" in call_kwargs["user_content"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_empty_stages(self, mock_validate):
        """Test rejection of plan with no stages."""
        mock_validate.return_value = []
        state = {"plan": {"stages": []}}
        
        result = plan_reviewer_node(state)
        
        assert result["workflow_phase"] == "plan_review"
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert "no stages defined" in result["planner_feedback"]
        assert "PLAN_ISSUE:" in result["planner_feedback"]
        # Should not increment replan_count when not provided
        assert "replan_count" not in result

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_missing_plan(self, mock_validate):
        """Test rejection when plan is missing entirely."""
        mock_validate.return_value = []
        state = {}
        
        result = plan_reviewer_node(state)
        
        assert result["workflow_phase"] == "plan_review"
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert "no stages defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_plan_none(self, mock_validate):
        """Test rejection when plan is None."""
        mock_validate.return_value = []
        state = {"plan": None}
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_missing_stages_key(self, mock_validate):
        """Test rejection when plan exists but stages key is missing."""
        mock_validate.return_value = []
        state = {"plan": {}}
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_stages_none(self, mock_validate):
        """Test rejection when stages is None."""
        mock_validate.return_value = []
        state = {"plan": {"stages": None}}
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_missing_stage_id(self, mock_validate):
        """Test rejection of stage without ID."""
        mock_validate.return_value = []
        state = {"plan": {"stages": [{"targets": ["t1"]}]}}  # Missing stage_id
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "missing 'stage_id'" in result["planner_feedback"]
        assert "PLAN_ISSUE:" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_empty_stage_id(self, mock_validate):
        """Test rejection of stage with empty string stage_id."""
        mock_validate.return_value = []
        state = {"plan": {"stages": [{"stage_id": "", "targets": ["t1"]}]}}
        
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
        assert "PLAN_ISSUE:" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_three_duplicate_stage_ids(self, mock_validate):
        """Test rejection when same stage ID appears three times."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": ["t1"]},
                    {"stage_id": "s1", "targets": ["t2"]},
                    {"stage_id": "s1", "targets": ["t3"]}
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
        assert "Stage 's1'" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_no_targets_missing_keys(self, mock_validate):
        """Test rejection when targets and target_details keys are missing."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1"}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no targets defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_allows_targets_only(self, mock_validate):
        """Test that stage with only targets (no target_details) is valid."""
        mock_validate.return_value = []
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {
                        "stages": [
                            {"stage_id": "s1", "targets": ["t1"]}
                        ]
                    }
                }
                
                result = plan_reviewer_node(state)
                
                assert result["last_plan_review_verdict"] == "approve"
                assert mock_llm.called  # Should reach LLM call

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_allows_target_details_only(self, mock_validate):
        """Test that stage with only target_details (no targets) is valid."""
        mock_validate.return_value = []
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {
                        "stages": [
                            {"stage_id": "s1", "target_details": [{"figure_id": "f1"}]}
                        ]
                    }
                }
                
                result = plan_reviewer_node(state)
                
                assert result["last_plan_review_verdict"] == "approve"
                assert mock_llm.called  # Should reach LLM call

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
        assert "Stage 's1'" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_multiple_missing_dependencies(self, mock_validate):
        """Test rejection when stage depends on multiple missing stages."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": ["t1"], "dependencies": ["s2", "s3", "s4"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result["planner_feedback"]
        assert "depends on missing stage" in feedback
        # Should mention all missing dependencies
        assert "s2" in feedback or "s3" in feedback or "s4" in feedback

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_allows_empty_dependencies(self, mock_validate):
        """Test that empty dependencies list is valid."""
        mock_validate.return_value = []
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {
                        "stages": [
                            {"stage_id": "s1", "targets": ["t1"], "dependencies": []}
                        ]
                    }
                }
                
                result = plan_reviewer_node(state)
                
                assert result["last_plan_review_verdict"] == "approve"
                assert mock_llm.called

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_allows_missing_dependencies_key(self, mock_validate):
        """Test that missing dependencies key is treated as empty (valid)."""
        mock_validate.return_value = []
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {
                        "stages": [
                            {"stage_id": "s1", "targets": ["t1"]}
                        ]
                    }
                }
                
                result = plan_reviewer_node(state)
                
                assert result["last_plan_review_verdict"] == "approve"
                assert mock_llm.called

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
        assert "Stage 's1'" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_circular_dependency_two_stage(self, mock_validate):
        """Test rejection of simple 2-stage circular dependency (A->B->A)."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "A", "targets": ["t"], "dependencies": ["B"]},
                    {"stage_id": "B", "targets": ["t"], "dependencies": ["A"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Circular dependencies" in result["planner_feedback"]

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
    def test_reviewer_blocking_circular_dependency_four_stage(self, mock_validate):
        """Test rejection of 4-stage circular dependency."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "A", "targets": ["t"], "dependencies": ["B"]},
                    {"stage_id": "B", "targets": ["t"], "dependencies": ["C"]},
                    {"stage_id": "C", "targets": ["t"], "dependencies": ["D"]},
                    {"stage_id": "D", "targets": ["t"], "dependencies": ["A"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Circular dependencies" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_multiple_circular_dependencies(self, mock_validate):
        """Test rejection when multiple independent cycles exist."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    # Cycle 1: A->B->A
                    {"stage_id": "A", "targets": ["t"], "dependencies": ["B"]},
                    {"stage_id": "B", "targets": ["t"], "dependencies": ["A"]},
                    # Cycle 2: C->D->C
                    {"stage_id": "C", "targets": ["t"], "dependencies": ["D"]},
                    {"stage_id": "D", "targets": ["t"], "dependencies": ["C"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Circular dependencies" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_allows_valid_dependency_chain(self, mock_validate):
        """Test that valid linear dependency chain passes."""
        mock_validate.return_value = []
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {
                        "stages": [
                            {"stage_id": "s1", "targets": ["t1"], "dependencies": []},
                            {"stage_id": "s2", "targets": ["t2"], "dependencies": ["s1"]},
                            {"stage_id": "s3", "targets": ["t3"], "dependencies": ["s2"]}
                        ]
                    }
                }
                
                result = plan_reviewer_node(state)
                
                assert result["last_plan_review_verdict"] == "approve"
                assert mock_llm.called

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_allows_valid_dependency_tree(self, mock_validate):
        """Test that valid dependency tree (multiple children of one parent) passes."""
        mock_validate.return_value = []
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {
                        "stages": [
                            {"stage_id": "s1", "targets": ["t1"], "dependencies": []},
                            {"stage_id": "s2", "targets": ["t2"], "dependencies": ["s1"]},
                            {"stage_id": "s3", "targets": ["t3"], "dependencies": ["s1"]}
                        ]
                    }
                }
                
                result = plan_reviewer_node(state)
                
                assert result["last_plan_review_verdict"] == "approve"
                assert mock_llm.called

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
        
        assert result["workflow_phase"] == "plan_review"
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["planner_feedback"] == "Please improve X"
        assert result["replan_count"] == 1

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_with_summary(self, mock_llm, mock_validate):
        """Test LLM rejection with summary but no feedback field."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "summary": "Plan needs improvement"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 0
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["planner_feedback"] == "Plan needs improvement"

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_no_feedback_no_summary(self, mock_llm, mock_validate):
        """Test LLM rejection with neither feedback nor summary."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 0
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["planner_feedback"] == ""  # Empty string fallback

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_increments_replan_count(self, mock_llm, mock_validate):
        """Test that replan_count increments correctly."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix it"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 1,
            "runtime_config": {"max_replans": 3}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["replan_count"] == 2

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_at_max_replan_count(self, mock_llm, mock_validate):
        """Test that replan_count does not exceed max_replans."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix it"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 2,
            "runtime_config": {"max_replans": 2}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["replan_count"] == 2  # Should not increment beyond max

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_exceeds_max_replan_count(self, mock_llm, mock_validate):
        """Test that replan_count does not increment when already exceeding max."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix it"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 5,  # Already exceeds max
            "runtime_config": {"max_replans": 2}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["replan_count"] == 5  # Should remain unchanged

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_missing_runtime_config(self, mock_llm, mock_validate):
        """Test replan_count increment when runtime_config is missing (uses MAX_REPLANS)."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix it"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 0
            # runtime_config missing
        }
        
        result = plan_reviewer_node(state)
        
        assert result["replan_count"] == 1  # Should increment

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_rejection_missing_max_replans_in_config(self, mock_llm, mock_validate):
        """Test replan_count increment when runtime_config exists but max_replans missing."""
        mock_validate.return_value = []
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix it"
        }
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 0,
            "runtime_config": {}  # Empty config
        }
        
        result = plan_reviewer_node(state)
        
        assert result["replan_count"] == 1  # Should increment (uses MAX_REPLANS default)

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_validate):
        """Test that LLM failure results in auto-approval."""
        mock_validate.return_value = []
        mock_llm.side_effect = Exception("LLM fail")
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["workflow_phase"] == "plan_review"
        assert result["last_plan_review_verdict"] == "approve"
        assert "planner_feedback" not in result
        assert "replan_count" not in result

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_failure_different_exception_types(self, mock_llm, mock_validate):
        """Test that different exception types are handled."""
        mock_validate.return_value = []
        
        for exc_type in [ValueError("test"), KeyError("test"), RuntimeError("test")]:
            mock_llm.side_effect = exc_type
            state = {
                "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
            }
            
            result = plan_reviewer_node(state)
            
            assert result["last_plan_review_verdict"] == "approve"

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_issues_from_validate_state(self, mock_validate):
        """Test that PLAN_ISSUE from validate_state_or_warn are treated as blocking."""
        mock_validate.return_value = [
            "PLAN_ISSUE: Some blocking issue",
            "WARNING: Some non-blocking warning"
        ]
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Some blocking issue" in result["planner_feedback"]
        # Non-blocking warnings should not appear in feedback
        assert "WARNING:" not in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_multiple_blocking_issues_combined(self, mock_validate):
        """Test that multiple blocking issues are all included in feedback."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": []},  # No targets
                    {"stage_id": "s1", "targets": ["t"]},  # Duplicate ID
                    {"stage_id": "s2", "targets": ["t"], "dependencies": ["s3"]}  # Missing dependency
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result["planner_feedback"]
        # Should contain all issues
        assert "no targets defined" in feedback or "Stage 's1'" in feedback
        assert "Duplicate stage ID" in feedback
        assert "depends on missing stage" in feedback

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_approve_verdict(self, mock_llm, mock_validate):
        """Test that approve verdict from LLM is handled correctly."""
        mock_validate.return_value = []
        mock_llm.return_value = {"verdict": "approve"}
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "replan_count": 1
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"
        assert "planner_feedback" not in result
        assert "replan_count" not in result  # Should not increment for approve

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_unexpected_verdict(self, mock_llm, mock_validate):
        """Test handling of unexpected verdict value from LLM.
        
        Unknown verdicts are normalized to 'approve' for safety, to avoid
        blocking workflow on malformed LLM responses.
        """
        mock_validate.return_value = []
        mock_llm.return_value = {"verdict": "unknown_verdict"}
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
        }
        
        result = plan_reviewer_node(state)
        
        # Unknown verdicts are normalized to 'approve' (logs warning)
        assert result["last_plan_review_verdict"] == "approve"
        # Should not increment replan_count for approve verdicts
        assert "replan_count" not in result

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_llm_missing_verdict_key(self, mock_llm, mock_validate):
        """Test handling when LLM response is missing verdict key.
        
        When verdict key is missing, the component defaults to 'approve'
        to avoid blocking workflow on malformed LLM responses.
        """
        mock_validate.return_value = []
        mock_llm.return_value = {"feedback": "Some feedback"}
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
        }
        
        result = plan_reviewer_node(state)
        
        # Missing verdict defaults to 'approve' (uses .get("verdict", "approve"))
        assert result["last_plan_review_verdict"] == "approve"
        # Should not increment replan_count for approve verdicts
        assert "replan_count" not in result

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_blocking_issues_verdict_structure(self, mock_validate):
        """Test that blocking issues create proper agent_output structure."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": []}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        # Verify structure matches what would come from LLM
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert "PLAN_ISSUE:" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_plan_json_in_user_content(self, mock_llm, mock_validate):
        """Test that plan is properly JSON-serialized in user content."""
        mock_validate.return_value = []
        mock_llm.return_value = {"verdict": "approve"}
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "s1", "targets": ["t1"], "dependencies": ["s2"]},
                    {"stage_id": "s2", "targets": ["t2"]}
                ]
            }
        }
        
        plan_reviewer_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        # Verify plan JSON is present
        assert '"stage_id": "s1"' in user_content
        assert '"stage_id": "s2"' in user_content
        # Verify dependencies are present (formatting may vary)
        assert '"dependencies"' in user_content
        assert '"s2"' in user_content

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_reviewer_assumptions_json_in_user_content(self, mock_llm, mock_validate):
        """Test that assumptions are properly JSON-serialized in user content."""
        mock_validate.return_value = []
        mock_llm.return_value = {"verdict": "approve"}
        state = {
            "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]},
            "assumptions": {
                "temperature": 300,
                "pressure": 1.0,
                "material": "Gold"
            }
        }
        
        plan_reviewer_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        # Verify assumptions JSON is present
        assert '"temperature": 300' in user_content
        assert '"pressure": 1.0' in user_content
        assert '"material": "Gold"' in user_content

    @patch("src.agents.planning.validate_state_or_warn")
    def test_reviewer_workflow_phase_always_set(self, mock_validate):
        """Test that workflow_phase is always set to plan_review."""
        mock_validate.return_value = []
        
        # Test with blocking issue
        state = {"plan": {"stages": []}}
        result = plan_reviewer_node(state)
        assert result["workflow_phase"] == "plan_review"
        
        # Test with LLM call
        mock_llm = MagicMock(return_value={"verdict": "approve"})
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm):
            with patch("src.agents.planning.build_agent_prompt", return_value="prompt"):
                state = {
                    "plan": {"stages": [{"stage_id": "s1", "targets": ["t"]}]}
                }
                result = plan_reviewer_node(state)
                assert result["workflow_phase"] == "plan_review"
