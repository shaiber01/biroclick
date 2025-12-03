"""
Unit tests for Planning Agent Module.
"""

import pytest
from unittest.mock import patch, MagicMock, ANY
from src.agents.planning import plan_node, plan_reviewer_node, adapt_prompts_node
from schemas.state import ReproState, MAX_REPLANS

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm_output():
    """Standard valid planner response."""
    return {
        "paper_id": "test_paper",
        "paper_domain": "plasmonics",
        "title": "Test Paper",
        "summary": "A test paper summary.",
        "stages": [
            {"stage_id": "stage_0_materials", "stage_type": "MATERIAL_VALIDATION", "targets": ["mat1"]},
            {"stage_id": "stage_1_sim", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig1"]}
        ],
        "targets": [{"figure_id": "Fig1"}],
        "extracted_parameters": [{"name": "param1", "value": 10}],
        "planned_materials": ["Gold"],
        "assumptions": {"assumption1": "true"}
    }

@pytest.fixture
def mock_state():
    """Standard input state."""
    return {
        "paper_text": "x" * 500,
        "paper_id": "test_paper",
        "replan_count": 0,
        "runtime_config": {"max_replans": 3}
    }

# ═══════════════════════════════════════════════════════════════════════
# plan_node Tests
# ═══════════════════════════════════════════════════════════════════════

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

class TestAdaptPromptsNode:
    """Tests for adapt_prompts_node."""

    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_success(self, mock_llm):
        """Test successful adaptation."""
        mock_llm.return_value = {
            "adaptations": ["a1"],
            "paper_domain": "domain"
        }
        
        state = {"paper_text": "text", "paper_domain": "old"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == ["a1"]
        assert result["paper_domain"] == "domain"

    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_failure_fallback(self, mock_llm):
        """Test fallback to empty list on failure."""
        mock_llm.side_effect = Exception("fail")
        
        state = {"paper_text": "text"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_truncates_text(self, mock_llm):
        """Verify paper text is truncated in user content."""
        mock_llm.return_value = {}
        long_text = "a" * 10000
        state = {"paper_text": long_text}
        
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        # Code uses [:5000] first then slices further to [:3000] in user_content
        assert len(long_text) > 5000
        assert len(user_content) < 5000
