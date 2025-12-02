"""
Unit tests for Planning Agent Module.
"""

import pytest
from unittest.mock import patch, MagicMock
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
        "extracted_parameters": [],
        "planned_materials": [],
        "assumptions": {}
    }

# ═══════════════════════════════════════════════════════════════════════
# plan_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlanNode:
    """Tests for plan_node function."""

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_basic_success(self, mock_check, mock_llm, mock_llm_output):
        """Test successful plan generation."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        state = {"paper_text": "x" * 200, "paper_id": "test_paper"}
        result = plan_node(state)
        
        assert result["workflow_phase"] == "planning"
        assert result["plan"]["paper_id"] == "test_paper"
        assert len(result["plan"]["stages"]) == 2
        assert result["progress"] is not None

    def test_plan_node_missing_text(self):
        """Test error when paper text is missing."""
        state = {"paper_text": "", "paper_id": "test_paper"}
        result = plan_node(state)
        
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result["awaiting_user_input"] is True

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_handles_replan(self, mock_check, mock_llm, mock_llm_output):
        """Test replan context injection."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        state = {"paper_text": "x" * 200, "replan_count": 1, "paper_id": "test_paper"}
        plan_node(state)
        
        # Verify prompt contains replan note
        call_kwargs = mock_llm.call_args[1]
        assert "Replan Attempt #1" in call_kwargs["system_prompt"]

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_preserves_paper_id(self, mock_check, mock_llm, mock_llm_output):
        """
        CRITICAL: Test that plan_node preserves input paper_id even if LLM omits it.
        This prevents 'outputs/unknown/' artifacts.
        """
        mock_check.return_value = None
        # LLM output missing paper_id
        mock_output_no_id = mock_llm_output.copy()
        del mock_output_no_id["paper_id"]
        mock_llm.return_value = mock_output_no_id
        
        input_id = "my_custom_paper_id"
        state = {"paper_text": "x" * 200, "paper_id": input_id}
        result = plan_node(state)
        
        assert result["plan"]["paper_id"] == input_id
        assert result["plan"]["paper_id"] != "unknown"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_handles_missing_state_paper_id(self, mock_check, mock_llm, mock_llm_output):
        """Test behavior when paper_id is missing from state (should fail gracefully or default)."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        state = {"paper_text": "x" * 200} # No paper_id
        result = plan_node(state)
        
        # Should use LLM provided ID or default to unknown, but NOT crash
        assert result["plan"]["paper_id"] == "test_paper" # From mock_llm_output

    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_bad_progress_structure(self, mock_check, mock_llm, mock_init_progress):
        """Test handling of invalid plan structure from LLM."""
        mock_check.return_value = None
        # Invalid stages (missing required fields)
        mock_llm.return_value = {
            "stages": [{"broken": "stage"}],
            "paper_id": "test"
        }
        # Mock initialize_progress_from_plan to raise exception
        mock_init_progress.side_effect = Exception("Missing stage_id")
        
        state = {"paper_text": "x" * 200, "paper_id": "test"}
        result = plan_node(state)
        
        # Fix: KeyError 'last_plan_review_verdict' was not present in plan_node output for this error path.
        # The plan_node now returns fields that supervisor will use to trigger replan.
        # The key 'last_plan_review_verdict' is set by plan_reviewer_node, not plan_node directly
        # unless plan_node detects an internal error that simulates a review failure.
        # The implementation of plan_node sets 'replan_count' and 'planner_feedback' on error.
        
        assert result.get("replan_count") == 1
        assert "progress initialization failed" in result.get("planner_feedback", "").lower()


# ═══════════════════════════════════════════════════════════════════════
# plan_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlanReviewerNode:
    """Tests for plan_reviewer_node function."""

    @patch("src.agents.planning.validate_state_or_warn")
    def test_plan_reviewer_blocks_empty_stages(self, mock_validate):
        """Test reviewer blocks plan with no stages locally."""
        mock_validate.return_value = []
        state = {"plan": {"stages": []}}
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages defined" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    def test_plan_reviewer_blocks_circular_deps(self, mock_validate):
        """Test reviewer blocks circular dependencies."""
        mock_validate.return_value = []
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "A", "dependencies": ["B"], "targets": ["t"]},
                    {"stage_id": "B", "dependencies": ["A"], "targets": ["t"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Circular dependencies" in result["planner_feedback"]

    @patch("src.agents.planning.validate_state_or_warn")
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_plan_reviewer_passes_valid_plan(self, mock_llm, mock_validate):
        """Test reviewer delegates to LLM for valid structure."""
        mock_validate.return_value = []
        mock_llm.return_value = {"verdict": "approve"}
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "A", "dependencies": [], "targets": ["t1"]}
                ]
            }
        }
        
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "approve"

    @patch("src.agents.planning.validate_state_or_warn")
    def test_plan_reviewer_replan_limit(self, mock_validate):
        """Test replan count incrementing and limiting."""
        mock_validate.return_value = []
        state = {
            "plan": {"stages": []}, # Force local rejection
            "replan_count": MAX_REPLANS - 1
        }
        
        # Should increment to MAX
        result = plan_reviewer_node(state)
        assert result["replan_count"] == MAX_REPLANS
        
        # Next call at max should stay at max
        state["replan_count"] = MAX_REPLANS
        result = plan_reviewer_node(state)
        assert result["replan_count"] == MAX_REPLANS


# ═══════════════════════════════════════════════════════════════════════
# adapt_prompts_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptPromptsNode:
    """Tests for adapt_prompts_node."""

    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_success(self, mock_llm):
        """Test successful prompt adaptation."""
        mock_llm.return_value = {
            "adaptations": ["Use more physics terms"],
            "paper_domain": "plasmonics"
        }
        
        state = {"paper_text": "test", "paper_domain": "unknown"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == ["Use more physics terms"]
        assert result["paper_domain"] == "plasmonics"

    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_fallback(self, mock_llm):
        """Test fallback on LLM failure."""
        mock_llm.side_effect = Exception("API Error")
        
        state = {"paper_text": "test"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
