"""
Tests for planner and plan reviewer validation logic.

This file tests:
- plan_node: Paper text validation, LLM error handling, progress initialization
- plan_reviewer_node: Plan structure validation, circular dependency detection,
  verdict normalization, replan counter management
"""

from unittest.mock import patch, MagicMock
import pytest

from src.agents import plan_node, plan_reviewer_node
from schemas.state import MAX_REPLANS

from tests.workflow.fixtures import MockResponseFactory


# =============================================================================
# PLAN NODE TESTS
# =============================================================================


class TestPlanNodePaperValidation:
    """Test plan_node validates paper text correctly."""

    def test_missing_paper_text_empty_string(self, base_state):
        """Test planner handles empty paper text."""
        base_state["paper_text"] = ""

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None
        assert len(result["pending_user_questions"]) > 0
        assert result["workflow_phase"] == "planning"
        # Verify the question mentions the character count
        assert "0 char" in result["pending_user_questions"][0].lower()

    def test_missing_paper_text_none(self, base_state):
        """Test planner handles None paper text."""
        base_state["paper_text"] = None

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None
        assert len(result["pending_user_questions"]) > 0

    def test_missing_paper_text_key_not_present(self, base_state):
        """Test planner handles missing paper_text key."""
        del base_state["paper_text"]

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None

    def test_short_paper_text(self, base_state):
        """Test planner handles insufficient paper text (<100 chars)."""
        base_state["paper_text"] = "Too short"

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None
        # Verify the question mentions the issue
        question = result["pending_user_questions"][0].lower()
        assert "too short" in question or "9 char" in question

    def test_paper_text_exactly_100_chars(self, base_state):
        """Test planner accepts paper text exactly at threshold."""
        # 100 characters should be accepted (the check is len < 100)
        base_state["paper_text"] = "x" * 100

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()
            result = plan_node(base_state)

        # Should not trigger missing_paper_text
        assert result.get("ask_user_trigger") != "missing_paper_text"
        assert result.get("ask_user_trigger") is None

    def test_paper_text_99_chars_rejected(self, base_state):
        """Test planner rejects paper text just below threshold."""
        base_state["paper_text"] = "x" * 99

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None

    def test_paper_text_whitespace_only(self, base_state):
        """Test planner handles whitespace-only paper text."""
        base_state["paper_text"] = "   \n\t   "

        result = plan_node(base_state)

        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None


class TestPlanNodeLLMInteraction:
    """Test plan_node LLM interaction and error handling."""

    def test_llm_failure_escalation(self, base_state):
        """Test planner escalates when LLM call fails."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API rate limit exceeded")
            result = plan_node(base_state)

        assert result["ask_user_trigger"] == "llm_error"
        assert result.get("ask_user_trigger") is not None
        assert "planner failed" in result["pending_user_questions"][0].lower()
        assert "rate limit" in result["pending_user_questions"][0].lower()

    def test_plan_extraction_from_llm_response(self, base_state):
        """Test that plan_node correctly extracts plan data from LLM response."""
        mock_response = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        assert "plan" in result
        assert result["plan"]["paper_domain"] == mock_response["paper_domain"]
        assert result["plan"]["title"] == mock_response["title"]
        assert len(result["plan"]["stages"]) == len(mock_response["stages"])

    def test_paper_id_preserved_from_state(self, base_state):
        """Test that paper_id from state is preserved over LLM response."""
        base_state["paper_id"] = "my_specific_paper_id"
        mock_response = MockResponseFactory.planner_response()
        mock_response["paper_id"] = "llm_suggested_id"

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        assert result["plan"]["paper_id"] == "my_specific_paper_id"

    def test_paper_id_fallback_to_llm_when_state_unknown(self, base_state):
        """Test that paper_id falls back to LLM when state has 'unknown'."""
        base_state["paper_id"] = "unknown"
        mock_response = MockResponseFactory.planner_response()
        mock_response["paper_id"] = "llm_suggested_id"

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        assert result["plan"]["paper_id"] == "llm_suggested_id"


class TestPlanNodeProgressInitialization:
    """Test plan_node progress initialization."""

    def test_progress_initialized_from_stages(self, base_state):
        """Test that progress is initialized from plan stages."""
        mock_response = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        assert "progress" in result
        assert result["progress"] is not None
        assert "stages" in result["progress"]
        # Progress should have same number of stages as plan
        assert len(result["progress"]["stages"]) == len(mock_response["stages"])

    def test_progress_initialization_failure_marks_needs_revision(self, base_state):
        """Test that progress init failure marks plan for revision."""
        mock_response = MockResponseFactory.planner_response()
        # Create invalid stage structure that will fail initialization
        mock_response["stages"] = [{"invalid": "structure"}]

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        # Should mark for revision due to init failure
        assert result.get("last_plan_review_verdict") == "needs_revision"
        assert "progress initialization failed" in result.get("planner_feedback", "").lower()


class TestPlanNodeReplanContext:
    """Test plan_node replan behavior."""

    def test_replan_count_increment_on_failure(self, base_state):
        """Test that replan count is incremented on progress init failure."""
        base_state["replan_count"] = 0
        mock_response = MockResponseFactory.planner_response()
        mock_response["stages"] = [{"invalid": "structure"}]

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        assert result.get("replan_count", 0) == 1

    def test_replan_count_respects_max_limit(self, base_state):
        """Test that replan count does not exceed max limit."""
        base_state["replan_count"] = MAX_REPLANS
        mock_response = MockResponseFactory.planner_response()
        mock_response["stages"] = [{"invalid": "structure"}]

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = plan_node(base_state)

        # Should not exceed max
        assert result.get("replan_count", 0) == MAX_REPLANS


# =============================================================================
# PLAN REVIEWER NODE TESTS - Structure Validation
# =============================================================================


class TestPlanReviewerStructureValidation:
    """Test plan_reviewer_node validates plan structure."""

    def test_empty_plan(self, base_state):
        """Test detection of empty plan."""
        base_state["plan"] = {}

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages" in result.get("planner_feedback", "").lower()

    def test_none_plan(self, base_state):
        """Test detection of None plan."""
        base_state["plan"] = None

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no stages" in result.get("planner_feedback", "").lower()

    def test_plan_with_no_stages_key(self, base_state):
        """Test detection of plan without stages key."""
        base_state["plan"] = {"title": "Test Plan"}

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_with_empty_stages_list(self, base_state):
        """Test detection of plan with empty stages list."""
        base_state["plan"] = {"stages": []}

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "at least one stage" in result.get("planner_feedback", "").lower()

    def test_plan_with_none_stages(self, base_state):
        """Test detection of plan with None stages."""
        base_state["plan"] = {"stages": None}

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"


class TestPlanReviewerStageValidation:
    """Test plan_reviewer_node validates individual stages."""

    def test_stage_missing_stage_id(self, base_state):
        """Test detection of stage without stage_id."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["stage_id"] = None

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "stage_id" in result.get("planner_feedback", "").lower()

    def test_stage_empty_stage_id(self, base_state):
        """Test detection of stage with empty stage_id."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["stage_id"] = ""

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_duplicate_stage_ids(self, base_state):
        """Test detection of duplicate stage IDs."""
        plan = MockResponseFactory.planner_response()
        # Make both stages have the same ID
        plan["stages"][0]["stage_id"] = "duplicate_id"
        plan["stages"][1]["stage_id"] = "duplicate_id"

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "duplicate" in result.get("planner_feedback", "").lower()

    def test_stage_not_a_dict(self, base_state):
        """Test detection of non-dict stage."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0] = "not a dict"

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "invalid stage structure" in result.get("planner_feedback", "").lower()


class TestPlanReviewerTargetValidation:
    """Test plan_reviewer_node validates stage targets."""

    def test_stage_no_targets(self, base_state):
        """Test detection of stages without targets."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["targets"] = []

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no targets" in result.get("planner_feedback", "").lower()

    def test_stage_none_targets(self, base_state):
        """Test detection of stage with None targets."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["targets"] = None
        # Also remove target_details to ensure no targets
        if "target_details" in plan["stages"][0]:
            plan["stages"][0]["target_details"] = None

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_stage_with_target_details_but_no_targets(self, base_state):
        """Test that target_details can satisfy the targets requirement."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["targets"] = []
        plan["stages"][0]["target_details"] = [{"figure_id": "Fig1"}]

        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = plan_reviewer_node(base_state)

        # Should NOT be rejected for no targets since target_details exists
        # But may have other validation from LLM
        if result["last_plan_review_verdict"] == "needs_revision":
            assert "no targets" not in result.get("planner_feedback", "").lower()


class TestPlanReviewerDependencyValidation:
    """Test plan_reviewer_node validates dependencies."""

    def test_dependency_on_nonexistent_stage(self, base_state):
        """Test detection of dependency on non-existent stage."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["dependencies"] = ["nonexistent_stage"]

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "missing stage" in result.get("planner_feedback", "").lower()

    def test_self_dependency(self, base_state):
        """Test detection of self-dependency."""
        plan = MockResponseFactory.planner_response()
        stage_id = plan["stages"][0]["stage_id"]
        plan["stages"][0]["dependencies"] = [stage_id]

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "depends on itself" in result.get("planner_feedback", "").lower()

    def test_none_dependencies_handled(self, base_state):
        """Test that None dependencies are handled gracefully."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["dependencies"] = None

        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = plan_reviewer_node(base_state)

        # Should not crash, and should pass if LLM approves
        assert result["last_plan_review_verdict"] in ["approve", "needs_revision"]


class TestPlanReviewerCircularDependencyDetection:
    """Test plan_reviewer_node detects circular dependencies."""

    def test_circular_dependency_direct(self, base_state):
        """Test detection of direct circular dependency (A -> B -> A)."""
        plan = MockResponseFactory.planner_response()
        # Create circular dependency: stage0 -> stage1 -> stage0
        plan["stages"][0]["dependencies"] = ["stage_1_extinction"]
        plan["stages"][1]["dependencies"] = ["stage_0_materials"]

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_circular_dependency_transitive(self, base_state):
        """Test detection of transitive circular dependency (A -> B -> C -> A)."""
        plan = MockResponseFactory.planner_response()
        # Add a third stage to create transitive cycle
        plan["stages"].append({
            "stage_id": "stage_2_analysis",
            "stage_type": "ANALYSIS",
            "name": "Analysis",
            "description": "Analyze results",
            "targets": ["analysis_target"],
            "dependencies": ["stage_1_extinction"],
        })
        # Create cycle: stage0 -> stage2 -> stage1 -> stage0
        plan["stages"][0]["dependencies"] = ["stage_2_analysis"]
        plan["stages"][1]["dependencies"] = ["stage_0_materials"]

        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_no_circular_dependency_valid_chain(self, base_state):
        """Test that valid dependency chain is not flagged as circular."""
        plan = MockResponseFactory.planner_response()
        # Valid chain: stage1 depends on stage0 (no cycle)
        plan["stages"][0]["dependencies"] = []
        plan["stages"][1]["dependencies"] = ["stage_0_materials"]

        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = plan_reviewer_node(base_state)

        # Should not be rejected for circular dependencies
        if result["last_plan_review_verdict"] == "needs_revision":
            assert "circular" not in result.get("planner_feedback", "").lower()


# =============================================================================
# PLAN REVIEWER NODE TESTS - Verdict Handling
# =============================================================================


class TestPlanReviewerVerdictNormalization:
    """Test plan_reviewer_node verdict normalization."""

    def test_verdict_pass_normalized_to_approve(self, base_state):
        """Test that 'pass' verdict is normalized to 'approve'."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve"

    def test_verdict_reject_normalized_to_needs_revision(self, base_state):
        """Test that 'reject' verdict is normalized to 'needs_revision'."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "reject", "summary": "Not OK", "feedback": "Fix it"}
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_verdict_approved_normalized_to_approve(self, base_state):
        """Test that 'approved' verdict is normalized to 'approve'."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "approved", "summary": "OK"}
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve"

    def test_verdict_accept_normalized_to_approve(self, base_state):
        """Test that 'accept' verdict is normalized to 'approve'."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "accept", "summary": "OK"}
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve"

    def test_unknown_verdict_defaults_to_needs_revision(self, base_state):
        """Test that unknown verdict defaults to 'needs_revision' for safety."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["replan_count"] = 0

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "unknown_verdict", "summary": "OK"}
            result = plan_reviewer_node(base_state)

        # Unknown verdicts should trigger revision, not auto-approve
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["replan_count"] == 1


class TestPlanReviewerReplanCounter:
    """Test plan_reviewer_node replan counter management."""

    def test_replan_count_incremented_on_needs_revision(self, base_state):
        """Test that replan count is incremented on needs_revision."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["replan_count"] = 0

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix plan"}
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result.get("replan_count", 0) == 1

    def test_replan_count_not_incremented_on_approve(self, base_state):
        """Test that replan count is not incremented on approve."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["replan_count"] = 1

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "approve"
        # Should not have replan_count in result when approving
        assert "replan_count" not in result

    def test_replan_count_respects_max_limit(self, base_state):
        """Test that replan count respects maximum limit."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["replan_count"] = MAX_REPLANS

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "needs_revision", "feedback": "Fix plan"}
            result = plan_reviewer_node(base_state)

        # Should not exceed max
        assert result.get("replan_count", MAX_REPLANS) == MAX_REPLANS

    def test_replan_count_incremented_for_blocking_issues(self, base_state):
        """Test that replan count is incremented for blocking structural issues."""
        plan = MockResponseFactory.planner_response()
        plan["stages"] = []  # Blocking issue: no stages
        base_state["plan"] = plan
        base_state["replan_count"] = 0

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        # For blocking issues, increment if replan_count exists in state
        assert result.get("replan_count", 0) == 1


class TestPlanReviewerLLMErrorHandling:
    """Test plan_reviewer_node LLM error handling."""

    def test_llm_error_defaults_to_needs_revision(self, base_state):
        """Test that LLM error results in needs_revision (safer than auto-approve)."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["replan_count"] = 0

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API error")
            result = plan_reviewer_node(base_state)

        # On LLM failure, default to needs_revision for safety
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["replan_count"] == 1


class TestPlanReviewerFeedbackExtraction:
    """Test plan_reviewer_node extracts feedback correctly."""

    def test_feedback_extracted_from_response(self, base_state):
        """Test that feedback is extracted from LLM response (via 'summary' key per schema)."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "summary": "The plan needs more detail about boundary conditions.",  # Schema uses "summary"
            }
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "boundary conditions" in result["planner_feedback"]

    def test_summary_used_as_fallback_for_feedback(self, base_state):
        """Test that summary is extracted from LLM response."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "summary": "Plan needs more simulation details.",
            }
            result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "simulation details" in result["planner_feedback"]


# =============================================================================
# PLAN REVIEWER NODE TESTS - Complex Scenarios
# =============================================================================


class TestPlanReviewerMultipleIssues:
    """Test plan_reviewer_node handles multiple issues."""

    def test_multiple_structural_issues_all_reported(self, base_state):
        """Test that multiple structural issues are all reported."""
        plan = {
            "stages": [
                # Stage 1: missing stage_id
                {"targets": ["Fig1"]},
                # Stage 2: no targets
                {"stage_id": "stage_1", "targets": []},
                # Stage 3: dependency on nonexistent
                {"stage_id": "stage_2", "targets": ["Fig2"], "dependencies": ["nonexistent"]},
            ]
        }
        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        feedback = result.get("planner_feedback", "").lower()
        # Check that multiple issues are mentioned
        assert "stage_id" in feedback or "missing" in feedback
        assert "no targets" in feedback or "target" in feedback


class TestPlanReviewerAwaitingUserInput:
    """Test plan_reviewer_node behavior when awaiting user input."""

    def test_skips_when_awaiting_user_input(self, base_state):
        """Test that plan_reviewer returns empty when awaiting user input."""
        base_state["awaiting_user_input"] = True
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan

        result = plan_reviewer_node(base_state)

        # Should return empty dict when awaiting user input
        assert result == {}
