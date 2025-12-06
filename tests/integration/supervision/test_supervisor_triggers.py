"""Supervisor trigger handling integration tests.

Tests the supervisor_node's handling of various ask_user_trigger types
and verifies the correct state mutations for each user response.
"""

from unittest.mock import patch
import pytest


# =============================================================================
# Material Checkpoint Tests
# =============================================================================

class TestSupervisorMaterialCheckpoint:
    """Supervisor ask_user trigger handling for material checkpoints."""

    def test_supervisor_handles_material_checkpoint_approval(self, base_state):
        """User APPROVES pending materials - materials should be validated."""
        from src.agents.supervision.supervisor import supervisor_node

        pending_materials = [
            {"name": "gold", "path": "/materials/Au.csv"},
            {"name": "silver", "path": "/materials/Ag.csv"},
        ]
        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = pending_materials
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        # Verify materials were moved from pending to validated
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["validated_materials"] == pending_materials
        assert result.get("pending_validated_materials") == []
        assert result.get("ask_user_trigger") is None
        # Verify feedback is set
        assert "approved" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_material_checkpoint_alternative_approval_keywords(
        self, base_state
    ):
        """Test various approval keywords: YES, OK, ACCEPT, etc."""
        from src.agents.supervision.supervisor import supervisor_node

        pending_materials = [{"name": "gold", "path": "/materials/Au.csv"}]
        base_state["pending_validated_materials"] = pending_materials
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        for keyword in ["YES", "OK", "ACCEPT", "CORRECT", "VALID", "PROCEED"]:
            base_state["ask_user_trigger"] = "material_checkpoint"
            base_state["user_responses"] = {"Q1": keyword}

            result = supervisor_node(base_state)

            assert result["supervisor_verdict"] == "ok_continue", (
                f"Keyword '{keyword}' should approve materials"
            )
            assert result["validated_materials"] == pending_materials

    def test_supervisor_material_checkpoint_without_pending_materials_escalates(
        self, base_state
    ):
        """User APPROVEs but no materials pending - should escalate."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = []
        base_state["pending_user_questions"] = ["Approve materials?"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "No materials were extracted" in questions[0]
        assert result.get("ask_user_trigger") is None
        # Both validated and pending should be empty
        assert result.get("validated_materials") == []
        assert result.get("pending_validated_materials") == []

    def test_supervisor_material_checkpoint_with_none_pending_materials(
        self, base_state
    ):
        """Pending materials is None - should handle gracefully."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = None
        base_state["pending_user_questions"] = ["Approve materials?"]

        result = supervisor_node(base_state)

        # Should escalate asking user since no materials
        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "No materials were extracted" in questions[0]

    def test_supervisor_material_checkpoint_rejection_with_database(self, base_state):
        """User REJECTS and requests database change."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "REJECT: CHANGE_DATABASE to Palik"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "replan_needed"
        assert result.get("pending_validated_materials") == []
        assert result.get("validated_materials") == []
        assert "database" in result.get("planner_feedback", "").lower()

    def test_supervisor_material_checkpoint_rejection_with_material(self, base_state):
        """User REJECTS and requests material change."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "REJECT: CHANGE_MATERIAL to silver"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "replan_needed"
        assert result.get("pending_validated_materials") == []
        assert result.get("validated_materials") == []
        assert "material" in result.get("planner_feedback", "").lower()

    def test_supervisor_material_checkpoint_rejection_without_specifics(
        self, base_state
    ):
        """User REJECTS without specifying what to change - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "REJECT"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve materials?"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "didn't specify" in questions[0].lower()

    def test_supervisor_material_checkpoint_need_help(self, base_state):
        """User requests HELP - should provide guidance."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "NEED_HELP"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve materials?"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "details" in questions[0].lower()

    def test_supervisor_material_checkpoint_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "maybe later"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve materials?"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "unclear" in questions[0].lower()


# =============================================================================
# Code Review Limit Tests
# =============================================================================

class TestSupervisorCodeReviewLimit:
    """Test code_review_limit trigger handling."""

    def test_supervisor_handles_code_review_limit_with_hint(self, base_state):
        """User provides HINT - should reset counter and provide feedback."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {
            "Q1": "PROVIDE_HINT: Try using mp.Medium instead"
        }
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("code_revision_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "User hint" in result.get("reviewer_feedback", "")
        assert "mp.Medium" in result.get("reviewer_feedback", "")
        assert result.get("ask_user_trigger") is None
        assert result.get("workflow_phase") == "supervision"
        assert result.get("pending_user_questions") in (None, [])

    def test_supervisor_handles_code_review_limit_hint_keyword_only(self, base_state):
        """Just HINT keyword (without PROVIDE_HINT) should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "HINT: Check the wavelength range"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 5

        result = supervisor_node(base_state)

        assert result.get("code_revision_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_code_review_limit_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        # Counter should NOT be reset when skipping
        assert result.get("code_revision_count") is None or result.get("code_revision_count") != 0

    def test_supervisor_handles_code_review_limit_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_code_review_limit_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "I'm not sure what to do"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Design Review Limit Tests
# =============================================================================

class TestSupervisorDesignReviewLimit:
    """Test design_review_limit trigger handling."""

    def test_supervisor_handles_design_review_limit_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "design_review_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Design review limit"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_design_review_limit_with_hint(self, base_state):
        """User provides HINT - should reset counter and continue."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "design_review_limit"
        base_state["user_responses"] = {
            "Q1": "PROVIDE_HINT: Use a finer mesh near the surface"
        }
        base_state["pending_user_questions"] = ["Design review limit"]
        base_state["design_revision_count"] = 4

        result = supervisor_node(base_state)

        assert result.get("design_revision_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "User hint" in result.get("reviewer_feedback", "")

    def test_supervisor_handles_design_review_limit_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "design_review_limit"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Design review limit"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_design_review_limit_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "design_review_limit"
        base_state["user_responses"] = {"Q1": "continue"}
        base_state["pending_user_questions"] = ["Design review limit"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Execution Failure Limit Tests
# =============================================================================

class TestSupervisorExecutionFailureLimit:
    """Test execution_failure_limit trigger handling."""

    def test_supervisor_handles_execution_failure_with_retry(self, base_state):
        """User provides RETRY_WITH_GUIDANCE - should reset counter."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {
            "Q1": "RETRY_WITH_GUIDANCE: Increase memory allocation"
        }
        base_state["pending_user_questions"] = ["Execution failed"]
        base_state["execution_failure_count"] = 2

        result = supervisor_node(base_state)

        assert result.get("execution_failure_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "User guidance" in result.get("supervisor_feedback", "")
        assert result.get("ask_user_trigger") is None

    def test_supervisor_handles_execution_failure_with_just_retry(self, base_state):
        """Just RETRY keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["Execution failed"]
        base_state["execution_failure_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("execution_failure_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_execution_failure_with_just_guidance(self, base_state):
        """Just GUIDANCE keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {"Q1": "GUIDANCE: try a smaller grid"}
        base_state["pending_user_questions"] = ["Execution failed"]
        base_state["execution_failure_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("execution_failure_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_execution_failure_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Execution failed"]
        base_state["execution_failure_count"] = 2
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_execution_failure_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Execution failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_execution_failure_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {"Q1": "hmm not sure"}
        base_state["pending_user_questions"] = ["Execution failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Physics Failure Limit Tests
# =============================================================================

class TestSupervisorPhysicsFailureLimit:
    """Test physics_failure_limit trigger handling."""

    def test_supervisor_handles_physics_failure_retry(self, base_state):
        """User requests RETRY - should reset counter."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "physics_failure_limit"
        base_state["user_responses"] = {"Q1": "RETRY with different parameters"}
        base_state["pending_user_questions"] = ["Physics check failed"]
        base_state["physics_failure_count"] = 2

        result = supervisor_node(base_state)

        assert result.get("physics_failure_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_physics_failure_accept_partial(self, base_state):
        """User requests ACCEPT_PARTIAL - should mark as partial success."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "physics_failure_limit"
        base_state["user_responses"] = {"Q1": "ACCEPT_PARTIAL"}
        base_state["pending_user_questions"] = ["Physics check failed"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_physics_failure_partial_keyword(self, base_state):
        """Just PARTIAL keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "physics_failure_limit"
        base_state["user_responses"] = {"Q1": "PARTIAL is fine"}
        base_state["pending_user_questions"] = ["Physics check failed"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_physics_failure_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "physics_failure_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Physics check failed"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_physics_failure_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "physics_failure_limit"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Physics check failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_physics_failure_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "physics_failure_limit"
        base_state["user_responses"] = {"Q1": "keep going"}
        base_state["pending_user_questions"] = ["Physics check failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Context Overflow Tests
# =============================================================================

class TestSupervisorContextOverflow:
    """Test context_overflow trigger handling."""

    def test_supervisor_handles_context_overflow_with_truncate(self, base_state):
        """User requests TRUNCATE - should truncate paper text."""
        from src.agents.supervision.supervisor import supervisor_node

        # Create text long enough to be truncated (> 15000 + 39 + 5000 = 20039)
        base_state["paper_text"] = "x" * 25000
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "TRUNCATE"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "paper_text" in result
        assert len(result["paper_text"]) < 25000
        assert "[TRUNCATED" in result["paper_text"]
        assert "Truncating" in result.get("supervisor_feedback", "")
        assert result.get("ask_user_trigger") is None

    def test_supervisor_handles_context_overflow_truncate_short_text(self, base_state):
        """TRUNCATE on already short text - should not truncate."""
        from src.agents.supervision.supervisor import supervisor_node

        # Create text short enough to not need truncation
        short_text = "x" * 10000
        base_state["paper_text"] = short_text
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "TRUNCATE"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        # Paper text should be preserved (not truncated)
        assert result.get("paper_text") == short_text
        assert "short enough" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_context_overflow_summarize(self, base_state):
        """User requests SUMMARIZE - should enable summarization."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["paper_text"] = "x" * 25000
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "SUMMARIZE"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "summariz" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_context_overflow_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["paper_text"] = "x" * 25000
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Context too long"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_context_overflow_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_context_overflow_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "whatever"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# LLM Error Tests
# =============================================================================

class TestSupervisorLLMError:
    """Test llm_error trigger handling."""

    def test_supervisor_handles_llm_error_with_retry(self, base_state):
        """User requests RETRY - should continue."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["LLM API failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "retry" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_llm_error_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["LLM API failed"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_llm_error_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["LLM API failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_llm_error_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "I don't know"}
        base_state["pending_user_questions"] = ["LLM API failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Replan Limit Tests
# =============================================================================

class TestSupervisorReplanLimit:
    """Test replan_limit trigger handling."""

    def test_supervisor_handles_replan_limit_force_accept(self, base_state):
        """User requests FORCE_ACCEPT - should accept plan."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "replan_limit"
        base_state["user_responses"] = {"Q1": "FORCE_ACCEPT"}
        base_state["pending_user_questions"] = ["Replan limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "force-accepted" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_replan_limit_accept_keyword(self, base_state):
        """Just ACCEPT keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "replan_limit"
        base_state["user_responses"] = {"Q1": "ACCEPT the current plan"}
        base_state["pending_user_questions"] = ["Replan limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_replan_limit_guidance(self, base_state):
        """User provides GUIDANCE - should reset counter and replan."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "replan_limit"
        base_state["user_responses"] = {"Q1": "GUIDANCE: Focus on the main figure"}
        base_state["pending_user_questions"] = ["Replan limit reached"]
        base_state["replan_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("replan_count") == 0
        assert result.get("supervisor_verdict") == "replan_with_guidance"
        assert "Focus on the main figure" in result.get("planner_feedback", "")

    def test_supervisor_handles_replan_limit_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "replan_limit"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Replan limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_replan_limit_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "replan_limit"
        base_state["user_responses"] = {"Q1": "maybe"}
        base_state["pending_user_questions"] = ["Replan limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Backtrack Approval Tests
# =============================================================================

class TestSupervisorBacktrackApproval:
    """Test backtrack_approval trigger handling."""

    def test_supervisor_handles_backtrack_approval_approve(self, base_state):
        """User APPROVES backtrack - should proceed with backtrack."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_approval"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_user_questions"] = ["Approve backtrack to stage_0?"]
        base_state["backtrack_decision"] = {
            "target_stage_id": "stage_0",
            "reason": "Results don't match"
        }
        base_state["plan"] = {
            "stages": [
                {"stage_id": "stage_0", "dependencies": []},
                {"stage_id": "stage_1", "dependencies": ["stage_0"]},
                {"stage_id": "stage_2", "dependencies": ["stage_1"]},
            ]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "backtrack_to_stage"
        assert result.get("backtrack_decision") is not None
        decision = result.get("backtrack_decision", {})
        assert decision.get("target_stage_id") == "stage_0"
        # Should have identified dependent stages
        assert "stages_to_invalidate" in decision
        assert "stage_1" in decision["stages_to_invalidate"]

    def test_supervisor_handles_backtrack_approval_yes_keyword(self, base_state):
        """Just YES keyword should also approve."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_approval"
        base_state["user_responses"] = {"Q1": "YES"}
        base_state["pending_user_questions"] = ["Approve backtrack?"]
        base_state["backtrack_decision"] = {
            "target_stage_id": "stage_0",
            "reason": "Test"
        }
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "dependencies": []}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "backtrack_to_stage"

    def test_supervisor_handles_backtrack_approval_reject(self, base_state):
        """User REJECTS backtrack - should continue normally."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_approval"
        base_state["user_responses"] = {"Q1": "REJECT"}
        base_state["pending_user_questions"] = ["Approve backtrack?"]
        base_state["backtrack_decision"] = {
            "target_stage_id": "stage_0",
            "reason": "Test"
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert result.get("backtrack_suggestion") is None

    def test_supervisor_handles_backtrack_approval_no_keyword(self, base_state):
        """NO keyword should also reject."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_approval"
        base_state["user_responses"] = {"Q1": "NO, continue please"}
        base_state["pending_user_questions"] = ["Approve backtrack?"]
        base_state["backtrack_decision"] = {
            "target_stage_id": "stage_0",
            "reason": "Test"
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_backtrack_approval_unclear_continues(self, base_state):
        """Unclear response should default to continue."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_approval"
        base_state["user_responses"] = {"Q1": "hmm let me think"}
        base_state["pending_user_questions"] = ["Approve backtrack?"]
        base_state["backtrack_decision"] = {
            "target_stage_id": "stage_0",
            "reason": "Test"
        }

        result = supervisor_node(base_state)

        # Default behavior for unclear is ok_continue
        assert result.get("supervisor_verdict") == "ok_continue"


# =============================================================================
# Deadlock Detected Tests
# =============================================================================

class TestSupervisorDeadlockDetected:
    """Test deadlock_detected trigger handling."""

    def test_supervisor_handles_deadlock_generate_report(self, base_state):
        """User requests GENERATE_REPORT - should stop and generate report."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "deadlock_detected"
        base_state["user_responses"] = {"Q1": "GENERATE_REPORT"}
        base_state["pending_user_questions"] = ["Deadlock detected"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_deadlock_report_keyword(self, base_state):
        """Just REPORT keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "deadlock_detected"
        base_state["user_responses"] = {"Q1": "REPORT and finish"}
        base_state["pending_user_questions"] = ["Deadlock detected"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_deadlock_replan(self, base_state):
        """User requests REPLAN - should trigger replanning."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "deadlock_detected"
        base_state["user_responses"] = {"Q1": "REPLAN"}
        base_state["pending_user_questions"] = ["Deadlock detected"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "replan_needed"
        assert "deadlock" in result.get("planner_feedback", "").lower()

    def test_supervisor_handles_deadlock_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "deadlock_detected"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Deadlock detected"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_deadlock_unclear_response(self, base_state):
        """User gives unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "deadlock_detected"
        base_state["user_responses"] = {"Q1": "what do you suggest?"}
        base_state["pending_user_questions"] = ["Deadlock detected"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Clarification Tests
# =============================================================================

class TestSupervisorClarification:
    """Test clarification trigger handling."""

    def test_supervisor_handles_clarification_with_response(self, base_state):
        """User provides clarification - should continue with feedback."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "clarification"
        base_state["user_responses"] = {
            "Q1": "The wavelength should be 550nm, not 650nm"
        }
        base_state["pending_user_questions"] = ["Please clarify the target wavelength"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "clarification" in result.get("supervisor_feedback", "").lower()
        assert "550nm" in result.get("supervisor_feedback", "")

    def test_supervisor_handles_clarification_empty_response(self, base_state):
        """User provides empty response - should continue anyway."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "clarification"
        base_state["user_responses"] = {"Q1": ""}
        base_state["pending_user_questions"] = ["Please clarify"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "no clarification" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_clarification_no_response(self, base_state):
        """No user response at all - should continue anyway."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "clarification"
        base_state["user_responses"] = {}
        base_state["pending_user_questions"] = ["Please clarify"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"


# =============================================================================
# Critical Error Retry Tests
# =============================================================================

class TestSupervisorCriticalErrorRetry:
    """Test critical error triggers (missing_paper_text, missing_stage_id, etc.)."""

    def test_supervisor_handles_missing_paper_text_retry(self, base_state):
        """User requests RETRY for missing paper text."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "missing_paper_text"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["Paper text is missing"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert "retry" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_missing_paper_text_stop(self, base_state):
        """User requests STOP for missing paper text."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "missing_paper_text"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Paper text is missing"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_missing_stage_id_retry(self, base_state):
        """User requests RETRY for missing stage ID."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "missing_stage_id"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["Stage ID is missing"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_progress_init_failed_stop(self, base_state):
        """User requests STOP for progress init failure."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "progress_init_failed"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Progress init failed"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_critical_error_unclear_response(self, base_state):
        """Unclear response to critical error - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "missing_paper_text"
        base_state["user_responses"] = {"Q1": "hmm"}
        base_state["pending_user_questions"] = ["Paper text is missing"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Planning Error Retry Tests
# =============================================================================

class TestSupervisorPlanningErrorRetry:
    """Test planning error triggers (no_stages_available, invalid_backtrack_target, etc.)."""

    def test_supervisor_handles_no_stages_available_replan(self, base_state):
        """User requests REPLAN for no stages available."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "no_stages_available"
        base_state["user_responses"] = {"Q1": "REPLAN"}
        base_state["pending_user_questions"] = ["No stages available"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "replan_needed"
        assert "REPLAN" in result.get("planner_feedback", "")

    def test_supervisor_handles_invalid_backtrack_target_stop(self, base_state):
        """User requests STOP for invalid backtrack target."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "invalid_backtrack_target"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Invalid backtrack target"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_backtrack_target_not_found_replan(self, base_state):
        """User requests REPLAN for backtrack target not found."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_target_not_found"
        base_state["user_responses"] = {"Q1": "REPLAN"}
        base_state["pending_user_questions"] = ["Backtrack target not found"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "replan_needed"

    def test_supervisor_handles_planning_error_unclear_response(self, base_state):
        """Unclear response to planning error - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "no_stages_available"
        base_state["user_responses"] = {"Q1": "continue anyway"}
        base_state["pending_user_questions"] = ["No stages available"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Backtrack Limit Tests
# =============================================================================

class TestSupervisorBacktrackLimit:
    """Test backtrack_limit trigger handling."""

    def test_supervisor_handles_backtrack_limit_force_continue(self, base_state):
        """User requests FORCE_CONTINUE - should continue."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_limit"
        base_state["user_responses"] = {"Q1": "FORCE_CONTINUE"}
        base_state["pending_user_questions"] = ["Backtrack limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        # Feedback should mention continuing (component says "Continuing")
        assert "continu" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_backtrack_limit_continue_keyword(self, base_state):
        """Just CONTINUE keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_limit"
        base_state["user_responses"] = {"Q1": "CONTINUE anyway"}
        base_state["pending_user_questions"] = ["Backtrack limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_backtrack_limit_force_keyword(self, base_state):
        """Just FORCE keyword should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_limit"
        base_state["user_responses"] = {"Q1": "FORCE it"}
        base_state["pending_user_questions"] = ["Backtrack limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_backtrack_limit_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_limit"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Backtrack limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_backtrack_limit_unclear_response(self, base_state):
        """Unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "backtrack_limit"
        base_state["user_responses"] = {"Q1": "I think we should go back"}
        base_state["pending_user_questions"] = ["Backtrack limit reached"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Invalid Backtrack Decision Tests
# =============================================================================

class TestSupervisorInvalidBacktrackDecision:
    """Test invalid_backtrack_decision trigger handling."""

    def test_supervisor_handles_invalid_backtrack_decision_continue(self, base_state):
        """User requests CONTINUE - should continue and clear decision."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "invalid_backtrack_decision"
        base_state["user_responses"] = {"Q1": "CONTINUE"}
        base_state["pending_user_questions"] = ["Invalid backtrack decision"]
        base_state["backtrack_decision"] = {
            "target_stage_id": "invalid_stage",
            "reason": "Test"
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert result.get("backtrack_decision") is None

    def test_supervisor_handles_invalid_backtrack_decision_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "invalid_backtrack_decision"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Invalid backtrack decision"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True

    def test_supervisor_handles_invalid_backtrack_decision_unclear(self, base_state):
        """Unclear response - should ask for clarification."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "invalid_backtrack_decision"
        base_state["user_responses"] = {"Q1": "fix it please"}
        base_state["pending_user_questions"] = ["Invalid backtrack decision"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "clarify" in questions[0].lower()


# =============================================================================
# Unknown Trigger Tests
# =============================================================================

class TestSupervisorUnknownTrigger:
    """Test handling of unknown triggers."""

    def test_supervisor_with_unknown_trigger_continues(self, base_state):
        """Unknown trigger should default to ok_continue."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "unknown_trigger_xyz"
        base_state["user_responses"] = {"Q1": "yes"}
        base_state["pending_user_questions"] = ["Unknown question"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        # Should mention the unknown trigger in feedback
        assert "unknown" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_with_empty_trigger_continues(self, base_state):
        """Empty string trigger should be treated as no trigger."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = ""
        base_state["user_responses"] = {}
        base_state["pending_user_questions"] = []
        base_state["current_stage_id"] = "stage_0"

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            return_value={"verdict": "ok_continue", "reasoning": "test"},
        ):
            result = supervisor_node(base_state)

        # Empty string should be falsy, so normal supervision should run
        assert result.get("supervisor_verdict") == "ok_continue"


# =============================================================================
# User Interaction Logging Tests
# =============================================================================

class TestSupervisorUserInteractionLogging:
    """Test that user interactions are properly logged to progress."""

    def test_supervisor_logs_user_interactions(self, base_state):
        """User interactions should be logged to progress."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {
            "Q1": "PROVIDE_HINT: Focus on resonance monitors"
        }
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}],
            "user_interactions": [],
        }

        result = supervisor_node(base_state)

        interactions = result.get("progress", {}).get("user_interactions", [])
        assert len(interactions) == 1
        entry = interactions[0]
        assert entry.get("interaction_type") == "code_review_limit"
        assert entry.get("context", {}).get("stage_id") == "stage_0"
        assert "PROVIDE_HINT" in entry.get("user_response", "")
        assert entry.get("question") == "Code review limit reached"
        # Should have an ID
        assert entry.get("id") is not None
        # Should have timestamp
        assert entry.get("timestamp") is not None

    def test_supervisor_logs_interaction_without_stage_id(self, base_state):
        """Interactions without current_stage_id should still be logged."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["LLM failed"]
        base_state["current_stage_id"] = None
        base_state["progress"] = {
            "stages": [],
            "user_interactions": [],
        }

        result = supervisor_node(base_state)

        interactions = result.get("progress", {}).get("user_interactions", [])
        assert len(interactions) == 1
        entry = interactions[0]
        assert entry.get("interaction_type") == "llm_error"
        assert entry.get("context", {}).get("stage_id") is None

    def test_supervisor_appends_to_existing_interactions(self, base_state):
        """New interactions should be appended to existing list."""
        from src.agents.supervision.supervisor import supervisor_node

        existing_interaction = {
            "id": "U1",
            "interaction_type": "material_checkpoint",
            "user_response": "APPROVE",
        }
        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Code review limit"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}],
            "user_interactions": [existing_interaction],
        }

        result = supervisor_node(base_state)

        interactions = result.get("progress", {}).get("user_interactions", [])
        assert len(interactions) == 2
        assert interactions[0] == existing_interaction
        assert interactions[1].get("id") == "U2"


# =============================================================================
# LLM Error Fallback Tests
# =============================================================================

class TestSupervisorLLMErrorFallback:
    """Test supervisor behavior when LLM calls fail (normal supervision path)."""

    def test_supervisor_defaults_on_llm_error(self, base_state):
        """LLM error during normal supervision should default to ok_continue."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            side_effect=RuntimeError("LLM API error"),
        ):
            result = supervisor_node(base_state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "unavailable" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_preserves_workflow_phase_on_llm_error(self, base_state):
        """LLM error should still set workflow_phase to supervision."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["current_stage_id"] = "stage_0"

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            side_effect=RuntimeError("LLM API error"),
        ):
            result = supervisor_node(base_state)

        assert result.get("workflow_phase") == "supervision"


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestSupervisorEdgeCases:
    """Test edge cases and robustness."""

    def test_supervisor_handles_none_user_responses(self, base_state):
        """None user_responses should be handled gracefully."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = None
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve?"]

        result = supervisor_node(base_state)

        # Should handle None gracefully and ask for clarification
        assert "supervisor_verdict" in result

    def test_supervisor_handles_non_dict_user_responses(self, base_state):
        """Non-dict user_responses should be handled gracefully."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = "APPROVE"  # Wrong type
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve?"]

        result = supervisor_node(base_state)

        # Should handle wrong type gracefully
        assert "supervisor_verdict" in result

    def test_supervisor_handles_empty_pending_questions(self, base_state):
        """Empty pending_user_questions should be handled."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/mat/Au.csv"}]
        base_state["pending_user_questions"] = []
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        # Should still process the trigger
        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_awaiting_user_input_flag(self, base_state):
        """If awaiting_user_input is set, context check should return early."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["awaiting_user_input"] = True
        base_state["ask_user_trigger"] = "material_checkpoint"

        result = supervisor_node(base_state)

        # Should return empty or minimal result when already awaiting
        # The actual behavior depends on check_context_or_escalate
        assert isinstance(result, dict)

    def test_supervisor_clears_ask_user_trigger_after_handling(self, base_state):
        """Trigger should be cleared after handling."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Limit reached"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("ask_user_trigger") is None

    def test_supervisor_sets_workflow_phase(self, base_state):
        """Workflow phase should always be set to supervision."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["Error"]

        result = supervisor_node(base_state)

        assert result.get("workflow_phase") == "supervision"


# =============================================================================
# Archive Errors Tests
# =============================================================================

class TestSupervisorArchiveErrors:
    """Test archive error handling and recovery."""

    def test_supervisor_retries_archive_errors(self, base_state):
        """Archive errors from previous runs should be retried."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["current_stage_id"] = "stage_1"
        base_state["archive_errors"] = [
            {"stage_id": "stage_0", "error": "Previous failure"}
        ]
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "status": "completed_success"},
                {"stage_id": "stage_1", "status": "in_progress"},
            ]
        }

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            return_value={"verdict": "ok_continue", "reasoning": "test"},
        ):
            with patch(
                "src.agents.supervision.supervisor.archive_stage_outputs_to_progress",
            ) as mock_archive:
                result = supervisor_node(base_state)

        # Should have attempted to archive for both the retry and current stage
        assert mock_archive.call_count >= 1

    def test_supervisor_handles_invalid_archive_errors_type(self, base_state):
        """Non-list archive_errors should be handled gracefully."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["current_stage_id"] = "stage_0"
        base_state["archive_errors"] = "not a list"  # Wrong type

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            return_value={"verdict": "ok_continue", "reasoning": "test"},
        ):
            result = supervisor_node(base_state)

        # Should reset archive_errors to empty list
        assert result.get("archive_errors") == []


# =============================================================================
# Reviewer Escalation Tests
# =============================================================================

class TestSupervisorReviewerEscalation:
    """Test reviewer_escalation trigger handling.
    
    This trigger occurs when a reviewer LLM explicitly requests user input
    via the escalate_to_user field in its output.
    """

    def test_supervisor_handles_reviewer_escalation_provide_guidance(self, base_state):
        """User provides PROVIDE_GUIDANCE - should set reviewer_feedback and route to retry."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {
            "Q1": "PROVIDE_GUIDANCE: Use the Drude model with adjusted parameters"
        }
        base_state["pending_user_questions"] = [
            "Which material model should I use: Drude or tabulated data?"
        ]
        base_state["reviewer_escalation_source"] = "code_reviewer"

        # Mock the LLM call in the trigger handler's SMART PATH
        # Return retry_code_review to trigger reviewer_feedback setting
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "retry_code_review", "summary": "User provided guidance."},
        ):
            result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "retry_code_review"
        assert "reviewer_feedback" in result
        assert "User guidance:" in result["reviewer_feedback"]
        assert "Drude model" in result["reviewer_feedback"]
        assert result.get("ask_user_trigger") is None
        assert result.get("workflow_phase") == "supervision"

    def test_supervisor_handles_reviewer_escalation_guidance_alias(self, base_state):
        """GUIDANCE alias should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {"Q1": "GUIDANCE: Focus on near-field enhancement"}
        base_state["pending_user_questions"] = ["What aspect should I prioritize?"]

        # Mock the LLM call in the trigger handler's SMART PATH
        # Return retry_code_review to trigger reviewer_feedback setting
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "retry_code_review", "summary": "User provided guidance."},
        ):
            result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "retry_code_review"
        assert "near-field enhancement" in result.get("reviewer_feedback", "")

    def test_supervisor_handles_reviewer_escalation_answer_alias(self, base_state):
        """ANSWER alias should also work."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {"Q1": "ANSWER: The spacing is 20nm period"}
        base_state["pending_user_questions"] = ["What is the array spacing?"]

        # Mock the LLM call in the trigger handler's SMART PATH
        # Return retry_code_review to trigger reviewer_feedback setting
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "retry_code_review", "summary": "User provided answer."},
        ):
            result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "retry_code_review"
        assert "20nm period" in result.get("reviewer_feedback", "")

    def test_supervisor_handles_reviewer_escalation_skip(self, base_state):
        """User requests SKIP - should skip the stage."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Question from reviewer"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "ok_continue"
        assert result.get("ask_user_trigger") is None

    def test_supervisor_handles_reviewer_escalation_stop(self, base_state):
        """User requests STOP - should stop workflow."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {"Q1": "STOP"}
        base_state["pending_user_questions"] = ["Question from reviewer"]

        result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "all_complete"
        assert result.get("should_stop") is True
        assert result.get("ask_user_trigger") is None

    def test_supervisor_handles_reviewer_escalation_freeform_response(self, base_state):
        """User gives free-form response - should accept as guidance (no keyword required)."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {"Q1": "I'm not sure what to do, but try the Drude model first"}
        base_state["pending_user_questions"] = ["Question from reviewer"]

        # Mock the LLM call in the trigger handler's SMART PATH
        # Return retry_code_review to trigger reviewer_feedback setting
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "retry_code_review", "summary": "User provided guidance."},
        ):
            result = supervisor_node(base_state)

        # Free-form responses should now be accepted as guidance
        assert result.get("supervisor_verdict") == "retry_code_review"
        # Verify reviewer feedback contains the user's response
        assert "reviewer_feedback" in result
        assert "try the Drude model first" in result["reviewer_feedback"]

    def test_supervisor_handles_reviewer_escalation_lowercase(self, base_state):
        """Keywords should work case-insensitively."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {"Q1": "provide_guidance: use tabulated data"}
        base_state["pending_user_questions"] = ["Question from reviewer"]

        # Mock the LLM call in the trigger handler's SMART PATH
        # Return retry_code_review to trigger reviewer_feedback setting
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "retry_code_review", "summary": "User provided guidance."},
        ):
            result = supervisor_node(base_state)

        assert result.get("supervisor_verdict") == "retry_code_review"
        assert "tabulated data" in result.get("reviewer_feedback", "")

    def test_reviewer_escalation_full_flow_code_reviewer(self, base_state):
        """End-to-end: code reviewer escalates -> user responds -> supervisor handles."""
        from src.agents.code import code_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node

        # Step 1: Code reviewer returns escalate_to_user
        with patch("src.agents.code.build_agent_prompt", return_value="Prompt"):
            with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
                mock_llm.return_value = {
                    "verdict": "needs_revision",
                    "escalate_to_user": "The design specifies Drude model but measured optical constants are available. Which should I use for best accuracy?",
                    "feedback": "Material model ambiguity",
                }
                
                base_state["code"] = "import meep as mp"
                base_state["design_description"] = {"materials": ["gold"]}
                
                reviewer_result = code_reviewer_node(base_state)
        
        # Verify reviewer sets up escalation correctly
        assert reviewer_result["ask_user_trigger"] == "reviewer_escalation"
        assert reviewer_result.get("ask_user_trigger") is not None
        assert "Drude model" in reviewer_result["pending_user_questions"][0]
        assert reviewer_result["reviewer_escalation_source"] == "code_reviewer"
        
        # Step 2: User provides guidance
        base_state.update(reviewer_result)
        base_state["user_responses"] = {
            "Q1": "PROVIDE_GUIDANCE: Use measured optical constants for better accuracy"
        }
        
        # Step 3: Supervisor handles the user response
        # Mock the LLM call in the trigger handler's SMART PATH
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "retry_code_review", "summary": "Re-review with guidance."},
        ):
            supervisor_result = supervisor_node(base_state)
        
        # Verify supervisor processed guidance correctly
        # Should route back to code_review since escalation came from there
        assert supervisor_result["supervisor_verdict"] == "retry_code_review"
        assert "reviewer_feedback" in supervisor_result
        assert "measured optical constants" in supervisor_result["reviewer_feedback"]
        assert supervisor_result.get("ask_user_trigger") is None
        # Workflow will route back to code reviewer with the guidance

    def test_reviewer_escalation_full_flow_design_reviewer(self, base_state):
        """End-to-end: design reviewer escalates -> user responds -> supervisor handles."""
        from src.agents.design import design_reviewer_node
        from src.agents.supervision.supervisor import supervisor_node

        # Step 1: Design reviewer returns escalate_to_user
        with patch("src.agents.base.check_context_or_escalate", return_value=None):
            with patch("src.agents.design.build_agent_prompt", return_value="Prompt"):
                with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
                    mock_llm.return_value = {
                        "verdict": "approve",
                        "escalate_to_user": "Should I use 2D or 3D simulation? 2D is faster but 3D may be more accurate.",
                        "issues": [],
                    }
                    
                    base_state["design_description"] = {"geometry": "nanorod"}
                    
                    reviewer_result = design_reviewer_node(base_state)
        
        # Verify reviewer sets up escalation correctly
        assert reviewer_result["ask_user_trigger"] == "reviewer_escalation"
        assert "2D or 3D" in reviewer_result["pending_user_questions"][0]
        assert reviewer_result["reviewer_escalation_source"] == "design_reviewer"
        
        # Step 2: User chooses to skip
        base_state.update(reviewer_result)
        base_state["user_responses"] = {"Q1": "SKIP_STAGE"}
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }
        
        # Step 3: Supervisor handles the skip
        supervisor_result = supervisor_node(base_state)
        
        # Verify supervisor processed skip correctly
        assert supervisor_result["supervisor_verdict"] == "ok_continue"
        assert supervisor_result.get("ask_user_trigger") is None

    def test_supervisor_logs_reviewer_escalation_interaction(self, base_state):
        """User interactions for reviewer_escalation should be logged to progress."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "reviewer_escalation"
        base_state["user_responses"] = {
            "Q1": "PROVIDE_GUIDANCE: Use the interpolated data"
        }
        base_state["pending_user_questions"] = ["Which data source?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}],
            "user_interactions": [],
        }

        # Mock the LLM call in the trigger handler's SMART PATH
        with patch(
            "src.agents.supervision.trigger_handlers.call_agent_with_metrics",
            return_value={"verdict": "ok_continue", "summary": "User provided guidance."},
        ):
            result = supervisor_node(base_state)

        interactions = result.get("progress", {}).get("user_interactions", [])
        assert len(interactions) == 1
        entry = interactions[0]
        assert entry.get("interaction_type") == "reviewer_escalation"
        assert entry.get("context", {}).get("stage_id") == "stage_0"
        assert "PROVIDE_GUIDANCE" in entry.get("user_response", "")
        assert entry.get("question") == "Which data source?"
