"""Supervisor trigger handling integration tests."""

from unittest.mock import patch


class TestSupervisorMaterialCheckpoint:
    """Supervisor ask_user trigger handling for material checkpoints."""

    def test_supervisor_handles_material_checkpoint(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        pending_materials = [{"name": "gold", "path": "/materials/Au.csv"}]
        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = pending_materials
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["validated_materials"] == pending_materials
        assert result.get("pending_validated_materials") == []
        assert result.get("ask_user_trigger") is None

    def test_supervisor_material_checkpoint_without_pending_materials_escalates(
        self, base_state
    ):
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

    def test_supervisor_defaults_on_llm_error(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["current_stage_id"] = "stage_0"
        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            side_effect=RuntimeError("LLM API error"),
        ):
            result = supervisor_node(base_state)

        assert result["supervisor_verdict"] == "ok_continue"

    def test_supervisor_with_unknown_trigger(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "unknown_trigger_xyz"
        base_state["user_responses"] = {"Q1": "yes"}
        base_state["pending_user_questions"] = ["Unknown question"]

        result = supervisor_node(base_state)
        assert "supervisor_verdict" in result


class TestSupervisorTriggerHandlers:
    """Test supervisor handles various ask_user_trigger types correctly."""

    def test_supervisor_handles_code_review_limit_with_hint(self, base_state):
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
        assert result.get("ask_user_trigger") is None
        assert result.get("workflow_phase") == "supervision"
        assert result.get("pending_user_questions") in (None, [])

    def test_supervisor_handles_design_review_limit_skip(self, base_state):
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

    def test_supervisor_handles_execution_failure_with_retry(self, base_state):
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

    def test_supervisor_handles_llm_error_with_retry(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["LLM API failed"]

        result = supervisor_node(base_state)
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "retry" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_context_overflow_with_truncate(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["paper_text"] = "x" * 25000
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "TRUNCATE"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)
        if "paper_text" in result:
            assert len(result["paper_text"]) < 25000
            assert "[TRUNCATED" in result["paper_text"]
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "Truncating" in result.get("supervisor_feedback", "")
        assert result.get("ask_user_trigger") is None

    def test_supervisor_logs_user_interactions(self, base_state):
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
        assert entry.get("question")

