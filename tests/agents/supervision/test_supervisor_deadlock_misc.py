"""Supervisor tests for deadlock and unknown trigger handling."""

from unittest.mock import patch, MagicMock

import pytest

from src.agents.supervision import supervisor_node


class TestDeadlockTrigger:
    """Tests for deadlock_detected trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_generates_report_on_generate_report(self, mock_call, mock_context):
        """Should generate report on GENERATE_REPORT and clear trigger."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
            "pending_user_questions": ["What should we do?"],
        }

        result = supervisor_node(state)

        # Verify verdict and stop flag
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        
        # Verify trigger is cleared
        assert result["ask_user_trigger"] is None
        
        # Verify normal supervision is skipped (no LLM call)
        mock_call.assert_not_called()
        
        # Verify workflow phase is set
        assert result["workflow_phase"] == "supervision"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_generates_report_on_report_keyword(self, mock_call, mock_context):
        """Should generate report when response contains REPORT keyword."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "Please generate REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_replans_on_replan(self, mock_call, mock_context):
        """Should trigger replan on REPLAN and set planner_feedback."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPLAN"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "replan_needed"
        assert "planner_feedback" in result
        assert "deadlock" in result["planner_feedback"].lower()
        assert "REPLAN" in result["planner_feedback"]
        assert result["ask_user_trigger"] is None
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_stops_on_stop(self, mock_call, mock_context):
        """Should stop on STOP and set should_stop flag."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "STOP"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        assert result["ask_user_trigger"] is None
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_asks_clarification_on_unclear(self, mock_call, mock_context):
        """Should ask clarification on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "I'm not sure"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert any("GENERATE_REPORT" in q or "REPLAN" in q or "STOP" in q 
                  for q in result["pending_user_questions"])
        assert result["ask_user_trigger"] is None
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_case_insensitive_responses(self, mock_call, mock_context):
        """Should handle case-insensitive user responses."""
        mock_context.return_value = None
        
        # Test lowercase
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "generate_report"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        result = supervisor_node(state)
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        
        # Test mixed case
        state["user_responses"] = {"Question": "RePlan"}
        result = supervisor_node(state)
        assert result["supervisor_verdict"] == "replan_needed"
        
        # Test uppercase
        state["user_responses"] = {"Question": "STOP"}
        result = supervisor_node(state)
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_logs_user_interaction(self, mock_call, mock_context):
        """Should log user interaction to progress."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
            "pending_user_questions": ["What should we do?"],
        }

        result = supervisor_node(state)

        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "deadlock_detected"
        assert interaction["context"]["stage_id"] == "stage1"
        assert interaction["context"]["agent"] == "SupervisorAgent"
        assert "GENERATE_REPORT" in interaction["user_response"]
        assert "id" in interaction
        assert "timestamp" in interaction

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_empty_user_responses(self, mock_call, mock_context):
        """Should handle empty user responses gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        # Empty response should trigger clarification request
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_multiple_responses(self, mock_call, mock_context):
        """Should use the last response when multiple responses exist."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {
                "Question1": "REPLAN",
                "Question2": "GENERATE_REPORT",  # Last response should be used
            },
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        # Should use last response (GENERATE_REPORT)
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_none_current_stage_id(self, mock_call, mock_context):
        """Should handle None current_stage_id."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": None,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        
        # Should still log interaction even with None stage_id
        if "progress" in result and "user_interactions" in result["progress"]:
            interaction = result["progress"]["user_interactions"][0]
            assert interaction["context"]["stage_id"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_skips_normal_supervision_when_handling_trigger(self, mock_call, mock_context):
        """Should skip normal supervision LLM call when handling trigger."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "STOP"},
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        result = supervisor_node(state)

        # Verify LLM was not called (normal supervision skipped)
        mock_call.assert_not_called()
        
        # Verify verdict was set by trigger handler
        assert result["supervisor_verdict"] == "all_complete"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_whitespace_in_responses(self, mock_call, mock_context):
        """Should handle responses with leading/trailing whitespace."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "  GENERATE_REPORT  "},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_partial_keyword_match(self, mock_call, mock_context):
        """Should handle partial keyword matches correctly."""
        mock_context.return_value = None
        
        # REPORT should match GENERATE_REPORT
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        result = supervisor_node(state)
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_missing_progress(self, mock_call, mock_context):
        """Should handle missing progress field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
        }

        # Should not crash
        result = supervisor_node(state)
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True


class TestUnknownTrigger:
    """Tests for unknown trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_unknown_trigger(self, mock_call, mock_context):
        """Should handle unknown trigger gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "some_unknown_trigger",
            "user_responses": {"Question": "Response"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in result
        assert "unknown" in result["supervisor_feedback"].lower()
        assert "some_unknown_trigger" in result["supervisor_feedback"]
        assert result["ask_user_trigger"] is None
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_unknown_trigger_clears_trigger(self, mock_call, mock_context):
        """Should clear trigger even for unknown triggers."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "completely_unknown_trigger_xyz",
            "user_responses": {"Question": "any response"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_unknown_trigger_skips_normal_supervision(self, mock_call, mock_context):
        """Should skip normal supervision for unknown triggers."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "unknown_trigger",
            "user_responses": {"Question": "response"},
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        result = supervisor_node(state)

        # Should not call LLM for normal supervision
        mock_call.assert_not_called()
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_unknown_trigger_with_empty_response(self, mock_call, mock_context):
        """Should handle unknown trigger with empty response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "unknown_trigger",
            "user_responses": {},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in result
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_unknown_trigger_logs_interaction(self, mock_call, mock_context):
        """Should log user interaction for unknown triggers."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "unknown_trigger",
            "user_responses": {"Question": "some response"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
            "pending_user_questions": ["Question?"],
        }

        result = supervisor_node(state)

        if "progress" in result and "user_interactions" in result["progress"]:
            assert len(result["progress"]["user_interactions"]) == 1
            interaction = result["progress"]["user_interactions"][0]
            assert interaction["interaction_type"] == "unknown_trigger"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_unknown_trigger_with_none_stage_id(self, mock_call, mock_context):
        """Should handle unknown trigger with None stage_id."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "unknown_trigger",
            "user_responses": {"Question": "response"},
            "current_stage_id": None,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None


class TestDeadlockTriggerEdgeCases:
    """Edge case tests for deadlock_detected trigger."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_invalid_user_responses_type(self, mock_call, mock_context):
        """Should handle invalid user_responses type gracefully."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": "not a dict",  # Invalid type
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        # Should handle gracefully (supervisor_node logs warning and uses empty dict)
        result = supervisor_node(state)
        
        # Should still clear trigger
        assert result["ask_user_trigger"] is None
        # Should ask for clarification since response is invalid
        assert result["supervisor_verdict"] == "ask_user"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_none_user_responses(self, mock_call, mock_context):
        """Should handle None user_responses."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": None,
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        result = supervisor_node(state)
        
        assert result["ask_user_trigger"] is None
        assert result["supervisor_verdict"] == "ask_user"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_response_with_only_whitespace(self, mock_call, mock_context):
        """Should handle response containing only whitespace."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "   \n\t  "},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_workflow_phase_set(self, mock_call, mock_context):
        """Should always set workflow_phase."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        result = supervisor_node(state)
        
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "supervision"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_missing_pending_user_questions(self, mock_call, mock_context):
        """Should handle missing pending_user_questions field."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        # Should still work and log interaction
        assert result["supervisor_verdict"] == "all_complete"
        if "progress" in result and "user_interactions" in result["progress"]:
            interaction = result["progress"]["user_interactions"][0]
            # Should handle missing pending_user_questions gracefully
            assert "question" in interaction

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_replan_does_not_set_should_stop(self, mock_call, mock_context):
        """REPLAN should NOT set should_stop flag."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPLAN"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        # REPLAN should NOT stop the workflow
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_ask_user_does_not_set_should_stop(self, mock_call, mock_context):
        """ask_user verdict should NOT set should_stop flag."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "unclear response"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        # ask_user should NOT stop the workflow
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_planner_feedback_set_for_replan(self, mock_call, mock_context):
        """REPLAN should set planner_feedback with user response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "REPLAN because of issue X"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "planner_feedback" in result
        assert "deadlock" in result["planner_feedback"].lower()
        # Should preserve original case of user response
        assert "issue X" in result["planner_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_clarification_question_content(self, mock_call, mock_context):
        """Clarification question should mention all valid options."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "invalid response"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        question_text = " ".join(result["pending_user_questions"])
        # Should mention all valid options
        assert "GENERATE_REPORT" in question_text or "REPORT" in question_text
        assert "REPLAN" in question_text
        assert "STOP" in question_text

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_context_escalation(self, mock_call, mock_context):
        """Should handle context escalation correctly."""
        # Context check returns escalation (awaiting_user_input)
        mock_context.return_value = {"awaiting_user_input": True}
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        result = supervisor_node(state)
        
        # Should return early with context escalation
        assert result.get("awaiting_user_input") is True
        # Should not process trigger when context escalates
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_context_state_update(self, mock_call, mock_context):
        """Should use updated state from context check."""
        # Context check returns state update
        mock_context.return_value = {"new_field": "value"}
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)
        
        # Should process trigger normally
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor._retry_archive_errors")
    def test_calls_archive_error_recovery(self, mock_retry, mock_call, mock_context):
        """Should call archive error recovery."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {"stages": []},
        }

        supervisor_node(state)
        
        # Should call archive error recovery
        mock_retry.assert_called_once()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_user_interaction_structure(self, mock_call, mock_context):
        """User interaction should have correct structure."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
            "pending_user_questions": ["What should we do?"],
        }

        result = supervisor_node(state)
        
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        
        interaction = result["progress"]["user_interactions"][0]
        # Verify all required fields
        assert "id" in interaction
        assert "timestamp" in interaction
        assert "interaction_type" in interaction
        assert "context" in interaction
        assert "question" in interaction
        assert "user_response" in interaction
        assert "impact" in interaction
        assert "alternatives_considered" in interaction
        
        # Verify context structure
        assert "stage_id" in interaction["context"]
        assert "agent" in interaction["context"]
        assert "reason" in interaction["context"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_interaction_id_format(self, mock_call, mock_context):
        """User interaction ID should follow correct format."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
        }

        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        # ID should start with 'U' followed by number
        assert interaction["id"].startswith("U")
        assert interaction["id"][1:].isdigit()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_timestamp_format(self, mock_call, mock_context):
        """User interaction timestamp should be ISO format."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
        }

        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        # Timestamp should be ISO format string
        assert isinstance(interaction["timestamp"], str)
        assert "T" in interaction["timestamp"]  # ISO format includes T
        assert "Z" in interaction["timestamp"] or "+" in interaction["timestamp"]  # Timezone indicator

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_handles_existing_user_interactions(self, mock_call, mock_context):
        """Should append to existing user interactions."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Question": "GENERATE_REPORT"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1", "interaction_type": "previous"}
                ]
            },
        }

        result = supervisor_node(state)
        
        assert len(result["progress"]["user_interactions"]) == 2
        assert result["progress"]["user_interactions"][0]["id"] == "U1"
        assert result["progress"]["user_interactions"][1]["interaction_type"] == "deadlock_detected"
        # New interaction should have ID U2
        assert result["progress"]["user_interactions"][1]["id"] == "U2"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_no_interaction_logged_when_no_trigger(self, mock_call, mock_context):
        """Should not log interaction when no trigger present."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": None,  # No trigger
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
        }

        result = supervisor_node(state)
        
        # Should not add user interaction when no trigger
        if "progress" in result and "user_interactions" in result["progress"]:
            assert len(result["progress"]["user_interactions"]) == 0

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    def test_verifies_no_interaction_logged_when_no_responses(self, mock_call, mock_context):
        """Should not log interaction when no user responses."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {},  # No responses
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": []
            },
        }

        result = supervisor_node(state)
        
        # Should not log interaction when no responses (even if trigger exists)
        # The code checks: if ask_user_trigger and user_responses
        if "progress" in result and "user_interactions" in result["progress"]:
            assert len(result["progress"]["user_interactions"]) == 0

