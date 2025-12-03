"""Miscellaneous trigger handler tests."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestUnknownTrigger:
    """Tests for unknown trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unknown_trigger(self, mock_context):
        """Should handle unknown trigger gracefully."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "some_unknown_trigger",
            "user_responses": {"Question": "Response"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "unknown" in result["supervisor_feedback"].lower()
        assert "some_unknown_trigger" in result["supervisor_feedback"]
        assert result.get("ask_user_trigger") is None  # Should be cleared

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unknown_trigger_with_empty_user_responses(self, mock_context):
        """Should handle unknown trigger with empty user responses."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "unknown_trigger_123",
            "user_responses": {},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "unknown" in result["supervisor_feedback"].lower()
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unknown_trigger_with_none_user_responses(self, mock_context):
        """Should handle unknown trigger with None user_responses."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "unknown_trigger_xyz",
            "user_responses": None,
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "unknown" in result["supervisor_feedback"].lower()
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_unknown_trigger_logs_interaction(self, mock_context):
        """Should log user interaction even for unknown triggers."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "unknown_trigger",
            "user_responses": {"Question": "Response"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "unknown_trigger"
        assert interaction["id"] == "U1"


class TestArchiveErrorRecovery:
    """Tests for archive error recovery logic."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry failed archive operations."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None  # Successful retry
        
        state = {
            "archive_errors": [{"stage_id": "stage1", "error": "Failed"}],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["archive_errors"] == []
        mock_archive.assert_called_once_with(state, "stage1")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_keeps_failed_retries(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should keep archive errors that still fail on retry."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.side_effect = Exception("Still failing")
        
        state = {
            "archive_errors": [{"stage_id": "stage1", "error": "Failed"}],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert "Failed" in result["archive_errors"][0]["error"] or "Still failing" in str(result["archive_errors"][0])
        mock_archive.assert_called_once_with(state, "stage1")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_multiple_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry multiple failed archive operations."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None
        
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Failed1"},
                {"stage_id": "stage2", "error": "Failed2"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["archive_errors"] == []
        assert mock_archive.call_count == 2
        mock_archive.assert_any_call(state, "stage1")
        mock_archive.assert_any_call(state, "stage2")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_partial_archive_failures(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle partial archive failures correctly."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        def archive_side_effect(state, stage_id):
            if stage_id == "stage1":
                raise Exception("Still failing")
            return None
        
        mock_archive.side_effect = archive_side_effect
        
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Failed1"},
                {"stage_id": "stage2", "error": "Failed2"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert mock_archive.call_count == 2

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_invalid_archive_errors_type(self, mock_prompt, mock_context, mock_call):
        """Should handle non-list archive_errors gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": "not a list",  # Invalid type
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["archive_errors"] == []

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_archive_errors_without_stage_id(self, mock_prompt, mock_context, mock_call):
        """Should preserve archive errors without stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [
                {"error": "No stage_id"},
                {"stage_id": "stage1", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should keep the error without stage_id
        assert len(result["archive_errors"]) >= 1
        errors_without_stage = [e for e in result["archive_errors"] if "stage_id" not in e]
        assert len(errors_without_stage) == 1

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_empty_archive_errors(self, mock_prompt, mock_context, mock_call):
        """Should handle empty archive_errors list."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["archive_errors"] == []

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_none_archive_errors(self, mock_prompt, mock_context, mock_call):
        """Should handle None archive_errors."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["archive_errors"] == []

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_invalid_archive_error_entry(self, mock_prompt, mock_context, mock_call):
        """Should handle invalid archive error entries (non-dict)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [
                "invalid entry",
                {"stage_id": "stage1", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should preserve invalid entries
        assert len(result["archive_errors"]) >= 1


class TestUserInteractionLogging:
    """Tests for user interaction logging."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction(self, mock_context):
        """Should log user interaction to progress."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Material question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "material_checkpoint"
        assert interaction["id"] == "U1"
        assert "timestamp" in interaction
        assert isinstance(interaction["timestamp"], str)
        assert interaction["context"]["stage_id"] == "stage0"
        assert interaction["context"]["agent"] == "SupervisorAgent"
        assert interaction["user_response"] == "APPROVE"
        assert "question" in interaction
        assert interaction["question"] == "Material question"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_empty_responses(self, mock_context):
        """Should log user interaction even with empty responses."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {},
            "pending_user_questions": ["Question?"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["user_response"] == ""
        assert interaction["interaction_type"] == "material_checkpoint"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_none_stage_id(self, mock_context):
        """Should log user interaction with None stage_id."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "RETRY"},
            "pending_user_questions": ["Question?"],
            "current_stage_id": None,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["context"]["stage_id"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_increments_id(self, mock_context):
        """Should increment interaction ID correctly."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "RETRY"},
            "pending_user_questions": ["Question?"],
            "current_stage_id": "stage1",
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1", "interaction_type": "previous"},
                    {"id": "U2", "interaction_type": "previous"},
                ],
            },
        }
        
        result = supervisor_node(state)
        
        assert len(result["progress"]["user_interactions"]) == 3
        assert result["progress"]["user_interactions"][2]["id"] == "U3"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_without_pending_questions(self, mock_context):
        """Should handle missing pending_user_questions gracefully."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "RETRY"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_preserves_existing_progress(self, mock_context):
        """Should preserve existing progress fields when logging interaction."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Question": "RETRY"},
            "current_stage_id": "stage1",
            "progress": {
                "stages": [{"stage_id": "stage0", "status": "completed"}],
                "user_interactions": [],
                "other_field": "preserved",
            },
        }
        
        result = supervisor_node(state)
        
        assert result["progress"]["other_field"] == "preserved"
        assert len(result["progress"]["stages"]) == 1
        assert len(result["progress"]["user_interactions"]) == 1


class TestInvalidUserResponses:
    """Tests for handling invalid user_responses."""

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle non-dict user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": "invalid string",  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()
        # Verify warning message contains expected text
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "invalid type" in warning_call_args.lower() or "expected dict" in warning_call_args.lower()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_list_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle list user_responses gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": ["invalid", "list"],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_logger.warning.assert_called()

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_none_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle None user_responses gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # None should be handled without warning (it's a valid case)


class TestHandleLlmAndDispatcherHandlers:
    """Direct handler tests for LLM errors and dispatcher fallbacks."""

    def test_handle_llm_error_retry(self, mock_state, mock_result):
        """Should retry on RETRY response."""
        user_input = {"q1": "RETRY"}
        initial_verdict = mock_result.get("supervisor_verdict")
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result
        assert "retrying" in mock_result["supervisor_feedback"].lower() or "acknowledged" in mock_result["supervisor_feedback"].lower()
        assert mock_result.get("should_stop") is not True

    def test_handle_llm_error_retry_case_insensitive(self, mock_state, mock_result):
        """Should handle RETRY case-insensitively."""
        user_input = {"q1": "retry"}
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_llm_error_retry_partial_match(self, mock_state, mock_result):
        """Should handle partial RETRY matches."""
        user_input = {"q1": "Please RETRY this"}
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_handle_llm_error_skip(self, mock_update, mock_state, mock_result):
        """Should skip stage on SKIP response."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        mock_update.assert_called_once_with(
            mock_state, mock_result, "stage1", "blocked",
            summary="Skipped by user after LLM error"
        )
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("should_stop") is not True

    @patch("src.agents.supervision.trigger_handlers._update_progress_with_error_handling")
    def test_handle_llm_error_skip_none_stage_id(self, mock_update, mock_state, mock_result):
        """Should handle SKIP with None stage_id."""
        user_input = {"q1": "SKIP"}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, None)
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_not_called()

    def test_handle_llm_error_stop(self, mock_state, mock_result):
        """Should stop workflow on STOP response."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_llm_error_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "What?"}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        assert isinstance(mock_result["pending_user_questions"], list)
        question = mock_result["pending_user_questions"][0]
        assert "RETRY" in question or "SKIP" in question or "STOP" in question

    def test_handle_llm_error_empty_response(self, mock_state, mock_result):
        """Should ask for clarification on empty response."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_llm_error_whitespace_only(self, mock_state, mock_result):
        """Should ask for clarification on whitespace-only response."""
        user_input = {"q1": "   "}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"

    def test_handle_llm_error_state_not_mutated(self, mock_state, mock_result):
        """Should not mutate state dict."""
        original_state = mock_state.copy()
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_llm_error(mock_state, mock_result, user_input, "stage1")
        
        assert mock_state == original_state

    def test_handle_trigger_dispatcher(self, mock_state, mock_result):
        """Should dispatch to correct handler."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_trigger("code_review_limit", mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_trigger_dispatcher_material_checkpoint(self, mock_state, mock_result):
        """Should dispatch material_checkpoint correctly."""
        user_input = {"q1": "APPROVE"}
        mock_state["pending_validated_materials"] = [{"material_id": "gold"}]
        
        trigger_handlers.handle_trigger("material_checkpoint", mock_state, mock_result, user_input, "stage0")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "validated_materials" in mock_result

    def test_handle_trigger_unknown(self, mock_state, mock_result):
        """Should handle unknown trigger gracefully."""
        user_input = {"q1": "foo"}
        
        trigger_handlers.handle_trigger("unknown_trigger_xyz", mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result
        assert "unknown" in mock_result["supervisor_feedback"].lower()
        assert "unknown_trigger_xyz" in mock_result["supervisor_feedback"]

    def test_handle_trigger_unknown_empty_string(self, mock_state, mock_result):
        """Should handle empty string trigger."""
        user_input = {"q1": "foo"}
        
        trigger_handlers.handle_trigger("", mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "unknown" in mock_result["supervisor_feedback"].lower()

    def test_handle_trigger_unknown_none(self, mock_state, mock_result):
        """Should handle None trigger."""
        user_input = {"q1": "foo"}
        
        trigger_handlers.handle_trigger(None, mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "unknown" in mock_result["supervisor_feedback"].lower()

    def test_handle_trigger_backtrack_approval_with_fn(self, mock_state, mock_result):
        """Should handle backtrack_approval with dependent stages function."""
        user_input = {"q1": "APPROVE"}
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}
        
        def get_dependent_stages(plan, target):
            return ["stage1", "stage2"]
        
        trigger_handlers.handle_trigger(
            "backtrack_approval", mock_state, mock_result, user_input, "stage1",
            get_dependent_stages_fn=get_dependent_stages
        )
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in mock_result
        assert mock_result["backtrack_decision"]["stages_to_invalidate"] == ["stage1", "stage2"]

    def test_handle_trigger_backtrack_approval_without_fn(self, mock_state, mock_result):
        """Should handle backtrack_approval without dependent stages function."""
        user_input = {"q1": "APPROVE"}
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}
        
        trigger_handlers.handle_trigger(
            "backtrack_approval", mock_state, mock_result, user_input, "stage1",
            get_dependent_stages_fn=None
        )
        
        assert mock_result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in mock_result
        # stages_to_invalidate should be empty list when fn is None
        assert mock_result["backtrack_decision"].get("stages_to_invalidate") == []

    def test_handle_trigger_backtrack_approval_reject(self, mock_state, mock_result):
        """Should handle backtrack_approval rejection."""
        user_input = {"q1": "REJECT"}
        mock_state["backtrack_decision"] = {"target_stage_id": "stage0"}
        
        trigger_handlers.handle_trigger(
            "backtrack_approval", mock_state, mock_result, user_input, "stage1"
        )
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("backtrack_suggestion") is None

    def test_handle_trigger_all_handlers_registered(self, mock_state, mock_result):
        """Should have handlers for all known triggers."""
        known_triggers = [
            "material_checkpoint",
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "context_overflow",
            "replan_limit",
            "backtrack_approval",
            "deadlock_detected",
            "llm_error",
            "clarification",
            "missing_paper_text",
            "missing_stage_id",
            "progress_init_failed",
            "no_stages_available",
            "invalid_backtrack_target",
            "backtrack_target_not_found",
            "backtrack_limit",
            "invalid_backtrack_decision",
        ]
        
        for trigger in known_triggers:
            user_input = {"q1": "test"}
            mock_result.clear()
            
            trigger_handlers.handle_trigger(trigger, mock_state, mock_result, user_input, "stage1")
            
            assert "supervisor_verdict" in mock_result, f"Trigger {trigger} did not set supervisor_verdict"
            assert mock_result["supervisor_verdict"] in [
                "ok_continue", "ask_user", "all_complete", "replan_needed", "backtrack_to_stage"
            ], f"Trigger {trigger} set invalid verdict: {mock_result['supervisor_verdict']}"

    def test_handle_trigger_state_not_mutated(self, mock_state, mock_result):
        """Should not mutate state dict."""
        original_state = mock_state.copy()
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_trigger("code_review_limit", mock_state, mock_result, user_input, "stage1")
        
        assert mock_state == original_state

    def test_handle_trigger_result_mutated(self, mock_state, mock_result):
        """Should mutate result dict in place."""
        user_input = {"q1": "STOP"}
        initial_result_id = id(mock_result)
        
        trigger_handlers.handle_trigger("code_review_limit", mock_state, mock_result, user_input, "stage1")
        
        assert id(mock_result) == initial_result_id  # Same object
        assert "supervisor_verdict" in mock_result  # Modified


class TestHandleClarification:
    """Tests for clarification handler."""

    def test_handle_clarification_with_response(self, mock_state, mock_result):
        """Should process clarification response."""
        user_input = {"q1": "This is a clarification"}
        
        trigger_handlers.handle_clarification(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result
        assert "clarification" in mock_result["supervisor_feedback"].lower()
        assert "This is a clarification" in mock_result["supervisor_feedback"]

    def test_handle_clarification_empty_response(self, mock_state, mock_result):
        """Should handle empty clarification response."""
        user_input = {"q1": ""}
        
        trigger_handlers.handle_clarification(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result
        assert "no clarification" in mock_result["supervisor_feedback"].lower()

    def test_handle_clarification_multiple_responses(self, mock_state, mock_result):
        """Should use last response for clarification."""
        user_input = {"q1": "First", "q2": "Last clarification"}
        
        trigger_handlers.handle_clarification(mock_state, mock_result, user_input, "stage1")
        
        assert "Last clarification" in mock_result["supervisor_feedback"]
        assert "First" not in mock_result["supervisor_feedback"]


class TestHandleCriticalErrorRetry:
    """Tests for critical error retry handler."""

    def test_handle_critical_error_retry(self, mock_state, mock_result):
        """Should retry on RETRY response."""
        user_input = {"q1": "RETRY"}
        
        trigger_handlers.handle_critical_error_retry(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result
        assert "retrying" in mock_result["supervisor_feedback"].lower() or "acknowledged" in mock_result["supervisor_feedback"].lower()

    def test_handle_critical_error_stop(self, mock_state, mock_result):
        """Should stop on STOP response."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_critical_error_retry(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_critical_error_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "Maybe?"}
        
        trigger_handlers.handle_critical_error_retry(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "RETRY" in question or "STOP" in question


class TestHandlePlanningErrorRetry:
    """Tests for planning error retry handler."""

    def test_handle_planning_error_replan(self, mock_state, mock_result):
        """Should replan on REPLAN response."""
        user_input = {"q1": "REPLAN"}
        
        trigger_handlers.handle_planning_error_retry(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "planner_feedback" in mock_result
        assert "replan" in mock_result["planner_feedback"].lower()

    def test_handle_planning_error_stop(self, mock_state, mock_result):
        """Should stop on STOP response."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_planning_error_retry(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_planning_error_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "Hmm?"}
        
        trigger_handlers.handle_planning_error_retry(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "REPLAN" in question or "STOP" in question


class TestHandleDeadlockDetected:
    """Tests for deadlock detected handler."""

    def test_handle_deadlock_generate_report(self, mock_state, mock_result):
        """Should generate report on GENERATE_REPORT response."""
        user_input = {"q1": "GENERATE_REPORT"}
        
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_report_keyword(self, mock_state, mock_result):
        """Should handle REPORT keyword."""
        user_input = {"q1": "REPORT"}
        
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_replan(self, mock_state, mock_result):
        """Should replan on REPLAN response."""
        user_input = {"q1": "REPLAN"}
        
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "replan_needed"
        assert "planner_feedback" in mock_result
        assert "deadlock" in mock_result["planner_feedback"].lower()

    def test_handle_deadlock_stop(self, mock_state, mock_result):
        """Should stop on STOP response."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_deadlock_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "What?"}
        
        trigger_handlers.handle_deadlock_detected(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "GENERATE_REPORT" in question or "REPLAN" in question or "STOP" in question


class TestHandleBacktrackLimit:
    """Tests for backtrack limit handler."""

    def test_handle_backtrack_limit_force_continue(self, mock_state, mock_result):
        """Should continue on FORCE_CONTINUE response."""
        user_input = {"q1": "FORCE_CONTINUE"}
        
        trigger_handlers.handle_backtrack_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in mock_result
        assert "continuing" in mock_result["supervisor_feedback"].lower() or "force" in mock_result["supervisor_feedback"].lower()

    def test_handle_backtrack_limit_force_keyword(self, mock_state, mock_result):
        """Should handle FORCE keyword."""
        user_input = {"q1": "FORCE"}
        
        trigger_handlers.handle_backtrack_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_backtrack_limit_continue_keyword(self, mock_state, mock_result):
        """Should handle CONTINUE keyword."""
        user_input = {"q1": "CONTINUE"}
        
        trigger_handlers.handle_backtrack_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"

    def test_handle_backtrack_limit_stop(self, mock_state, mock_result):
        """Should stop on STOP response."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_backtrack_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_backtrack_limit_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "Maybe?"}
        
        trigger_handlers.handle_backtrack_limit(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "FORCE_CONTINUE" in question or "STOP" in question


class TestHandleInvalidBacktrackDecision:
    """Tests for invalid backtrack decision handler."""

    def test_handle_invalid_backtrack_continue(self, mock_state, mock_result):
        """Should continue on CONTINUE response."""
        user_input = {"q1": "CONTINUE"}
        mock_result["backtrack_decision"] = {"invalid": "decision"}
        
        trigger_handlers.handle_invalid_backtrack_decision(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result.get("backtrack_decision") is None

    def test_handle_invalid_backtrack_stop(self, mock_state, mock_result):
        """Should stop on STOP response."""
        user_input = {"q1": "STOP"}
        
        trigger_handlers.handle_invalid_backtrack_decision(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_invalid_backtrack_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown response."""
        user_input = {"q1": "Hmm?"}
        
        trigger_handlers.handle_invalid_backtrack_decision(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        question = mock_result["pending_user_questions"][0]
        assert "CONTINUE" in question or "STOP" in question
