"""Supervisor tests for context overflow and backtrack approval triggers."""

from unittest.mock import patch, MagicMock

from src.agents.supervision import supervisor_node


class TestContextOverflowTrigger:
    """Tests for context_overflow trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_continues_on_summarize(self, mock_context):
        """Should continue with summarization on SUMMARIZE."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SUMMARIZE"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "summariz" in result["supervisor_feedback"].lower()
        assert result["ask_user_trigger"] is None  # Should be cleared
        assert "workflow_phase" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_truncates_paper_on_truncate(self, mock_context):
        """Should truncate paper on TRUNCATE."""
        mock_context.return_value = None
        original_text = "A" * 30000
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": original_text,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in result["paper_text"]
        assert len(result["paper_text"]) < len(original_text)
        # Exact truncation: first 15000 + marker + last 5000
        marker = "\n\n... [TRUNCATED BY USER REQUEST] ...\n\n"
        expected_length = 15000 + len(marker) + 5000
        assert len(result["paper_text"]) == expected_length
        assert result["paper_text"].startswith("A" * 15000)
        assert result["paper_text"].endswith("A" * 5000)
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_truncates_paper_at_exact_boundary(self, mock_context):
        """Should truncate paper just over 20039 char boundary (15000 + marker + 5000)."""
        mock_context.return_value = None
        # Truncation threshold is 15000 + 39 (marker) + 5000 = 20039
        # Only truncate if original > truncated result
        original_text = "A" * 20040  # Just over truncated threshold
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": original_text,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in result["paper_text"]
        marker = "\n\n... [TRUNCATED BY USER REQUEST] ...\n\n"
        expected_length = 15000 + len(marker) + 5000
        assert len(result["paper_text"]) == expected_length

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_paper_exactly_at_threshold(self, mock_context):
        """Should NOT truncate paper at or below 20039 char threshold."""
        mock_context.return_value = None
        # Truncation threshold is 15000 + 39 (marker) + 5000 = 20039
        # Original must be LONGER than truncated result for truncation to occur
        original_text = "A" * 20039  # Exactly at truncated threshold
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": original_text,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        # Should NOT truncate since len(20039) is not > truncated_length(20039)
        # Truncating would not reduce the text size
        assert "short enough" in result["supervisor_feedback"].lower()
        assert result.get("paper_text") == original_text  # Unchanged

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_paper_just_below_threshold(self, mock_context):
        """Should handle paper just below 20000 char threshold."""
        mock_context.return_value = None
        original_text = "A" * 19999  # Just below threshold
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": original_text,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "short enough" in result["supervisor_feedback"].lower()
        assert result.get("paper_text") == original_text  # Should remain unchanged

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_short_paper_on_truncate(self, mock_context):
        """Should handle short paper gracefully on TRUNCATE."""
        mock_context.return_value = None
        original_text = "Short paper"
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": original_text,
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "short enough" in result["supervisor_feedback"].lower()
        assert result.get("paper_text") == original_text  # Should remain unchanged

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_missing_paper_text_on_truncate(self, mock_context):
        """Should handle missing paper_text gracefully on TRUNCATE."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert "short enough" in result["supervisor_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage on SKIP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        # Verify stage status update was called
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][1] == "stage1"  # stage_id
        assert call_args[0][2] == "blocked"  # status
        assert "context overflow" in call_args[1]["summary"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip_without_current_stage(self, mock_update, mock_context):
        """Should handle SKIP when current_stage_id is missing."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SKIP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None
        # Should not call update_progress_stage_status when no current_stage_id
        mock_update.assert_not_called()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "STOP"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "all_complete"
        assert result.get("should_stop") is True
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_asks_user_on_unclear_response(self, mock_context):
        """Should ask user again on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "maybe"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert any("SUMMARIZE" in q.upper() or "TRUNCATE" in q.upper() for q in result["pending_user_questions"])

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses dict."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction(self, mock_context):
        """Should log user interaction to progress."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SUMMARIZE"},
            "pending_user_questions": ["What should we do?"],
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "context_overflow"
        assert interaction["user_response"] == "SUMMARIZE"
        assert "timestamp" in interaction
        assert "id" in interaction


class TestBacktrackApprovalTrigger:
    """Tests for backtrack_approval trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_on_approve(self, mock_context):
        """Should approve backtrack on APPROVE."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE the backtrack"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "stage1"
        assert "stages_to_invalidate" in result["backtrack_decision"]
        assert "stage2" in result["backtrack_decision"]["stages_to_invalidate"]
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_complex_dependencies(self, mock_context):
        """Should handle complex transitive dependency chains."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage0"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                    {"stage_id": "stage3", "dependencies": ["stage2"]},
                    {"stage_id": "stage4", "dependencies": ["stage1"]},  # Also depends on stage1
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        invalidated = result["backtrack_decision"]["stages_to_invalidate"]
        # All stages except stage0 should be invalidated
        assert "stage1" in invalidated
        assert "stage2" in invalidated
        assert "stage3" in invalidated
        assert "stage4" in invalidated
        assert "stage0" not in invalidated
        assert len(invalidated) == 4

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_no_dependents(self, mock_context):
        """Should handle backtrack to stage with no dependents."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage2"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        invalidated = result["backtrack_decision"]["stages_to_invalidate"]
        assert isinstance(invalidated, list)
        assert len(invalidated) == 0  # No dependents

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_missing_backtrack_decision(self, mock_context):
        """Should handle APPROVE when backtrack_decision is missing."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        # Should still set verdict but may not have stages_to_invalidate
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # backtrack_decision may not be set if missing from state

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_missing_target_stage_id(self, mock_context):
        """Should handle APPROVE when target_stage_id is missing."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {},  # Missing target_stage_id
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # Should not crash, but stages_to_invalidate may not be set

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_approves_backtrack_with_missing_plan(self, mock_context):
        """Should handle APPROVE when plan is missing."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        # Should not crash even if plan is missing

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejects_backtrack_on_reject(self, mock_context):
        """Should reject backtrack on REJECT."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "REJECT"},
            "backtrack_suggestion": {"target": "stage1"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["backtrack_suggestion"] is None
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_rejects_backtrack_with_various_reject_keywords(self, mock_context):
        """Should reject on various rejection keywords."""
        mock_context.return_value = None
        rejection_responses = ["REJECT", "NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]
        
        for response in rejection_responses:
            state = {
                "ask_user_trigger": "backtrack_approval",
                "user_responses": {"Question": response},
                "backtrack_suggestion": {"target": "stage1"},
                "progress": {"stages": [], "user_interactions": []},
            }

            result = supervisor_node(state)

            assert result["supervisor_verdict"] == "ok_continue", f"Failed for response: {response}"
            assert result["backtrack_suggestion"] is None, f"Failed for response: {response}"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_defaults_to_continue_on_unclear(self, mock_context):
        """Should default to continue on unclear response."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "maybe"},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["ask_user_trigger"] is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_empty_user_responses(self, mock_context):
        """Should handle empty user_responses dict."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {},
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_on_approve(self, mock_context):
        """Should log user interaction when approving backtrack."""
        mock_context.return_value = None
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": {"target_stage_id": "stage1"},
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                ]
            },
            "pending_user_questions": ["Approve backtrack?"],
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["interaction_type"] == "backtrack_approval"
        assert interaction["user_response"] == "APPROVE"
        assert "timestamp" in interaction
        assert "id" in interaction

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_preserves_backtrack_decision_fields(self, mock_context):
        """Should preserve all fields in backtrack_decision when approving."""
        mock_context.return_value = None
        original_decision = {
            "target_stage_id": "stage1",
            "reason": "Test reason",
            "extra_field": "should be preserved",
        }
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Question": "APPROVE"},
            "backtrack_decision": original_decision,
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }

        result = supervisor_node(state)

        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert result["backtrack_decision"]["target_stage_id"] == "stage1"
        assert result["backtrack_decision"]["reason"] == "Test reason"
        assert result["backtrack_decision"]["extra_field"] == "should be preserved"
        assert "stages_to_invalidate" in result["backtrack_decision"]

