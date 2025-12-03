"""Context overflow trigger tests."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.supervision import supervisor_node, trigger_handlers


class TestContextOverflowTrigger:
    """Tests for context_overflow trigger handling."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_continues_on_summarize(self, mock_context):
        """Should continue with summarization on SUMMARIZE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SUMMARIZE"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "summariz" in result["supervisor_feedback"].lower()
        assert result["supervisor_feedback"] == "Applying feedback summarization for context management."

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_truncates_paper_on_truncate(self, mock_context):
        """Should truncate paper on TRUNCATE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": "A" * 30000,  # Long paper
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in result["paper_text"]
        assert len(result["paper_text"]) < 30000
        assert result["paper_text"].startswith("A" * 15000)
        assert result["paper_text"].endswith("A" * 5000)
        assert result["supervisor_feedback"] == "Truncating paper to first 15k and last 5k chars."

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_handles_short_paper_on_truncate(self, mock_context):
        """Should handle short paper gracefully on TRUNCATE."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "TRUNCATE"},
            "paper_text": "Short paper",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "short enough" in result["supervisor_feedback"].lower()
        assert result["paper_text"] == "Short paper"
        assert result["supervisor_feedback"] == "Paper already short enough, proceeding."

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_skips_stage_on_skip(self, mock_update, mock_context):
        """Should skip stage on SKIP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "SKIP"},
            "current_stage_id": "stage1",
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        mock_update.assert_called_once_with(
            state, "stage1", "blocked", summary="Skipped due to context overflow"
        )

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stops_on_stop(self, mock_context):
        """Should stop on STOP."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result.get("should_stop") is True


class TestHandleContextOverflow:
    """Direct handler tests for context overflow mitigation."""

    def test_handle_context_overflow_summarize(self, mock_state, mock_result):
        """Should apply summarization."""
        user_input = {"q1": "SUMMARIZE"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["supervisor_feedback"] == "Applying feedback summarization for context management."

    def test_handle_context_overflow_summarize_case_insensitive(self, mock_state, mock_result):
        """Should handle SUMMARIZE case insensitively."""
        user_input = {"q1": "summarize"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "summarization" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_summarize_partial_match(self, mock_state, mock_result):
        """Should handle partial matches like 'SUMMARIZE NOW'."""
        user_input = {"q1": "SUMMARIZE NOW"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "summarization" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_summarize_multiple_responses(self, mock_state, mock_result):
        """Should use last response when multiple responses provided."""
        user_input = {"q1": "TRUNCATE", "q2": "SUMMARIZE"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "summarization" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_truncate_long(self, mock_state, mock_result):
        """Should truncate long paper text."""
        long_text = "a" * 25000
        mock_state["paper_text"] = long_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in mock_result["paper_text"]
        assert len(mock_result["paper_text"]) < 25000
        assert mock_result["paper_text"].startswith(long_text[:15000])
        assert mock_result["paper_text"].endswith(long_text[-5000:])
        assert mock_result["supervisor_feedback"] == "Truncating paper to first 15k and last 5k chars."
        # Verify exact truncation format
        expected_length = 15000 + len("\n\n... [TRUNCATED BY USER REQUEST] ...\n\n") + 5000
        assert len(mock_result["paper_text"]) == expected_length

    def test_handle_context_overflow_truncate_exact_boundary(self, mock_state, mock_result):
        """Should not truncate at exactly 20000 characters since truncation would make it longer.
        
        Truncated length = 15000 + 39 (marker) + 5000 = 20039 chars.
        So we only truncate if original > 20039 to ensure truncation actually reduces size.
        """
        exact_boundary_text = "a" * 20000
        mock_state["paper_text"] = exact_boundary_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        # Should NOT truncate since 20000 <= 20039 (truncated length)
        assert "[TRUNCATED" not in mock_result["paper_text"]
        assert mock_result["paper_text"] == exact_boundary_text
        assert "short enough" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_truncate_one_above_effective_boundary(self, mock_state, mock_result):
        """Should truncate when text is 20040 chars (one above effective boundary of 20039)."""
        above_boundary_text = "a" * 20040
        mock_state["paper_text"] = above_boundary_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in mock_result["paper_text"]
        # Truncation should reduce size: 20040 -> 20039
        assert len(mock_result["paper_text"]) < len(above_boundary_text)
        assert len(mock_result["paper_text"]) == 20039  # 15000 + 39 + 5000
        assert mock_result["paper_text"].startswith(above_boundary_text[:15000])
        assert mock_result["paper_text"].endswith(above_boundary_text[-5000:])

    def test_handle_context_overflow_truncate_one_below_boundary(self, mock_state, mock_result):
        """Should not truncate if text is 19999 characters (one below boundary)."""
        short_text = "a" * 19999
        mock_state["paper_text"] = short_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["paper_text"] == short_text
        assert "[TRUNCATED" not in mock_result["paper_text"]
        assert "short enough" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_truncate_short(self, mock_state, mock_result):
        """Should preserve paper_text even when text is short."""
        short_text = "a" * 100
        mock_state["paper_text"] = short_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["paper_text"] == short_text
        assert "[TRUNCATED" not in mock_result["paper_text"]
        assert mock_result["supervisor_feedback"] == "Paper already short enough, proceeding."

    def test_handle_context_overflow_truncate_empty_text(self, mock_state, mock_result):
        """Should handle empty paper_text gracefully."""
        mock_state["paper_text"] = ""
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["paper_text"] == ""
        assert "short enough" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_truncate_missing_paper_text(self, mock_state, mock_result):
        """Should handle missing paper_text key gracefully."""
        if "paper_text" in mock_state:
            del mock_state["paper_text"]
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert mock_result["paper_text"] == ""
        assert "short enough" in mock_result["supervisor_feedback"].lower()

    def test_handle_context_overflow_skip_with_stage_id(self, mock_state, mock_result):
        """Should update stage status when SKIP with current_stage_id."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_called_once_with(
                mock_state, "stage1", "blocked", summary="Skipped due to context overflow"
            )

    def test_handle_context_overflow_skip_without_stage_id(self, mock_state, mock_result):
        """Should not update stage status when SKIP without current_stage_id."""
        user_input = {"q1": "SKIP"}
        
        with patch("src.agents.supervision.trigger_handlers.update_progress_stage_status") as mock_update:
            trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, None)
            
            assert mock_result["supervisor_verdict"] == "ok_continue"
            mock_update.assert_not_called()

    def test_handle_context_overflow_stop(self, mock_state, mock_result):
        """Should stop workflow on STOP."""
        user_input = {"q1": "STOP"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_context_overflow_stop_case_insensitive(self, mock_state, mock_result):
        """Should handle STOP case insensitively."""
        user_input = {"q1": "stop"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "all_complete"
        assert mock_result["should_stop"] is True

    def test_handle_context_overflow_unknown(self, mock_state, mock_result):
        """Should ask for clarification on unknown input."""
        user_input = {"q1": "What is this?"}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1
        assert "SUMMARIZE" in mock_result["pending_user_questions"][0]
        assert "TRUNCATE" in mock_result["pending_user_questions"][0]
        assert "SKIP_STAGE" in mock_result["pending_user_questions"][0]
        assert "STOP" in mock_result["pending_user_questions"][0]

    def test_handle_context_overflow_empty_user_responses(self, mock_state, mock_result):
        """Should ask for clarification on empty user responses."""
        user_input = {}
        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")
        
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result

    def test_handle_context_overflow_none_user_responses(self, mock_state, mock_result):
        """Should handle None user_responses gracefully by asking for clarification."""
        trigger_handlers.handle_context_overflow(mock_state, mock_result, None, "stage1")
        
        # parse_user_response returns "" for None, which falls into "else" branch
        assert mock_result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in mock_result
        assert len(mock_result["pending_user_questions"]) == 1

    def test_handle_context_overflow_very_long_paper(self, mock_state, mock_result):
        """Should truncate very long paper correctly."""
        very_long_text = "x" * 100000
        mock_state["paper_text"] = very_long_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        assert mock_result["supervisor_verdict"] == "ok_continue"
        assert "[TRUNCATED BY USER REQUEST]" in mock_result["paper_text"]
        assert len(mock_result["paper_text"]) < 100000
        assert mock_result["paper_text"].startswith(very_long_text[:15000])
        assert mock_result["paper_text"].endswith(very_long_text[-5000:])
        # Verify exact truncation format
        expected_length = 15000 + len("\n\n... [TRUNCATED BY USER REQUEST] ...\n\n") + 5000
        assert len(mock_result["paper_text"]) == expected_length

    def test_handle_context_overflow_truncate_preserves_exact_lengths(self, mock_state, mock_result):
        """Should preserve exactly 15000 chars at start and 5000 at end."""
        long_text = "a" * 30000
        mock_state["paper_text"] = long_text
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(mock_state, mock_result, user_input, "stage1")

        truncated = mock_result["paper_text"]
        marker = "\n\n... [TRUNCATED BY USER REQUEST] ...\n\n"
        
        # Verify structure: first 15k + marker + last 5k
        assert truncated.startswith(long_text[:15000])
        assert truncated.endswith(long_text[-5000:])
        assert marker in truncated
        
        # Verify exact lengths
        parts = truncated.split(marker)
        assert len(parts) == 2
        assert len(parts[0]) == 15000
        assert len(parts[1]) == 5000

    def test_handle_context_overflow_does_not_modify_state(self, mock_state, mock_result):
        """Should not modify state dict, only result dict."""
        original_state = dict(mock_state)
        original_state["paper_text"] = "a" * 25000
        user_input = {"q1": "TRUNCATE"}

        trigger_handlers.handle_context_overflow(original_state, mock_result, user_input, "stage1")

        # State should be unchanged
        assert original_state["paper_text"] == "a" * 25000
        # Result should be modified
        assert "paper_text" in mock_result
        assert mock_result["paper_text"] != original_state["paper_text"]
