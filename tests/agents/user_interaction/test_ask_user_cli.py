"""Core functionality tests for ask_user_node.

These tests verify the basic behavior of ask_user_node using the LangGraph
interrupt() mechanism for human-in-the-loop workflows.

Note: Tests for signal handling, timeout, EOF, and non-interactive mode have been
removed as those features are now handled by runner.py, not ask_user_node.

Note: Keyword validation tests have been removed as validation is now handled
by supervisor's trigger handlers, not ask_user_node. The node only checks for
empty responses.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.agents.user_interaction import ask_user_node


class TestAskUserNode:
    """Tests for ask_user_node function."""

    def test_returns_not_awaiting_when_no_questions(self):
        """Should clear ask_user state when no questions pending."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "context_overflow",
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert result["workflow_phase"] == "awaiting_user"
        # Verify no other keys are set
        assert "user_responses" not in result
        assert "pending_user_questions" not in result

    def test_returns_not_awaiting_when_no_questions_missing_keys(self):
        """Should handle missing state keys gracefully."""
        state = {
            "pending_user_questions": [],
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert result["workflow_phase"] == "awaiting_user"

    def test_returns_not_awaiting_when_questions_is_none(self):
        """Should handle None pending_user_questions."""
        state = {
            "pending_user_questions": None,
            "ask_user_trigger": "context_overflow",
        }
        
        # None should be treated as falsy (no questions), returning early
        result = ask_user_node(state)
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.user_interaction.interrupt")
    def test_collects_user_response(self, mock_interrupt):
        """Should collect user response via interrupt."""
        mock_interrupt.return_value = "User response"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper",
        }
        
        result = ask_user_node(state)
        
        assert result.get("ask_user_trigger") is None
        assert "user_responses" in result
        assert result["user_responses"]["Question?"] == "User response"
        assert result["workflow_phase"] == "awaiting_user"
        assert result["pending_user_questions"] == []

    @patch("src.agents.user_interaction.interrupt")
    def test_clears_pending_questions_on_success(self, mock_interrupt):
        """Should clear pending_user_questions upon successful response collection."""
        mock_interrupt.return_value = "Response"
        
        state = {
            "pending_user_questions": ["Q1"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper",
        }
        
        result = ask_user_node(state)
        assert result["pending_user_questions"] == []
        assert result.get("ask_user_trigger") is None
        assert result["workflow_phase"] == "awaiting_user"

    @patch("src.agents.user_interaction.interrupt")
    def test_stores_response_for_first_question(self, mock_interrupt):
        """Should store response mapped to first question."""
        mock_interrupt.return_value = "Answer"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper",
        }
        
        result = ask_user_node(state)
        
        assert result["user_responses"]["Question?"] == "Answer"
        assert result.get("ask_user_trigger") is None


class TestMergeExistingResponses:
    """Tests for merging with existing user responses."""

    @patch("src.agents.user_interaction.interrupt")
    def test_merges_with_existing_responses(self, mock_interrupt):
        """Should merge new responses with existing ones."""
        mock_interrupt.return_value = "new answer"
        
        state = {
            "pending_user_questions": ["New question?"],
            "ask_user_trigger": "test",
            "user_responses": {"Previous question": "previous answer"},
            "paper_id": "test_paper",
        }
        
        result = ask_user_node(state)
        
        assert result["user_responses"]["Previous question"] == "previous answer"
        assert result["user_responses"]["New question?"] == "new answer"
        assert len(result["user_responses"]) == 2
        assert result.get("ask_user_trigger") is None
        assert result["pending_user_questions"] == []

    @patch("src.agents.user_interaction.interrupt")
    def test_merges_when_existing_responses_is_none(self, mock_interrupt):
        """Should handle None existing user_responses."""
        mock_interrupt.return_value = "new answer"
        
        state = {
            "pending_user_questions": ["New question?"],
            "ask_user_trigger": "test",
            "user_responses": None,
            "paper_id": "test_paper",
        }
        
        # None should be treated as empty dict
        result = ask_user_node(state)
        assert result["user_responses"]["New question?"] == "new answer"

    @patch("src.agents.user_interaction.interrupt")
    def test_merges_when_existing_responses_missing(self, mock_interrupt):
        """Should handle missing user_responses key."""
        mock_interrupt.return_value = "new answer"
        
        state = {
            "pending_user_questions": ["New question?"],
            "ask_user_trigger": "test",
            "paper_id": "test_paper",
        }
        
        result = ask_user_node(state)
        
        assert result["user_responses"]["New question?"] == "new answer"
        assert len(result["user_responses"]) == 1


class TestSafetyNet:
    """Tests for safety net behavior when trigger is missing."""

    @patch("src.agents.user_interaction.interrupt")
    def test_sets_unknown_escalation_when_trigger_missing(self, mock_interrupt):
        """Should set unknown_escalation trigger when ask_user_trigger is missing."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": ["Some question"],
            # No ask_user_trigger
        }
        
        result = ask_user_node(state)
        
        # Safety net should set "unknown_escalation" as trigger
        assert result["ask_user_trigger"] == "unknown_escalation"

    @patch("src.agents.user_interaction.interrupt")
    def test_regenerates_questions_with_workflow_recovery(self, mock_interrupt):
        """Bug #3: Should regenerate questions with WORKFLOW RECOVERY when trigger missing."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": ["Original question"],
            # No ask_user_trigger - safety net will trigger
        }
        
        ask_user_node(state)
        
        # Verify interrupt was called with regenerated questions
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["trigger"] == "unknown_escalation"
        # Questions should contain WORKFLOW RECOVERY
        assert "WORKFLOW RECOVERY" in payload["questions"][0]

    @patch("src.agents.user_interaction.interrupt")
    def test_preserves_original_context_in_regenerated_questions(self, mock_interrupt):
        """Should preserve original question context when regenerating."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": ["Important context about the error"],
            # No ask_user_trigger
        }
        
        ask_user_node(state)
        
        payload = mock_interrupt.call_args[0][0]
        # Original context should be included
        assert "Important context about the error" in payload["questions"][0]

    @patch("src.agents.user_interaction.interrupt")
    def test_strips_old_options_from_regenerated_questions(self, mock_interrupt):
        """Should strip old Options: section when regenerating questions."""
        mock_interrupt.return_value = "RETRY"
        
        state = {
            "pending_user_questions": ["Question text\n\nOptions:\n- OLD_OPTION_1\n- OLD_OPTION_2"],
            # No ask_user_trigger
        }
        
        ask_user_node(state)
        
        payload = mock_interrupt.call_args[0][0]
        questions = payload["questions"][0]
        # Old options should be stripped
        assert "OLD_OPTION_1" not in questions
        assert "OLD_OPTION_2" not in questions
        # But question text should remain
        assert "Question text" in questions


class TestInterruptIntegration:
    """Tests for interrupt() integration."""

    @patch("src.agents.user_interaction.interrupt")
    def test_interrupt_payload_structure(self, mock_interrupt):
        """Should call interrupt with correct payload structure."""
        mock_interrupt.return_value = "APPROVE"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "material_checkpoint",
            "paper_id": "paper123",
        }
        
        ask_user_node(state)
        
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        
        assert payload["trigger"] == "material_checkpoint"
        assert payload["questions"] == ["Question?"]
        assert payload["paper_id"] == "paper123"

    @patch("src.agents.user_interaction.interrupt")
    def test_uses_unknown_paper_id_when_missing(self, mock_interrupt):
        """Should use 'unknown' as paper_id when not in state."""
        mock_interrupt.return_value = "APPROVE"
        
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
            # No paper_id
        }
        
        ask_user_node(state)
        
        payload = mock_interrupt.call_args[0][0]
        assert payload["paper_id"] == "unknown"

    @patch("src.agents.user_interaction.interrupt")
    def test_interrupt_not_called_when_no_questions(self, mock_interrupt):
        """Should not call interrupt when there are no questions."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test",
        }
        
        ask_user_node(state)
        
        mock_interrupt.assert_not_called()
