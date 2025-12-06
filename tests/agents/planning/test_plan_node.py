"""Tests for plan_node."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from src.agents.planning import plan_node
from schemas.state import MAX_REPLANS, ReproState


@pytest.fixture(name="mock_state")
def plan_state_alias(plan_state):
    return plan_state


@pytest.fixture(name="mock_llm_output")
def planner_output_alias(planner_llm_output):
    return planner_llm_output


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
        
        # Verify plan structure completeness
        assert "title" in result["plan"]
        assert "summary" in result["plan"]
        assert "targets" in result["plan"]
        assert "extracted_parameters" in result["plan"]

    def test_plan_node_missing_text(self, mock_state):
        """Test error when paper text is empty."""
        mock_state["paper_text"] = ""
        result = plan_node(mock_state)
        
        assert result["workflow_phase"] == "planning"
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None
        assert len(result["pending_user_questions"]) == 1
        assert "missing or too short" in result["pending_user_questions"][0]
        assert "0 characters" in result["pending_user_questions"][0]

    def test_plan_node_none_text(self, mock_state):
        """Test error when paper text is None."""
        mock_state["paper_text"] = None
        result = plan_node(mock_state)
        
        assert result["workflow_phase"] == "planning"
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None
        assert "0 characters" in result["pending_user_questions"][0]

    def test_plan_node_short_text(self, mock_state):
        """Test error when paper text is too short (< 100 chars)."""
        mock_state["paper_text"] = "Too short"
        result = plan_node(mock_state)
        
        assert result["workflow_phase"] == "planning"
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None
        assert "too short" in result["pending_user_questions"][0]

    def test_plan_node_whitespace_only_text(self, mock_state):
        """Test error when paper text is only whitespace."""
        mock_state["paper_text"] = "   \n\t   "
        result = plan_node(mock_state)
        
        assert result["workflow_phase"] == "planning"
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None

    def test_plan_node_exactly_100_chars_boundary(self, mock_state, mock_llm_output):
        """Test boundary condition: exactly 100 characters should pass."""
        mock_state["paper_text"] = "x" * 100
        mock_check = patch("src.agents.planning.check_context_or_escalate", return_value=None)
        mock_llm = patch("src.agents.planning.call_agent_with_metrics", return_value=mock_llm_output)
        mock_init = patch("src.agents.planning.initialize_progress_from_plan", return_value={**mock_state, "progress": []})
        mock_sync = patch("src.agents.planning.sync_extracted_parameters", return_value={**mock_state, "progress": []})
        
        with mock_check, mock_llm, mock_init, mock_sync:
            result = plan_node(mock_state)
            # Should not trigger missing_paper_text error
            assert result.get("ask_user_trigger") != "missing_paper_text"

    def test_plan_node_99_chars_fails(self, mock_state):
        """Test boundary condition: 99 characters should fail."""
        mock_state["paper_text"] = "x" * 99
        result = plan_node(mock_state)
        
        assert result["ask_user_trigger"] == "missing_paper_text"
        assert result.get("ask_user_trigger") is not None

    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_context_escalation(self, mock_check, mock_state):
        """Test immediate return on context check escalation."""
        escalation = {"ask_user_trigger": "context_overflow", "reason": "context"}
        mock_check.return_value = escalation
        
        result = plan_node(mock_state)
        assert result == escalation
        assert result.get("ask_user_trigger") is not None

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_context_non_blocking_update(self, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test context check returns non-blocking state updates."""
        # Context check returns metrics update, not blocking escalation
        mock_check.return_value = {"metrics": {"tokens": 100}}
        mock_llm.return_value = mock_llm_output
        mock_init = patch("src.agents.planning.initialize_progress_from_plan", return_value={**mock_state, "progress": []})
        mock_sync = patch("src.agents.planning.sync_extracted_parameters", return_value={**mock_state, "progress": []})
        
        with mock_init, mock_sync:
            result = plan_node(mock_state)
            # Should continue processing, not return early
            assert result["workflow_phase"] == "planning"
            assert "plan" in result

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_llm_failure(self, mock_check, mock_llm, mock_state):
        """Test handling of LLM failure."""
        mock_check.return_value = None
        mock_llm.side_effect = Exception("LLM Error")
        
        result = plan_node(mock_state)
        
        assert result["workflow_phase"] == "planning"
        assert result["ask_user_trigger"] == "llm_error"
        assert result.get("ask_user_trigger") is not None
        assert len(result["pending_user_questions"]) == 1
        assert "LLM Error" in result["pending_user_questions"][0]

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_llm_failure_different_exception_types(self, mock_check, mock_llm, mock_state):
        """Test handling of different LLM exception types."""
        mock_check.return_value = None
        
        for exc_type, exc_msg in [
            (ValueError, "Invalid response format"),
            (KeyError, "Missing required key"),
            (RuntimeError, "API timeout"),
            (ConnectionError, "Network error"),
        ]:
            mock_llm.side_effect = exc_type(exc_msg)
            result = plan_node(mock_state)
            
            assert result["workflow_phase"] == "planning"
            assert result["ask_user_trigger"] == "llm_error"
            assert result.get("ask_user_trigger") is not None
            assert exc_msg in result["pending_user_questions"][0]

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
        # Verify plan is still in result despite initialization failure
        assert "plan" in result
        assert result["plan"]["stages"] == mock_llm_output["stages"]

    @patch("src.agents.planning.sync_extracted_parameters")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_sync_extracted_parameters_failure(self, mock_check, mock_llm, mock_init, mock_sync, mock_llm_output, mock_state):
        """Test handling of sync_extracted_parameters failure."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_init.return_value = {**mock_state, "progress": ["stage1"]}
        mock_sync.side_effect = Exception("Sync failed")
        
        result = plan_node(mock_state)
        
        # sync_extracted_parameters failure is caught by the same try-except as initialize_progress_from_plan
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Progress initialization failed" in result["planner_feedback"]
        assert "Sync failed" in result["planner_feedback"]
        assert result["replan_count"] == 1  # Incremented from 0
        # Verify plan is still in result despite sync failure
        assert "plan" in result
        assert result["plan"]["stages"] == mock_llm_output["stages"]

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
        assert result["last_plan_review_verdict"] == "needs_revision"

    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_progress_init_failure_replan_count_below_max(self, mock_check, mock_llm, mock_init, mock_llm_output, mock_state):
        """Test replan count increments when below max."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_init.side_effect = Exception("Bad Structure")
        
        mock_state["replan_count"] = 1
        mock_state["runtime_config"] = {"max_replans": 5}
        
        result = plan_node(mock_state)
        
        assert result["replan_count"] == 2  # Should increment

    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_progress_init_failure_replan_count_at_max(self, mock_check, mock_llm, mock_init, mock_llm_output, mock_state):
        """Test replan count does not increment when at max."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_init.side_effect = Exception("Bad Structure")
        
        mock_state["replan_count"] = 5
        mock_state["runtime_config"] = {"max_replans": 5}
        
        result = plan_node(mock_state)
        
        assert result["replan_count"] == 5  # Should not increment

    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_progress_init_failure_replan_count_uses_default_max(self, mock_check, mock_llm, mock_init, mock_llm_output, mock_state):
        """Test replan count uses MAX_REPLANS default when runtime_config missing."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_init.side_effect = Exception("Bad Structure")
        
        mock_state["replan_count"] = MAX_REPLANS - 1
        # No runtime_config
        
        result = plan_node(mock_state)
        
        assert result["replan_count"] == MAX_REPLANS  # Should increment to MAX_REPLANS

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_replan_prompt_injection(self, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that replan count triggers prompt modification."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_state["replan_count"] = 2
        mock_init = patch("src.agents.planning.initialize_progress_from_plan", return_value={**mock_state, "progress": []})
        mock_sync = patch("src.agents.planning.sync_extracted_parameters", return_value={**mock_state, "progress": []})
        
        with mock_init, mock_sync:
            plan_node(mock_state)
            
            call_kwargs = mock_llm.call_args[1]
            assert "Replan Attempt #2" in call_kwargs["system_prompt"]

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_no_replan_prompt_injection_when_zero(self, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that replan count of 0 does not trigger prompt modification."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        mock_state["replan_count"] = 0
        mock_init = patch("src.agents.planning.initialize_progress_from_plan", return_value={**mock_state, "progress": []})
        mock_sync = patch("src.agents.planning.sync_extracted_parameters", return_value={**mock_state, "progress": []})
        
        with mock_init, mock_sync:
            plan_node(mock_state)
            
            call_kwargs = mock_llm.call_args[1]
            assert "Replan Attempt" not in call_kwargs["system_prompt"]

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

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_paper_id_unknown_in_state(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that paper_id 'unknown' in state uses LLM output."""
        mock_check.return_value = None
        mock_output = mock_llm_output.copy()
        mock_output["paper_id"] = "llm_provided_id"
        mock_llm.return_value = mock_output
        
        mock_init.return_value = mock_state
        mock_sync.return_value = mock_state
        
        mock_state["paper_id"] = "unknown"
        result = plan_node(mock_state)
        
        assert result["plan"]["paper_id"] == "llm_provided_id"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_paper_id_none_in_state(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that paper_id None in state uses LLM output."""
        mock_check.return_value = None
        mock_output = mock_llm_output.copy()
        mock_output["paper_id"] = "llm_provided_id"
        mock_llm.return_value = mock_output
        
        mock_init.return_value = mock_state
        mock_sync.return_value = mock_state
        
        mock_state["paper_id"] = None
        result = plan_node(mock_state)
        
        assert result["plan"]["paper_id"] == "llm_provided_id"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_paper_id_missing_from_state_and_llm(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that missing paper_id defaults to 'unknown'."""
        mock_check.return_value = None
        mock_output = mock_llm_output.copy()
        del mock_output["paper_id"]
        mock_llm.return_value = mock_output
        
        mock_init.return_value = mock_state
        mock_sync.return_value = mock_state
        
        if "paper_id" in mock_state:
            del mock_state["paper_id"]
        result = plan_node(mock_state)
        
        assert result["plan"]["paper_id"] == "unknown"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_llm_output_missing_fields(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test handling of LLM output with missing optional fields."""
        mock_check.return_value = None
        mock_output = {
            "stages": [{"stage_id": "stage1"}],
            # Missing: paper_domain, title, summary, targets, extracted_parameters, planned_materials, assumptions
        }
        mock_llm.return_value = mock_output
        
        mock_init.return_value = {**mock_state, "progress": []}
        mock_sync.return_value = {**mock_state, "progress": []}
        
        result = plan_node(mock_state)
        
        # Should use defaults for missing fields
        assert result["plan"]["paper_domain"] == "other"
        assert result["plan"]["title"] == ""
        assert result["plan"]["summary"] == ""
        assert result["plan"]["targets"] == []
        assert result["plan"]["extracted_parameters"] == []
        assert result["planned_materials"] == []
        assert result["assumptions"] == {}
        assert result["paper_domain"] == "other"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_llm_output_empty_stages(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test handling of LLM output with empty stages list."""
        mock_check.return_value = None
        mock_output = mock_llm_output.copy()
        mock_output["stages"] = []
        mock_llm.return_value = mock_output
        
        mock_init.return_value = {**mock_state, "progress": []}
        mock_sync.return_value = {**mock_state, "progress": []}
        
        result = plan_node(mock_state)
        
        # Should not call initialize_progress_from_plan when stages is empty
        assert result["plan"]["stages"] == []
        # Progress initialization should be skipped
        mock_init.assert_not_called()

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    def test_plan_node_llm_output_no_plan(self, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test handling when LLM returns None or invalid plan."""
        mock_check.return_value = None
        mock_llm.return_value = None
        
        # This should cause an AttributeError when accessing .get() on None
        # Test documents current behavior - might reveal a bug
        with pytest.raises((AttributeError, TypeError)):
            plan_node(mock_state)

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_progress_reset_on_replan(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that progress is reset to None on replan."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        # State has existing progress
        mock_state["progress"] = ["old_stage1", "old_stage2"]
        mock_state["replan_count"] = 1
        
        updated_state = {**mock_state, "progress": None}
        mock_init.return_value = {**updated_state, "progress": ["new_stage1"]}
        mock_sync.return_value = {**updated_state, "progress": ["new_stage1"]}
        
        result = plan_node(mock_state)
        
        # Verify initialize_progress_from_plan was called with progress=None
        call_args = mock_init.call_args[0][0]
        assert call_args.get("progress") is None

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_progress_preserved_when_no_replan(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that progress reset only happens when replan_count > 0."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        # State has existing progress but no replan
        mock_state["progress"] = ["existing_stage"]
        mock_state["replan_count"] = 0
        
        mock_init.return_value = {**mock_state, "progress": ["existing_stage", "new_stage"]}
        mock_sync.return_value = {**mock_state, "progress": ["existing_stage", "new_stage"]}
        
        result = plan_node(mock_state)
        
        # Verify initialize_progress_from_plan was called
        mock_init.assert_called_once()

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_verify_llm_call_arguments(self, mock_sync, mock_init, mock_check, mock_llm, mock_llm_output, mock_state):
        """Test that LLM is called with correct arguments."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        mock_init.return_value = {**mock_state, "progress": []}
        mock_sync.return_value = {**mock_state, "progress": []}
        
        plan_node(mock_state)
        
        # Verify LLM was called with correct agent_name
        assert mock_llm.called
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["agent_name"] == "planner"
        assert "system_prompt" in call_kwargs
        assert "user_content" in call_kwargs
        assert "state" in call_kwargs

    @patch("src.agents.planning.log_agent_call")
    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.check_context_or_escalate")
    @patch("src.agents.planning.initialize_progress_from_plan")
    @patch("src.agents.planning.sync_extracted_parameters")
    def test_plan_node_metrics_logging(self, mock_sync, mock_init, mock_check, mock_llm, mock_log, mock_llm_output, mock_state):
        """Test that metrics are logged correctly."""
        mock_check.return_value = None
        mock_llm.return_value = mock_llm_output
        
        mock_init.return_value = {**mock_state, "progress": []}
        mock_sync.return_value = {**mock_state, "progress": []}
        
        result = plan_node(mock_state)
        
        # Verify metrics logging was called
        assert mock_log.called
        # log_agent_call returns a function that's called with (state, result)
        log_func = mock_log.return_value
        assert callable(log_func)


# ═══════════════════════════════════════════════════════════════════════
# plan_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════
