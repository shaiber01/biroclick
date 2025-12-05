"""analysis_limit trigger tests."""

from unittest.mock import patch

import pytest

from src.agents.supervision import supervisor_node
from src.graph import route_after_supervisor


class TestAnalysisLimitTrigger:
    """Tests for analysis_limit trigger handling via supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_hint_resets_count_and_sets_retry_analyze_verdict(self, mock_context):
        """
        PROVIDE_HINT for analysis_limit should:
        1. Reset analysis_revision_count to 0
        2. Set analysis_feedback with the user's hint
        3. Set supervisor_verdict to 'retry_analyze' (not 'ok_continue')
        """
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "analysis_limit",
            "user_responses": {"Question": "PROVIDE_HINT: note that hwhm is not fwhm"},
            "analysis_revision_count": 5,
        }
        
        result = supervisor_node(state)
        
        # Verify counter reset
        assert result["analysis_revision_count"] == 0
        # Verify verdict is retry_analyze (NOT ok_continue)
        assert result["supervisor_verdict"] == "retry_analyze"
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None
        # Verify analysis feedback contains the hint
        assert "analysis_feedback" in result
        assert "hwhm is not fwhm" in result["analysis_feedback"]
        # Verify should_stop is NOT set
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_hint_lowercase(self, mock_context):
        """Should handle lowercase provide_hint."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "analysis_limit",
            "user_responses": {"Question": "provide_hint: try smaller wavelength range"},
            "analysis_revision_count": 3,
        }
        
        result = supervisor_node(state)
        
        assert result["analysis_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_analyze"
        assert "smaller wavelength range" in result["analysis_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_hint_keyword_only(self, mock_context):
        """Should handle HINT keyword (without PROVIDE_HINT prefix)."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "analysis_limit",
            "user_responses": {"Question": "HINT: check the units"},
            "analysis_revision_count": 2,
        }
        
        result = supervisor_node(state)
        
        assert result["analysis_revision_count"] == 0
        assert result["supervisor_verdict"] == "retry_analyze"
        assert "check the units" in result["analysis_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.trigger_handlers.update_progress_stage_status")
    def test_accept_partial_continues_with_ok_continue(self, mock_update, mock_context):
        """ACCEPT_PARTIAL should set ok_continue verdict."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "analysis_limit",
            "user_responses": {"Question": "ACCEPT_PARTIAL"},
            "current_stage_id": "stage0",
            "progress": {"stages": [{"stage_id": "stage0", "status": "in_progress"}]},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("should_stop") is not True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_stop_sets_all_complete(self, mock_context):
        """STOP should set all_complete verdict."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "analysis_limit",
            "user_responses": {"Question": "STOP"},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True


class TestRetryAnalyzeRouting:
    """Tests for routing when retry_analyze verdict is set."""

    def test_retry_analyze_routes_to_analyze(self):
        """
        route_after_supervisor should route to 'analyze' when verdict is 'retry_analyze'.
        
        This is the critical test that verifies the fix for the bug where
        PROVIDE_HINT for analysis_limit didn't actually retry analysis.
        """
        state = {
            "supervisor_verdict": "retry_analyze",
            "current_stage_type": "MATERIAL_VALIDATION",
        }
        
        result = route_after_supervisor(state)
        
        assert result == "analyze"

    def test_retry_analyze_routes_to_analyze_non_material_validation(self):
        """retry_analyze should route to analyze for any stage type."""
        state = {
            "supervisor_verdict": "retry_analyze",
            "current_stage_type": "SINGLE_STRUCTURE",
        }
        
        result = route_after_supervisor(state)
        
        assert result == "analyze"

    def test_ok_continue_still_routes_to_material_checkpoint_for_stage0(self):
        """
        ok_continue should still route to material_checkpoint for MATERIAL_VALIDATION
        stages when validated_materials is empty (this is the expected behavior
        for ACCEPT_PARTIAL, not PROVIDE_HINT).
        """
        state = {
            "supervisor_verdict": "ok_continue",
            "current_stage_type": "MATERIAL_VALIDATION",
            "validated_materials": [],
        }
        
        result = route_after_supervisor(state)
        
        assert result == "material_checkpoint"

    def test_ok_continue_routes_to_select_stage_when_materials_validated(self):
        """ok_continue routes to select_stage when materials are validated."""
        state = {
            "supervisor_verdict": "ok_continue",
            "current_stage_type": "MATERIAL_VALIDATION",
            "validated_materials": [{"name": "Al", "source": "test"}],
        }
        
        result = route_after_supervisor(state)
        
        assert result == "select_stage"


class TestAnalysisLimitEndToEnd:
    """End-to-end tests for analysis_limit with PROVIDE_HINT."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_provide_hint_full_flow(self, mock_context):
        """
        Full flow test: user provides hint -> supervisor sets retry_analyze -> routes to analyze.
        
        This test verifies the complete fix for the bug where PROVIDE_HINT
        would route to material_checkpoint instead of retrying analysis.
        """
        mock_context.return_value = None
        
        # Step 1: Supervisor processes the user response
        state = {
            "ask_user_trigger": "analysis_limit",
            "user_responses": {"Q": "PROVIDE_HINT: HWHM is half of FWHM"},
            "analysis_revision_count": 2,
            "current_stage_type": "MATERIAL_VALIDATION",
            "validated_materials": [],  # Not yet validated
        }
        
        supervisor_result = supervisor_node(state)
        
        # Verify supervisor sets the right verdict
        assert supervisor_result["supervisor_verdict"] == "retry_analyze"
        assert supervisor_result["analysis_revision_count"] == 0
        assert "HWHM" in supervisor_result["analysis_feedback"]
        
        # Step 2: Apply supervisor result to state and route
        updated_state = {**state, **supervisor_result}
        route_result = route_after_supervisor(updated_state)
        
        # Verify routing goes to analyze (not material_checkpoint!)
        assert route_result == "analyze", (
            f"Expected 'analyze' but got '{route_result}'. "
            "PROVIDE_HINT should route to analyze, not material_checkpoint."
        )
