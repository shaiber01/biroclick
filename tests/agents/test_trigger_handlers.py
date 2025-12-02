"""Unit tests for src/agents/supervision/trigger_handlers.py"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.supervision.trigger_handlers import (
    handle_critical_error_retry,
    handle_planning_error_retry,
    handle_backtrack_limit,
    handle_invalid_backtrack_decision,
    handle_clarification,
    handle_material_checkpoint
)

class TestTriggerHandlers:
    """Tests for new trigger handlers."""

    def test_handle_critical_error_retry(self):
        """Should handle RETRY/STOP for critical errors."""
        state = {}
        result = {}
        
        # Test RETRY
        handle_critical_error_retry(state, result, {"q1": "Please RETRY the operation"}, "stage1")
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Retrying" in result["supervisor_feedback"]
        
        # Test STOP
        result = {}
        handle_critical_error_retry(state, result, {"q1": "I want to STOP now"}, "stage1")
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True
        
        # Test Unknown
        result = {}
        handle_critical_error_retry(state, result, {"q1": "What?"}, "stage1")
        assert result["supervisor_verdict"] == "ask_user"

    def test_handle_planning_error_retry(self):
        """Should handle REPLAN/STOP for planning errors."""
        state = {}
        result = {}
        
        # Test REPLAN
        handle_planning_error_retry(state, result, {"q1": "Please REPLAN the workflow"}, "stage1")
        assert result["supervisor_verdict"] == "replan_needed"
        assert "User requested replan" in result["planner_feedback"]
        
        # Test STOP
        result = {}
        handle_planning_error_retry(state, result, {"q1": "STOP execution"}, "stage1")
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    def test_handle_backtrack_limit(self):
        """Should handle FORCE_CONTINUE/STOP for backtrack limit."""
        state = {}
        result = {}
        
        # Test FORCE_CONTINUE
        handle_backtrack_limit(state, result, {"q1": "FORCE_CONTINUE please"}, "stage1")
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Continuing despite backtrack limit" in result["supervisor_feedback"]
        
        # Test STOP
        result = {}
        handle_backtrack_limit(state, result, {"q1": "STOP"}, "stage1")
        assert result["supervisor_verdict"] == "all_complete"

    def test_handle_invalid_backtrack_decision(self):
        """Should handle CONTINUE/STOP for invalid backtrack decision."""
        state = {}
        result = {"backtrack_decision": {"some": "data"}}
        
        # Test CONTINUE
        handle_invalid_backtrack_decision(state, result, {"q1": "CONTINUE regardless"}, "stage1")
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["backtrack_decision"] is None  # Should be cleared
        
        # Test STOP
        result = {"backtrack_decision": {"some": "data"}}
        handle_invalid_backtrack_decision(state, result, {"q1": "STOP"}, "stage1")
        assert result["supervisor_verdict"] == "all_complete"

    def test_handle_clarification(self):
        """Should append user clarification to feedback."""
        state = {}
        result = {}
        
        # With response
        handle_clarification(state, result, {"q1": "The material thickness is 50nm"}, "stage1")
        assert result["supervisor_verdict"] == "ok_continue"
        assert "The material thickness is 50nm" in result["supervisor_feedback"]
        
        # Empty response
        result = {}
        handle_clarification(state, result, {}, "stage1")
        assert result["supervisor_verdict"] == "ok_continue"
        assert "No clarification provided" in result["supervisor_feedback"]

    def test_handle_material_checkpoint_edge_cases(self):
        """Test material checkpoint with mixed signals."""
        state = {}
        result = {}
        
        # APPROVE but also REJECT keywords (Ambiguous -> REJECT priority in text usually?)
        # Logic says: if is_approval and not is_rejection
        
        # "I APPROVE the plan but REJECT the material" -> Should fall through to CHANGE_MATERIAL check or rejection
        user_input = {"q1": "I APPROVE the general idea but REJECT this material, please CHANGE_MATERIAL"}
        handle_material_checkpoint(state, result, user_input, "stage1")
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "User indicated wrong material" in result["planner_feedback"]
        
        # "APPROVE" only
        result = {}
        state = {"pending_validated_materials": ["mat1"]}
        handle_material_checkpoint(state, result, {"q1": "APPROVE"}, "stage1")
        assert result["supervisor_verdict"] == "ok_continue"

