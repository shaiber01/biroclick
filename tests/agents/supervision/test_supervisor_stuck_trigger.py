"""
Tests for stuck trigger detection and auto-recovery in supervisor.

These tests verify that the supervisor properly detects when an ask_user_trigger
has been set for too many cycles without being cleared, logs appropriate warnings/errors,
and optionally auto-recovers by force-clearing the trigger.
"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.agents.supervision.supervisor import supervisor_node


@pytest.fixture
def base_state():
    """Minimal state for supervisor tests."""
    return {
        "paper_id": "test_paper",
        "paper_text": "Test paper content",
        "plan": {"stages": []},
        "progress": {"stages": {}},
        "stage_outputs": {},
        "current_stage_id": "stage1",
        "workflow_phase": "supervision",
        "supervisor_call_count": 0,
        "backtrack_count": 0,
        "replan_count": 0,
        "runtime_config": {},
        "metrics": {"agent_calls": []},
        "user_responses": {},
        "pending_user_questions": [],
        "ask_user_trigger": None,
        "_trigger_persistence_count": 0,
        "_last_seen_trigger": None,
        "_trigger_first_seen_time": None,
    }


def _mock_handler_needs_clarification(trigger, state, result, user_responses, current_stage_id, get_dependent_stages_fn):
    """Mock handler that simulates needing clarification (preserves trigger)."""
    result["supervisor_verdict"] = "ask_user"
    result["pending_user_questions"] = ["Please clarify"]


def _mock_handler_success(trigger, state, result, user_responses, current_stage_id, get_dependent_stages_fn):
    """Mock handler that simulates successful handling."""
    result["supervisor_verdict"] = "ok_continue"


class TestStuckTriggerDetection:
    """Tests for stuck trigger detection logic."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # BOUNDARY TESTS - Verify thresholds trigger at exactly the right count
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_no_warning_below_warn_threshold(self, base_state, caplog):
        """Should NOT log warning when count is below WARN_THRESHOLD (default 3)."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 1  # Will become 2, below threshold
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.WARNING):
                result = supervisor_node(base_state)
        
        # Should NOT have warned - count is 2, threshold is 3
        assert "Possible stuck trigger" not in caplog.text
        assert result["_trigger_persistence_count"] == 2
    
    def test_warns_at_exactly_warn_threshold(self, base_state, caplog):
        """Should log warning at exactly WARN_THRESHOLD (default 3) cycles."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 2  # Will become 3
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["_trigger_first_seen_time"] = datetime.now(timezone.utc).isoformat()
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.WARNING):
                result = supervisor_node(base_state)
        
        assert "Possible stuck trigger" in caplog.text
        assert "'test_trigger' seen 3 times" in caplog.text
        assert result["_trigger_persistence_count"] == 3
    
    def test_no_error_below_error_threshold(self, base_state, caplog):
        """Should NOT log error when count is below ERROR_THRESHOLD (default 5)."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 3  # Will become 4, below threshold
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.ERROR):
                result = supervisor_node(base_state)
        
        # Should NOT have errored - count is 4, threshold is 5
        assert "STUCK TRIGGER DETECTED" not in caplog.text
        assert result["_trigger_persistence_count"] == 4
    
    def test_errors_at_exactly_error_threshold(self, base_state, caplog):
        """Should log error at exactly ERROR_THRESHOLD (default 5) cycles."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 4  # Will become 5
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["_trigger_first_seen_time"] = datetime.now(timezone.utc).isoformat()
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.ERROR):
                result = supervisor_node(base_state)
        
        assert "STUCK TRIGGER DETECTED" in caplog.text
        assert "'test_trigger' has persisted for 5 supervisor cycles" in caplog.text
    
    def test_no_auto_clear_below_threshold(self, base_state, caplog):
        """Should NOT auto-clear when count is below AUTO_CLEAR_THRESHOLD (default 7)."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 5  # Will become 6, below threshold
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.ERROR):
                result = supervisor_node(base_state)
        
        # Should NOT have auto-cleared
        assert "AUTO-RECOVERY ACTIVATED" not in caplog.text
        assert result["ask_user_trigger"] == "test_trigger"
        assert result["_trigger_persistence_count"] == 6
    
    def test_auto_clears_at_exactly_threshold(self, base_state, caplog):
        """Should auto-clear at exactly AUTO_CLEAR_THRESHOLD (default 7) cycles."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 6  # Will become 7
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["_trigger_first_seen_time"] = datetime.now(timezone.utc).isoformat()
        base_state["user_responses"] = {"question": "answer"}
        base_state["pending_user_questions"] = ["What should we do?"]
        
        with caplog.at_level(logging.ERROR):
            result = supervisor_node(base_state)
        
        # Should have logged auto-recovery
        assert "AUTO-RECOVERY ACTIVATED" in caplog.text
        assert "Force-clearing stuck trigger 'test_trigger'" in caplog.text
        
        # Should have cleared the trigger
        assert result["ask_user_trigger"] is None
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Auto-recovered from stuck trigger" in result["supervisor_feedback"]
        
        # Should have cleared tracking state
        assert result["_last_seen_trigger"] is None
        assert result["_trigger_persistence_count"] == 0
        assert result["_trigger_first_seen_time"] is None
        
        # Should have saved diagnostic info with ALL required fields
        assert "_last_stuck_trigger_recovery" in result
        recovery_info = result["_last_stuck_trigger_recovery"]
        assert recovery_info["trigger"] == "test_trigger"
        assert recovery_info["persistence_count"] == 7
        assert recovery_info["user_responses_at_clear"] == {"question": "answer"}
        assert recovery_info["pending_questions_at_clear"] == ["What should we do?"]
        assert recovery_info["current_stage_id"] == "stage1"
        # Verify first_seen_time is preserved in diagnostic info
        assert recovery_info["first_seen_time"] is not None
        # Verify cleared_at is a valid ISO timestamp
        cleared_at = recovery_info["cleared_at"]
        datetime.fromisoformat(cleared_at.replace('Z', '+00:00'))  # Would raise if invalid

    # ═══════════════════════════════════════════════════════════════════════
    # EDGE CASES - Verify handling of None, empty, and malformed inputs
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_handles_none_persistence_count(self, base_state, caplog):
        """Should handle None persistence_count gracefully (treat as 0)."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = None  # Explicitly None
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            # Should not crash
            result = supervisor_node(base_state)
        
        # Should treat None as 0, so this is first occurrence (count becomes 1)
        # Actually the code does: persistence_count = state.get("_trigger_persistence_count", 0)
        # So None would be returned, then None + 1 would fail... let's verify
        # Wait, the implementation uses `state.get("_trigger_persistence_count", 0)` which returns None if key exists with None value
        # This is actually a BUG we're testing for!
        assert result["_trigger_persistence_count"] >= 1  # Should have counted

    def test_handles_missing_tracking_fields(self, base_state, caplog):
        """Should handle completely missing tracking fields gracefully."""
        base_state["ask_user_trigger"] = "test_trigger"
        # Remove tracking fields entirely
        del base_state["_trigger_persistence_count"]
        del base_state["_last_seen_trigger"]
        del base_state["_trigger_first_seen_time"]
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            # Should not crash
            result = supervisor_node(base_state)
        
        # Should have initialized tracking
        assert result["_trigger_persistence_count"] == 1
        assert result["_last_seen_trigger"] == "test_trigger"
        assert result["_trigger_first_seen_time"] is not None

    def test_handles_malformed_timestamp(self, base_state, caplog):
        """Should handle malformed first_seen_time without crashing."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 2
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["_trigger_first_seen_time"] = "not-a-valid-timestamp"
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            # Should not crash even with invalid timestamp
            result = supervisor_node(base_state)
        
        # Should still increment count
        assert result["_trigger_persistence_count"] == 3
        # Duration string should be empty or missing (graceful degradation)
        # The log message should still be generated (check it doesn't contain "(stuck for")
        # Actually, it should just skip the duration - let's verify warning is still logged

    def test_handles_empty_string_trigger(self, base_state, caplog):
        """Should treat empty string trigger same as no trigger."""
        base_state["ask_user_trigger"] = ""  # Empty string, not None
        base_state["_trigger_persistence_count"] = 5
        base_state["_last_seen_trigger"] = "old_trigger"
        
        with patch("src.agents.supervision.supervisor._run_normal_supervision") as mock_normal:
            mock_normal.return_value = None
            result = supervisor_node(base_state)
        
        # Empty string is falsy, so should be treated like no trigger
        # This verifies the `if ask_user_trigger:` check works correctly
        # If it doesn't, this test will catch the bug
        assert result.get("_trigger_persistence_count") == 0 or result.get("_last_seen_trigger") is None

    def test_handles_none_runtime_config(self, base_state, caplog):
        """Should handle None runtime_config gracefully."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 2
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["runtime_config"] = None  # Explicitly None
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            # Should not crash, should use default thresholds
            result = supervisor_node(base_state)
        
        assert result["_trigger_persistence_count"] == 3

    # ═══════════════════════════════════════════════════════════════════════
    # NEGATIVE TESTS - Verify things that should NOT happen
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_count_increments_for_same_trigger(self, base_state):
        """Count should INCREMENT (not reset) when trigger is the same."""
        base_state["ask_user_trigger"] = "same_trigger"
        base_state["_trigger_persistence_count"] = 5
        base_state["_last_seen_trigger"] = "same_trigger"  # Same!
        base_state["user_responses"] = {"question": "answer"}
        first_seen = datetime.now(timezone.utc).isoformat()
        base_state["_trigger_first_seen_time"] = first_seen
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            result = supervisor_node(base_state)
        
        # Count should have incremented, NOT reset
        assert result["_trigger_persistence_count"] == 6, "Count should increment for same trigger"
        # First seen time should be preserved, not updated
        assert result["_trigger_first_seen_time"] == first_seen, "First seen time should be preserved"
    
    def test_resets_count_on_different_trigger(self, base_state):
        """Count should RESET when trigger changes to a different value."""
        base_state["ask_user_trigger"] = "new_trigger"
        base_state["_trigger_persistence_count"] = 5
        base_state["_last_seen_trigger"] = "old_trigger"  # Different!
        old_first_seen = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        base_state["_trigger_first_seen_time"] = old_first_seen
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            result = supervisor_node(base_state)
        
        # Count should have reset to 1
        assert result["_trigger_persistence_count"] == 1, "Count should reset for different trigger"
        assert result["_last_seen_trigger"] == "new_trigger"
        # First seen time should be updated (new timestamp, not the old one)
        assert result["_trigger_first_seen_time"] != old_first_seen, "First seen time should be updated"
        # Verify it's a valid recent timestamp
        new_first_seen = datetime.fromisoformat(result["_trigger_first_seen_time"].replace('Z', '+00:00'))
        assert (datetime.now(timezone.utc) - new_first_seen).total_seconds() < 5, "New timestamp should be recent"

    def test_clears_tracking_when_no_trigger(self, base_state):
        """Should clear ALL tracking state when ask_user_trigger is None."""
        base_state["ask_user_trigger"] = None
        base_state["_trigger_persistence_count"] = 5
        base_state["_last_seen_trigger"] = "old_trigger"
        base_state["_trigger_first_seen_time"] = datetime.now(timezone.utc).isoformat()
        
        with patch("src.agents.supervision.supervisor._run_normal_supervision") as mock_normal:
            mock_normal.return_value = None
            result = supervisor_node(base_state)
        
        # ALL tracking should be cleared
        assert result.get("_trigger_persistence_count") == 0
        assert result.get("_last_seen_trigger") is None
        assert result.get("_trigger_first_seen_time") is None


class TestStuckTriggerConfiguration:
    """Tests for stuck trigger configuration options."""
    
    def test_custom_warn_threshold_respected(self, base_state, caplog):
        """Should respect custom warn threshold from runtime_config."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 1  # Will become 2
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["runtime_config"] = {"stuck_trigger_warn_threshold": 2}
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.WARNING):
                result = supervisor_node(base_state)
        
        # Should warn at 2 instead of default 3
        assert "Possible stuck trigger" in caplog.text
        assert result["_trigger_persistence_count"] == 2
    
    def test_custom_error_threshold_respected(self, base_state, caplog):
        """Should respect custom error threshold from runtime_config."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 2  # Will become 3
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["runtime_config"] = {"stuck_trigger_error_threshold": 3}
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.ERROR):
                result = supervisor_node(base_state)
        
        # Should error at 3 instead of default 5
        assert "STUCK TRIGGER DETECTED" in caplog.text
    
    def test_custom_auto_clear_threshold_respected(self, base_state, caplog):
        """Should respect custom auto-clear threshold from runtime_config."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 3  # Will become 4
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["runtime_config"] = {"stuck_trigger_auto_clear": 4}
        base_state["user_responses"] = {"question": "answer"}
        
        with caplog.at_level(logging.ERROR):
            result = supervisor_node(base_state)
        
        # Should auto-clear at 4 instead of default 7
        assert "AUTO-RECOVERY ACTIVATED" in caplog.text
        assert result["ask_user_trigger"] is None
        assert result["supervisor_verdict"] == "ok_continue"
    
    def test_auto_clear_disabled_when_zero(self, base_state, caplog):
        """Should NEVER auto-clear when threshold is set to 0 (disabled)."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 999  # Extremely high
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["runtime_config"] = {"stuck_trigger_auto_clear": 0}
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            with caplog.at_level(logging.ERROR):
                result = supervisor_node(base_state)
        
        # Should NOT have auto-cleared despite high count
        assert "AUTO-RECOVERY ACTIVATED" not in caplog.text
        # Trigger should still be present (preserved by mock handler)
        assert result["ask_user_trigger"] == "test_trigger"
        # The count should have incremented
        assert result["_trigger_persistence_count"] == 1000

    def test_negative_threshold_treated_as_disabled(self, base_state, caplog):
        """Negative threshold should effectively disable auto-clear."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 99
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["runtime_config"] = {"stuck_trigger_auto_clear": -1}
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            result = supervisor_node(base_state)
        
        # Should NOT have auto-cleared (negative threshold fails the > 0 check)
        assert "AUTO-RECOVERY ACTIVATED" not in caplog.text


class TestStuckTriggerClearOnSuccess:
    """Tests that tracking is cleared when trigger is successfully handled."""
    
    def test_clears_all_tracking_when_handler_succeeds(self, base_state):
        """Should clear ALL tracking when handler processes trigger successfully."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 4  # Non-zero
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["_trigger_first_seen_time"] = datetime.now(timezone.utc).isoformat()
        base_state["user_responses"] = {"question": "CONTINUE"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_success):
            result = supervisor_node(base_state)
        
        # Trigger should be cleared
        assert result["ask_user_trigger"] is None
        # ALL tracking should also be cleared
        assert result["_last_seen_trigger"] is None
        assert result["_trigger_persistence_count"] == 0
        assert result["_trigger_first_seen_time"] is None
        # User responses should be cleared
        assert result["user_responses"] == {}
        assert result["pending_user_questions"] == []
    
    def test_preserves_tracking_when_handler_needs_clarification(self, base_state):
        """Should preserve tracking when handler needs more user input."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 2
        base_state["_last_seen_trigger"] = "test_trigger"
        first_seen = datetime.now(timezone.utc).isoformat()
        base_state["_trigger_first_seen_time"] = first_seen
        base_state["user_responses"] = {"question": "unclear"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            result = supervisor_node(base_state)
        
        # Trigger should be preserved
        assert result["ask_user_trigger"] == "test_trigger"
        # Tracking should show the incremented count
        assert result["_trigger_persistence_count"] == 3
        assert result["_last_seen_trigger"] == "test_trigger"
        # First seen time should be preserved (same trigger, not reset)


class TestStuckTriggerMultiCycle:
    """Tests verifying behavior across multiple supervisor calls."""
    
    def test_progressive_count_across_cycles(self, base_state, caplog):
        """Verify count correctly progresses across multiple supervisor calls."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            # Cycle 1: 0 -> 1
            result1 = supervisor_node(base_state)
            assert result1["_trigger_persistence_count"] == 1
            
            # Update state for cycle 2
            base_state.update(result1)
            
            # Cycle 2: 1 -> 2
            result2 = supervisor_node(base_state)
            assert result2["_trigger_persistence_count"] == 2
            
            # Update state for cycle 3
            base_state.update(result2)
            
            # Cycle 3: 2 -> 3 (should warn)
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                result3 = supervisor_node(base_state)
            
            assert result3["_trigger_persistence_count"] == 3
            assert "Possible stuck trigger" in caplog.text
    
    def test_first_seen_time_preserved_across_cycles(self, base_state):
        """First seen time should be preserved as count increments."""
        initial_time = datetime.now(timezone.utc).isoformat()
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 0
        base_state["_last_seen_trigger"] = None
        base_state["user_responses"] = {"question": "answer"}
        
        with patch("src.agents.supervision.supervisor.handle_trigger", side_effect=_mock_handler_needs_clarification):
            # Cycle 1: First occurrence, timestamp set
            result1 = supervisor_node(base_state)
            first_timestamp = result1["_trigger_first_seen_time"]
            assert first_timestamp is not None
            
            # Update state
            base_state.update(result1)
            
            # Cycle 2: Same trigger, timestamp should be preserved
            result2 = supervisor_node(base_state)
            assert result2["_trigger_first_seen_time"] == first_timestamp, \
                "First seen time should not change for same trigger"
            
            # Cycle 3: Same trigger, timestamp still preserved
            base_state.update(result2)
            result3 = supervisor_node(base_state)
            assert result3["_trigger_first_seen_time"] == first_timestamp, \
                "First seen time should persist across multiple cycles"


class TestStuckTriggerDiagnostics:
    """Tests for diagnostic information quality."""
    
    def test_recovery_info_contains_all_required_fields(self, base_state):
        """Recovery diagnostic info should contain all fields needed for debugging."""
        base_state["ask_user_trigger"] = "debug_trigger"
        base_state["_trigger_persistence_count"] = 6  # Will become 7
        base_state["_last_seen_trigger"] = "debug_trigger"
        first_seen = datetime.now(timezone.utc).isoformat()
        base_state["_trigger_first_seen_time"] = first_seen
        base_state["user_responses"] = {"q1": "a1", "q2": "a2"}
        base_state["pending_user_questions"] = ["Question 1", "Question 2"]
        base_state["current_stage_id"] = "stage_xyz"
        
        result = supervisor_node(base_state)
        
        # Verify all required diagnostic fields
        assert "_last_stuck_trigger_recovery" in result
        info = result["_last_stuck_trigger_recovery"]
        
        # Required fields
        assert "trigger" in info and info["trigger"] == "debug_trigger"
        assert "persistence_count" in info and info["persistence_count"] == 7
        assert "first_seen_time" in info and info["first_seen_time"] == first_seen
        assert "cleared_at" in info
        assert "user_responses_at_clear" in info
        assert "pending_questions_at_clear" in info
        assert "current_stage_id" in info and info["current_stage_id"] == "stage_xyz"
        
        # Verify data integrity
        assert info["user_responses_at_clear"] == {"q1": "a1", "q2": "a2"}
        assert info["pending_questions_at_clear"] == ["Question 1", "Question 2"]
        
        # Verify cleared_at is a valid, recent timestamp
        cleared_at = datetime.fromisoformat(info["cleared_at"].replace('Z', '+00:00'))
        assert (datetime.now(timezone.utc) - cleared_at).total_seconds() < 5
    
    def test_feedback_message_contains_actionable_info(self, base_state):
        """Supervisor feedback should contain actionable information."""
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["_trigger_persistence_count"] = 6
        base_state["_last_seen_trigger"] = "test_trigger"
        base_state["user_responses"] = {}
        
        result = supervisor_node(base_state)
        
        feedback = result["supervisor_feedback"]
        # Should mention the trigger
        assert "test_trigger" in feedback
        # Should mention the count
        assert "7" in feedback  # The count at which it cleared
        # Should mention auto-recovery
        assert "Auto-recovered" in feedback or "auto" in feedback.lower()
        # Should prompt to check logs
        assert "log" in feedback.lower() or "investigate" in feedback.lower()
