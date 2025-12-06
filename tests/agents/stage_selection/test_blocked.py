"""Tests for blocked stage handling."""

from unittest.mock import patch, call

import pytest

from src.agents.stage_selection import select_stage_node
from tests.agents.stage_selection.utils import create_stage

class TestBlockedStageHandling:
    """Tests for blocked stage handling."""

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_unblocks_stage_when_deps_satisfied(self, mock_update):
        """Should unblock stage when dependencies are now satisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Verify unblocking call with exact parameters
        mock_update.assert_called_once_with(state, "stage1", "not_started", summary="Unblocked: Dependencies now satisfied")
        
        # Verify full return value structure
        assert result["current_stage_id"] == "stage1"
        assert result["workflow_phase"] == "stage_selection"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        assert "stage_start_time" in result
        assert isinstance(result["stage_start_time"], str)
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        # Should not trigger user interaction
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_unblocks_stage_with_partial_completion(self, mock_update):
        """Should unblock stage when dependency is completed_partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "partial"},
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called_once_with(state, "stage1", "not_started", summary="Unblocked: Dependencies now satisfied")
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_keeps_stage_blocked_when_deps_not_satisfied(self):
        """Should keep stage blocked when dependencies not satisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (the dependency) instead of stage1
        assert result["current_stage_id"] == "stage0"
        assert result["workflow_phase"] == "stage_selection"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        # stage1 should remain blocked (no unblocking call)

    def test_keeps_stage_blocked_when_dep_in_progress(self):
        """Should keep stage blocked when dependency is in_progress."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "in_progress"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should not select anything (stage0 is in_progress, stage1 is blocked)
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_keeps_stage_blocked_when_dep_invalidated(self):
        """Should keep stage blocked when dependency is invalidated.
        
        Note: Invalidated stages are SKIPPED (waiting for dependency to complete).
        They need to be marked as 'needs_rerun' to be re-executed.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "invalidated"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Invalidated stages are SKIPPED per the docstring:
        # "invalidated" - waiting for dependency to complete
        # Neither stage can be selected: stage0 is invalidated (skipped), stage1 is blocked
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_missing_deps(self, mock_update):
        """Should block stage with missing dependencies and detect deadlock.
        
        When a stage has missing dependencies (dependencies that don't exist in the
        stages list), it can never run. If this is the only stage, it IS a deadlock.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["nonexistent"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Verify blocking call with exact parameters
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[0] == state  # state passed as first arg
        assert args[1] == "stage1"
        assert args[2] == "blocked"
        assert "summary" in kwargs
        assert "Missing dependencies" in kwargs["summary"]
        assert "nonexistent" in kwargs["summary"]
        
        # Verify return value - should return None since no stage can be selected
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        # With only one stage that has missing dependencies, it's a deadlock
        # The stage can never run because its dependencies don't exist
        assert result.get("ask_user_trigger") == "deadlock_detected"
        assert result.get("ask_user_trigger") is not None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_multiple_missing_deps(self, mock_update):
        """Should block stage with multiple missing dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["missing1", "missing2"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[1] == "stage1"
        assert args[2] == "blocked"
        assert "missing1" in kwargs["summary"] or "missing2" in kwargs["summary"]
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_does_not_reblock_already_blocked_stage(self, mock_update):
        """Should not call update_progress_stage_status if stage is already blocked."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["nonexistent"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should not call update since stage is already blocked
        mock_update.assert_not_called()
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_failed_dep(self, mock_update):
        """Should skip stage if dependency failed (does not block, just skips)."""
        # If dependency failed, it's just "dependencies not satisfied".
        # The code checks `missing_deps`. If dep exists but is failed, it sets `deps_satisfied = False`.
        # It does NOT block it permanently unless `missing_deps` is true.
        # So it just skips it.
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should not block (dependency exists, just failed)
        mock_update.assert_not_called()
        # Should return None since no stage can be selected
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_missing_stage_type(self, mock_update):
        """Should block stage that has no stage_type field."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage1",
                        "status": "not_started",
                        "dependencies": [],
                        # Missing stage_type
                    },
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should block stage with missing stage_type
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[1] == "stage1"
        assert args[2] == "blocked"
        assert "Missing stage_type field" in kwargs["summary"] or "stage_type" in kwargs["summary"]
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_unknown_stage_type(self, mock_update):
        """Should block stage with unknown/unrecognized stage_type."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "UNKNOWN_TYPE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should block stage with unknown type
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[1] == "stage1"
        assert args[2] == "blocked"
        assert "Unknown stage type" in kwargs["summary"] or "UNKNOWN_TYPE" in kwargs["summary"]
        assert result["current_stage_id"] is None

    def test_blocks_stage_when_validation_hierarchy_not_passed(self):
        """Should skip stage when validation hierarchy requirement not met."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "not_done"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (material validation) first
        assert result["current_stage_id"] == "stage0"
        # stage1 should be skipped due to validation hierarchy

    def test_allows_stage_when_validation_hierarchy_partial(self):
        """Should allow stage when validation hierarchy is partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "partial"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 since material validation is partial (acceptable)
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_blocks_stage_when_lower_type_order_not_completed(self):
        """Should skip stage when lower-order stage type exists but not completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "ARRAY_SYSTEM", "not_started"),  # Higher order type
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (lower order) first, not stage1
        assert result["current_stage_id"] == "stage0"
        # stage1 should be skipped due to type order enforcement

    def test_allows_stage_when_lower_type_order_completed(self):
        """Should allow stage when lower-order stage type is completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 since material validation is completed
        assert result["current_stage_id"] == "stage1"

    def test_handles_multiple_blocked_stages(self):
        """Should handle multiple blocked stages correctly."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "blocked", ["stage1"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (the root dependency)
        assert result["current_stage_id"] == "stage0"
        # stage1 and stage2 should remain blocked

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_deadlock_detection_all_blocked(self, mock_update):
        """Should detect deadlock when all stages are permanently blocked."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["nonexistent"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "blocked", ["nonexistent2"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should detect deadlock and trigger user interaction
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result.get("ask_user_trigger") is not None
        assert len(result["pending_user_questions"]) > 0
        assert "Deadlock detected" in result["pending_user_questions"][0]
        assert "stage1" in result["pending_user_questions"][0] or "stage2" in result["pending_user_questions"][0]

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_deadlock_detection_mixed_blocked_and_failed(self, mock_update):
        """Should detect deadlock when stages are blocked or failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_failed"),
                    create_stage("stage2", "ARRAY_SYSTEM", "blocked", ["nonexistent"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should detect deadlock
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result.get("ask_user_trigger") is not None

    def test_no_deadlock_when_potentially_runnable_exists(self):
        """Should not detect deadlock when potentially runnable stages exist."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["nonexistent"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0, not detect deadlock
        assert result["current_stage_id"] == "stage0"
        assert result.get("ask_user_trigger") is None

    def test_handles_empty_dependencies_list(self):
        """Should handle stage with empty dependencies list correctly."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started", []),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (no dependencies means it's eligible)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_handles_none_dependencies(self):
        """Should handle stage with None dependencies gracefully."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        stage["dependencies"] = None
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Should handle None gracefully (treat as empty)
        assert result["current_stage_id"] == "stage0"

    def test_handles_missing_dependencies_field(self):
        """Should handle stage with missing dependencies field."""
        stage = {
            "stage_id": "stage0",
            "stage_type": "MATERIAL_VALIDATION",
            "status": "not_started",
            # Missing dependencies field
        }
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Should handle missing field gracefully
        assert result["current_stage_id"] == "stage0"

    def test_blocks_stage_when_prerequisite_type_missing(self):
        """Should block stages when prerequisite stage types don't exist in the plan.
        
        This tests the fix for a bug where stages were silently skipped (not blocked)
        when their prerequisite stage types didn't exist, leading to undetected deadlocks.
        
        Note: This test doesn't mock update_progress_stage_status so the actual
        status gets updated and cascading blocks work correctly.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "PARAMETER_SWEEP", "not_started"),
                    create_stage("stage1", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "parameter_sweeps": "not_done",
                "arrays_systems": "not_done",
                "single_structure": "not_done",  # Required for PARAMETER_SWEEP
            },
        }
        
        result = select_stage_node(state)
        
        # PARAMETER_SWEEP requires SINGLE_STRUCTURE but no SINGLE_STRUCTURE stage exists
        # COMPLEX_PHYSICS requires PARAMETER_SWEEP or ARRAY_SYSTEM but neither can complete
        # Both stages should be BLOCKED (not just skipped) because the prerequisite
        # stage types don't exist or are blocked - they can never run
        
        # Verify both stages were blocked in state
        assert state["progress"]["stages"][0]["status"] == "blocked"
        assert "SINGLE_STRUCTURE" in state["progress"]["stages"][0].get("summary", "")
        
        assert state["progress"]["stages"][1]["status"] == "blocked"
        summary1 = state["progress"]["stages"][1].get("summary", "")
        assert "PARAMETER_SWEEP" in summary1 or "ARRAY_SYSTEM" in summary1
        
        # Should detect deadlock since all stages are now blocked
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result.get("ask_user_trigger") is not None
        assert len(result["pending_user_questions"]) > 0
        assert "Deadlock" in result["pending_user_questions"][0]

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_skips_stage_when_prerequisite_type_exists_but_not_completed(self, mock_update):
        """Should skip (not block) stage when prerequisite type exists but isn't completed yet.
        
        This is the normal case: the prerequisite stage exists but hasn't run yet,
        so we skip the dependent stage temporarily (it can run later).
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    # Include MATERIAL_VALIDATION so SINGLE_STRUCTURE doesn't get blocked
                    create_stage("mat_val", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage0", "SINGLE_STRUCTURE", "not_started", ["mat_val"]),
                    create_stage("stage1", "PARAMETER_SWEEP", "not_started", ["stage0"]),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "parameter_sweeps": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        # PARAMETER_SWEEP requires SINGLE_STRUCTURE to be passed/partial
        # SINGLE_STRUCTURE exists but is not_started, so PARAMETER_SWEEP is skipped (not blocked)
        # Should NOT block PARAMETER_SWEEP because SINGLE_STRUCTURE can still complete
        mock_update.assert_not_called()
        
        # Should select MATERIAL_VALIDATION (the root prerequisite)
        assert result["current_stage_id"] == "mat_val"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_single_structure_when_material_validation_missing(self, mock_update):
        """Should block SINGLE_STRUCTURE when MATERIAL_VALIDATION doesn't exist.
        
        Note: Since we mock update_progress_stage_status, the actual status in state
        isn't updated, so deadlock detection won't trigger. This test verifies the
        blocking call is made correctly.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        # SINGLE_STRUCTURE requires MATERIAL_VALIDATION which doesn't exist
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[1] == "stage0"
        assert args[2] == "blocked"
        assert "MATERIAL_VALIDATION" in kwargs["summary"]
        
        assert result["current_stage_id"] is None
        # Note: Deadlock detection doesn't trigger because mock prevents actual status update
        # The stage status remains "not_started" in memory, so it's counted as potentially_runnable

    def test_blocks_single_structure_and_triggers_deadlock_without_mock(self):
        """Should block SINGLE_STRUCTURE and trigger deadlock when MATERIAL_VALIDATION doesn't exist.
        
        This test doesn't mock update_progress_stage_status so the actual status
        gets updated and deadlock detection triggers correctly.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        # SINGLE_STRUCTURE requires MATERIAL_VALIDATION which doesn't exist
        # Stage should be blocked and deadlock detected
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result.get("ask_user_trigger") is not None
        assert "Deadlock" in result["pending_user_questions"][0]
        
        # Verify stage was actually blocked in state
        assert state["progress"]["stages"][0]["status"] == "blocked"

    def test_complex_physics_allowed_when_prerequisite_met(self):
        """Should allow COMPLEX_PHYSICS when prerequisite is met."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "PARAMETER_SWEEP", "completed_success"),
                    create_stage("stage1", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "parameter_sweeps": "passed",
                "arrays_systems": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        # Should allow COMPLEX_PHYSICS since PARAMETER_SWEEP is passed
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "COMPLEX_PHYSICS"

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_verifies_exact_blocking_summary_format(self, mock_update):
        """Should verify exact format of blocking summary message."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["missing_dep"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        summary = kwargs["summary"]
        
        # Verify summary contains required information
        assert "Blocked:" in summary or "Missing dependencies" in summary
        assert "missing_dep" in summary
        assert result["current_stage_id"] is None

    def test_handles_stage_with_partial_dependency_satisfaction(self):
        """Should handle stage where some dependencies are satisfied, others not."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage2", "SINGLE_STRUCTURE", "not_started", ["stage0", "stage1"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 (not stage2, since stage1 dependency is not satisfied)
        assert result["current_stage_id"] == "stage1"
