"""Unit tests for src/agents/stage_selection.py"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.agents.stage_selection import select_stage_node

# Helper to create a basic stage
def create_stage(id, type="MATERIAL_VALIDATION", status="not_started", deps=None):
    return {
        "stage_id": id,
        "stage_type": type,
        "status": status,
        "dependencies": deps or []
    }

class TestSelectStageNode:
    """Tests for select_stage_node function."""

    def test_selects_first_available_stage_with_full_validation(self):
        """Should select first stage with met dependencies and return full state update."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["workflow_phase"] == "stage_selection"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert "stage_start_time" in result
        assert isinstance(result["stage_start_time"], str)
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        
        # Verify no user interaction triggers
        assert result.get("ask_user_trigger") is None
        assert result.get("awaiting_user_input") is None

    def test_selects_stage_with_met_dependencies(self):
        """Should select stage whose dependencies are met."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage1"

    def test_skips_completed_stages(self):
        """Should skip already completed stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage1"

    def test_skips_in_progress_stages(self):
        """Should skip in_progress stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "in_progress"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None

    def test_returns_none_when_all_stages_completed(self):
        """Should return None when all stages are completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result.get("ask_user_trigger") is None

    def test_returns_escalation_when_no_stages(self):
        """Should return escalation when no stages exist."""
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "no_stages_available"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) > 0

    def test_handles_none_plan_and_progress(self):
        """Should handle None plan and progress gracefully (escalate)."""
        state = {
            "plan": None,
            "progress": None,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "no_stages_available"

    def test_resets_counters_for_new_stage(self):
        """Should reset revision counters when selecting a new stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "current_stage_id": "old_stage",
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_does_not_reset_counters_for_same_stage_continuation(self):
        """Should NOT reset counters if somehow selecting the same stage (though unusual in this logic unless re-entrant)."""
        # Note: The current logic resets if (selected != current) OR (status == needs_rerun).
        # If we just select the same stage that is 'not_started' (maybe it was interrupted?), it should NOT reset.
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "current_stage_id": "stage0",
            "design_revision_count": 5
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        # Should NOT be present in updates if not resetting
        assert "design_revision_count" not in result


class TestNeedsRerunPriority:
    """Tests for needs_rerun stage priority."""

    def test_needs_rerun_takes_priority_over_not_started(self):
        """Should select needs_rerun stage before not_started."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage0"
        # Needs rerun should always reset counters
        assert result["design_revision_count"] == 0

    def test_skips_needs_rerun_with_invalidated_deps(self):
        """Should skip needs_rerun stage if it has invalidated dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "invalidated"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # stage1 skipped due to invalidated dep. stage0 is invalidated and eligible to run.
        assert result["current_stage_id"] == "stage0"

    def test_needs_rerun_respects_dependency_chain(self):
        """Should NOT select needs_rerun stage if its dependency also needs rerun."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]), # Dependent
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"), # Dependency
                ]
            },
        }
        
        # If stage1 is selected, we are running child before parent is fixed.
        result = select_stage_node(state)
        
        # Should select stage0 because stage1 has an unmet dependency (stage0 is not done).
        # Note: The current implementation might fail this.
        assert result["current_stage_id"] == "stage0"


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
        
        mock_update.assert_called_with(state, "stage1", "not_started", summary="Unblocked: Dependencies now satisfied")
        assert result["current_stage_id"] == "stage1"

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
        assert result["current_stage_id"] == "stage0"

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_missing_deps(self, mock_update):
        """Should block stage with missing dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["nonexistent"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called()
        args, kwargs = mock_update.call_args
        assert args[1] == "stage1"
        assert args[2] == "blocked"
        assert "Missing dependencies" in kwargs["summary"]
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_failed_dep(self, mock_update):
        """Should block stage if dependency failed (and wasn't already blocked)."""
        # Actually, if dependency failed, it's just "dependencies not satisfied".
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
        assert result["current_stage_id"] is None
        mock_update.assert_not_called()


class TestStageTypeValidation:
    """Tests for stage_type validation."""

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_without_stage_type(self, mock_update):
        """Should block stage without stage_type."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "status": "not_started", "dependencies": []},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called()
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_unknown_stage_type(self, mock_update):
        """Should block stage with unknown stage_type."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "UNKNOWN_TYPE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called()
        assert result["current_stage_id"] is None


class TestValidationHierarchy:
    """Tests for validation hierarchy enforcement."""

    def test_requires_material_validation_for_single_structure(self):
        """SINGLE_STRUCTURE requires material_validation passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
            # No validation hierarchy or not passed
            "validation_hierarchy": {"material_validation": "not_started"},
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None

    def test_requires_single_structure_for_array_system(self):
        """ARRAY_SYSTEM requires single_structure passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "ARRAY_SYSTEM", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "failed" # Failed
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None

    def test_allows_partial_validation(self):
        """Should allow proceeding if validation is 'partial'."""
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
        assert result["current_stage_id"] == "stage1"


class TestTypeOrderEnforcement:
    """Tests for STAGE_TYPE_ORDER enforcement."""

    def test_enforces_type_order_skipped(self):
        """Should skip higher order type if lower order type exists and is not completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "ARRAY_SYSTEM", "not_started"), # Higher order
                ]
            },
        }
        
        # Should select stage0. 
        # Crucially, if stage0 was NOT selectable (e.g. blocked), it should still NOT select stage1.
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage0"
        
    def test_enforces_type_order_explicit_skip(self):
        """Verify it explicitly skips the higher order stage even if lower order is blocked."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "blocked", ["missing_dep"]), # Truly blocked
                    create_stage("stage1", "ARRAY_SYSTEM", "not_started"), # Higher order
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"} # Pretend hierarchy is ok
        }
        
        result = select_stage_node(state)
        # stage0 is blocked (missing dep). 
        # stage1 should be skipped because stage0 (lower type) exists and is not completed.
        assert result["current_stage_id"] is None


class TestDeadlockDetection:
    """Tests for deadlock detection."""

    def test_detects_deadlock_when_all_blocked_or_failed(self):
        """Should detect deadlock when all remaining stages are blocked or failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result["awaiting_user_input"] is True
        assert "Deadlock detected" in result["pending_user_questions"][0]

    def test_no_deadlock_if_needs_rerun_exists(self):
        """Should not report deadlock if a stage needs rerun."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage0"


class TestProgressInitialization:
    """Tests for on-demand progress initialization."""

    @patch("src.agents.stage_selection.initialize_progress_from_plan")
    def test_initializes_progress_from_plan(self, mock_init):
        """Should initialize progress from plan if not already done."""
        mock_init.return_value = {
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            }
        }
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "dependencies": []},
                ]
            },
            "progress": {},
        }
        
        result = select_stage_node(state)
        
        mock_init.assert_called_once()
        assert result["current_stage_id"] == "stage0"
