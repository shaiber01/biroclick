"""Unit tests for src/agents/stage_selection.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.stage_selection import select_stage_node


class TestSelectStageNode:
    """Tests for select_stage_node function."""

    def test_selects_first_available_stage(self):
        """Should select first stage with met dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"

    def test_selects_stage_with_met_dependencies(self):
        """Should select stage whose dependencies are met."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage0"]},
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
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage0"]},
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should skip completed stage0 and select stage1
        assert result["current_stage_id"] == "stage1"

    def test_skips_in_progress_stages(self):
        """Should skip in_progress stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "in_progress", "dependencies": []},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # No stages available (stage0 is in_progress)
        assert result["current_stage_id"] is None

    def test_returns_none_when_all_stages_completed(self):
        """Should return None when all stages are completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "completed_success", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None

    def test_returns_escalation_when_no_stages(self):
        """Should return escalation when no stages exist."""
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        
        # Returns escalation for user
        assert result["current_stage_id"] is None
        assert result["awaiting_user_input"] is True

    def test_handles_empty_plan(self):
        """Should handle empty plan with escalation."""
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None

    def test_handles_missing_plan(self):
        """Should handle missing plan/progress with escalation."""
        state = {
            "plan": {},  # Empty dict, not None
            "progress": {},  # Empty dict, not None
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["awaiting_user_input"] is True

    def test_resets_counters_for_new_stage(self):
        """Should reset revision counters when selecting a new stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                ]
            },
            "current_stage_id": None,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0

    def test_selects_needs_rerun_stage(self):
        """Should prioritize stages that need re-run."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "needs_rerun", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"


class TestNeedsRerunPriority:
    """Tests for needs_rerun stage priority."""

    def test_needs_rerun_takes_priority_over_not_started(self):
        """Should select needs_rerun stage before not_started."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": []},
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "needs_rerun", "dependencies": []},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # needs_rerun has priority regardless of order
        assert result["current_stage_id"] == "stage0"

    def test_skips_needs_rerun_with_invalidated_deps(self):
        """Should skip needs_rerun stage if it has invalidated dependencies.
        
        The needs_rerun loop (priority 1) skips stage1 because stage0 is invalidated.
        Then the not_started loop (priority 2) processes stage0 since "invalidated"
        stages are eligible for selection (they're waiting to be re-run).
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "invalidated", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "needs_rerun", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # stage1 needs_rerun but has invalidated dependency, so skip it in priority 1.
        # stage0 is invalidated but has no dependencies, so it's selected in priority 2.
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
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "blocked", "dependencies": ["stage0"]},
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Stage1 should be unblocked and selected
        mock_update.assert_called()
        assert result["current_stage_id"] == "stage1"

    def test_keeps_stage_blocked_when_deps_not_satisfied(self):
        """Should keep stage blocked when dependencies not satisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "blocked", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (not_started), stage1 stays blocked
        assert result["current_stage_id"] == "stage0"


class TestMissingDependencies:
    """Tests for missing dependencies handling."""

    @patch("src.agents.stage_selection.update_progress_stage_status")
    @patch("src.agents.stage_selection.get_progress_stage")
    def test_blocks_stage_with_missing_deps(self, mock_get_progress, mock_update):
        """Should block stage with missing dependencies."""
        mock_get_progress.return_value = {"status": "not_started"}
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["nonexistent_stage"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should block the stage due to missing dependency
        mock_update.assert_called()
        assert result["current_stage_id"] is None


class TestStageTypeValidation:
    """Tests for stage_type validation."""

    @patch("src.agents.stage_selection.update_progress_stage_status")
    @patch("src.agents.stage_selection.get_progress_stage")
    def test_blocks_stage_without_stage_type(self, mock_get_progress, mock_update):
        """Should block stage without stage_type."""
        mock_get_progress.return_value = {"status": "not_started"}
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "status": "not_started", "dependencies": []},  # Missing stage_type
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should block due to missing stage_type
        mock_update.assert_called()
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    @patch("src.agents.stage_selection.get_progress_stage")
    def test_blocks_stage_with_unknown_stage_type(self, mock_get_progress, mock_update):
        """Should block stage with unknown stage_type."""
        mock_get_progress.return_value = {"status": "not_started"}
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "UNKNOWN_TYPE", "status": "not_started", "dependencies": []},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should block due to unknown stage_type
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
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": []},
                ]
            },
            "validation_hierarchy": {"material_validation": "not_started"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 first (material_validation)
        assert result["current_stage_id"] == "stage0"

    def test_allows_single_structure_when_material_passed(self):
        """SINGLE_STRUCTURE allowed when material_validation passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage0"]},
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 (single_structure)
        assert result["current_stage_id"] == "stage1"


class TestTypeOrderEnforcement:
    """Tests for STAGE_TYPE_ORDER enforcement."""

    def test_enforces_type_order(self):
        """Should enforce stage type order."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "ARRAY_SYSTEM", "status": "not_started", "dependencies": []},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select MATERIAL_VALIDATION first per type order
        assert result["current_stage_id"] == "stage0"

    def test_allows_higher_type_when_lower_completed(self):
        """Should allow higher type when lower type is completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "completed_success", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "stage_type": "ARRAY_SYSTEM", "status": "not_started", "dependencies": ["stage1"]},
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        # Should allow ARRAY_SYSTEM since lower types are completed
        assert result["current_stage_id"] == "stage2"


class TestDeadlockDetection:
    """Tests for deadlock detection."""

    def test_detects_deadlock_when_all_blocked(self):
        """Should detect deadlock when all remaining stages are blocked with unsatisfied deps."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_failed", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "blocked", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "stage_type": "ARRAY_SYSTEM", "status": "blocked", "dependencies": ["stage1"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # All blocked with unsatisfied deps (stage0 failed, not success/partial)
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result["awaiting_user_input"] is True

    def test_detects_deadlock_with_failed_stages(self):
        """Should detect deadlock when stages are failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_failed", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "blocked", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["ask_user_trigger"] == "deadlock_detected"

    def test_no_deadlock_with_potentially_runnable(self):
        """Should not report deadlock if there are potentially runnable stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "blocked", "dependencies": ["stage0"]},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0, not report deadlock
        assert result["current_stage_id"] == "stage0"
        assert result.get("ask_user_trigger") is None


class TestProgressInitialization:
    """Tests for on-demand progress initialization."""

    @patch("src.agents.stage_selection.initialize_progress_from_plan")
    def test_initializes_progress_from_plan(self, mock_init):
        """Should initialize progress from plan if not already done."""
        mock_init.return_value = {
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                ]
            }
        }
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "dependencies": []},
                ]
            },
            "progress": {},  # Empty progress
        }
        
        result = select_stage_node(state)
        
        mock_init.assert_called_once()
        assert result["current_stage_id"] == "stage0"

    @patch("src.agents.stage_selection.initialize_progress_from_plan")
    def test_handles_init_failure(self, mock_init):
        """Should handle progress initialization failure."""
        mock_init.side_effect = Exception("Init failed")
        
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION"},
                ]
            },
            "progress": {},  # Empty progress
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "progress_init_failed"
        assert result["awaiting_user_input"] is True


class TestStageOutputReset:
    """Tests for stage output resetting."""

    def test_resets_stage_outputs_on_selection(self):
        """Should reset stage_outputs when selecting a new stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                ]
            },
            "stage_outputs": {"previous": "data"},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["stage_outputs"] == {}

    def test_clears_run_error_on_selection(self):
        """Should clear run_error when selecting a new stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                ]
            },
            "run_error": "Previous error",
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["run_error"] is None


class TestCounterReset:
    """Tests for counter resetting on stage selection."""

    def test_resets_all_counters_for_needs_rerun(self):
        """Should reset all counters when selecting needs_rerun stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "needs_rerun", "dependencies": []},
                ]
            },
            "current_stage_id": "stage0",
            "design_revision_count": 5,
            "code_revision_count": 3,
            "execution_failure_count": 2,
            "physics_failure_count": 1,
            "analysis_revision_count": 4,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_resets_counters_for_different_stage(self):
        """Should reset counters when selecting a different stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage0"]},
                ]
            },
            "current_stage_id": "stage0",
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage1"
        assert result["design_revision_count"] == 0


class TestComplexPhysicsStage:
    """Tests for COMPLEX_PHYSICS stage type."""

    def test_allows_complex_physics_after_param_sweep(self):
        """COMPLEX_PHYSICS allowed after parameter_sweep passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "completed_success", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "stage_type": "PARAMETER_SWEEP", "status": "completed_success", "dependencies": ["stage1"]},
                    {"stage_id": "stage3", "stage_type": "COMPLEX_PHYSICS", "status": "not_started", "dependencies": ["stage2"]},
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
                "parameter_sweep": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage3"

    def test_allows_complex_physics_after_array_system(self):
        """COMPLEX_PHYSICS allowed after array_system passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE", "status": "completed_success", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "stage_type": "ARRAY_SYSTEM", "status": "completed_success", "dependencies": ["stage1"]},
                    {"stage_id": "stage3", "stage_type": "COMPLEX_PHYSICS", "status": "not_started", "dependencies": ["stage2"]},
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
                "array_system": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage3"
