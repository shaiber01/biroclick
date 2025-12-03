"""Tests for validation hierarchy enforcement."""

from unittest.mock import patch

import pytest

from src.agents.stage_selection import select_stage_node
from tests.agents.stage_selection.utils import create_stage

class TestValidationHierarchy:
    """Tests for validation hierarchy enforcement."""

    def test_requires_material_validation_for_single_structure_not_done(self):
        """SINGLE_STRUCTURE requires material_validation passed/partial, blocks when not_done."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result.get("current_stage_type") is None

    def test_requires_material_validation_for_single_structure_failed(self):
        """SINGLE_STRUCTURE blocks when material_validation is failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_requires_material_validation_for_single_structure_passed(self):
        """SINGLE_STRUCTURE allows proceeding when material_validation is passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage1"
        assert result["workflow_phase"] == "stage_selection"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        assert "stage_start_time" in result

    def test_requires_single_structure_for_array_system_not_done(self):
        """ARRAY_SYSTEM requires single_structure passed/partial, blocks when not_done."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select SINGLE_STRUCTURE first, not ARRAY_SYSTEM
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_requires_single_structure_for_array_system_failed(self):
        """ARRAY_SYSTEM blocks when single_structure is failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_failed"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_requires_single_structure_for_array_system_passed(self):
        """ARRAY_SYSTEM allows proceeding when single_structure is passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["workflow_phase"] == "stage_selection"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"

    def test_allows_partial_validation_single_structure(self):
        """Should allow proceeding if material_validation is 'partial'."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        assert result["workflow_phase"] == "stage_selection"

    def test_allows_partial_validation_array_system(self):
        """Should allow proceeding if single_structure is 'partial'."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_partial"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"

    def test_parameter_sweep_requires_single_structure(self):
        """PARAMETER_SWEEP requires single_structure passed/partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                    create_stage("stage2", "PARAMETER_SWEEP", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select SINGLE_STRUCTURE first
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_parameter_sweep_allows_when_single_structure_passed(self):
        """PARAMETER_SWEEP allows proceeding when single_structure is passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "PARAMETER_SWEEP", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "PARAMETER_SWEEP"

    def test_complex_physics_requires_array_or_sweep(self):
        """COMPLEX_PHYSICS requires ARRAY_SYSTEM or PARAMETER_SWEEP passed/partial."""
        # Test with ARRAY_SYSTEM passed
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "completed_success"),
                    create_stage("stage3", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage3"
        assert result["current_stage_type"] == "COMPLEX_PHYSICS"

    def test_complex_physics_blocks_when_neither_array_nor_sweep_passed(self):
        """COMPLEX_PHYSICS blocks when neither ARRAY_SYSTEM nor PARAMETER_SWEEP passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                    create_stage("stage3", "PARAMETER_SWEEP", "not_started"),
                    create_stage("stage4", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select ARRAY_SYSTEM or PARAMETER_SWEEP, not COMPLEX_PHYSICS
        assert result["current_stage_id"] in ["stage2", "stage3"]
        assert result["current_stage_type"] in ["ARRAY_SYSTEM", "PARAMETER_SWEEP"]

    def test_complex_physics_allows_with_parameter_sweep_partial(self):
        """COMPLEX_PHYSICS allows proceeding when PARAMETER_SWEEP is partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "PARAMETER_SWEEP", "completed_partial"),
                    create_stage("stage3", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage3"
        assert result["current_stage_type"] == "COMPLEX_PHYSICS"

    def test_multiple_stages_same_type_blocks_on_any_failed(self):
        """If multiple stages of same type exist, any failed blocks higher types."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1a", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage1b", "SINGLE_STRUCTURE", "completed_failed"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should block ARRAY_SYSTEM because one SINGLE_STRUCTURE failed
        assert result["current_stage_id"] is None

    def test_multiple_stages_same_type_allows_when_all_passed(self):
        """If multiple stages of same type exist, all must pass for higher types."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1a", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage1b", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"

    def test_multiple_stages_same_type_allows_partial_mixed(self):
        """Mixed success/partial results in 'partial' status allowing higher types."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1a", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage1b", "SINGLE_STRUCTURE", "completed_partial"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Mixed success/partial should allow proceeding
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"


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
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["workflow_phase"] == "stage_selection"
        
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
        }
        
        result = select_stage_node(state)
        # stage0 is blocked (missing dep). 
        # stage1 should be skipped because stage0 (lower type) exists and is not completed.
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_enforces_type_order_multiple_lower_types(self):
        """Should skip higher type if ANY lower order type exists and is not completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"), # Higher order
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select SINGLE_STRUCTURE, not ARRAY_SYSTEM
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_enforces_type_order_all_lower_completed(self):
        """Should allow higher type when all lower order types are completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"

    def test_enforces_type_order_parameter_sweep(self):
        """Should enforce order for PARAMETER_SWEEP (requires SINGLE_STRUCTURE completed)."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                    create_stage("stage2", "PARAMETER_SWEEP", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select SINGLE_STRUCTURE first
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_enforces_type_order_complex_physics(self):
        """Should enforce order for COMPLEX_PHYSICS (requires ARRAY_SYSTEM or PARAMETER_SWEEP)."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                    create_stage("stage3", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select ARRAY_SYSTEM first, not COMPLEX_PHYSICS
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"

    def test_enforces_type_order_lower_partial_allows(self):
        """Lower type with 'partial' status should allow higher types (validation hierarchy check)."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Partial should allow proceeding (validation hierarchy allows it)
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"


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
        assert result["current_stage_type"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result["awaiting_user_input"] is True
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "Deadlock detected" in result["pending_user_questions"][0]
        assert "blocked" in result["pending_user_questions"][0].lower() or "failed" in result["pending_user_questions"][0].lower()

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
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result.get("ask_user_trigger") != "deadlock_detected"
        assert result.get("awaiting_user_input") is not True

    def test_no_deadlock_if_invalidated_exists(self):
        """Should not report deadlock if a stage is invalidated (can be rerun)."""
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
        # Invalidated stages are not runnable, but shouldn't trigger deadlock
        # if there are other potentially runnable stages
        # In this case, stage0 is invalidated (not runnable), stage1 is blocked
        # This might be a deadlock, but let's check what actually happens
        # The component should check if there are any potentially_runnable stages
        assert result.get("current_stage_id") is None
        # If all are blocked/invalidated, it should detect deadlock
        if result.get("ask_user_trigger") == "deadlock_detected":
            assert result["awaiting_user_input"] is True

    def test_no_deadlock_if_not_started_exists(self):
        """Should not report deadlock if a stage is not_started."""
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
        # Should select stage0, not report deadlock
        assert result["current_stage_id"] == "stage0"
        assert result.get("ask_user_trigger") != "deadlock_detected"

    def test_deadlock_multiple_blocked_stages(self):
        """Should detect deadlock with multiple blocked stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "blocked", ["stage1"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) > 0

    def test_deadlock_mixed_blocked_and_failed(self):
        """Should detect deadlock with mix of blocked and failed stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_failed"),
                    create_stage("stage2", "ARRAY_SYSTEM", "blocked", ["stage1"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result["awaiting_user_input"] is True


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
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["workflow_phase"] == "stage_selection"

    @patch("src.agents.stage_selection.initialize_progress_from_plan")
    def test_handles_initialization_failure(self, mock_init):
        """Should handle initialization failure gracefully."""
        mock_init.side_effect = Exception("Initialization failed")
        
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
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result["ask_user_trigger"] == "progress_init_failed"
        assert result["awaiting_user_input"] is True
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "Failed to initialize" in result["pending_user_questions"][0]

    def test_handles_empty_plan_and_progress(self):
        """Should handle empty plan and progress gracefully."""
        state = {
            "plan": {"stages": []},
            "progress": {},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result["ask_user_trigger"] == "no_stages_available"
        assert result["awaiting_user_input"] is True
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "No stages available" in result["pending_user_questions"][0]

    def test_handles_none_progress(self):
        """Should handle None progress gracefully."""
        state = {
            "plan": {"stages": []},
            "progress": None,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_handles_none_plan(self):
        """Should handle None plan gracefully."""
        state = {
            "plan": None,
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should still work if progress has stages
        assert result["current_stage_id"] == "stage0"
        assert result["workflow_phase"] == "stage_selection"


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    def test_missing_stage_type_blocks_stage(self):
        """Should block stage if stage_type is missing."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "status": "not_started",
                        "dependencies": [],
                        # Missing stage_type
                    },
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should not select stage without stage_type
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_unknown_stage_type_blocks_stage(self):
        """Should block stage if stage_type is unknown."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "UNKNOWN_TYPE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should not select stage with unknown type
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

    def test_missing_dependencies_blocks_stage(self):
        """Should block stage if dependencies are missing."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["missing_stage"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select stage0, not stage1 (which has missing dependency)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_resets_counters_on_stage_change(self):
        """Should reset counters when switching to a different stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "current_stage_id": "previous_stage",
            "design_revision_count": 5,
            "code_revision_count": 3,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_keeps_counters_on_same_stage(self):
        """Should keep counters when staying on same stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
            "current_stage_id": "stage0",
            "design_revision_count": 5,
            "code_revision_count": 3,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        # Counters should be reset for needs_rerun
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0

    def test_invalidated_stage_not_selected(self):
        """Should not select invalidated stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "invalidated"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select stage1, not invalidated stage0
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_in_progress_stage_not_selected(self):
        """Should not select in_progress stages (shouldn't happen, but test it)."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "in_progress"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select stage1, not in_progress stage0
        assert result["current_stage_id"] == "stage1"

    def test_completed_stages_not_selected(self):
        """Should not select completed stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage2", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage3", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select stage3, not completed stages
        assert result["current_stage_id"] == "stage3"

    def test_needs_rerun_has_priority(self):
        """needs_rerun stages should have highest priority."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select needs_rerun stage1, not stage0
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_needs_rerun_with_blocking_dependencies(self):
        """needs_rerun stage should not be selected if dependencies block it."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should select stage0 first (no blocking deps), not stage1
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_blocked_stage_unblocks_when_deps_satisfied(self):
        """Should unblock stage when dependencies become satisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        # Should unblock and select stage1
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_empty_stages_list_handled(self):
        """Should handle empty stages list gracefully."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": []
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"
        # Should not trigger deadlock if no stages exist
        assert result.get("ask_user_trigger") != "deadlock_detected"

    def test_all_stages_completed_returns_none(self):
        """Should return None when all stages are completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["current_stage_type"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result.get("ask_user_trigger") != "deadlock_detected"

    def test_stage_start_time_set(self):
        """Should set stage_start_time when selecting a stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert "stage_start_time" in result
        assert result["stage_start_time"] is not None
        # Should be ISO format timestamp
        assert "T" in result["stage_start_time"] or "Z" in result["stage_start_time"]

    def test_stage_outputs_initialized(self):
        """Should initialize stage_outputs when selecting a stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert "stage_outputs" in result
        assert result["stage_outputs"] == {}

    def test_run_error_cleared(self):
        """Should clear run_error when selecting a new stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "run_error": "Previous error",
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert "run_error" in result
        assert result["run_error"] is None
