"""Tests for validation hierarchy enforcement."""

from unittest.mock import patch

import pytest

from src.agents.stage_selection import select_stage_node
from tests.agents.stage_selection.utils import create_stage

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
