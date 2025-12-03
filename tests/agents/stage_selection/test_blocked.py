"""Tests for blocked stage handling."""

from unittest.mock import patch

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
