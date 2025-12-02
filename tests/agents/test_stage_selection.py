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

