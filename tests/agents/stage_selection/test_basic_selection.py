"""Tests for basic stage selection logic."""

from unittest.mock import patch

import pytest

from src.agents.stage_selection import select_stage_node
from tests.agents.stage_selection.utils import create_stage

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
