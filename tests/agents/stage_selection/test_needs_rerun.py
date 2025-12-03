"""Tests for needs-rerun stage prioritization."""

import pytest

from src.agents.stage_selection import select_stage_node
from tests.agents.stage_selection.utils import create_stage

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
