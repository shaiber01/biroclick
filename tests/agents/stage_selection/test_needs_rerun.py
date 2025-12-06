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
        
        # Verify correct stage selected
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["workflow_phase"] == "stage_selection"
        
        # Needs rerun should always reset counters
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0
        
        # Verify other required fields
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        assert result["analysis_summary"] is None
        assert result["analysis_overall_classification"] is None
        
        # Verify no user interaction triggers
        assert result.get("ask_user_trigger") is None

    def test_skips_needs_rerun_with_invalidated_deps(self):
        """Should skip needs_rerun stage if it has invalidated dependencies.
        
        Note: Invalidated stages are also skipped per the docstring:
        "invalidated" - waiting for dependency to complete
        """
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
        
        # stage1 skipped due to invalidated dep
        # stage0 is invalidated and also skipped (invalidated stages are skipped)
        # No stage can be selected
        assert result["current_stage_id"] is None
        assert result["workflow_phase"] == "stage_selection"

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
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        # Should NOT select stage1
        assert result["current_stage_id"] != "stage1"
        
        # Verify counters are reset for stage0
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_skips_needs_rerun_with_blocked_deps(self):
        """Should skip needs_rerun stage if dependency is blocked."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "blocked"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should NOT select stage1 because dependency is blocked
        assert result["current_stage_id"] != "stage1"
        # stage0 is blocked, so nothing should be selected
        assert result["current_stage_id"] is None

    def test_skips_needs_rerun_with_in_progress_deps(self):
        """Should skip needs_rerun stage if dependency is in_progress."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "in_progress"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should NOT select stage1 because dependency is in_progress
        assert result["current_stage_id"] != "stage1"
        # stage0 is in_progress, so nothing should be selected
        assert result["current_stage_id"] is None

    def test_skips_needs_rerun_with_not_started_deps(self):
        """Should skip needs_rerun stage if dependency is not_started."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should NOT select stage1 because dependency is not_started
        assert result["current_stage_id"] != "stage1"
        # Should select stage0 instead
        assert result["current_stage_id"] == "stage0"

    def test_selects_needs_rerun_with_completed_deps(self):
        """Should select needs_rerun stage if dependencies are completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 because dependency is completed
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        # Verify counters are reset
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_selects_needs_rerun_with_partial_completed_deps(self):
        """Should select needs_rerun stage if dependencies are completed_partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "partial"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 because dependency is completed_partial
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        # Verify counters are reset
        assert result["design_revision_count"] == 0

    def test_selects_needs_rerun_with_no_dependencies(self):
        """Should select needs_rerun stage with no dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (needs_rerun takes priority)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        # Verify counters are reset
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_selects_first_needs_rerun_when_multiple_exist(self):
        """Should select first needs_rerun stage when multiple exist (order matters)."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage2", "SINGLE_STRUCTURE", "needs_rerun"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "needs_rerun"),
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select first needs_rerun stage encountered (stage2)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        # Verify counters are reset
        assert result["design_revision_count"] == 0

    def test_resets_counters_even_when_same_stage_id(self):
        """Should reset counters for needs_rerun even if current_stage_id matches."""
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
        
        # Should still select stage0
        assert result["current_stage_id"] == "stage0"
        # Should reset counters even though current_stage_id matches (because status is needs_rerun)
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_needs_rerun_bypasses_validation_hierarchy(self):
        """Should select needs_rerun stage even if validation hierarchy not satisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "failed"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (not_started), not stage1 (needs_rerun with unmet dep)
        assert result["current_stage_id"] == "stage0"
        # But if stage0 was completed, stage1 should be selected even with failed hierarchy
        state["progress"]["stages"][0] = create_stage("stage0", "MATERIAL_VALIDATION", "completed_success")
        
        result2 = select_stage_node(state)
        # Should select stage1 even though validation hierarchy is failed
        assert result2["current_stage_id"] == "stage1"

    def test_needs_rerun_with_multiple_dependencies_all_completed(self):
        """Should select needs_rerun stage when all multiple dependencies are completed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage2", "SINGLE_STRUCTURE", "needs_rerun", ["stage0", "stage1"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should select stage2 because all dependencies are completed
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        assert result["design_revision_count"] == 0

    def test_skips_needs_rerun_with_one_incomplete_dep(self):
        """Should skip needs_rerun stage if any dependency is incomplete."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage2", "SINGLE_STRUCTURE", "needs_rerun", ["stage0", "stage1"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should NOT select stage2 because stage1 is not completed
        assert result["current_stage_id"] != "stage2"
        # Should select stage1 instead
        assert result["current_stage_id"] == "stage1"

    def test_skips_needs_rerun_with_missing_dependency(self):
        """Should skip needs_rerun stage if dependency doesn't exist."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["nonexistent"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should NOT select stage1 because dependency doesn't exist
        assert result["current_stage_id"] != "stage1"
        # Should return None since no valid stage to run
        assert result["current_stage_id"] is None

    def test_needs_rerun_takes_priority_over_completed_failed(self):
        """Should select needs_rerun stage even when other stages are completed_failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage1 (needs_rerun) even though stage0 is completed_failed
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        assert result["design_revision_count"] == 0

    def test_needs_rerun_with_empty_dependencies_list(self):
        """Should handle needs_rerun stage with empty dependencies list."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun", []),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["design_revision_count"] == 0

    def test_needs_rerun_with_none_dependencies(self):
        """Should handle needs_rerun stage with None dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "needs_rerun",
                        "dependencies": None,
                    }
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (None dependencies treated as no dependencies)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["design_revision_count"] == 0

    def test_needs_rerun_with_missing_dependencies_field(self):
        """Should handle needs_rerun stage with missing dependencies field."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "needs_rerun",
                    }
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (missing dependencies treated as no dependencies)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["design_revision_count"] == 0

    def test_needs_rerun_returns_all_required_fields(self):
        """Should return all required fields for needs_rerun stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Verify all required fields are present
        assert "current_stage_id" in result
        assert "current_stage_type" in result
        assert "workflow_phase" in result
        assert "stage_outputs" in result
        assert "run_error" in result
        assert "analysis_summary" in result
        assert "analysis_overall_classification" in result
        assert "design_revision_count" in result
        assert "code_revision_count" in result
        assert "execution_failure_count" in result
        assert "physics_failure_count" in result
        assert "analysis_revision_count" in result
        
        # Verify field values
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result["workflow_phase"] == "stage_selection"
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        assert result["analysis_summary"] is None
        assert result["analysis_overall_classification"] is None
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_needs_rerun_complex_dependency_chain(self):
        """Should handle complex dependency chain with needs_rerun stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "needs_rerun", ["stage1"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (root dependency)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        
        # After stage0 completes, should select stage1
        state["progress"]["stages"][0] = create_stage("stage0", "MATERIAL_VALIDATION", "completed_success")
        result2 = select_stage_node(state)
        assert result2["current_stage_id"] == "stage1"
        
        # After stage1 completes, should select stage2
        state["progress"]["stages"][1] = create_stage("stage1", "SINGLE_STRUCTURE", "completed_success", ["stage0"])
        result3 = select_stage_node(state)
        assert result3["current_stage_id"] == "stage2"

    def test_needs_rerun_with_completed_failed_dependency(self):
        """Should skip needs_rerun stage if dependency is completed_failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should NOT select stage1 because dependency is completed_failed (not completed_success/partial)
        assert result["current_stage_id"] != "stage1"
        # Should return None since stage0 is completed_failed and stage1 can't run
        assert result["current_stage_id"] is None
