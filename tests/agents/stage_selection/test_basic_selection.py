"""Tests for basic stage selection logic."""

from unittest.mock import patch, MagicMock

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
        assert len(result["stage_start_time"]) > 0  # Should be non-empty ISO timestamp
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        
        # Verify no user interaction triggers
        assert result.get("ask_user_trigger") is None
        assert result.get("awaiting_user_input") is None
        assert result.get("pending_user_questions") is None
        
        # Verify counters are reset for new stage
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

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
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        assert result["workflow_phase"] == "stage_selection"
        assert "stage_start_time" in result
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None

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
        assert result["current_stage_id"] != "stage0"  # Explicitly verify skipped

    def test_skips_completed_partial_stages(self):
        """Should skip completed_partial stages."""
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
        assert result["current_stage_id"] != "stage0"

    def test_skips_completed_failed_stages(self):
        """Should skip completed_failed stages."""
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
        # stage1 should not be selected because stage0 dependency is not satisfied
        assert result["current_stage_id"] is None

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
        assert result["workflow_phase"] == "stage_selection"

    def test_skips_invalidated_stages(self):
        """Should skip invalidated stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),  # Must be completed for SINGLE_STRUCTURE
                    create_stage("stage1", "SINGLE_STRUCTURE", "invalidated"),  # This should be skipped
                    create_stage("stage2", "SINGLE_STRUCTURE", "not_started"),  # This should be selected
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_id"] != "stage1"  # Invalidated stage should be skipped

    def test_skips_blocked_stages(self):
        """Should skip blocked stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),  # Must be completed for SINGLE_STRUCTURE
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked"),  # This should be skipped
                    create_stage("stage2", "SINGLE_STRUCTURE", "not_started"),  # This should be selected
                ]
            },
        }
        
        result = select_stage_node(state)
        assert result["current_stage_id"] == "stage2"
        assert result["current_stage_id"] != "stage1"  # Blocked stage should be skipped

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
        assert result["current_stage_type"] is None
        assert result["workflow_phase"] == "stage_selection"
        assert result.get("ask_user_trigger") is None
        assert result.get("awaiting_user_input") is None

    def test_returns_escalation_when_no_stages(self):
        """Should return escalation when no stages exist."""
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["current_stage_type"] is None
        assert result["ask_user_trigger"] == "no_stages_available"
        assert result["awaiting_user_input"] is True
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0] or "No stages" in result["pending_user_questions"][0]

    def test_handles_none_plan_and_progress(self):
        """Should handle None plan and progress gracefully (escalate)."""
        state = {
            "plan": None,
            "progress": None,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["current_stage_type"] is None
        assert result["ask_user_trigger"] == "no_stages_available"
        assert result["awaiting_user_input"] is True

    def test_handles_empty_plan_stages_dict(self):
        """Should handle plan with empty stages dict."""
        state = {
            "plan": {},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        assert result["ask_user_trigger"] == "no_stages_available"

    def test_handles_missing_plan_key(self):
        """Should handle missing plan key."""
        state = {
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        assert result["ask_user_trigger"] == "no_stages_available"

    def test_handles_missing_progress_key(self):
        """Should handle missing progress key."""
        state = {
            "plan": {"stages": []},
        }
        
        result = select_stage_node(state)
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
            "design_revision_count": 10,
            "code_revision_count": 5,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_clears_all_stage_data_for_new_stage(self):
        """Should clear ALL stage-specific data when moving to a new stage.
        
        This is critical for the revision mode behavior: when starting a fresh stage,
        there should be no stale code, design, feedback, or verdicts from the previous
        stage. The code generator's revision mode depends on feedback being cleared.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "current_stage_id": "old_stage",  # Different stage = reset
            # Counters from previous stage
            "design_revision_count": 10,
            "code_revision_count": 5,
            "execution_failure_count": 3,
            "physics_failure_count": 2,
            "analysis_revision_count": 1,
            # Feedback from previous stage
            "reviewer_feedback": "Old reviewer feedback",
            "physics_feedback": "Old physics feedback",
            "execution_feedback": "Old execution feedback",
            "analysis_feedback": "Old analysis feedback",
            "design_feedback": "Old design feedback",
            "comparison_feedback": "Old comparison feedback",
            # Stage working data from previous stage
            "code": "# Old code from previous stage\nimport meep as mp",
            "design_description": {"old": "design from previous stage"},
            "performance_estimate": {"runtime_minutes": 60},
            "analysis_result_reports": [{"old": "analysis"}],
            # Verdicts from previous stage
            "last_design_review_verdict": "approve",
            "last_code_review_verdict": "approve",
            "execution_verdict": "pass",
            "physics_verdict": "pass",
            "comparison_verdict": "approve",
            "reviewer_issues": [{"severity": "major", "description": "old issue"}],
            "execution_warnings": ["old warning"],
            "physics_warnings": ["old physics warning"],
            # Structured agent output from previous stage
            "execution_status": {"completed": True, "exit_code": 0},
            "execution_files_check": {"expected_files": ["old.csv"]},
            "physics_conservation_checks": [{"law": "energy", "status": "pass"}],
            "physics_value_range_checks": [{"quantity": "T", "status": "pass"}],
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        
        # ─── Counters should be reset ───────────────────────────────────────
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        assert result["analysis_revision_count"] == 0
        
        # ─── Feedback fields should be cleared ──────────────────────────────
        # These are critical for the code generator's revision mode detection
        assert result["reviewer_feedback"] is None, "reviewer_feedback must be cleared for fresh generation"
        assert result["physics_feedback"] is None, "physics_feedback must be cleared for fresh generation"
        assert result["execution_feedback"] is None, "execution_feedback must be cleared for fresh generation"
        assert result["analysis_feedback"] is None
        assert result["design_feedback"] is None
        assert result["comparison_feedback"] is None
        
        # ─── Stage working data should be cleared ───────────────────────────
        # code being None ensures code generator doesn't see stale code
        assert result["code"] is None, "code must be cleared to prevent stale code leaking to new stage"
        assert result["design_description"] is None, "design_description must be cleared for new stage"
        assert result["performance_estimate"] is None
        assert result["analysis_result_reports"] == []
        
        # ─── Verdicts should be cleared ─────────────────────────────────────
        assert result["last_design_review_verdict"] is None
        assert result["last_code_review_verdict"] is None
        assert result["execution_verdict"] is None
        assert result["physics_verdict"] is None
        assert result["comparison_verdict"] is None
        assert result["reviewer_issues"] == []
        assert result["execution_warnings"] == []
        assert result["physics_warnings"] == []
        
        # ─── Structured agent output should be cleared ──────────────────────
        assert result["execution_status"] is None
        assert result["execution_files_check"] is None
        assert result["physics_conservation_checks"] is None
        assert result["physics_value_range_checks"] is None

    def test_does_not_reset_counters_for_same_stage_continuation(self):
        """Should NOT reset counters if selecting the same stage that is not_started."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "current_stage_id": "stage0",
            "design_revision_count": 5,
            "code_revision_count": 3,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        # Should NOT reset counters for same stage (unless needs_rerun)
        assert "design_revision_count" not in result
        assert "code_revision_count" not in result

    def test_does_not_clear_stage_data_for_same_stage_continuation(self):
        """Should NOT clear stage data when continuing the same stage.
        
        This preserves the code generator's ability to see previous code during
        revision loops within the same stage.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                ]
            },
            "current_stage_id": "stage0",  # Same stage = no reset
            # These should NOT be cleared
            "code": "# Existing code\nimport meep as mp",
            "design_description": {"current": "design"},
            "reviewer_feedback": "Please fix the bug",
            "physics_feedback": "T > 1.0 detected",
            "execution_feedback": None,
            "last_code_review_verdict": "needs_revision",
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        
        # These should NOT be in result (not cleared for same stage)
        assert "code" not in result, "code should not be cleared for same stage continuation"
        assert "design_description" not in result
        assert "reviewer_feedback" not in result
        assert "physics_feedback" not in result
        assert "execution_feedback" not in result
        assert "last_code_review_verdict" not in result
        assert "design_revision_count" not in result
        assert "code_revision_count" not in result

    def test_priority_1_selects_needs_rerun_stage(self):
        """Priority 1: Should select needs_rerun stage before not_started stages."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),  # Must be completed for validation hierarchy
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                    create_stage("stage2", "MATERIAL_VALIDATION", "not_started"),  # Another stage that could be selected
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # needs_rerun should be selected even though stage2 is available (validation hierarchy allows it)
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_id"] != "stage2"  # needs_rerun has priority
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"
        # Counters should be reset for needs_rerun
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0

    def test_priority_1_needs_rerun_resets_counters(self):
        """Priority 1: needs_rerun stage should reset counters even if same stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
            "current_stage_id": "stage0",
            "design_revision_count": 10,
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        # Should reset even for same stage if needs_rerun
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0

    def test_priority_1_needs_rerun_clears_all_stage_data(self):
        """Priority 1: needs_rerun should clear ALL stage data even for same stage.
        
        This is critical: when a stage is marked for rerun (e.g., after backtrack),
        it must start completely fresh with no stale data from the previous attempt.
        """
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
            "current_stage_id": "stage0",  # Same stage, but needs_rerun = reset
            # All of this should be cleared
            "design_revision_count": 10,
            "code_revision_count": 5,
            "code": "# Old code that failed\nimport meep",
            "design_description": {"old": "design that had issues"},
            "reviewer_feedback": "Old feedback from failed attempt",
            "physics_feedback": "Physics issues from before",
            "execution_feedback": "Execution problems",
            "last_code_review_verdict": "needs_revision",
            "execution_verdict": "fail",
            "physics_verdict": "fail",
            "reviewer_issues": [{"severity": "blocking", "description": "old"}],
            "execution_warnings": ["old warning"],
            "physics_warnings": ["old physics warning"],
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        
        # All counters should be reset
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["physics_failure_count"] == 0
        
        # All feedback should be cleared
        assert result["reviewer_feedback"] is None, "needs_rerun must clear reviewer_feedback"
        assert result["physics_feedback"] is None, "needs_rerun must clear physics_feedback"
        assert result["execution_feedback"] is None, "needs_rerun must clear execution_feedback"
        
        # Stage working data should be cleared
        assert result["code"] is None, "needs_rerun must clear code"
        assert result["design_description"] is None, "needs_rerun must clear design_description"
        
        # Verdicts should be cleared
        assert result["last_code_review_verdict"] is None
        assert result["execution_verdict"] is None
        assert result["physics_verdict"] is None
        assert result["reviewer_issues"] == []
        assert result["execution_warnings"] == []
        assert result["physics_warnings"] == []

    def test_priority_1_skips_needs_rerun_with_blocking_deps(self):
        """Priority 1: Should skip needs_rerun stage if dependencies are blocking."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "invalidated"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "needs_rerun", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # stage1 should be skipped because stage0 is invalidated (blocking)
        # stage2 should be skipped because dependencies not met
        assert result["current_stage_id"] is None

    def test_priority_1_skips_needs_rerun_with_needs_rerun_dep(self):
        """Priority 1: Should skip needs_rerun stage if dependency also needs_rerun."""
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
        
        # stage0 should be selected first (no blocking deps)
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_id"] != "stage1"

    def test_priority_1_skips_needs_rerun_with_not_started_dep(self):
        """Priority 1: Should skip needs_rerun stage if dependency is not_started."""
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
        
        # stage0 should be selected first
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_id"] != "stage1"

    def test_priority_1_skips_needs_rerun_with_in_progress_dep(self):
        """Priority 1: Should skip needs_rerun stage if dependency is in_progress."""
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
        
        # stage1 should be skipped because stage0 is in_progress (blocking)
        assert result["current_stage_id"] is None

    def test_priority_1_skips_needs_rerun_with_blocked_dep(self):
        """Priority 1: Should skip needs_rerun stage if dependency is blocked."""
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
        
        # stage1 should be skipped because stage0 is blocked
        assert result["current_stage_id"] is None

    def test_dependency_checking_requires_completed_success(self):
        """Should only select stage if dependencies are completed_success or completed_partial."""
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
        
        # stage1 should not be selected because stage0 is not completed_success/partial
        assert result["current_stage_id"] is None

    def test_dependency_checking_allows_completed_partial(self):
        """Should allow dependencies with completed_partial status."""
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

    def test_dependency_checking_handles_missing_dependency(self):
        """Should handle missing dependency by blocking stage."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["nonexistent_stage"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Stage should be blocked due to missing dependency
        assert result["current_stage_id"] is None

    def test_dependency_checking_handles_empty_dependencies(self):
        """Should select stage with no dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started", []),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"

    def test_dependency_checking_handles_none_dependencies(self):
        """Should handle None dependencies list."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        stage["dependencies"] = None
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Should handle None gracefully (treat as no dependencies)
        assert result["current_stage_id"] == "stage0"

    def test_dependency_checking_handles_missing_dependencies_key(self):
        """Should handle missing dependencies key."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        del stage["dependencies"]
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Should handle missing key gracefully
        assert result["current_stage_id"] == "stage0"

    def test_validation_hierarchy_blocks_single_structure_without_material_validation(self):
        """Should block SINGLE_STRUCTURE if MATERIAL_VALIDATION not passed/partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Only stage0 should be selected
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_id"] != "stage1"

    def test_validation_hierarchy_allows_single_structure_with_passed_material(self):
        """Should allow SINGLE_STRUCTURE if MATERIAL_VALIDATION is passed."""
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
        
        # stage1 should be selectable
        assert result["current_stage_id"] == "stage1"

    def test_validation_hierarchy_allows_single_structure_with_partial_material(self):
        """Should allow SINGLE_STRUCTURE if MATERIAL_VALIDATION is partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_partial"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started"),
                ]
            },
            "validation_hierarchy": {"material_validation": "partial"},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage1"

    def test_validation_hierarchy_blocks_array_system_without_single_structure(self):
        """Should block ARRAY_SYSTEM if SINGLE_STRUCTURE not passed/partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "ARRAY_SYSTEM", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        # stage1 should not be selected
        assert result["current_stage_id"] is None

    def test_validation_hierarchy_allows_array_system_with_single_structure(self):
        """Should allow ARRAY_SYSTEM if SINGLE_STRUCTURE is passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage2"

    def test_validation_hierarchy_blocks_parameter_sweep_without_single_structure(self):
        """Should block PARAMETER_SWEEP if SINGLE_STRUCTURE not passed/partial."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "PARAMETER_SWEEP", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None

    def test_validation_hierarchy_allows_parameter_sweep_with_single_structure(self):
        """Should allow PARAMETER_SWEEP if SINGLE_STRUCTURE is passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "PARAMETER_SWEEP", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage2"

    def test_validation_hierarchy_complex_physics_requires_array_or_sweep(self):
        """Should allow COMPLEX_PHYSICS if ARRAY_SYSTEM or PARAMETER_SWEEP is passed/partial."""
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
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage3"

    def test_validation_hierarchy_complex_physics_blocks_without_array_or_sweep(self):
        """Should block COMPLEX_PHYSICS if neither ARRAY_SYSTEM nor PARAMETER_SWEEP is passed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "COMPLEX_PHYSICS", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None

    def test_stage_type_order_enforcement_blocks_higher_before_lower(self):
        """Should enforce STAGE_TYPE_ORDER - cannot select higher type before lower completes."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "ARRAY_SYSTEM", "not_started"),  # Higher type
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 (lower type) first, not stage1
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_id"] != "stage1"

    def test_stage_type_order_enforcement_allows_after_lower_completes(self):
        """Should allow higher type after lower type completes."""
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
        
        assert result["current_stage_id"] == "stage1"

    def test_stage_type_order_enforcement_checks_all_lower_types(self):
        """Should check all lower types in order, not just immediate predecessor."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "not_started"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success"),
                    create_stage("stage2", "PARAMETER_SWEEP", "not_started"),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "not_done",
                "single_structure": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        # Should select stage0 first (MATERIAL_VALIDATION), not stage2
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_id"] != "stage2"

    def test_blocked_stage_unblocking_when_deps_satisfied(self):
        """Should unblock blocked stage when dependencies become satisfied."""
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
        
        # stage1 should be unblocked and selected
        assert result["current_stage_id"] == "stage1"

    def test_blocked_stage_remains_blocked_when_deps_not_satisfied(self):
        """Should keep stage blocked when dependencies are not satisfied."""
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
        
        # stage0 should be selected, stage1 should remain blocked
        assert result["current_stage_id"] == "stage0"
        assert result["current_stage_id"] != "stage1"

    def test_missing_stage_type_blocks_stage(self):
        """Should block stage with missing stage_type field."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        del stage["stage_type"]
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Stage should be blocked due to missing stage_type
        assert result["current_stage_id"] is None

    def test_none_stage_type_blocks_stage(self):
        """Should block stage with None stage_type."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        stage["stage_type"] = None
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None

    def test_unknown_stage_type_blocks_stage(self):
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
        
        # Unknown stage type should be blocked
        assert result["current_stage_id"] is None

    def test_complex_physics_stage_type_allowed(self):
        """Should allow COMPLEX_PHYSICS stage type (special case)."""
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
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        # COMPLEX_PHYSICS should be allowed if validation hierarchy allows
        assert result["current_stage_id"] == "stage3"

    def test_deadlock_detection_all_blocked(self):
        """Should detect deadlock when all remaining stages are blocked."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "blocked"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "blocked", ["stage0"]),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"
        assert result["awaiting_user_input"] is True
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert "Deadlock" in result["pending_user_questions"][0] or "blocked" in result["pending_user_questions"][0]

    def test_deadlock_detection_all_failed(self):
        """Should detect deadlock when all remaining stages are completed_failed."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_failed"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        # Should detect deadlock (all stages failed, none runnable)
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "deadlock_detected"

    def test_deadlock_detection_mixed_blocked_and_failed(self):
        """Should detect deadlock with mix of blocked and failed stages."""
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
        
        assert result["ask_user_trigger"] == "deadlock_detected"

    def test_no_deadlock_when_potentially_runnable_exists(self):
        """Should not detect deadlock when potentially runnable stages exist."""
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
        
        # Should select stage0, not detect deadlock
        assert result["current_stage_id"] == "stage0"
        assert result.get("ask_user_trigger") != "deadlock_detected"

    def test_progress_initialization_from_plan(self):
        """Should initialize progress from plan if progress stages empty."""
        plan_stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        
        state = {
            "plan": {"stages": [plan_stage]},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        
        # Should initialize progress and select stage
        assert result["current_stage_id"] == "stage0"

    @patch('src.agents.stage_selection.initialize_progress_from_plan')
    def test_progress_initialization_failure_handling(self, mock_init):
        """Should handle progress initialization failure gracefully."""
        mock_init.side_effect = Exception("Initialization failed")
        
        state = {
            "plan": {"stages": [create_stage("stage0", "MATERIAL_VALIDATION", "not_started")]},
            "progress": {"stages": []},
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] is None
        assert result["ask_user_trigger"] == "progress_init_failed"
        assert result["awaiting_user_input"] is True
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0

    def test_complex_dependency_chain(self):
        """Should handle complex multi-level dependency chain."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started", ["stage1"]),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage2"

    def test_multiple_stages_same_dependencies(self):
        """Should select first eligible stage when multiple stages have same dependencies."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                    create_stage("stage2", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # Should select first eligible stage (stage1)
        assert result["current_stage_id"] == "stage1"

    def test_stage_with_multiple_dependencies(self):
        """Should require all dependencies to be satisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "completed_success", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started", ["stage0", "stage1"]),
                ]
            },
            "validation_hierarchy": {
                "material_validation": "passed",
                "single_structure": "passed",
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage2"

    def test_stage_with_multiple_dependencies_one_unsatisfied(self):
        """Should not select stage if any dependency is unsatisfied."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "completed_success"),
                    create_stage("stage1", "SINGLE_STRUCTURE", "not_started", ["stage0"]),
                    create_stage("stage2", "ARRAY_SYSTEM", "not_started", ["stage0", "stage1"]),
                ]
            },
            "validation_hierarchy": {"material_validation": "passed"},
        }
        
        result = select_stage_node(state)
        
        # stage2 should not be selected because stage1 is not completed
        assert result["current_stage_id"] == "stage1"
        assert result["current_stage_id"] != "stage2"

    def test_needs_rerun_stage_does_not_include_stage_start_time(self):
        """Priority 1: needs_rerun stage should not include stage_start_time."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        # needs_rerun path does not set stage_start_time
        assert "stage_start_time" not in result

    def test_not_started_stage_includes_stage_start_time(self):
        """Priority 2: not_started stage should include stage_start_time."""
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
        assert isinstance(result["stage_start_time"], str)

    def test_result_includes_analysis_fields_for_needs_rerun(self):
        """Priority 1: needs_rerun result should include analysis fields."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "MATERIAL_VALIDATION", "needs_rerun"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert "analysis_summary" in result
        assert result["analysis_summary"] is None
        assert "analysis_overall_classification" in result
        assert result["analysis_overall_classification"] is None

    def test_result_does_not_include_analysis_fields_for_not_started(self):
        """Priority 2: not_started result should not include analysis fields."""
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
        assert "analysis_summary" not in result
        assert "analysis_overall_classification" not in result

    def test_empty_stages_list_after_initialization(self):
        """Should handle case where stages list is empty after initialization."""
        with patch('src.agents.stage_selection.initialize_progress_from_plan') as mock_init:
            mock_init.return_value = {"progress": {"stages": []}}
            
            state = {
                "plan": {"stages": [create_stage("stage0", "MATERIAL_VALIDATION", "not_started")]},
                "progress": {"stages": []},
            }
            
            result = select_stage_node(state)
            
            assert result["current_stage_id"] is None
            assert result["ask_user_trigger"] == "no_stages_available"
            assert result["awaiting_user_input"] is True

    def test_stage_id_must_be_present(self):
        """Should handle stage without stage_id field."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        del stage["stage_id"]
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Stage without stage_id should be skipped
        assert result["current_stage_id"] is None

    def test_stage_id_none_handling(self):
        """Should handle stage with None stage_id."""
        stage = create_stage("stage0", "MATERIAL_VALIDATION", "not_started")
        stage["stage_id"] = None
        
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [stage]
            },
        }
        
        result = select_stage_node(state)
        
        # Stage with None stage_id should be skipped
        assert result["current_stage_id"] is None
