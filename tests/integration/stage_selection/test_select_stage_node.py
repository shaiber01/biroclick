"""Integration tests for stage selection edge cases."""

from unittest.mock import patch

import pytest

from tests.integration.helpers.state_factories import make_plan, make_progress, make_stage


class TestStageSelectionEdgeCases:
    """Stage selection edge-case handling."""

    def test_select_stage_node_selects_valid_stage(self, base_state, valid_plan):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0",
                    "MATERIAL_VALIDATION",
                    status="not_started",
                    dependencies=[],
                )
            ]
        )

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result.get("workflow_phase") == "stage_selection"
        # Verify state is cleared for new stage
        assert result.get("stage_outputs") == {}
        assert result.get("run_error") is None
        # Verify counters are reset
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0
        assert result.get("execution_failure_count") == 0
        assert result.get("physics_failure_count") == 0
        assert result.get("analysis_revision_count") == 0
        # Verify stage_start_time is set
        assert "stage_start_time" in result
        assert isinstance(result["stage_start_time"], str)
        assert "T" in result["stage_start_time"]  # ISO format check

    def test_select_stage_respects_validation_hierarchy(self, base_state):
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"], dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", targets=["Fig2"], dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage(
                "stage_0",
                "MATERIAL_VALIDATION",
                targets=["Fig1"],
                status="not_started",
                dependencies=[],
            ),
            make_stage(
                "stage_1",
                "SINGLE_STRUCTURE",
                targets=["Fig2"],
                status="not_started",
                dependencies=["stage_0"],
            ),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Must select stage_0 first because stage_1 depends on it
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        # stage_1 should NOT be selected yet
        assert result["current_stage_id"] != "stage_1"

    def test_select_stage_skips_completed_stages(self, base_state):
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"]),
            make_stage(
                "stage_1", "SINGLE_STRUCTURE", targets=["Fig2"], dependencies=["stage_0"]
            ),
        ]
        progress_stages = [
            make_stage(
                "stage_0",
                "MATERIAL_VALIDATION",
                status="completed_success",
                dependencies=[],
            ),
            make_stage(
                "stage_1",
                "SINGLE_STRUCTURE",
                status="not_started",
                dependencies=["stage_0"],
            ),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Should skip completed stage_0 and select stage_1
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_select_stage_detects_deadlock(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"])]
        )
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0",
                    "MATERIAL_VALIDATION",
                    status="completed_failed",
                    dependencies=[],
                )
            ]
        )

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "deadlock_detected"
        assert result.get("awaiting_user_input") is True
        assert result.get("current_stage_id") is None
        assert result.get("current_stage_type") is None
        # Check that pending_user_questions contains useful info
        questions = result.get("pending_user_questions", [])
        assert len(questions) > 0
        assert "deadlock" in questions[0].lower() or "blocked" in questions[0].lower()

    def test_blocked_stage_with_unsatisfied_deps_stays_blocked(self, base_state):
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"], dependencies=[]),
            make_stage(
                "stage_1", "SINGLE_STRUCTURE", targets=["Fig2"], dependencies=["stage_0"]
            ),
        ]
        progress_stages = [
            make_stage(
                "stage_0",
                "MATERIAL_VALIDATION",
                status="not_started",
                dependencies=[],
            ),
            make_stage(
                "stage_1",
                "SINGLE_STRUCTURE",
                status="blocked",
                dependencies=["stage_0"],
            ),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Should select stage_0, not the blocked stage_1
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_select_stage_resets_counters_on_new_stage(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"])]
        )
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0",
                    "MATERIAL_VALIDATION",
                    status="not_started",
                    dependencies=[],
                )
            ]
        )
        base_state["current_stage_id"] = None
        base_state["design_revision_count"] = 5
        base_state["code_revision_count"] = 5
        base_state["execution_failure_count"] = 3
        base_state["physics_failure_count"] = 2
        base_state["analysis_revision_count"] = 4

        result = select_stage_node(base_state)
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0
        assert result.get("execution_failure_count") == 0
        assert result.get("physics_failure_count") == 0
        assert result.get("analysis_revision_count") == 0

    def test_select_stage_sets_stage_start_time_and_clears_outputs(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"])]
        )
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started")]
        )
        base_state["stage_outputs"] = {"files": ["/tmp/old.csv"]}
        base_state["run_error"] = "stale failure"

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"
        assert result.get("stage_outputs") == {}
        assert result.get("run_error") is None
        start_time = result.get("stage_start_time")
        assert isinstance(start_time, str) and "T" in start_time

    def test_select_stage_reports_progress_init_failure(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", targets=["Fig1"])]
        )
        base_state["progress"] = {}

        with patch(
            "src.agents.stage_selection.initialize_progress_from_plan",
            side_effect=RuntimeError("boom"),
        ):
            result = select_stage_node(base_state)

        assert result.get("ask_user_trigger") == "progress_init_failed"
        assert result.get("awaiting_user_input") is True
        assert result.get("current_stage_id") is None
        assert result.get("current_stage_type") is None
        questions = result.get("pending_user_questions", [])
        assert questions and "Failed to initialize progress" in questions[0]


class TestNeedsRerunPriority:
    """Tests for needs_rerun priority (Priority 1 selection)."""

    def test_needs_rerun_selected_over_not_started(self, base_state):
        """needs_rerun stages should be selected before not_started stages."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="needs_rerun", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_needs_rerun_resets_counters(self, base_state):
        """Selecting a needs_rerun stage should reset all counters."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[])]
        )
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="needs_rerun", dependencies=[])]
        )
        base_state["current_stage_id"] = "stage_0"  # Same stage, but needs_rerun
        base_state["design_revision_count"] = 3
        base_state["code_revision_count"] = 2
        base_state["execution_failure_count"] = 1

        result = select_stage_node(base_state)
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0
        assert result.get("execution_failure_count") == 0

    def test_needs_rerun_with_blocking_deps_not_selected(self, base_state):
        """needs_rerun stage with blocking dependencies should not be selected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            # stage_0 is not complete, so stage_1 can't run
            make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="needs_rerun", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Should select stage_0 (not_started with no deps) instead of blocked needs_rerun
        assert result["current_stage_id"] == "stage_0"

    def test_needs_rerun_with_completed_deps_selected(self, base_state):
        """needs_rerun stage with completed dependencies should be selected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="needs_rerun", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_needs_rerun_with_partial_deps_selected(self, base_state):
        """needs_rerun stage with completed_partial dependencies should be selected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_partial", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="needs_rerun", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_1"


class TestValidationHierarchy:
    """Tests for validation hierarchy enforcement."""

    def test_single_structure_requires_material_validation_passed(self, base_state):
        """SINGLE_STRUCTURE cannot run until MATERIAL_VALIDATION passes."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=[]),  # No explicit dep
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=[]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # MATERIAL_VALIDATION must complete before SINGLE_STRUCTURE
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_array_system_requires_single_structure_passed(self, base_state):
        """ARRAY_SYSTEM cannot run until SINGLE_STRUCTURE passes."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", dependencies=["stage_1"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", status="not_started", dependencies=["stage_1"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # SINGLE_STRUCTURE must complete before ARRAY_SYSTEM
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_parameter_sweep_requires_single_structure_passed(self, base_state):
        """PARAMETER_SWEEP cannot run until SINGLE_STRUCTURE passes."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "PARAMETER_SWEEP", dependencies=["stage_1"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
            make_stage("stage_2", "PARAMETER_SWEEP", status="not_started", dependencies=["stage_1"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_hierarchy_allows_partial_completion(self, base_state):
        """Hierarchy should allow progress when prerequisite stage is completed_partial."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_partial", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # completed_partial should allow dependent stage to run
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"


class TestComplexPhysicsStage:
    """Tests for COMPLEX_PHYSICS stage type special handling."""

    def test_complex_physics_requires_param_sweep_or_array_system(self, base_state):
        """COMPLEX_PHYSICS requires either PARAMETER_SWEEP or ARRAY_SYSTEM to pass."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "PARAMETER_SWEEP", dependencies=["stage_1"]),
            make_stage("stage_3", "COMPLEX_PHYSICS", dependencies=["stage_2"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success", dependencies=["stage_0"]),
            make_stage("stage_2", "PARAMETER_SWEEP", status="not_started", dependencies=["stage_1"]),
            make_stage("stage_3", "COMPLEX_PHYSICS", status="not_started", dependencies=["stage_2"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # PARAMETER_SWEEP must complete before COMPLEX_PHYSICS
        assert result["current_stage_id"] == "stage_2"
        assert result["current_stage_type"] == "PARAMETER_SWEEP"

    def test_complex_physics_can_run_after_param_sweep_passes(self, base_state):
        """COMPLEX_PHYSICS can run when PARAMETER_SWEEP is completed."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "PARAMETER_SWEEP", dependencies=["stage_1"]),
            make_stage("stage_3", "COMPLEX_PHYSICS", dependencies=["stage_2"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success", dependencies=["stage_0"]),
            make_stage("stage_2", "PARAMETER_SWEEP", status="completed_success", dependencies=["stage_1"]),
            make_stage("stage_3", "COMPLEX_PHYSICS", status="not_started", dependencies=["stage_2"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_3"
        assert result["current_stage_type"] == "COMPLEX_PHYSICS"

    def test_complex_physics_can_run_after_array_system_passes(self, base_state):
        """COMPLEX_PHYSICS can run when ARRAY_SYSTEM is completed."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", dependencies=["stage_1"]),
            make_stage("stage_3", "COMPLEX_PHYSICS", dependencies=["stage_2"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", status="completed_success", dependencies=["stage_1"]),
            make_stage("stage_3", "COMPLEX_PHYSICS", status="not_started", dependencies=["stage_2"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_3"
        assert result["current_stage_type"] == "COMPLEX_PHYSICS"


class TestBlockedStageUnblocking:
    """Tests for blocked stage unblocking logic."""

    def test_blocked_stage_unblocked_when_deps_satisfied(self, base_state):
        """Blocked stage should be unblocked when dependencies become satisfied."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="blocked", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # stage_1 should be unblocked and selected
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_blocked_stage_stays_blocked_without_deps(self, base_state):
        """Blocked stage without dependencies (blocked for other reason) should stay blocked."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
        ]
        progress_stages = [
            # Blocked for budget or other non-dependency reason
            make_stage("stage_0", "MATERIAL_VALIDATION", status="blocked", dependencies=[]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Deadlock - no runnable stages
        assert result.get("ask_user_trigger") == "deadlock_detected"

    def test_blocked_stage_stays_blocked_with_failed_deps(self, base_state):
        """Blocked stage with failed dependencies should stay blocked."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_failed", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="blocked", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Both stages are permanently blocked/failed
        assert result.get("ask_user_trigger") == "deadlock_detected"


class TestMissingOrUnknownStageType:
    """Tests for missing and unknown stage_type handling."""

    def test_stage_without_stage_type_is_blocked(self, base_state):
        """Stage without stage_type should be blocked."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            {"stage_id": "stage_1", "targets": [], "dependencies": ["stage_0"]},  # No stage_type
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            {"stage_id": "stage_1", "status": "not_started", "dependencies": ["stage_0"]},  # No stage_type
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Stage without stage_type should be skipped/blocked
        # Result depends on whether any other valid stage exists
        # If no valid stages, it should detect deadlock
        assert result.get("current_stage_id") is None or result.get("ask_user_trigger") is not None

    def test_unknown_stage_type_is_blocked(self, base_state):
        """Stage with unknown stage_type should be blocked."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "UNKNOWN_TYPE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "UNKNOWN_TYPE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Unknown stage type should be blocked, not selected
        assert result.get("current_stage_id") != "stage_1" or result.get("ask_user_trigger") is not None


class TestEmptyPlanAndProgress:
    """Tests for empty plan and progress handling."""

    def test_empty_plan_and_progress_triggers_no_stages_available(self, base_state):
        """Empty plan and progress should trigger no_stages_available error."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {}
        base_state["progress"] = {}

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "no_stages_available"
        assert result.get("awaiting_user_input") is True
        assert result.get("current_stage_id") is None
        assert result.get("current_stage_type") is None
        questions = result.get("pending_user_questions", [])
        assert len(questions) > 0

    def test_plan_with_empty_stages_triggers_no_stages_available(self, base_state):
        """Plan with empty stages list should trigger no_stages_available error."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {"stages": []}
        base_state["progress"] = {"stages": []}

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "no_stages_available"
        assert result.get("current_stage_id") is None

    def test_none_plan_handled_gracefully(self, base_state):
        """None plan should be handled gracefully."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = None
        base_state["progress"] = None

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "no_stages_available"


class TestMissingDependencies:
    """Tests for missing dependency handling."""

    def test_stage_with_missing_dependency_id_is_blocked(self, base_state):
        """Stage referencing non-existent dependency should be blocked."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),  # stage_0 doesn't exist
        ]
        progress_stages = [
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Stage with missing dependency should be blocked
        assert result.get("current_stage_id") != "stage_1" or result.get("ask_user_trigger") is not None

    def test_none_dependencies_treated_as_empty(self, base_state):
        """None dependencies should be treated as empty list."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION"),
        ]
        progress_stages = [
            {
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "status": "not_started",
                "dependencies": None,  # Explicitly None
            }
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Should be selected despite None dependencies
        assert result["current_stage_id"] == "stage_0"


class TestStatusHandling:
    """Tests for various stage status handling."""

    def test_in_progress_stage_is_skipped(self, base_state):
        """in_progress stages should be skipped."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="in_progress", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Neither should be selected: stage_0 is in_progress, stage_1's dep isn't satisfied
        assert result.get("current_stage_id") is None or result.get("ask_user_trigger") is not None

    def test_invalidated_stage_is_skipped(self, base_state):
        """invalidated stages should be skipped."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="invalidated", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # stage_0 is invalidated (skipped), stage_1's dep isn't satisfied
        # Should detect deadlock or no runnable stages
        assert result.get("current_stage_id") is None or result.get("ask_user_trigger") is not None

    def test_completed_partial_allows_dependent_stage(self, base_state):
        """completed_partial should allow dependent stages to run."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_partial", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # completed_partial counts as "satisfied" for dependencies
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"

    def test_completed_failed_blocks_dependent_stage(self, base_state):
        """completed_failed should block dependent stages."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_failed", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # stage_1 cannot run because its dependency failed
        # Should detect deadlock
        assert result.get("ask_user_trigger") == "deadlock_detected"


class TestStageTypeOrderEnforcement:
    """Tests for stage type order enforcement."""

    def test_higher_type_waits_for_lower_type(self, base_state):
        """Higher stage types must wait for lower types to complete."""
        from src.agents.stage_selection import select_stage_node

        # Plan with both MATERIAL_VALIDATION and SINGLE_STRUCTURE
        # Even without explicit dependencies, SINGLE_STRUCTURE should wait
        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=[]),  # No explicit dep
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=[]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # MATERIAL_VALIDATION (lower in order) should be selected first
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"

    def test_array_system_waits_for_single_structure(self, base_state):
        """ARRAY_SYSTEM must wait for SINGLE_STRUCTURE to complete."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", dependencies=[]),  # No explicit dep on stage_1
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", status="not_started", dependencies=[]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Even without explicit dep, ARRAY_SYSTEM should wait for SINGLE_STRUCTURE
        assert result["current_stage_id"] == "stage_1"
        assert result["current_stage_type"] == "SINGLE_STRUCTURE"


class TestAllStagesCompleted:
    """Tests for completion scenarios."""

    def test_all_stages_completed_returns_none(self, base_state):
        """When all stages are completed, current_stage_id should be None."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # All stages completed - no more stages to select
        assert result.get("current_stage_id") is None
        assert result.get("current_stage_type") is None
        # Should NOT trigger deadlock or error
        assert result.get("ask_user_trigger") is None
        assert result.get("awaiting_user_input") is None or result.get("awaiting_user_input") is False

    def test_mix_of_completed_success_and_partial(self, base_state):
        """Mix of completed_success and completed_partial should still complete."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_partial", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # All stages completed (partial or success)
        assert result.get("current_stage_id") is None
        assert result.get("ask_user_trigger") is None


class TestCounterResetBehavior:
    """Tests for counter reset behavior in various scenarios."""

    def test_counters_not_reset_when_same_stage_selected(self, base_state):
        """Counters should NOT be reset if same stage is selected again (not needs_rerun)."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[])]
        )
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=[])]
        )
        base_state["current_stage_id"] = "stage_0"  # Same stage already selected
        base_state["design_revision_count"] = 3
        base_state["code_revision_count"] = 2

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"
        # Since it's the same stage and not needs_rerun, counters may or may not reset
        # depending on implementation - let's verify the behavior
        # According to the code: reset_counters = (selected_stage_id != current_stage_id) or (status == "needs_rerun")
        # For not_started same stage, counters should NOT be reset
        # Note: The result only includes counters if reset_counters is True
        assert "design_revision_count" not in result or result["design_revision_count"] == 3

    def test_counters_reset_when_different_stage_selected(self, base_state):
        """Counters should be reset when a different stage is selected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)
        base_state["current_stage_id"] = "stage_0"  # Previous stage
        base_state["design_revision_count"] = 5
        base_state["code_revision_count"] = 4

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_1"  # Different stage
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0

    def test_feedback_fields_cleared_on_stage_transition(self, base_state):
        """Feedback fields should be cleared when transitioning to a new stage.
        
        This tests the bug fix for issue 4: feedback from Stage N should not
        leak into Stage N+1. All feedback fields must be explicitly cleared
        to None when selecting a different stage.
        """
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)
        base_state["current_stage_id"] = "stage_0"  # Previous stage
        
        # Simulate stale feedback from Stage 0
        base_state["reviewer_feedback"] = "Old feedback from stage 0 code review"
        base_state["physics_feedback"] = "Old physics validation feedback"
        base_state["execution_feedback"] = "Old execution error message"
        base_state["analysis_feedback"] = "Old analysis revision feedback"
        base_state["design_feedback"] = "Old design flaw feedback"
        base_state["comparison_feedback"] = "Old comparison feedback"

        result = select_stage_node(base_state)
        
        # Verify we selected the new stage
        assert result["current_stage_id"] == "stage_1"
        
        # Critical: ALL feedback fields must be cleared to None
        # If any of these are not None, feedback from Stage 0 could confuse
        # agents working on Stage 1
        assert result.get("reviewer_feedback") is None, (
            f"reviewer_feedback should be None, got: {result.get('reviewer_feedback')!r}"
        )
        assert result.get("physics_feedback") is None, (
            f"physics_feedback should be None, got: {result.get('physics_feedback')!r}"
        )
        assert result.get("execution_feedback") is None, (
            f"execution_feedback should be None, got: {result.get('execution_feedback')!r}"
        )
        assert result.get("analysis_feedback") is None, (
            f"analysis_feedback should be None, got: {result.get('analysis_feedback')!r}"
        )
        assert result.get("design_feedback") is None, (
            f"design_feedback should be None, got: {result.get('design_feedback')!r}"
        )
        assert result.get("comparison_feedback") is None, (
            f"comparison_feedback should be None, got: {result.get('comparison_feedback')!r}"
        )

    def test_planner_feedback_preserved_on_stage_transition(self, base_state):
        """Planner feedback should NOT be cleared on stage transitions.
        
        Unlike other feedback fields, planner_feedback is used for replanning
        which can span multiple stages. It should be preserved during stage
        transitions so the planner can reference it if needed.
        """
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)
        base_state["current_stage_id"] = "stage_0"
        
        # Set planner feedback that should be preserved
        base_state["planner_feedback"] = "Important feedback for replanning"
        base_state["supervisor_feedback"] = "Supervisor decision context"

        result = select_stage_node(base_state)
        
        assert result["current_stage_id"] == "stage_1"
        
        # planner_feedback and supervisor_feedback should NOT be in result
        # (meaning they are preserved from state, not overwritten)
        assert "planner_feedback" not in result, (
            "planner_feedback should not be cleared - it's used for replanning"
        )
        assert "supervisor_feedback" not in result, (
            "supervisor_feedback should not be cleared - it's used for supervision"
        )


class TestProgressInitialization:
    """Tests for progress initialization from plan."""

    def test_progress_initialized_from_plan_on_demand(self, base_state):
        """Progress should be initialized from plan if not present."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[], targets=["Fig1"])]
        )
        # Progress exists but has no stages
        base_state["progress"] = {"stages": []}

        result = select_stage_node(base_state)
        # The initialization should happen and a stage should be selected
        # OR it should return an error about initialization
        # Based on code: empty stages after init attempt triggers no_stages_available
        assert result.get("current_stage_id") == "stage_0" or result.get("ask_user_trigger") is not None

    def test_progress_missing_stages_key_handled(self, base_state):
        """Missing stages key in progress should be handled."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[])]
        )
        base_state["progress"] = {}  # No stages key

        # This should trigger initialization from plan
        result = select_stage_node(base_state)
        # Should either initialize and select, or return appropriate error
        assert result.get("current_stage_id") is not None or result.get("ask_user_trigger") is not None


class TestEdgeCasesWithMultipleStages:
    """Tests for edge cases with multiple stages."""

    def test_multiple_not_started_stages_selects_first(self, base_state):
        """First eligible not_started stage should be selected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "MATERIAL_VALIDATION", dependencies=[]),  # Also mat val
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=[]),
            make_stage("stage_1", "MATERIAL_VALIDATION", status="not_started", dependencies=[]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # First stage should be selected (stage_0)
        assert result["current_stage_id"] == "stage_0"

    def test_multiple_needs_rerun_selects_first(self, base_state):
        """First eligible needs_rerun stage should be selected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "MATERIAL_VALIDATION", dependencies=[]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="needs_rerun", dependencies=[]),
            make_stage("stage_1", "MATERIAL_VALIDATION", status="needs_rerun", dependencies=[]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # First needs_rerun stage should be selected
        assert result["current_stage_id"] == "stage_0"

    def test_chain_of_dependencies(self, base_state):
        """Chain of dependencies should be respected."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", dependencies=["stage_1"]),
            make_stage("stage_3", "PARAMETER_SWEEP", dependencies=["stage_2"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success", dependencies=["stage_0"]),
            make_stage("stage_2", "ARRAY_SYSTEM", status="not_started", dependencies=["stage_1"]),
            make_stage("stage_3", "PARAMETER_SWEEP", status="not_started", dependencies=["stage_2"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # stage_2 should be selected (next in chain)
        assert result["current_stage_id"] == "stage_2"
        assert result["current_stage_type"] == "ARRAY_SYSTEM"


class TestDeadlockScenarios:
    """Tests for various deadlock scenarios."""

    def test_all_stages_blocked_triggers_deadlock(self, base_state):
        """All stages blocked should trigger deadlock."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="blocked", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="blocked", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "deadlock_detected"
        questions = result.get("pending_user_questions", [])
        assert len(questions) > 0

    def test_mix_of_failed_and_blocked_triggers_deadlock(self, base_state):
        """Mix of completed_failed and blocked stages should trigger deadlock."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_failed", dependencies=[]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="blocked", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "deadlock_detected"

    def test_circular_dependencies_cause_deadlock(self, base_state):
        """Circular dependencies should cause deadlock (no stage can run)."""
        from src.agents.stage_selection import select_stage_node

        plan_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=["stage_1"]),
            make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
        ]
        progress_stages = [
            make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=["stage_1"]),
            make_stage("stage_1", "SINGLE_STRUCTURE", status="not_started", dependencies=["stage_0"]),
        ]
        base_state["plan"] = make_plan(plan_stages)
        base_state["progress"] = make_progress(progress_stages)

        result = select_stage_node(base_state)
        # Both stages have unsatisfied deps - should detect no runnable stages
        # This may trigger deadlock or just return None stage
        assert result.get("current_stage_id") is None


class TestWorkflowPhaseOutput:
    """Tests for workflow_phase output."""

    def test_workflow_phase_always_set(self, base_state):
        """workflow_phase should always be set to 'stage_selection'."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", dependencies=[])]
        )
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="not_started", dependencies=[])]
        )

        result = select_stage_node(base_state)
        assert result.get("workflow_phase") == "stage_selection"

    def test_workflow_phase_set_on_error(self, base_state):
        """workflow_phase should be set even on error paths."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {}
        base_state["progress"] = {}

        result = select_stage_node(base_state)
        assert result.get("workflow_phase") == "stage_selection"

    def test_workflow_phase_set_on_deadlock(self, base_state):
        """workflow_phase should be set on deadlock."""
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = make_plan(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_failed")]
        )
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_failed", dependencies=[])]
        )

        result = select_stage_node(base_state)
        assert result.get("workflow_phase") == "stage_selection"
