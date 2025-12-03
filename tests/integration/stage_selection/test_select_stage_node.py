"""Integration tests for stage selection edge cases."""

from unittest.mock import patch

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
        assert result["current_stage_id"] == "stage_0"

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
        assert result["current_stage_id"] == "stage_1"

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
        assert result["current_stage_id"] == "stage_0"

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

        result = select_stage_node(base_state)
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0

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
        questions = result.get("pending_user_questions", [])
        assert questions and "Failed to initialize progress" in questions[0]

