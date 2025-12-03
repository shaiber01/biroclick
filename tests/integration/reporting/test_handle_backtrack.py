"""Integration tests for handle_backtrack_node behavior."""

from tests.integration.helpers.state_factories import make_plan, make_progress, make_stage


class TestHandleBacktrackNode:
    """Integration checks for handle_backtrack_node."""

    def test_backtrack_marks_target_as_needs_rerun(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0",
                    "MATERIAL_VALIDATION",
                    status="completed_success",
                    outputs=["some_output"],
                    discrepancies=["some_discrepancy"],
                )
            ]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
            "reason": "Need to revalidate materials",
        }

        result = handle_backtrack_node(base_state)
        progress = result.get("progress", {})
        stages = progress.get("stages", [])
        target_stage = next((s for s in stages if s["stage_id"] == "stage_0"), None)

        assert target_stage is not None
        assert target_stage["status"] == "needs_rerun"
        assert target_stage["outputs"] == []
        assert target_stage["discrepancies"] == []
        assert result["current_stage_id"] == "stage_0"
        assert result["backtrack_decision"] is None

    def test_backtrack_invalidates_dependent_stages(self, base_state):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = make_plan(
            [
                make_stage("stage_0", "MATERIAL_VALIDATION"),
                make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
            ]
        )
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0", "MATERIAL_VALIDATION", status="completed_success"
                ),
                make_stage(
                    "stage_1", "SINGLE_STRUCTURE", status="completed_success"
                ),
            ]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": ["stage_1"],
            "reason": "Material validation needs rerun",
        }

        result = handle_backtrack_node(base_state)
        stages = result.get("progress", {}).get("stages", [])

        stage_1 = next((s for s in stages if s["stage_id"] == "stage_1"), None)
        assert stage_1 is not None
        assert stage_1["status"] == "invalidated"
        assert result["invalidated_stages"] == ["stage_1"]

    def test_backtrack_increments_counter(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 0
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 1

        base_state["backtrack_count"] = 5
        base_state["runtime_config"] = {"max_backtracks": 10}
        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 6

    def test_backtrack_clears_working_state(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        base_state["code"] = "print('old code')"
        base_state["design_description"] = {"old": "design"}
        base_state["stage_outputs"] = {"files": ["/old/file.csv"]}
        base_state["run_error"] = "Some error"
        base_state["analysis_summary"] = "Some summary"
        base_state["supervisor_verdict"] = "approve"
        base_state["last_design_review_verdict"] = "approved"
        base_state["last_code_review_verdict"] = "approved"

        result = handle_backtrack_node(base_state)

        assert result.get("code") is None
        assert result.get("design_description") is None
        assert result.get("stage_outputs") == {}
        assert result.get("run_error") is None
        assert result.get("analysis_summary") is None
        assert result.get("supervisor_verdict") is None
        assert result.get("last_design_review_verdict") is None
        assert result.get("last_code_review_verdict") is None
        assert result.get("workflow_phase") == "backtracking"

    def test_backtrack_rejects_missing_decision(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress([])
        base_state["backtrack_decision"] = None

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result

        base_state["backtrack_decision"] = {"accepted": False}
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision"

    def test_backtrack_rejects_empty_target(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_target"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_target_not_found(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_999",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "backtrack_target_not_found"
        assert result.get("awaiting_user_input") is True
        assert any("stage_999" in q for q in result.get("pending_user_questions", []))

    def test_backtrack_respects_max_limit(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 2
        base_state["runtime_config"] = {"max_backtracks": 2}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "backtrack_limit"
        assert result.get("workflow_phase") == "backtracking_limit"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_to_material_validation_clears_materials(
        self, base_state, valid_plan
    ):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0", "MATERIAL_VALIDATION", status="completed_success"
                )
            ]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        base_state["validated_materials"] = ["Gold"]
        base_state["pending_validated_materials"] = ["Silver"]

        result = handle_backtrack_node(base_state)

        assert result.get("validated_materials") == []
        assert result.get("pending_validated_materials") == []

    def test_backtrack_to_non_material_preserves_materials(
        self, base_state, valid_plan
    ):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_1", "SINGLE_STRUCTURE", status="completed_success"
                )
            ]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_1",
            "stages_to_invalidate": [],
        }
        materials = ["Gold"]
        base_state["validated_materials"] = materials

        result = handle_backtrack_node(base_state)

        assert "validated_materials" not in result

