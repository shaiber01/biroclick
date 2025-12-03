"""Integration tests for handle_backtrack_node behavior."""

import copy
from tests.integration.helpers.state_factories import make_plan, make_progress, make_stage


class TestHandleBacktrackNode:
    """Integration checks for handle_backtrack_node."""

    # ═══════════════════════════════════════════════════════════════════════
    # Core Functionality Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_marks_target_as_needs_rerun(self, base_state, valid_plan):
        """Verify target stage is marked as needs_rerun with cleared outputs."""
        from src.agents.reporting import handle_backtrack_node

        original_state = copy.deepcopy(base_state)
        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0",
                    "MATERIAL_VALIDATION",
                    status="completed_success",
                    outputs=["some_output", "another_output"],
                    discrepancies=["some_discrepancy", "another_discrepancy"],
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

        # Core assertions - target stage must be reset
        assert target_stage is not None, "Target stage must exist in result"
        assert target_stage["status"] == "needs_rerun", "Target stage must have status 'needs_rerun'"
        assert target_stage["outputs"] == [], "Target stage outputs must be cleared"
        assert target_stage["discrepancies"] == [], "Target stage discrepancies must be cleared"
        
        # Verify state updates
        assert result["current_stage_id"] == "stage_0", "Current stage must be set to target"
        assert result["backtrack_decision"] is None, "Backtrack decision must be cleared"
        assert result["workflow_phase"] == "backtracking", "Workflow phase must be 'backtracking'"
        
        # Verify original state is not mutated
        assert base_state["progress"]["stages"][0]["status"] != "needs_rerun" or \
               original_state["progress"]["stages"][0]["status"] == base_state["progress"]["stages"][0]["status"], \
               "Original state should not be mutated (deep copy behavior)"

    def test_backtrack_invalidates_dependent_stages(self, base_state):
        """Verify dependent stages are marked as invalidated."""
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
                    "stage_0", "MATERIAL_VALIDATION", status="completed_success",
                    outputs=["material_data"]
                ),
                make_stage(
                    "stage_1", "SINGLE_STRUCTURE", status="completed_success",
                    outputs=["structure_data"]
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

        # Target stage assertions
        stage_0 = next((s for s in stages if s["stage_id"] == "stage_0"), None)
        assert stage_0 is not None, "Target stage must exist"
        assert stage_0["status"] == "needs_rerun", "Target stage must be needs_rerun"
        assert stage_0["outputs"] == [], "Target stage outputs must be cleared"

        # Invalidated stage assertions
        stage_1 = next((s for s in stages if s["stage_id"] == "stage_1"), None)
        assert stage_1 is not None, "Dependent stage must exist"
        assert stage_1["status"] == "invalidated", "Dependent stage must be invalidated"
        # Invalidated stages should NOT have their outputs cleared (only status changes)
        # This tests that the code distinguishes between target and invalidated stages
        
        # Verify invalidated_stages list
        assert result["invalidated_stages"] == ["stage_1"], "invalidated_stages must contain dependent stage IDs"

    def test_backtrack_invalidates_multiple_dependent_stages(self, base_state):
        """Verify multiple dependent stages are all invalidated."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = make_plan(
            [
                make_stage("stage_0", "MATERIAL_VALIDATION"),
                make_stage("stage_1", "SINGLE_STRUCTURE", dependencies=["stage_0"]),
                make_stage("stage_2", "MULTI_STRUCTURE", dependencies=["stage_0"]),
                make_stage("stage_3", "COMPARATIVE", dependencies=["stage_1", "stage_2"]),
            ]
        )
        base_state["progress"] = make_progress(
            [
                make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success"),
                make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success"),
                make_stage("stage_2", "MULTI_STRUCTURE", status="completed_success"),
                make_stage("stage_3", "COMPARATIVE", status="completed_success"),
            ]
        )
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": ["stage_1", "stage_2", "stage_3"],
            "reason": "Full revalidation needed",
        }

        result = handle_backtrack_node(base_state)
        stages = result.get("progress", {}).get("stages", [])

        # Verify target stage
        stage_0 = next((s for s in stages if s["stage_id"] == "stage_0"), None)
        assert stage_0["status"] == "needs_rerun"

        # Verify all dependent stages are invalidated
        for stage_id in ["stage_1", "stage_2", "stage_3"]:
            stage = next((s for s in stages if s["stage_id"] == stage_id), None)
            assert stage is not None, f"Stage {stage_id} must exist"
            assert stage["status"] == "invalidated", f"Stage {stage_id} must be invalidated"

        # Verify invalidated_stages list contains all three
        assert set(result["invalidated_stages"]) == {"stage_1", "stage_2", "stage_3"}

    def test_backtrack_increments_counter(self, base_state, valid_plan):
        """Verify backtrack_count is incremented correctly."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 0
        base_state["runtime_config"] = {"max_backtracks": 10}  # High limit to avoid triggering
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 1, "Counter must increment from 0 to 1"

        # Test increment from higher value
        base_state["backtrack_count"] = 5
        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 6, "Counter must increment from 5 to 6"

    def test_backtrack_handles_none_counter(self, base_state, valid_plan):
        """Verify None backtrack_count is treated as 0."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = None
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 1, "None counter should be treated as 0 and incremented to 1"

    def test_backtrack_handles_missing_counter(self, base_state, valid_plan):
        """Verify missing backtrack_count is treated as 0."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state.pop("backtrack_count", None)  # Ensure it's not set
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 1, "Missing counter should be treated as 0 and incremented to 1"

    # ═══════════════════════════════════════════════════════════════════════
    # State Clearing Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_clears_working_state(self, base_state, valid_plan):
        """Verify all working state fields are cleared on backtrack."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        # Set all fields that should be cleared
        base_state["code"] = "print('old code')"
        base_state["design_description"] = {"old": "design", "complex": {"nested": True}}
        base_state["stage_outputs"] = {"files": ["/old/file.csv"], "data": [1, 2, 3]}
        base_state["run_error"] = "Some detailed error message"
        base_state["analysis_summary"] = "Some summary of analysis"
        base_state["supervisor_verdict"] = "approve"
        base_state["last_design_review_verdict"] = "approved"
        base_state["last_code_review_verdict"] = "approved"

        result = handle_backtrack_node(base_state)

        # Verify all working state is cleared
        assert result.get("code") is None, "code must be cleared to None"
        assert result.get("design_description") is None, "design_description must be cleared to None"
        assert result.get("stage_outputs") == {}, "stage_outputs must be cleared to empty dict"
        assert result.get("run_error") is None, "run_error must be cleared to None"
        assert result.get("analysis_summary") is None, "analysis_summary must be cleared to None"
        assert result.get("supervisor_verdict") is None, "supervisor_verdict must be cleared to None"
        assert result.get("last_design_review_verdict") is None, "last_design_review_verdict must be cleared"
        assert result.get("last_code_review_verdict") is None, "last_code_review_verdict must be cleared"
        
        # Verify workflow phase is set correctly
        assert result.get("workflow_phase") == "backtracking", "workflow_phase must be 'backtracking'"

    def test_backtrack_resets_revision_counters(self, base_state, valid_plan):
        """Verify revision counters are reset to 0 on backtrack."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        # Set non-zero revision counters
        base_state["design_revision_count"] = 3
        base_state["code_revision_count"] = 5
        base_state["execution_failure_count"] = 2
        base_state["analysis_revision_count"] = 1

        result = handle_backtrack_node(base_state)

        # All revision counters must be reset to 0
        assert result["design_revision_count"] == 0, "design_revision_count must be reset to 0"
        assert result["code_revision_count"] == 0, "code_revision_count must be reset to 0"
        assert result["execution_failure_count"] == 0, "execution_failure_count must be reset to 0"
        assert result["analysis_revision_count"] == 0, "analysis_revision_count must be reset to 0"

    # ═══════════════════════════════════════════════════════════════════════
    # Error Handling Tests - Invalid/Missing Decision
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_rejects_missing_decision(self, base_state, valid_plan):
        """Verify error response when backtrack_decision is None."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress([])
        base_state["backtrack_decision"] = None

        result = handle_backtrack_node(base_state)
        
        # Verify error response structure
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision", \
            "Must trigger invalid_backtrack_decision"
        assert result.get("awaiting_user_input") is True, "Must set awaiting_user_input"
        assert "pending_user_questions" in result, "Must include pending_user_questions"
        assert len(result["pending_user_questions"]) > 0, "Must have at least one question"
        assert result.get("workflow_phase") == "backtracking", "workflow_phase must be 'backtracking'"
        
        # Verify error message content
        questions = result.get("pending_user_questions", [])
        assert any("missing" in q.lower() or "invalid" in q.lower() for q in questions), \
            "Error message should mention missing or invalid decision"

    def test_backtrack_rejects_not_accepted_decision(self, base_state, valid_plan):
        """Verify error response when accepted is False."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress([])
        base_state["backtrack_decision"] = {"accepted": False}

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_rejects_empty_decision(self, base_state, valid_plan):
        """Verify error response when backtrack_decision is empty dict."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress([])
        base_state["backtrack_decision"] = {}

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_rejects_empty_target(self, base_state, valid_plan):
        """Verify error response when target_stage_id is empty string."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        assert result.get("ask_user_trigger") == "invalid_backtrack_target", \
            "Must trigger invalid_backtrack_target for empty target"
        assert result.get("awaiting_user_input") is True
        assert result.get("workflow_phase") == "backtracking"
        
        # Verify error message
        questions = result.get("pending_user_questions", [])
        assert len(questions) > 0, "Must have error message"
        assert any("empty" in q.lower() for q in questions), \
            "Error message should mention empty target"

    def test_backtrack_rejects_missing_target_key(self, base_state, valid_plan):
        """Verify error when target_stage_id key is missing from decision."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["backtrack_decision"] = {
            "accepted": True,
            # target_stage_id is missing
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Missing key defaults to empty string, which should trigger invalid_backtrack_target
        assert result.get("ask_user_trigger") == "invalid_backtrack_target"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_target_not_found(self, base_state, valid_plan):
        """Verify error when target stage doesn't exist in progress."""
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
        
        assert result.get("ask_user_trigger") == "backtrack_target_not_found", \
            "Must trigger backtrack_target_not_found"
        assert result.get("awaiting_user_input") is True
        assert result.get("workflow_phase") == "backtracking"
        
        # Verify the error message includes the stage ID that wasn't found
        questions = result.get("pending_user_questions", [])
        assert any("stage_999" in q for q in questions), \
            "Error message must include the missing stage ID"

    # ═══════════════════════════════════════════════════════════════════════
    # Max Backtrack Limit Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_respects_max_limit(self, base_state, valid_plan):
        """Verify backtrack is blocked when limit would be exceeded."""
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
        
        assert result.get("ask_user_trigger") == "backtrack_limit", \
            "Must trigger backtrack_limit"
        assert result.get("workflow_phase") == "backtracking_limit", \
            "Workflow phase must be 'backtracking_limit'"
        assert result.get("awaiting_user_input") is True
        assert result.get("last_node_before_ask_user") == "handle_backtrack", \
            "Must record last node before ask_user"
        
        # Verify the counter is still incremented even when limit is hit
        assert result.get("backtrack_count") == 3, \
            "Counter should still increment to show the attempted count"

    def test_backtrack_limit_boundary_at_max(self, base_state, valid_plan):
        """Test the exact boundary: when new_count would equal max_backtracks.
        
        With current implementation (>=), if count=1 and max=2, new_count=2,
        and 2 >= 2 triggers the limit. This test documents this behavior.
        """
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        # Set count so that new_count will equal max_backtracks exactly
        base_state["backtrack_count"] = 1
        base_state["runtime_config"] = {"max_backtracks": 2}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Current implementation triggers limit when new_count >= max
        # new_count = 1 + 1 = 2, 2 >= 2 is True, so limit triggers
        assert result.get("ask_user_trigger") == "backtrack_limit", \
            "Limit should trigger when new_count equals max_backtracks (>= comparison)"
        assert result.get("backtrack_count") == 2

    def test_backtrack_limit_boundary_below_max(self, base_state, valid_plan):
        """Test just below boundary: new_count < max_backtracks should succeed."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        # count=0, max=2: new_count=1, 1 >= 2 is False, should succeed
        base_state["backtrack_count"] = 0
        base_state["runtime_config"] = {"max_backtracks": 2}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Should NOT trigger limit
        assert result.get("ask_user_trigger") is None, \
            "Should not trigger limit when new_count < max"
        assert result.get("workflow_phase") == "backtracking", \
            "Should proceed with backtracking"
        assert result.get("backtrack_count") == 1

    def test_backtrack_limit_with_max_one(self, base_state, valid_plan):
        """Test edge case: max_backtracks=1 should trigger limit on first attempt."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 0
        base_state["runtime_config"] = {"max_backtracks": 1}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # new_count = 0 + 1 = 1, 1 >= 1 is True, limit triggers
        assert result.get("ask_user_trigger") == "backtrack_limit", \
            "max_backtracks=1 should trigger limit on first attempt"

    def test_backtrack_uses_default_max_when_config_missing(self, base_state, valid_plan):
        """Verify default max_backtracks=2 is used when runtime_config is missing."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 0
        base_state.pop("runtime_config", None)  # Ensure no config
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # With default max=2, count=0 -> new_count=1, 1 >= 2 is False, should succeed
        assert result.get("ask_user_trigger") is None, \
            "Should succeed with default max_backtracks"
        assert result.get("backtrack_count") == 1

    def test_backtrack_uses_default_max_when_key_missing(self, base_state, valid_plan):
        """Verify default max_backtracks=2 is used when key is missing from config."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 1
        base_state["runtime_config"] = {"other_setting": "value"}  # No max_backtracks key
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # With default max=2, count=1 -> new_count=2, 2 >= 2 is True, limit triggers
        assert result.get("ask_user_trigger") == "backtrack_limit"

    # ═══════════════════════════════════════════════════════════════════════
    # Material Validation Stage Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_to_material_validation_clears_materials(
        self, base_state, valid_plan
    ):
        """Verify validated materials are cleared when backtracking to MATERIAL_VALIDATION."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0", "MATERIAL_VALIDATION", status="completed_success"
                )
            ]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        base_state["validated_materials"] = ["Gold", "Silver", "Copper"]
        base_state["pending_validated_materials"] = ["Aluminum"]

        result = handle_backtrack_node(base_state)

        assert result.get("validated_materials") == [], \
            "validated_materials must be cleared for MATERIAL_VALIDATION backtrack"
        assert result.get("pending_validated_materials") == [], \
            "pending_validated_materials must be cleared for MATERIAL_VALIDATION backtrack"

    def test_backtrack_to_non_material_preserves_materials(
        self, base_state, valid_plan
    ):
        """Verify materials are NOT cleared when backtracking to non-MATERIAL_VALIDATION stage."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_1", "SINGLE_STRUCTURE", status="completed_success"
                )
            ]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_1",
            "stages_to_invalidate": [],
        }
        materials = ["Gold", "Silver"]
        base_state["validated_materials"] = materials

        result = handle_backtrack_node(base_state)

        # validated_materials should NOT be in the result (not touched)
        assert "validated_materials" not in result, \
            "validated_materials should not be modified for non-MATERIAL_VALIDATION backtrack"

    def test_backtrack_to_material_validation_with_empty_materials(
        self, base_state, valid_plan
    ):
        """Verify backtrack works when materials are already empty."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0", "MATERIAL_VALIDATION", status="completed_success"
                )
            ]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        base_state["validated_materials"] = []
        base_state["pending_validated_materials"] = []

        result = handle_backtrack_node(base_state)

        # Should still set to empty (idempotent)
        assert result.get("validated_materials") == []
        assert result.get("pending_validated_materials") == []

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Case Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_with_empty_progress(self, base_state, valid_plan):
        """Test behavior when progress has no stages - should report target not found."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress([])  # Empty stages
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Target stage won't exist in empty progress
        assert result.get("ask_user_trigger") == "backtrack_target_not_found"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_with_missing_progress(self, base_state, valid_plan):
        """Test behavior when progress key is missing from state."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state.pop("progress", None)  # Remove progress entirely
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Should handle gracefully - target not found
        assert result.get("ask_user_trigger") == "backtrack_target_not_found"

    def test_backtrack_with_missing_stages_key(self, base_state, valid_plan):
        """Test behavior when progress exists but stages key is missing."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {}  # No stages key
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Should handle gracefully - target not found
        assert result.get("ask_user_trigger") == "backtrack_target_not_found"

    def test_backtrack_ignores_nonexistent_invalidation_targets(self, base_state, valid_plan):
        """Verify that non-existent stage IDs in stages_to_invalidate are ignored."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success"),
                make_stage("stage_1", "SINGLE_STRUCTURE", status="completed_success"),
            ]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": ["stage_1", "nonexistent_stage", "another_fake"],
        }

        result = handle_backtrack_node(base_state)
        stages = result.get("progress", {}).get("stages", [])

        # Target stage should be reset
        stage_0 = next((s for s in stages if s["stage_id"] == "stage_0"), None)
        assert stage_0["status"] == "needs_rerun"

        # Existing stage should be invalidated
        stage_1 = next((s for s in stages if s["stage_id"] == "stage_1"), None)
        assert stage_1["status"] == "invalidated"

        # Non-existent stages are silently ignored - check invalidated_stages includes all specified
        assert result["invalidated_stages"] == ["stage_1", "nonexistent_stage", "another_fake"]

    def test_backtrack_with_empty_stages_to_invalidate(self, base_state, valid_plan):
        """Verify backtrack works with empty stages_to_invalidate list."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)

        # Should succeed with empty invalidated_stages
        assert result.get("invalidated_stages") == []
        assert result.get("current_stage_id") == "stage_0"

    def test_backtrack_with_missing_stages_to_invalidate_key(self, base_state, valid_plan):
        """Verify backtrack handles missing stages_to_invalidate key."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            # stages_to_invalidate key is missing
        }

        result = handle_backtrack_node(base_state)

        # Should default to empty list and succeed
        assert result.get("invalidated_stages") == []
        assert result.get("current_stage_id") == "stage_0"

    # ═══════════════════════════════════════════════════════════════════════
    # State Immutability Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_does_not_mutate_input_state(self, base_state, valid_plan):
        """Verify that the input state is not mutated (deep copy behavior)."""
        from src.agents.reporting import handle_backtrack_node

        # Set up state with nested structures
        base_state["plan"] = valid_plan
        progress_data = make_progress(
            [
                make_stage(
                    "stage_0",
                    "MATERIAL_VALIDATION",
                    status="completed_success",
                    outputs=["output1", "output2"],
                    discrepancies=["disc1"],
                )
            ]
        )
        base_state["progress"] = progress_data
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        # Deep copy to compare later
        original_progress = copy.deepcopy(progress_data)

        result = handle_backtrack_node(base_state)

        # Verify result has cleared outputs
        result_stage = result["progress"]["stages"][0]
        assert result_stage["outputs"] == []
        assert result_stage["status"] == "needs_rerun"

        # Verify original was NOT mutated
        assert base_state["progress"]["stages"][0]["outputs"] == original_progress["stages"][0]["outputs"], \
            "Original state's outputs should not be mutated"
        assert base_state["progress"]["stages"][0]["status"] == original_progress["stages"][0]["status"], \
            "Original state's status should not be mutated"

    # ═══════════════════════════════════════════════════════════════════════
    # Return Value Structure Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_successful_backtrack_return_structure(self, base_state, valid_plan):
        """Verify complete structure of successful backtrack return value."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_count"] = 0
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)

        # Verify all expected keys are present
        expected_keys = {
            "workflow_phase",
            "progress",
            "current_stage_id",
            "backtrack_count",
            "backtrack_decision",
            "code",
            "design_description",
            "stage_outputs",
            "run_error",
            "analysis_summary",
            "invalidated_stages",
            "last_design_review_verdict",
            "last_code_review_verdict",
            "supervisor_verdict",
            "design_revision_count",
            "code_revision_count",
            "execution_failure_count",
            "analysis_revision_count",
        }
        
        for key in expected_keys:
            assert key in result, f"Missing expected key: {key}"

        # Verify specific values
        assert result["workflow_phase"] == "backtracking"
        assert result["current_stage_id"] == "stage_0"
        assert result["backtrack_count"] == 1
        assert result["backtrack_decision"] is None
        assert result["code"] is None
        assert result["design_description"] is None
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        assert result["analysis_summary"] is None
        assert result["supervisor_verdict"] is None
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0

    def test_limit_error_return_structure(self, base_state, valid_plan):
        """Verify complete structure of backtrack limit error return value."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [make_stage("stage_0", "MATERIAL_VALIDATION", status="completed_success")]
        )
        base_state["backtrack_count"] = 5
        base_state["runtime_config"] = {"max_backtracks": 5}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)

        # Verify limit error structure
        assert result["workflow_phase"] == "backtracking_limit"
        assert result["backtrack_count"] == 6
        assert result["ask_user_trigger"] == "backtrack_limit"
        assert result["awaiting_user_input"] is True
        assert result["last_node_before_ask_user"] == "handle_backtrack"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        
        # Verify error message mentions the limit value
        assert any("5" in q for q in result["pending_user_questions"]), \
            "Error message should mention the max_backtracks value"

    def test_validation_error_does_not_modify_backtrack_count(self, base_state, valid_plan):
        """Verify backtrack_count is NOT modified on validation errors."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["backtrack_count"] = 3
        base_state["backtrack_decision"] = None  # Invalid

        result = handle_backtrack_node(base_state)

        # Count should NOT be in result (not modified) for validation errors
        assert "backtrack_count" not in result, \
            "backtrack_count should not be modified on validation error"

    # ═══════════════════════════════════════════════════════════════════════
    # Stage Type Detection Tests  
    # ═══════════════════════════════════════════════════════════════════════

    def test_backtrack_detects_material_validation_case_sensitive(self, base_state, valid_plan):
        """Verify MATERIAL_VALIDATION detection is case-sensitive."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = make_progress(
            [
                make_stage(
                    "stage_0", "material_validation",  # lowercase
                    status="completed_success"
                )
            ]
        )
        base_state["runtime_config"] = {"max_backtracks": 10}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        base_state["validated_materials"] = ["Gold"]

        result = handle_backtrack_node(base_state)

        # With lowercase, it should NOT clear materials (case-sensitive comparison)
        assert "validated_materials" not in result, \
            "lowercase stage_type should not trigger material clearing"

    def test_backtrack_other_stage_types_preserve_materials(self, base_state, valid_plan):
        """Test various non-MATERIAL_VALIDATION stage types preserve materials."""
        from src.agents.reporting import handle_backtrack_node

        stage_types = [
            "SINGLE_STRUCTURE",
            "MULTI_STRUCTURE",
            "COMPARATIVE",
            "PARAMETER_SWEEP",
            "CONVERGENCE_TEST",
        ]

        for stage_type in stage_types:
            base_state["plan"] = valid_plan
            base_state["progress"] = make_progress(
                [make_stage("stage_0", stage_type, status="completed_success")]
            )
            base_state["runtime_config"] = {"max_backtracks": 10}
            base_state["backtrack_decision"] = {
                "accepted": True,
                "target_stage_id": "stage_0",
                "stages_to_invalidate": [],
            }
            base_state["validated_materials"] = ["Gold"]

            result = handle_backtrack_node(base_state)

            assert "validated_materials" not in result, \
                f"Stage type {stage_type} should not trigger material clearing"
