"""Tests for handle_backtrack_node."""

import pytest

from src.agents.reporting import handle_backtrack_node


class TestHandleBacktrackNode:
    """Tests for handle_backtrack_node function."""

    def test_backtracks_to_stage(self):
        """Should handle backtrack to a specific stage and update status."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "outputs": ["data"], "discrepancies": [{"param": "wavelength"}]},
                    {"stage_id": "stage2", "status": "completed_success"},
                    {"stage_id": "stage3", "status": "pending"}
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["current_stage_id"] == "stage1"
        assert result["backtrack_count"] == 1
        assert result["invalidated_stages"] == ["stage2"]
        assert result["backtrack_decision"] is None
        
        # Verify stage updates
        new_stages = result["progress"]["stages"]
        stage1 = next(s for s in new_stages if s["stage_id"] == "stage1")
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        stage3 = next(s for s in new_stages if s["stage_id"] == "stage3")
        
        assert stage1["status"] == "needs_rerun"
        assert stage1["outputs"] == [] # Should be cleared
        assert stage1["discrepancies"] == [] # Should be cleared
        
        assert stage2["status"] == "invalidated"
        
        assert stage3["status"] == "pending" # Unaffected

    def test_errors_on_missing_decision(self):
        """Should error when backtrack_decision is missing."""
        state = {"backtrack_decision": None}
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result.get("ask_user_trigger") is not None
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0]
        assert "missing or invalid" in result["pending_user_questions"][0].lower()

    def test_errors_on_unaccepted_decision(self):
        """Should error when backtrack_decision is not accepted."""
        state = {
            "backtrack_decision": {
                "accepted": False,
                "target_stage_id": "stage1"
            }
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result.get("ask_user_trigger") is not None
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0]
        assert "missing or invalid" in result["pending_user_questions"][0].lower()

    def test_errors_on_missing_target(self):
        """Should error when target_stage_id is empty."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "",
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result.get("ask_user_trigger") is not None
        assert result["ask_user_trigger"] == "invalid_backtrack_target"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0]
        assert "empty" in result["pending_user_questions"][0].lower()

    def test_increments_backtrack_count(self):
        """Should increment backtrack count."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 1,
        }
        
        result = handle_backtrack_node(state)
        
        assert result["backtrack_count"] == 2
    
    def test_backtrack_count_defaults_to_zero(self):
        """Should default backtrack_count to 0 if missing."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            # backtrack_count not present
        }
        
        result = handle_backtrack_node(state)
        
        assert result["backtrack_count"] == 1  # 0 + 1
    
    def test_backtrack_count_none_defaults_to_zero(self):
        """Should handle None backtrack_count as 0."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": None,
        }
        
        result = handle_backtrack_node(state)
        
        assert result["backtrack_count"] == 1  # 0 + 1

    def test_clears_all_downstream_state(self):
        """Should clear all relevant downstream state fields on backtrack."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "code": "old code",
            "design_description": {"old": "design"},
            "stage_outputs": {"stage1": "output", "stage2": "output2"},
            "run_error": "Some error",
            "analysis_summary": "Summary",
            "invalidated_stages": ["old_stage"],
            "last_design_review_verdict": "approved",
            "last_code_review_verdict": "approved",
            "supervisor_verdict": "backtrack",
            # Counters that should be reset
            "design_revision_count": 3,
            "code_revision_count": 2,
            "execution_failure_count": 1,
            "analysis_revision_count": 1,
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["code"] is None
        assert result["design_description"] is None
        assert result["stage_outputs"] == {}
        assert result["run_error"] is None
        assert result["analysis_summary"] is None
        assert result["last_design_review_verdict"] is None
        assert result["last_code_review_verdict"] is None
        assert result["supervisor_verdict"] is None
        assert result["backtrack_decision"] is None
        assert result["invalidated_stages"] == ["stage2"]  # Should be set to stages_to_invalidate
        
        # Verify counters are reset
        assert result["design_revision_count"] == 0
        assert result["code_revision_count"] == 0
        assert result["execution_failure_count"] == 0
        assert result["analysis_revision_count"] == 0

    def test_handles_backtrack_limit(self):
        """Should handle backtrack limit reached."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 2,  # Will become 3 after increment
            "runtime_config": {"max_backtracks": 2},
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking_limit"
        assert result.get("ask_user_trigger") is not None
        assert result["ask_user_trigger"] == "backtrack_limit"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "limit" in result["pending_user_questions"][0].lower()
        assert result["last_node_before_ask_user"] == "handle_backtrack"
    
    def test_backtrack_limit_at_exact_boundary(self):
        """Should trigger limit when backtrack_count equals max_backtracks after increment."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 1,  # Will become 2 after increment
            "runtime_config": {"max_backtracks": 2},
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking_limit"
        assert result["backtrack_count"] == 2
        assert result.get("ask_user_trigger") is not None
    
    def test_backtrack_limit_defaults_to_two(self):
        """Should use default max_backtracks=2 when runtime_config is missing."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 2,  # Will become 3 after increment
            # runtime_config missing
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking_limit"
        assert result["backtrack_count"] == 3
    
    def test_backtrack_limit_with_missing_max_backtracks(self):
        """Should use default max_backtracks=2 when runtime_config exists but max_backtracks missing."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "backtrack_count": 2,
            "runtime_config": {},  # max_backtracks missing
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking_limit"
        assert result["backtrack_count"] == 3

    def test_backtrack_to_stage_0_clears_materials(self):
        """Should clear materials when backtracking to Stage 0 (MATERIAL_VALIDATION)."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage0",
                "stages_to_invalidate": ["stage1"],
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0", 
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_success"
                    },
                    {"stage_id": "stage1", "status": "completed_success"},
                ]
            },
            "validated_materials": ["gold.py"],
            "pending_validated_materials": ["silver.py"]
        }
        
        result = handle_backtrack_node(state)
        
        assert result["current_stage_id"] == "stage0"
        assert result["validated_materials"] == []
        assert result["pending_validated_materials"] == []
    
    def test_backtrack_to_non_material_stage_preserves_materials(self):
        """Should NOT clear materials when backtracking to non-MATERIAL_VALIDATION stage."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0", 
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage1",
                        "stage_type": "DESIGN",
                        "status": "completed_success"
                    },
                ]
            },
            "validated_materials": ["gold.py"],
            "pending_validated_materials": ["silver.py"]
        }
        
        result = handle_backtrack_node(state)
        
        assert result["current_stage_id"] == "stage1"
        # Materials should NOT be cleared when backtracking to non-material stage
        assert "validated_materials" not in result or result.get("validated_materials") != []
        assert "pending_validated_materials" not in result or result.get("pending_validated_materials") != []
    
    def test_backtrack_to_stage_without_stage_type_does_not_clear_materials(self):
        """Should NOT clear materials when target stage has no stage_type field."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage0",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        # stage_type missing
                        "status": "completed_success"
                    },
                ]
            },
            "validated_materials": ["gold.py"],
            "pending_validated_materials": ["silver.py"]
        }
        
        result = handle_backtrack_node(state)
        
        assert result["current_stage_id"] == "stage0"
        # Materials should NOT be cleared if stage_type is missing
        assert "validated_materials" not in result or result.get("validated_materials") != []
        assert "pending_validated_materials" not in result or result.get("pending_validated_materials") != []

    def test_errors_on_missing_target_in_progress(self):
        """Should error when target stage is not in progress history."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stageX", # Not in progress
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["ask_user_trigger"] == "backtrack_target_not_found"
        assert result.get("ask_user_trigger") is not None
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert "stageX" in result["pending_user_questions"][0]
        assert "not found" in result["pending_user_questions"][0].lower()
    
    def test_errors_on_empty_progress_stages(self):
        """Should error when progress.stages is empty."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": []
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["ask_user_trigger"] == "backtrack_target_not_found"
        assert result.get("ask_user_trigger") is not None
    
    def test_errors_on_missing_progress(self):
        """Should error when progress is missing."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            # progress missing
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["ask_user_trigger"] == "backtrack_target_not_found"
        assert result.get("ask_user_trigger") is not None
    
    def test_errors_on_missing_progress_stages(self):
        """Should error when progress.stages is missing."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {},
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["ask_user_trigger"] == "backtrack_target_not_found"
        assert result.get("ask_user_trigger") is not None

    def test_deep_copies_progress(self):
        """Should deep copy progress to avoid side effects on input state."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "outputs": ["data"], "discrepancies": [{"param": "x"}]}
                ]
            }
        }
        
        original_outputs = state["progress"]["stages"][0]["outputs"]
        original_discrepancies = state["progress"]["stages"][0]["discrepancies"]
        
        result = handle_backtrack_node(state)
        
        # Modify result progress deeply
        result["progress"]["stages"][0]["status"] = "modified"
        result["progress"]["stages"][0]["outputs"].append("new_data")
        result["progress"]["stages"][0]["discrepancies"].append({"param": "y"})
        
        # Input state should remain unchanged
        assert state["progress"]["stages"][0]["status"] == "completed_success"
        assert state["progress"]["stages"][0]["outputs"] == original_outputs
        assert state["progress"]["stages"][0]["discrepancies"] == original_discrepancies
        assert len(state["progress"]["stages"][0]["outputs"]) == 1
        assert len(state["progress"]["stages"][0]["discrepancies"]) == 1
    
    def test_handles_multiple_stages_to_invalidate(self):
        """Should invalidate multiple stages correctly."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2", "stage3", "stage4"],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success"},
                    {"stage_id": "stage2", "status": "completed_success"},
                    {"stage_id": "stage3", "status": "completed_success"},
                    {"stage_id": "stage4", "status": "completed_success"},
                    {"stage_id": "stage5", "status": "pending"},
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["invalidated_stages"] == ["stage2", "stage3", "stage4"]
        
        new_stages = result["progress"]["stages"]
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        stage3 = next(s for s in new_stages if s["stage_id"] == "stage3")
        stage4 = next(s for s in new_stages if s["stage_id"] == "stage4")
        stage5 = next(s for s in new_stages if s["stage_id"] == "stage5")
        
        assert stage2["status"] == "invalidated"
        assert stage3["status"] == "invalidated"
        assert stage4["status"] == "invalidated"
        assert stage5["status"] == "pending"  # Unaffected
    
    def test_handles_nonexistent_stages_to_invalidate(self):
        """Should handle stages_to_invalidate containing non-existent stages."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2", "nonexistent_stage"],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success"},
                    {"stage_id": "stage2", "status": "completed_success"},
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        # Should still set invalidated_stages to the provided list
        assert result["invalidated_stages"] == ["stage2", "nonexistent_stage"]
        
        new_stages = result["progress"]["stages"]
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        assert stage2["status"] == "invalidated"
        
        # Non-existent stage should not cause error, just be ignored in status updates
    
    def test_handles_target_in_stages_to_invalidate(self):
        """Should handle case where target stage is also in stages_to_invalidate."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage1", "stage2"],  # stage1 is target
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "outputs": ["data"]},
                    {"stage_id": "stage2", "status": "completed_success"},
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["invalidated_stages"] == ["stage1", "stage2"]
        
        new_stages = result["progress"]["stages"]
        stage1 = next(s for s in new_stages if s["stage_id"] == "stage1")
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        
        # Target stage should be marked as needs_rerun (not invalidated)
        assert stage1["status"] == "needs_rerun"
        assert stage1["outputs"] == []
        assert stage2["status"] == "invalidated"
    
    def test_handles_empty_stages_to_invalidate(self):
        """Should handle empty stages_to_invalidate list."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success"},
                    {"stage_id": "stage2", "status": "completed_success"},
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["invalidated_stages"] == []
        
        new_stages = result["progress"]["stages"]
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        assert stage2["status"] == "completed_success"  # Unchanged
    
    def test_handles_missing_stages_to_invalidate(self):
        """Should handle missing stages_to_invalidate field."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                # stages_to_invalidate missing
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success"},
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["invalidated_stages"] == []
    
    def test_clears_outputs_even_when_missing(self):
        """Should set outputs to [] even if stage has no outputs field initially."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success"}  # No outputs field
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        new_stages = result["progress"]["stages"]
        stage1 = next(s for s in new_stages if s["stage_id"] == "stage1")
        assert stage1["outputs"] == []
        assert stage1["discrepancies"] == []
    
    def test_clears_discrepancies_even_when_missing(self):
        """Should set discrepancies to [] even if stage has no discrepancies field initially."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "outputs": ["data"]}  # No discrepancies field
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        new_stages = result["progress"]["stages"]
        stage1 = next(s for s in new_stages if s["stage_id"] == "stage1")
        assert stage1["discrepancies"] == []
        assert stage1["outputs"] == []
