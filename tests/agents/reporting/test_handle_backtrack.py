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
                    {"stage_id": "stage1", "status": "completed_success", "outputs": ["data"]},
                    {"stage_id": "stage2", "status": "completed_success"},
                    {"stage_id": "stage3", "status": "pending"}
                ]
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking"
        assert result["current_stage_id"] == "stage1"
        assert result["backtrack_count"] == 1
        
        # Verify stage updates
        new_stages = result["progress"]["stages"]
        stage1 = next(s for s in new_stages if s["stage_id"] == "stage1")
        stage2 = next(s for s in new_stages if s["stage_id"] == "stage2")
        stage3 = next(s for s in new_stages if s["stage_id"] == "stage3")
        
        assert stage1["status"] == "needs_rerun"
        assert stage1["outputs"] == [] # Should be cleared
        assert "discrepancies" in stage1 and stage1["discrepancies"] == [] # Should be cleared
        
        assert stage2["status"] == "invalidated"
        
        assert stage3["status"] == "pending" # Unaffected

    def test_errors_on_missing_decision(self):
        """Should error when backtrack_decision is missing."""
        state = {"backtrack_decision": None}
        
        result = handle_backtrack_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"

    def test_errors_on_unaccepted_decision(self):
        """Should error when backtrack_decision is not accepted."""
        state = {
            "backtrack_decision": {
                "accepted": False,
                "target_stage_id": "stage1"
            }
        }
        
        result = handle_backtrack_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "invalid_backtrack_decision"

    def test_errors_on_missing_target(self):
        """Should error when target_stage_id is empty."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "",
            },
        }
        
        result = handle_backtrack_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "invalid_backtrack_target"

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

    def test_clears_all_downstream_state(self):
        """Should clear all relevant downstream state fields on backtrack."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            },
            "code": "old code",
            "design_description": {"old": "design"},
            "stage_outputs": {"stage1": "output"},
            "run_error": "Some error",
            "analysis_summary": "Summary",
            "invalidated_stages": [],
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
            "backtrack_count": 3,
            "runtime_config": {"max_backtracks": 2},
        }
        
        result = handle_backtrack_node(state)
        
        assert result["workflow_phase"] == "backtracking_limit"
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "backtrack_limit"

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
        
        assert result["ask_user_trigger"] == "backtrack_target_not_found"
        assert result["awaiting_user_input"] is True

    def test_deep_copies_progress(self):
        """Should deep copy progress to avoid side effects on input state."""
        state = {
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": [],
            },
            "progress": {
                "stages": [{"stage_id": "stage1", "status": "completed_success"}]
            }
        }
        
        result = handle_backtrack_node(state)
        
        # Modify result progress
        result["progress"]["stages"][0]["status"] = "modified"
        
        # Input state should remain unchanged
        assert state["progress"]["stages"][0]["status"] == "completed_success"
