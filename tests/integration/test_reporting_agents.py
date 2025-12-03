from unittest.mock import patch, MagicMock
import pytest
import copy

class TestHandleBacktrackNode:
    """Integration checks for handle_backtrack_node."""

    def test_backtrack_marks_target_as_needs_rerun(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                    "outputs": ["some_output"],
                    "discrepancies": ["some_discrepancy"]
                }
            ]
        }
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
        assert target_stage["status"] == "needs_rerun", "Target stage status should be 'needs_rerun'"
        assert target_stage["outputs"] == [], "Target stage outputs should be cleared"
        assert target_stage["discrepancies"] == [], "Target stage discrepancies should be cleared"
        assert result["current_stage_id"] == "stage_0", "Current stage ID should be updated to target"
        assert result["backtrack_decision"] is None, "Backtrack decision should be consumed (set to None)"

    def test_backtrack_invalidates_dependent_stages(self, base_state):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION"},
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "dependencies": ["stage_0"],
                },
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "status": "completed_success",
                },
            ]
        }
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
        assert stage_1["status"] == "invalidated", "Dependent stage should be invalidated"
        
        # Verify invalidated_stages is correctly set in result
        assert result["invalidated_stages"] == ["stage_1"]

    def test_backtrack_increments_counter(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = 0
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 1
        
        # Test increment from existing value
        base_state["backtrack_count"] = 5
        # Ensure max_backtracks is high enough to avoid limit trigger
        base_state["runtime_config"] = {"max_backtracks": 10}
        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 6

    def test_backtrack_clears_working_state(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        # Pollute state with working data
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
        base_state["progress"] = {"stages": []}
        base_state["backtrack_decision"] = None

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        
        # Also test with decision present but accepted=False/None (though schema implies it should be True if present in decision logic usually)
        base_state["backtrack_decision"] = {"accepted": False}
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision"

    def test_backtrack_rejects_empty_target(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = valid_plan
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "", # Empty
            "stages_to_invalidate": []
        }
        
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "invalid_backtrack_target"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_target_not_found(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_999", # Does not exist
            "stages_to_invalidate": []
        }
        
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "backtrack_target_not_found"
        assert result.get("awaiting_user_input") is True
        assert any("stage_999" in q for q in result.get("pending_user_questions", []))

    def test_backtrack_respects_max_limit(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = 2
        base_state["runtime_config"] = {"max_backtracks": 2}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        
        # Should exceed limit (2 -> 3)
        assert result.get("ask_user_trigger") == "backtrack_limit"
        assert result.get("workflow_phase") == "backtracking_limit"
        assert result.get("awaiting_user_input") is True

    def test_backtrack_to_material_validation_clears_materials(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                }
            ]
        }
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

    def test_backtrack_to_non_material_preserves_materials(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "status": "completed_success",
                }
            ]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_1",
            "stages_to_invalidate": [],
        }
        materials = ["Gold"]
        base_state["validated_materials"] = materials

        result = handle_backtrack_node(base_state)
        
        # Should NOT be present in result (updates only) or preserved if copied
        # The node returns a dict of updates. If 'validated_materials' is not in result, it is preserved.
        # But if the code explicitly sets it to empty list only for MATERIAL_VALIDATION, 
        # we must ensure it DOES NOT set it for others.
        assert "validated_materials" not in result


class TestReportGeneratorCompleteness:
    """Verify generate_report_node produces complete reports."""

    def test_generate_report_node_creates_report(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        mock_response = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
            "conclusions": {"main_physics_reproduced": True, "key_findings": ["Test finding"]},
            "paper_citation": {"title": "Test Paper", "authors": "Test Author"},
        }

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
            ]
        }
        base_state["paper_id"] = "test_paper_id"

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)

        assert result.get("workflow_complete") is True
        assert result["workflow_phase"] == "reporting"
        assert "metrics" in result and "token_summary" in result["metrics"]
        assert result["executive_summary"] == mock_response["executive_summary"]
        assert result["paper_citation"] == mock_response["paper_citation"]
        assert result["report_conclusions"] == mock_response["conclusions"]

    def test_report_includes_token_summary(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
                {"agent_name": "designer", "input_tokens": 2000, "output_tokens": 800},
            ]
        }
        mock_response = {
            "executive_summary": {"overall_assessment": []},
            "conclusions": {"main_physics_reproduced": True},
        }

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)

        metrics = result.get("metrics", {})
        token_summary = metrics.get("token_summary", {})
        assert token_summary.get("total_input_tokens") == 3000
        assert token_summary.get("total_output_tokens") == 1300
        
        # Check cost calculation (input * 3.0 + output * 15.0) / 1_000_000
        expected_cost = (3000 * 3.0 + 1300 * 15.0) / 1_000_000
        assert token_summary.get("estimated_cost") == pytest.approx(expected_cost)

    def test_report_generation_handles_llm_failure(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        
        # Ensure these are NOT in state initially to check fallback
        if "executive_summary" in base_state: del base_state["executive_summary"]
        
        # Mock exception
        with patch("src.agents.reporting.call_agent_with_metrics", side_effect=Exception("LLM Error")):
            result = generate_report_node(base_state)
            
        assert result["workflow_complete"] is True
        # Should still have defaults populated
        assert "executive_summary" in result
        assert result["executive_summary"]["overall_assessment"] is not None
        assert "paper_citation" in result

    def test_report_generation_quantitative_summary(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node
        
        base_state["plan"] = valid_plan
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_1",
                "target_figure": "Fig 1",
                "status": "pass",
                "precision_requirement": "high",
                "quantitative_metrics": {
                    "peak_position_error_percent": 0.5,
                    "normalized_rmse_percent": 1.2,
                    "correlation": 0.99,
                    "n_points_compared": 100
                }
            }
        ]
        
        mock_response = {} # Empty response to rely on pre-calculation
        
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
            
        summary = result.get("quantitative_summary")
        assert summary is not None
        assert len(summary) == 1
        row = summary[0]
        assert row["stage_id"] == "stage_1"
        assert row["peak_position_error_percent"] == 0.5
        assert row["normalized_rmse_percent"] == 1.2

    def test_report_generation_populates_missing_structures(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node
        
        base_state["plan"] = valid_plan
        # Ensure missing
        base_state.pop("paper_citation", None)
        base_state.pop("executive_summary", None)
        base_state["paper_title"] = "My Paper"
        
        mock_response = {}
        
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
            
        assert result["paper_citation"]["title"] == "My Paper"
        assert result["paper_citation"]["authors"] == "Unknown"
        assert "executive_summary" in result

    def test_report_includes_rich_state_in_context(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node
        
        base_state["plan"] = valid_plan
        base_state["figure_comparisons"] = [{"fig": "1", "diff": "small"}]
        base_state["assumptions"] = {"param": "value"}
        base_state["discrepancies"] = [{"parameter": "p1", "classification": "minor"}]
        
        mock_response = {}
        
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response) as mock_call:
            generate_report_node(base_state)
            
            # Verify that the context sent to LLM includes these details
            args, kwargs = mock_call.call_args
            user_content = kwargs.get("user_content", "")
            
            assert "Figure Comparisons" in user_content
            assert "Assumptions" in user_content
            assert "Discrepancies" in user_content
            assert "p1" in user_content # content of discrepancy
