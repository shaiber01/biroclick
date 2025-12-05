"""Tests for generate_report_node."""

import copy
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.reporting import generate_report_node


class TestGenerateReportNode:
    """Tests for generate_report_node function."""

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_generates_report_on_success(self, mock_prompt, mock_call):
        """Should generate final report with all expected fields."""
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "Pass"}]},
            "key_findings": ["Finding 1"],
            "recommendations": ["Recommendation 1"],
            "paper_citation": {"title": "Generated Title", "authors": "Author", "journal": "Journal", "year": 2024},
            "assumptions": {"assumed": "true"},
            "figure_comparisons": [{"fig": "1"}],
            "systematic_discrepancies": ["disc1"],
            "conclusions": {"main_physics_reproduced": True, "key_findings": ["Conclusion 1"]}
        }
        
        original_state = {
            "paper_id": "test_paper",
            "progress": {"stages": [{"stage_id": "stage1", "status": "completed_success"}]},
            "metrics": {"agent_calls": []},
            "paper_citation": {"title": "Original Title"} 
        }
        state = copy.deepcopy(original_state)
        
        result = generate_report_node(state)
        
        # Verify state was not mutated
        assert state == original_state
        
        # Verify workflow fields
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True
        
        # Verify all agent output fields are mapped correctly
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Test"
        assert result["executive_summary"]["overall_assessment"][0]["status"] == "Pass"
        assert result["paper_citation"]["title"] == "Generated Title"
        assert result["paper_citation"]["authors"] == "Author"
        assert result["paper_citation"]["journal"] == "Journal"
        assert result["paper_citation"]["year"] == 2024
        assert result["assumptions"] == {"assumed": "true"}
        assert result["figure_comparisons"] == [{"fig": "1"}]
        assert result["systematic_discrepancies_identified"] == ["disc1"]
        assert result["report_conclusions"]["main_physics_reproduced"] is True
        assert result["report_conclusions"]["key_findings"] == ["Conclusion 1"]
        
        # Verify LLM was called correctly
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["agent_name"] == "report"
        assert call_kwargs["system_prompt"] == "system prompt"
        assert call_kwargs["schema_name"] == "report_output_schema"
        assert call_kwargs["state"] == state
        
        # Verify metrics are present
        assert "metrics" in result
        assert "token_summary" in result["metrics"]

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_constructs_user_content_correctly(self, mock_prompt, mock_call):
        """Should verify user content includes all state components."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "paper_title": "Test Paper",
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "completed_success", "summary": "Done well"},
                    {"stage_id": "stage2", "status": "completed_failure", "summary": "Failed"},
                    {"stage_id": "stage3", "status": "in_progress"}  # No summary
                ]
            },
            "figure_comparisons": [{"fig_id": "fig1", "status": "match"}, {"fig_id": "fig2"}],
            "assumptions": {"temp": "300K", "pressure": "1 atm"},
            "discrepancies": [
                {"parameter": "gap", "classification": "minor", "likely_cause": "noise"},
                {"parameter": "wavelength", "classification": "major", "likely_cause": "calibration"}
            ]
        }
        
        generate_report_node(state)
        
        # Verify call arguments
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Verify header
        assert user_content.startswith("# GENERATE REPRODUCTION REPORT")
        
        # Verify paper ID
        assert "Paper ID: test_paper" in user_content
        
        # Verify stage summary section
        assert "## Stage Summary" in user_content
        assert "stage1: completed_success - Done well" in user_content
        assert "stage2: completed_failure - Failed" in user_content
        assert "stage3: in_progress - No summary" in user_content
        
        # Verify figure comparisons (should be limited to 5)
        assert "## Figure Comparisons" in user_content
        assert '"fig_id": "fig1"' in user_content
        assert '"fig_id": "fig2"' in user_content
        
        # Verify assumptions
        assert "## Assumptions" in user_content
        assert '"temp": "300K"' in user_content
        assert '"pressure": "1 atm"' in user_content
        
        # Verify discrepancies (should be limited to 5 and show count)
        assert "## Discrepancies (2 total)" in user_content
        assert "- gap: minor - noise" in user_content
        assert "- wavelength: major - calibration" in user_content
        
        # Verify build_agent_prompt was called
        mock_prompt.assert_called_once_with("report_generator", state)

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_llm_error(self, mock_prompt, mock_call):
        """Should handle LLM call failure gracefully and return partial results."""
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "paper_id": "test_paper",
            "completed_stages": ["stage1"],
            "paper_title": "Fallback Title"
        }
        
        result = generate_report_node(state)
        
        # Verify workflow completes even on error
        assert result["workflow_phase"] == "reporting"
        assert result["workflow_complete"] is True
        
        # Verify default paper_citation is created
        assert "paper_citation" in result
        assert result["paper_citation"]["title"] == "Fallback Title"
        assert result["paper_citation"]["authors"] == "Unknown"
        assert result["paper_citation"]["journal"] == "Unknown"
        assert result["paper_citation"]["year"] == 2023
        
        # Verify default executive_summary is created
        assert "executive_summary" in result
        assert isinstance(result["executive_summary"]["overall_assessment"], list)
        assert len(result["executive_summary"]["overall_assessment"]) == 2
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Material Properties"
        assert result["executive_summary"]["overall_assessment"][0]["status"] == "Reproduced"
        assert result["executive_summary"]["overall_assessment"][0]["status_icon"] == "✅"
        assert result["executive_summary"]["overall_assessment"][1]["aspect"] == "Geometric Resonances"
        assert result["executive_summary"]["overall_assessment"][1]["status"] == "Partial"
        assert result["executive_summary"]["overall_assessment"][1]["status_icon"] == "⚠️"
        
        # Verify metrics are still computed
        assert "metrics" in result
        assert result["metrics"]["token_summary"]["total_input_tokens"] == 0
        assert result["metrics"]["token_summary"]["total_output_tokens"] == 0
        assert result["metrics"]["token_summary"]["estimated_cost"] == 0.0

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_calculates_metrics_cost(self, mock_prompt, mock_call):
        """Should correctly calculate token costs in metrics summary."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        # Cost formula in code: (input * 3.0 + output * 15.0) / 1_000_000
        state = {
            "paper_id": "test_paper",
            "metrics": {
                "agent_calls": [
                    {"input_tokens": 1000, "output_tokens": 100}, # 3000 + 1500 = 4500
                    {"input_tokens": 2000, "output_tokens": 200}, # 6000 + 3000 = 9000
                ],
                "stage_metrics": [{"stage": "stage1"}],  # Should be preserved
                "other_field": "preserved"
            }
        }
        # Total input: 3000, Total output: 300
        # Total cost = (3000 * 3 + 300 * 15) / 1M = (9000 + 4500) / 1M = 0.0135
        
        result = generate_report_node(state)
        
        metrics = result["metrics"]
        
        # Verify token summary
        assert "token_summary" in metrics
        summary = metrics["token_summary"]
        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 300
        assert summary["estimated_cost"] == pytest.approx(0.0135)
        
        # Verify original metrics fields are preserved
        assert metrics["agent_calls"] == state["metrics"]["agent_calls"]
        assert metrics["stage_metrics"] == [{"stage": "stage1"}]
        assert metrics["other_field"] == "preserved"

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_builds_quantitative_summary(self, mock_prompt, mock_call):
        """Should build quantitative summary table from reports."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "analysis_result_reports": [
                {
                    "stage_id": "stage1",
                    "target_figure": "Fig1",
                    "status": "match",
                    "precision_requirement": "acceptable",
                    "quantitative_metrics": {
                        "peak_position_error_percent": 1.5,
                        "normalized_rmse_percent": 2.0,
                        "correlation": 0.95,
                        "n_points_compared": 100
                    }
                },
                {
                    # Partial data
                    "stage_id": "stage2",
                    "status": "fail",
                    "quantitative_metrics": None
                }
            ]
        }
        
        result = generate_report_node(state)
        
        summary = result.get("quantitative_summary", [])
        assert len(summary) == 2
        
        # Verify first row has all fields
        row1 = summary[0]
        assert row1["stage_id"] == "stage1"
        assert row1["figure_id"] == "Fig1"
        assert row1["status"] == "match"
        assert row1["precision_requirement"] == "acceptable"
        assert row1["peak_position_error_percent"] == 1.5
        assert row1["normalized_rmse_percent"] == 2.0
        assert row1["correlation"] == 0.95
        assert row1["n_points_compared"] == 100
        
        # Verify second row handles missing metrics
        row2 = summary[1]
        assert row2["stage_id"] == "stage2"
        assert row2["status"] == "fail"
        assert row2["figure_id"] is None
        assert row2["precision_requirement"] is None
        assert row2["peak_position_error_percent"] is None
        assert row2["normalized_rmse_percent"] is None
        assert row2["correlation"] is None
        assert row2["n_points_compared"] is None
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_empty_quantitative_reports(self, mock_prompt, mock_call):
        """Should handle empty analysis_result_reports."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "analysis_result_reports": []
        }
        
        result = generate_report_node(state)
        
        # Should not have quantitative_summary when empty
        assert "quantitative_summary" not in result
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_missing_quantitative_reports(self, mock_prompt, mock_call):
        """Should handle missing analysis_result_reports."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper"
            # No analysis_result_reports
        }
        
        result = generate_report_node(state)
        
        # Should not have quantitative_summary when missing
        assert "quantitative_summary" not in result
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_quantitative_metrics_as_empty_dict(self, mock_prompt, mock_call):
        """Should handle quantitative_metrics as empty dict."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "analysis_result_reports": [
                {
                    "stage_id": "stage1",
                    "target_figure": "Fig1",
                    "quantitative_metrics": {}  # Empty dict, not None
                }
            ]
        }
        
        result = generate_report_node(state)
        
        summary = result["quantitative_summary"]
        assert len(summary) == 1
        row = summary[0]
        assert row["stage_id"] == "stage1"
        assert row["figure_id"] == "Fig1"
        assert row["peak_position_error_percent"] is None
        assert row["correlation"] is None
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_missing_fields_in_quantitative_reports(self, mock_prompt, mock_call):
        """Should handle missing fields in analysis_result_reports."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "analysis_result_reports": [
                {
                    "stage_id": "stage1"
                    # Missing target_figure, status, precision_requirement, quantitative_metrics
                }
            ]
        }
        
        result = generate_report_node(state)
        
        summary = result["quantitative_summary"]
        assert len(summary) == 1
        row = summary[0]
        assert row["stage_id"] == "stage1"
        assert row["figure_id"] is None
        assert row["status"] is None
        assert row["precision_requirement"] is None
        assert row["peak_position_error_percent"] is None
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_empty_stages(self, mock_prompt, mock_call):
        """Should handle empty stages list in user content."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "progress": {
                "stages": []
            }
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        assert "## Stage Summary" in user_content
        # Should not crash with empty stages
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_missing_progress(self, mock_prompt, mock_call):
        """Should handle missing progress in user content."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper"
            # No progress
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        assert "## Stage Summary" in user_content
        # Should not crash with missing progress
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_limits_figure_comparisons_to_five(self, mock_prompt, mock_call):
        """Should limit figure_comparisons to 5 in user content."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "figure_comparisons": [
                {"fig_id": f"fig{i}"} for i in range(10)
            ]
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should only include first 5
        assert '"fig_id": "fig0"' in user_content
        assert '"fig_id": "fig4"' in user_content
        assert '"fig_id": "fig5"' not in user_content
        assert '"fig_id": "fig9"' not in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_limits_discrepancies_to_five(self, mock_prompt, mock_call):
        """Should limit discrepancies to 5 in user content but show total count."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "discrepancies": [
                {"parameter": f"param{i}", "classification": "minor"} for i in range(10)
            ]
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should show total count
        assert "## Discrepancies (10 total)" in user_content
        # Should only include first 5
        assert "- param0:" in user_content
        assert "- param4:" in user_content
        assert "- param5:" not in user_content
        assert "- param9:" not in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_missing_discrepancy_fields(self, mock_prompt, mock_call):
        """Should handle missing fields in discrepancies."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "discrepancies": [
                {"parameter": "param1"},  # Missing classification and likely_cause
                {"parameter": "param2", "classification": "minor"},  # Missing likely_cause
                {"parameter": "param3", "likely_cause": "noise"}  # Missing classification
            ]
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should handle None values gracefully
        assert "- param1:" in user_content or "param1" in user_content
        assert "- param2:" in user_content or "param2" in user_content
        assert "- param3:" in user_content or "param3" in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_empty_figure_comparisons(self, mock_prompt, mock_call):
        """Should handle empty figure_comparisons."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "figure_comparisons": []
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should not include figure comparisons section when empty
        assert "## Figure Comparisons" not in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_empty_assumptions(self, mock_prompt, mock_call):
        """Should handle empty assumptions."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "assumptions": {}
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should not include assumptions section when empty
        assert "## Assumptions" not in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_empty_discrepancies(self, mock_prompt, mock_call):
        """Should handle empty discrepancies."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "discrepancies": []
        }
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should not include discrepancies section when empty
        assert "## Discrepancies" not in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_user_content_with_missing_paper_id(self, mock_prompt, mock_call):
        """Should handle missing paper_id in user content."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {}  # No paper_id
        
        generate_report_node(state)
        
        args, kwargs = mock_call.call_args
        user_content = kwargs.get("user_content")
        
        # Should use "unknown" as fallback
        assert "Paper ID: unknown" in user_content
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_maps_systematic_discrepancies_correctly(self, mock_prompt, mock_call):
        """Should map systematic_discrepancies to systematic_discrepancies_identified."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "systematic_discrepancies": [
                {"name": "Systematic Shift", "description": "Red shift"}
            ]
        }
        
        state = {"paper_id": "test_paper"}
        
        result = generate_report_node(state)
        
        assert "systematic_discrepancies_identified" in result
        assert result["systematic_discrepancies_identified"][0]["name"] == "Systematic Shift"
        assert result["systematic_discrepancies_identified"][0]["description"] == "Red shift"
        # Should not have systematic_discrepancies key
        assert "systematic_discrepancies" not in result
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_maps_conclusions_correctly(self, mock_prompt, mock_call):
        """Should map conclusions to report_conclusions."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "conclusions": {
                "main_physics_reproduced": True,
                "key_findings": ["Finding 1", "Finding 2"]
            }
        }
        
        state = {"paper_id": "test_paper"}
        
        result = generate_report_node(state)
        
        assert "report_conclusions" in result
        assert result["report_conclusions"]["main_physics_reproduced"] is True
        assert result["report_conclusions"]["key_findings"] == ["Finding 1", "Finding 2"]
        # Should not have conclusions key
        assert "conclusions" not in result
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_does_not_overwrite_when_agent_output_is_empty_dict(self, mock_prompt, mock_call):
        """Should preserve existing fields when agent returns empty dict (falsy values don't overwrite)."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}  # Empty output - falsy values
        
        state = {
            "paper_id": "test_paper",
            "paper_citation": {"title": "Original"},
            "executive_summary": {"overall_assessment": [{"aspect": "Original"}]},
            "assumptions": {"param": "value"},
            "figure_comparisons": [{"fig": "1"}]
        }
        
        result = generate_report_node(state)
        
        # All fields from state should be preserved when agent output is empty
        assert result["paper_citation"]["title"] == "Original"
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Original"
        assert "assumptions" in result
        assert result["assumptions"] == {"param": "value"}
        assert "figure_comparisons" in result
        assert result["figure_comparisons"] == [{"fig": "1"}]
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_falsy_agent_output_values(self, mock_prompt, mock_call):
        """Should NOT overwrite with falsy values (empty dicts/lists are falsy)."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "executive_summary": {},  # Empty dict - falsy
            "paper_citation": {},  # Empty dict - falsy
            "assumptions": {},  # Empty dict - falsy
            "figure_comparisons": [],  # Empty list - falsy
            "systematic_discrepancies": [],  # Empty list - falsy
            "conclusions": {}  # Empty dict - falsy
        }
        
        state = {
            "paper_id": "test_paper",
            "paper_citation": {"title": "Original"},
            "executive_summary": {"overall_assessment": []}
        }
        
        result = generate_report_node(state)
        
        # Empty dicts/lists are falsy, so they should NOT overwrite existing values
        # The code uses `if agent_output.get("key"):` which is False for empty dict/list
        assert result["executive_summary"] == {"overall_assessment": []}  # Preserved from state
        assert result["paper_citation"]["title"] == "Original"  # Preserved from state
        
        # Empty dicts/lists don't overwrite because they're falsy
        # But assumptions/figure_comparisons weren't in state initially, so they won't be in result
        assert "assumptions" not in result
        assert "figure_comparisons" not in result
        assert "systematic_discrepancies_identified" not in result
        assert "report_conclusions" not in result
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_state_immutability(self, mock_prompt, mock_call):
        """Should not mutate input state."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": [{"aspect": "New"}]},
            "paper_citation": {"title": "New Title"}
        }
        
        original_state = {
            "paper_id": "test_paper",
            "progress": {"stages": [{"stage_id": "stage1"}]},
            "metrics": {"agent_calls": []},
            "paper_citation": {"title": "Original"},
            "executive_summary": {"overall_assessment": [{"aspect": "Original"}]}
        }
        state = copy.deepcopy(original_state)
        
        result = generate_report_node(state)
        
        # Verify state was not mutated
        assert state == original_state
        # Verify result has new values
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "New"
        assert result["paper_citation"]["title"] == "New Title"
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_workflow_complete_always_true(self, mock_prompt, mock_call):
        """Should always set workflow_complete to True."""
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("Error")
        
        state = {"paper_id": "test_paper"}
        
        result = generate_report_node(state)
        
        assert result["workflow_complete"] is True
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_workflow_phase_always_reporting(self, mock_prompt, mock_call):
        """Should always set workflow_phase to 'reporting'."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {"paper_id": "test_paper"}
        
        result = generate_report_node(state)
        
        assert result["workflow_phase"] == "reporting"
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_preserves_assumptions_and_figure_comparisons_from_state(self, mock_prompt, mock_call):
        """Should preserve assumptions and figure_comparisons from state when agent doesn't provide them."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}  # Agent doesn't provide assumptions/figure_comparisons
        
        state = {
            "paper_id": "test_paper",
            "assumptions": {"temperature": "300K", "pressure": "1 atm"},
            "figure_comparisons": [
                {"figure_id": "Fig1", "status": "match"},
                {"figure_id": "Fig2", "status": "partial"}
            ]
        }
        
        result = generate_report_node(state)
        
        # Assumptions and figure_comparisons should be preserved from state
        assert "assumptions" in result
        assert result["assumptions"] == {"temperature": "300K", "pressure": "1 atm"}
        assert "figure_comparisons" in result
        assert result["figure_comparisons"] == [
            {"figure_id": "Fig1", "status": "match"},
            {"figure_id": "Fig2", "status": "partial"}
        ]

    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_missing_metrics(self, mock_prompt, mock_call):
        """Should handle missing metrics in state gracefully."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {"paper_id": "test_paper"} # No metrics key
        
        result = generate_report_node(state)
        
        # Verify metrics structure is created with defaults
        assert "metrics" in result
        assert isinstance(result["metrics"], dict)
        assert "token_summary" in result["metrics"]
        assert result["metrics"]["token_summary"]["total_input_tokens"] == 0
        assert result["metrics"]["token_summary"]["total_output_tokens"] == 0
        assert result["metrics"]["token_summary"]["estimated_cost"] == 0.0
        assert result["metrics"]["agent_calls"] == []
        assert result["metrics"]["stage_metrics"] == []
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_none_tokens_in_metrics(self, mock_prompt, mock_call):
        """Should handle None values in token counts gracefully."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "metrics": {
                "agent_calls": [
                    {"input_tokens": None, "output_tokens": 100},
                    {"input_tokens": 2000, "output_tokens": None},
                    {"input_tokens": 500, "output_tokens": 50},
                ]
            }
        }
        
        result = generate_report_node(state)
        
        # None values should be treated as 0
        summary = result["metrics"]["token_summary"]
        assert summary["total_input_tokens"] == 2500  # 0 + 2000 + 500
        assert summary["total_output_tokens"] == 150  # 100 + 0 + 50
        assert summary["estimated_cost"] == pytest.approx((2500 * 3.0 + 150 * 15.0) / 1_000_000)
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_missing_tokens_in_agent_calls(self, mock_prompt, mock_call):
        """Should handle missing token fields in agent_calls."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "metrics": {
                "agent_calls": [
                    {},  # No token fields
                    {"input_tokens": 1000},  # Missing output_tokens
                    {"output_tokens": 200},  # Missing input_tokens
                    {"input_tokens": 500, "output_tokens": 50},
                ]
            }
        }
        
        result = generate_report_node(state)
        
        summary = result["metrics"]["token_summary"]
        assert summary["total_input_tokens"] == 1500  # 0 + 1000 + 0 + 500
        assert summary["total_output_tokens"] == 250  # 0 + 0 + 200 + 50
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_empty_agent_calls(self, mock_prompt, mock_call):
        """Should handle empty agent_calls list."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "metrics": {
                "agent_calls": []
            }
        }
        
        result = generate_report_node(state)
        
        summary = result["metrics"]["token_summary"]
        assert summary["total_input_tokens"] == 0
        assert summary["total_output_tokens"] == 0
        assert summary["estimated_cost"] == 0.0
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_handles_partial_agent_output(self, mock_prompt, mock_call):
        """Should handle partial agent output - only some fields present."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test"}]},
            # Missing other fields
        }
        
        state = {
            "paper_id": "test_paper",
            "paper_citation": {"title": "Original", "authors": "Author"},
            "executive_summary": {"overall_assessment": [{"aspect": "Original"}]}
        }
        
        result = generate_report_node(state)
        
        # Only executive_summary should be updated
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Test"
        # paper_citation should remain from state (not overwritten)
        assert result["paper_citation"]["title"] == "Original"
        assert result["paper_citation"]["authors"] == "Author"
        # Missing fields should not be in result
        assert "assumptions" not in result or result.get("assumptions") == state.get("assumptions")
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_preserves_existing_paper_citation_when_agent_does_not_provide(self, mock_prompt, mock_call):
        """Should preserve existing paper_citation when agent doesn't provide one."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "executive_summary": {"overall_assessment": []}
            # No paper_citation in output
        }
        
        state = {
            "paper_id": "test_paper",
            "paper_citation": {
                "title": "Existing Title",
                "authors": "Existing Author",
                "journal": "Existing Journal",
                "year": 2023
            }
        }
        
        result = generate_report_node(state)
        
        assert result["paper_citation"]["title"] == "Existing Title"
        assert result["paper_citation"]["authors"] == "Existing Author"
        assert result["paper_citation"]["journal"] == "Existing Journal"
        assert result["paper_citation"]["year"] == 2023
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_creates_default_paper_citation_when_missing(self, mock_prompt, mock_call):
        """Should create default paper_citation when missing from state."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "paper_title": "Test Title"
        }
        
        result = generate_report_node(state)
        
        assert result["paper_citation"]["title"] == "Test Title"
        assert result["paper_citation"]["authors"] == "Unknown"
        assert result["paper_citation"]["journal"] == "Unknown"
        assert result["paper_citation"]["year"] == 2023
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_creates_default_paper_citation_without_title(self, mock_prompt, mock_call):
        """Should create default paper_citation even without paper_title."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper"
            # No paper_title
        }
        
        result = generate_report_node(state)
        
        assert result["paper_citation"]["title"] == "Unknown"
        assert result["paper_citation"]["authors"] == "Unknown"
        assert result["paper_citation"]["journal"] == "Unknown"
        assert result["paper_citation"]["year"] == 2023
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_preserves_existing_executive_summary_when_agent_does_not_provide(self, mock_prompt, mock_call):
        """Should preserve existing executive_summary when agent doesn't provide one."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper",
            "executive_summary": {
                "overall_assessment": [
                    {"aspect": "Custom Aspect", "status": "Custom Status"}
                ]
            }
        }
        
        result = generate_report_node(state)
        
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Custom Aspect"
        assert result["executive_summary"]["overall_assessment"][0]["status"] == "Custom Status"
    
    @patch("src.agents.reporting.call_agent_with_metrics")
    @patch("src.agents.reporting.build_agent_prompt")
    def test_creates_default_executive_summary_when_missing(self, mock_prompt, mock_call):
        """Should create default executive_summary when missing from state."""
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}
        
        state = {
            "paper_id": "test_paper"
        }
        
        result = generate_report_node(state)
        
        assert "executive_summary" in result
        assert isinstance(result["executive_summary"]["overall_assessment"], list)
        assert len(result["executive_summary"]["overall_assessment"]) == 2
        assert result["executive_summary"]["overall_assessment"][0]["aspect"] == "Material Properties"
        assert result["executive_summary"]["overall_assessment"][1]["aspect"] == "Geometric Resonances"
