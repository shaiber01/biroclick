"""
Comprehensive tests for results analysis workflow nodes.

Tests results_analyzer_node and comparison_validator_node with rigorous assertions
to catch bugs in classification logic, state updates, and integration flow.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from src.agents import results_analyzer_node, comparison_validator_node
from src.agents.constants import AnalysisClassification

from tests.workflow.fixtures import MockResponseFactory


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def analysis_ready_state(base_state, tmp_path):
    """State prepared for analysis with output files."""
    # Create dummy output file
    output_file = tmp_path / "output.csv"
    output_file.write_text("wavelength,extinction\n400,0.1\n500,0.5\n700,0.9")
    
    base_state["current_stage_id"] = "stage_1_extinction"
    base_state["stage_outputs"] = {"files": [str(output_file)]}
    
    # Add plan with target details
    plan = MockResponseFactory.planner_response()
    plan["stages"][1]["target_details"] = [
        {"figure_id": "Fig1", "precision_requirement": "acceptable"}
    ]
    base_state["plan"] = plan
    
    return base_state


@pytest.fixture
def validated_analysis_state(analysis_ready_state, tmp_path):
    """State with completed analysis results for validator testing."""
    analysis_ready_state["analysis_result_reports"] = [
        {
            "result_id": "stage_1_extinction_Fig1",
            "stage_id": "stage_1_extinction",
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "expected_outputs": ["output.csv"],
            "matched_output": str(tmp_path / "output.csv"),
            "precision_requirement": "acceptable",
            "digitized_data_path": None,
            "validation_criteria": [],
            "quantitative_metrics": {"peak_position_error_percent": 0.5},
            "criteria_failures": [],
            "notes": "Output identified.",
        }
    ]
    analysis_ready_state["figure_comparisons"] = [
        {
            "figure_id": "Fig1",
            "stage_id": "stage_1_extinction",
            "title": "Extinction spectrum",
            "classification": AnalysisClassification.MATCH,
            "paper_image_path": None,
            "reproduction_image_path": str(tmp_path / "output.csv"),
            "comparison_table": [],
            "shape_comparison": [],
            "reason_for_difference": "",
        }
    ]
    return analysis_ready_state


# ═══════════════════════════════════════════════════════════════════════
# ResultsAnalyzerNode Tests
# ═══════════════════════════════════════════════════════════════════════


class TestResultsAnalyzerClassification:
    """Test results analyzer classification logic."""

    def test_excellent_match_classification(self, analysis_ready_state, tmp_path):
        """Test EXCELLENT_MATCH when error is below 1%."""
        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm, \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=["img.png"]), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.3,
                 "peak_position_paper": 700.0,
                 "peak_position_sim": 702.1
             }), \
             patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"):
            
            mock_llm.return_value = MockResponseFactory.analyzer_response(
                AnalysisClassification.EXCELLENT_MATCH
            )

            result = results_analyzer_node(analysis_ready_state)

            # STRICT assertions
            assert result["analysis_overall_classification"] == AnalysisClassification.EXCELLENT_MATCH
            assert result["workflow_phase"] == "analysis"
            assert "figure_comparisons" in result
            assert len(result["figure_comparisons"]) > 0
            
            summary = result["analysis_summary"]
            assert summary["totals"]["matches"] == 1
            assert summary["totals"]["missing"] == 0
            assert summary["totals"]["mismatch"] == 0

    def test_acceptable_match_classification(self, analysis_ready_state):
        """Test ACCEPTABLE_MATCH when error is 1-5%."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 3.5,  # 1-5% = acceptable
                 "peak_position_paper": 700.0,
                 "peak_position_sim": 724.5
             }), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            # 3.5% error should be PARTIAL_MATCH according to classification_from_metrics
            report = result["analysis_result_reports"][0]
            assert report["status"] in [
                AnalysisClassification.PARTIAL_MATCH,
                AnalysisClassification.ACCEPTABLE_MATCH
            ]
            assert report["quantitative_metrics"]["peak_position_error_percent"] == 3.5

    def test_poor_match_when_missing_outputs(self, analysis_ready_state):
        """Test POOR_MATCH when output files are missing."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value=None), \
             patch("src.agents.analysis.match_expected_files", return_value=None):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            # Missing outputs MUST result in POOR_MATCH
            assert result["analysis_overall_classification"] == AnalysisClassification.POOR_MATCH
            summary = result["analysis_summary"]
            assert summary["totals"]["missing"] >= 1
            assert "Fig1" in summary["missing_targets"]

    def test_mismatch_when_high_error(self, analysis_ready_state):
        """Test MISMATCH when error exceeds acceptable threshold."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 15.0,  # High error
                 "peak_position_paper": 700.0,
                 "peak_position_sim": 805.0
             }), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            summary = result["analysis_summary"]
            report = result["analysis_result_reports"][0]
            
            # 15% error should be MISMATCH
            assert report["status"] == AnalysisClassification.MISMATCH
            assert "Fig1" in summary["mismatch_targets"]
            assert summary["discrepancies_logged"] > 0


class TestResultsAnalyzerErrorHandling:
    """Test error handling in results analyzer."""

    def test_missing_stage_id_triggers_user_escalation(self, base_state):
        """Missing current_stage_id should trigger user escalation."""
        base_state["current_stage_id"] = None
        base_state["plan"] = MockResponseFactory.planner_response()

        result = results_analyzer_node(base_state)

        # STRICT: Must trigger user escalation, not silent failure
        assert result["workflow_phase"] == "analysis"
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0]

    def test_empty_stage_id_triggers_user_escalation(self, base_state):
        """Empty string current_stage_id should trigger user escalation."""
        base_state["current_stage_id"] = ""
        base_state["plan"] = MockResponseFactory.planner_response()

        result = results_analyzer_node(base_state)

        # STRICT: Empty string should be treated same as None
        assert result["workflow_phase"] == "analysis"
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"

    def test_missing_stage_outputs_returns_fail(self, analysis_ready_state):
        """Missing stage_outputs should return execution fail."""
        analysis_ready_state["stage_outputs"] = None

        result = results_analyzer_node(analysis_ready_state)

        # STRICT: Must indicate execution failure
        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED

    def test_empty_stage_outputs_files_returns_fail(self, analysis_ready_state):
        """Empty stage_outputs.files list should return execution fail."""
        analysis_ready_state["stage_outputs"] = {"files": []}

        result = results_analyzer_node(analysis_ready_state)

        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert "output" in result["run_error"].lower() or "missing" in result["run_error"].lower()

    def test_all_files_missing_from_disk_returns_fail(self, analysis_ready_state):
        """When all listed files don't exist on disk, should return fail."""
        analysis_ready_state["stage_outputs"] = {"files": ["nonexistent1.csv", "nonexistent2.csv"]}

        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.is_file", return_value=False), \
             patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            assert result["execution_verdict"] == "fail"
            assert "run_error" in result
            assert "missing" in result["run_error"].lower() or "not exist" in result["run_error"].lower()

    def test_no_targets_returns_no_targets_classification(self, analysis_ready_state):
        """Stage with no targets should return NO_TARGETS classification."""
        analysis_ready_state["plan"]["stages"][1]["targets"] = []
        analysis_ready_state["plan"]["stages"][1].pop("target_details", None)

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
            summary = result["analysis_summary"]
            assert summary["overall_classification"] == AnalysisClassification.NO_TARGETS
            assert summary["totals"]["targets"] == 0


class TestResultsAnalyzerDigitizedData:
    """Test digitized data requirements."""

    def test_excellent_precision_without_digitized_data_blocks(self, analysis_ready_state):
        """Excellent precision requirement without digitized data should block analysis."""
        # Clear any existing digitized paths from paper_figures
        analysis_ready_state["paper_figures"] = [
            {"id": "Fig1", "description": "Extinction spectrum"}  # No digitized_data_path
        ]
        analysis_ready_state["plan"]["stages"][1]["target_details"] = [
            {"figure_id": "Fig1", "precision_requirement": "excellent"}
        ]
        analysis_ready_state["plan"]["targets"] = [
            {"figure_id": "Fig1", "precision_requirement": "excellent"}
        ]
        # No digitized_data_path provided anywhere

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            summary = result["analysis_summary"]
            
            # Should be pending/blocked due to missing digitized data
            assert "Fig1" in summary["pending_targets"] or "Fig1" in summary["mismatch_targets"]
            assert summary["discrepancies_logged"] > 0
            
            # Verify report indicates missing digitized data
            report = next((r for r in result["analysis_result_reports"] if r["target_figure"] == "Fig1"), None)
            assert report is not None
            assert "digitized" in str(report.get("criteria_failures", [])).lower() or \
                   report["status"] == "missing_digitized_data"

    def test_excellent_precision_with_digitized_data_succeeds(self, analysis_ready_state, tmp_path):
        """Excellent precision with digitized data should compute quantitative metrics."""
        ref_file = tmp_path / "reference.csv"
        ref_file.write_text("wavelength,extinction\n400,0.1\n500,0.5\n700,0.9")
        
        # Clear any existing digitized paths from paper_figures
        analysis_ready_state["paper_figures"] = [
            {"id": "Fig1", "description": "Extinction spectrum"}  # No digitized_data_path here
        ]
        analysis_ready_state["plan"]["stages"][1]["target_details"] = [
            {"figure_id": "Fig1", "precision_requirement": "excellent", "digitized_data_path": str(ref_file)}
        ]
        analysis_ready_state["plan"]["targets"] = [
            {"figure_id": "Fig1", "precision_requirement": "excellent", "digitized_data_path": str(ref_file)}
        ]

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5,
                 "peak_position_paper": 700.0,
                 "peak_position_sim": 703.5
             }):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            report = result["analysis_result_reports"][0]
            assert report["precision_requirement"] == "excellent"
            assert report["digitized_data_path"] == str(ref_file)
            assert report["quantitative_metrics"]["peak_position_error_percent"] == 0.5


class TestResultsAnalyzerMultipleTargets:
    """Test handling of multiple targets."""

    def test_processes_all_targets_not_just_first(self, analysis_ready_state, tmp_path):
        """Must process ALL targets, not return early after first."""
        output2 = tmp_path / "output2.csv"
        output2.write_text("wavelength,extinction\n400,0.2\n600,0.8")
        
        analysis_ready_state["plan"]["stages"][1]["targets"] = ["Fig1", "Fig2"]
        analysis_ready_state["plan"]["stages"][1]["target_details"] = [
            {"figure_id": "Fig1", "precision_requirement": "acceptable"},
            {"figure_id": "Fig2", "precision_requirement": "acceptable"}
        ]
        analysis_ready_state["stage_outputs"]["files"].append(str(output2))

        def match_side_effect(files, target_id):
            if "Fig1" in target_id:
                return str(files[0]) if files else None
            return str(output2)

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_expected_files", return_value=None), \
             patch("src.agents.analysis.match_output_file", side_effect=match_side_effect), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5
             }):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            summary = result["analysis_summary"]
            
            # STRICT: Both targets must be processed
            assert summary["totals"]["targets"] == 2
            assert len(result["analysis_result_reports"]) == 2
            
            target_figures = {r["target_figure"] for r in result["analysis_result_reports"]}
            assert "Fig1" in target_figures
            assert "Fig2" in target_figures

    def test_feedback_targets_prioritized(self, analysis_ready_state, tmp_path):
        """Targets mentioned in feedback should be processed first."""
        output2 = tmp_path / "output2.csv"
        output2.write_text("wavelength,extinction\n400,0.2\n600,0.8")
        
        analysis_ready_state["plan"]["stages"][1]["targets"] = ["Fig1", "Fig2"]
        analysis_ready_state["plan"]["stages"][1]["target_details"] = [
            {"figure_id": "Fig1", "precision_requirement": "acceptable"},
            {"figure_id": "Fig2", "precision_requirement": "acceptable"}
        ]
        analysis_ready_state["analysis_feedback"] = "Please recheck Fig2"

        processed_order = []
        def match_side_effect(files, target_id):
            processed_order.append(target_id)
            return "output.csv"

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_expected_files", return_value=None), \
             patch("src.agents.analysis.match_output_file", side_effect=match_side_effect), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5
             }):

            result = results_analyzer_node(analysis_ready_state)

            # STRICT: Fig2 should be processed FIRST due to feedback
            assert len(processed_order) == 2
            assert processed_order[0] == "Fig2"
            assert "Fig1" in processed_order


class TestResultsAnalyzerValidationCriteria:
    """Test validation criteria handling."""

    def test_criteria_failure_results_in_mismatch(self, analysis_ready_state):
        """Failed validation criteria should result in MISMATCH."""
        analysis_ready_state["plan"]["stages"][1]["validation_criteria"] = [
            "Fig1: peak within 1%"
        ]

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 5.0,  # Fails 1% criteria
                 "peak_position_paper": 700.0,
                 "peak_position_sim": 735.0
             }):

            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            summary = result["analysis_summary"]
            report = result["analysis_result_reports"][0]
            
            # STRICT: Criteria failure must result in MISMATCH
            assert report["status"] == AnalysisClassification.MISMATCH
            assert len(report["criteria_failures"]) > 0
            assert "Fig1" in summary["mismatch_targets"]

    def test_criteria_with_missing_metrics_fails(self, analysis_ready_state):
        """Validation criteria without required metrics should fail."""
        analysis_ready_state["plan"]["stages"][1]["validation_criteria"] = [
            "Fig1: peak within 1%"
        ]

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={}):  # No metrics

            result = results_analyzer_node(analysis_ready_state)

            report = result["analysis_result_reports"][0]
            
            # Should have criteria failure due to missing metric
            assert len(report["criteria_failures"]) > 0
            assert any("missing metric" in f.lower() for f in report["criteria_failures"])


class TestResultsAnalyzerContextEscalation:
    """Test context escalation handling."""

    def test_context_overflow_returns_awaiting_user(self, analysis_ready_state):
        """Context overflow should return awaiting_user_input state."""
        escalation = {
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context limit exceeded"]
        }

        with patch("src.agents.analysis.check_context_or_escalate", return_value=escalation):
            result = results_analyzer_node(analysis_ready_state)

            assert result["awaiting_user_input"] is True
            assert result["ask_user_trigger"] == "context_overflow"
            assert len(result["pending_user_questions"]) > 0


class TestResultsAnalyzerStateUpdates:
    """Test state update correctness."""

    def test_filters_existing_comparisons_by_stage(self, analysis_ready_state):
        """Existing comparisons for other stages should be preserved."""
        analysis_ready_state["figure_comparisons"] = [
            {"figure_id": "Fig0", "stage_id": "stage_0_materials", "classification": "match"},
            {"figure_id": "Fig1", "stage_id": "stage_1_extinction", "classification": "old"},
        ]

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5
             }):

            result = results_analyzer_node(analysis_ready_state)

            # Old stage_0 comparison should be preserved
            stage_0_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_0_materials"]
            assert len(stage_0_comps) == 1
            
            # Current stage comparison should be new (not "old")
            stage_1_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_1_extinction"]
            assert len(stage_1_comps) >= 1
            assert stage_1_comps[0]["classification"] != "old"

    def test_analysis_feedback_preserved_when_unresolved(self, analysis_ready_state):
        """analysis_feedback should be preserved when targets are unresolved."""
        analysis_ready_state["analysis_feedback"] = "Previous feedback"

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value=None):  # No match = unresolved

            result = results_analyzer_node(analysis_ready_state)

            summary = result["analysis_summary"]
            if summary["unresolved_targets"]:
                assert result["analysis_feedback"] == "Previous feedback"

    def test_analysis_feedback_cleared_when_resolved(self, analysis_ready_state):
        """analysis_feedback should be cleared when all targets resolved."""
        analysis_ready_state["analysis_feedback"] = "Previous feedback"

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.3
             }):

            result = results_analyzer_node(analysis_ready_state)

            summary = result["analysis_summary"]
            if not summary["unresolved_targets"]:
                assert result["analysis_feedback"] is None


class TestResultsAnalyzerLLMIntegration:
    """Test LLM visual analysis integration."""

    def test_llm_updates_classification(self, analysis_ready_state):
        """LLM response should update overall classification."""
        llm_response = {
            "overall_classification": AnalysisClassification.EXCELLENT_MATCH,
            "summary": "Visual comparison confirms excellent match",
            "figure_comparisons": [{
                "figure_id": "Fig1",
                "shape_comparison": ["Peak shape matches well"],
                "reason_for_difference": "No significant differences"
            }]
        }

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5
             }), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=["fig1.png"]), \
             patch("src.agents.analysis.build_user_content_for_analyzer", return_value="content"), \
             patch("src.agents.analysis.call_agent_with_metrics", return_value=llm_response):

            result = results_analyzer_node(analysis_ready_state)

            assert result["analysis_overall_classification"] == AnalysisClassification.EXCELLENT_MATCH
            summary = result["analysis_summary"]
            assert "llm_qualitative_analysis" in summary
            
            comp = result["figure_comparisons"][0]
            assert comp["shape_comparison"] == ["Peak shape matches well"]

    def test_llm_exception_handled_gracefully(self, analysis_ready_state):
        """LLM exceptions should not crash analysis."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5
             }), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=["fig1.png"]), \
             patch("src.agents.analysis.build_user_content_for_analyzer", return_value="content"), \
             patch("src.agents.analysis.call_agent_with_metrics", side_effect=Exception("LLM error")):

            # Should not raise
            result = results_analyzer_node(analysis_ready_state)

            assert result["workflow_phase"] == "analysis"
            assert len(result["analysis_result_reports"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# ComparisonValidatorNode Tests
# ═══════════════════════════════════════════════════════════════════════


class TestComparisonValidatorVerdict:
    """Test comparison validator verdict logic."""

    def test_approve_when_all_comparisons_match(self, validated_analysis_state):
        """Should approve when all comparisons are present and match."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["workflow_phase"] == "comparison_validation"
            assert result["comparison_verdict"] == "approve"
            assert result["comparison_feedback"] == "All required comparisons present."
            assert result["analysis_feedback"] is None
            # Should NOT increment revision count on approve
            assert "analysis_revision_count" not in result

    def test_needs_revision_when_comparison_missing(self, validated_analysis_state):
        """Should need revision when comparison is missing for target."""
        validated_analysis_state["figure_comparisons"] = []  # No comparisons
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["workflow_phase"] == "comparison_validation"
            assert result["comparison_verdict"] == "needs_revision"
            assert "Fig1" in result["comparison_feedback"]
            assert result["analysis_feedback"] == result["comparison_feedback"]
            assert result["analysis_revision_count"] == 1

    def test_needs_revision_when_report_missing(self, validated_analysis_state):
        """Should need revision when quantitative report is missing."""
        validated_analysis_state["analysis_result_reports"] = []  # No reports
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "needs_revision"
            assert "Missing quantitative reports" in result["comparison_feedback"]
            assert result["analysis_revision_count"] == 1

    def test_needs_revision_when_pending_classification(self, validated_analysis_state):
        """Should need revision when comparisons are pending validation."""
        validated_analysis_state["figure_comparisons"][0]["classification"] = AnalysisClassification.PENDING_VALIDATION
        validated_analysis_state["analysis_result_reports"][0]["status"] = AnalysisClassification.PENDING_VALIDATION
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "needs_revision"
            assert "pending" in result["comparison_feedback"].lower()
            assert result["analysis_revision_count"] == 1

    def test_needs_revision_when_mismatch_classification(self, validated_analysis_state):
        """Should need revision when comparisons have mismatch classification."""
        validated_analysis_state["figure_comparisons"][0]["classification"] = AnalysisClassification.MISMATCH
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "needs_revision"
            assert result["analysis_revision_count"] == 1


class TestComparisonValidatorRevisionCount:
    """Test revision count handling."""

    def test_increments_revision_count_on_needs_revision(self, validated_analysis_state):
        """Should increment analysis_revision_count on needs_revision."""
        validated_analysis_state["figure_comparisons"] = []
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["analysis_revision_count"] == 1

    def test_respects_max_revisions(self, validated_analysis_state):
        """Should not increment beyond max revisions."""
        validated_analysis_state["figure_comparisons"] = []
        validated_analysis_state["analysis_revision_count"] = 3
        validated_analysis_state["runtime_config"] = {"max_analysis_revisions": 3}

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            # Should be capped at max
            assert result["analysis_revision_count"] == 3

    def test_uses_default_max_when_config_missing(self, validated_analysis_state):
        """Should use default max when runtime_config is missing."""
        validated_analysis_state["figure_comparisons"] = []
        validated_analysis_state["analysis_revision_count"] = 0
        validated_analysis_state["runtime_config"] = None

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["analysis_revision_count"] == 1


class TestComparisonValidatorNoTargets:
    """Test handling when stage has no targets."""

    def test_approve_when_no_targets_defined(self, validated_analysis_state):
        """Should approve when stage has no reproducible targets."""
        validated_analysis_state["plan"]["stages"][1]["targets"] = []
        validated_analysis_state["plan"]["stages"][1].pop("target_details", None)
        validated_analysis_state["figure_comparisons"] = []

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "approve"
            assert "no reproducible targets" in result["comparison_feedback"].lower()
            assert result["analysis_feedback"] is None


class TestComparisonValidatorReportValidation:
    """Test analysis report validation."""

    def test_detects_high_error_for_match_status(self, validated_analysis_state):
        """Should detect inconsistency: match status with high error."""
        validated_analysis_state["analysis_result_reports"][0]["quantitative_metrics"]["peak_position_error_percent"] = 15.0
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            # High error should flag issue
            assert result["comparison_verdict"] == "needs_revision"
            assert "Fig1" in result["comparison_feedback"]

    def test_detects_missing_metrics_for_excellent_precision(self, validated_analysis_state):
        """Should detect missing metrics for excellent precision requirement."""
        validated_analysis_state["analysis_result_reports"][0]["precision_requirement"] = "excellent"
        validated_analysis_state["analysis_result_reports"][0]["quantitative_metrics"] = {}
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "needs_revision"
            assert "excellent precision requires quantitative metrics" in result["comparison_feedback"]


class TestComparisonValidatorStageFiltering:
    """Test stage-based filtering."""

    def test_ignores_comparisons_for_other_stages(self, validated_analysis_state):
        """Should only consider comparisons for current stage."""
        validated_analysis_state["current_stage_id"] = "stage_1_extinction"
        validated_analysis_state["figure_comparisons"] = [
            {"stage_id": "stage_0_materials", "figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            # Should treat as missing for current stage
            assert result["comparison_verdict"] == "needs_revision"
            assert "Fig1" in result["comparison_feedback"]

    def test_ignores_reports_for_other_stages(self, validated_analysis_state):
        """Should only consider reports for current stage."""
        validated_analysis_state["current_stage_id"] = "stage_1_extinction"
        validated_analysis_state["analysis_result_reports"][0]["stage_id"] = "stage_0_materials"
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            # Should detect missing report for current stage
            assert result["comparison_verdict"] == "needs_revision"
            assert "Missing quantitative reports" in result["comparison_feedback"]


class TestComparisonValidatorEdgeCases:
    """Test edge cases."""

    def test_handles_none_current_stage_id(self, validated_analysis_state):
        """Should handle None current_stage_id gracefully."""
        validated_analysis_state["current_stage_id"] = None
        validated_analysis_state["figure_comparisons"] = []

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            # Should approve with no targets message
            assert result["comparison_verdict"] == "approve"
            assert "no reproducible targets" in result["comparison_feedback"].lower()

    def test_handles_none_figure_comparisons(self, validated_analysis_state):
        """Should handle None figure_comparisons gracefully."""
        validated_analysis_state["figure_comparisons"] = None
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "needs_revision"

    def test_handles_none_plan(self, validated_analysis_state):
        """Should handle None plan gracefully."""
        validated_analysis_state["plan"] = None
        validated_analysis_state["figure_comparisons"] = []

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            assert result["comparison_verdict"] == "approve"
            assert "no reproducible targets" in result["comparison_feedback"].lower()

    def test_skips_processing_when_awaiting_user_input(self, validated_analysis_state):
        """Should skip processing when awaiting_user_input is True."""
        validated_analysis_state["awaiting_user_input"] = True
        validated_analysis_state["figure_comparisons"] = []

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
            result = comparison_validator_node(validated_analysis_state)

            # Should return empty dict (no state changes) - this preserves awaiting_user_input=True
            # in the merged state since LangGraph merges the result dict into state
            assert result == {}
            # Verify workflow_phase was NOT set (processing was skipped)
            assert "workflow_phase" not in result

    def test_truncates_feedback_when_many_issues(self, validated_analysis_state):
        """Should truncate feedback when many issues exist."""
        validated_analysis_state["plan"]["stages"][1]["targets"] = ["F1", "F2", "F3", "F4", "F5"]
        validated_analysis_state["figure_comparisons"] = []
        validated_analysis_state["analysis_revision_count"] = 0

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.validate_analysis_reports", return_value=["I1", "I2", "I3", "I4"]):
            result = comparison_validator_node(validated_analysis_state)

            assert "more)" in result["comparison_feedback"]


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests: Analyzer -> Validator Flow
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzerValidatorIntegration:
    """Test integration between analyzer and validator."""

    def test_successful_flow_analyzer_to_validator(self, analysis_ready_state):
        """Test full successful flow from analyzer to validator."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5,
                 "peak_position_paper": 700.0,
                 "peak_position_sim": 703.5
             }):

            # Run analyzer
            analyzer_result = results_analyzer_node(analysis_ready_state)

            # Merge analyzer results into state
            merged_state = {**analysis_ready_state, **analyzer_result}

            # Run validator
            with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
                validator_result = comparison_validator_node(merged_state)

            # Verify full flow succeeded
            assert analyzer_result["workflow_phase"] == "analysis"
            assert validator_result["workflow_phase"] == "comparison_validation"
            assert validator_result["comparison_verdict"] == "approve"

    def test_revision_flow_when_analysis_fails(self, analysis_ready_state):
        """Test that failed analysis triggers revision in validator."""
        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value=None):  # Missing output

            # Run analyzer - should have missing targets
            analyzer_result = results_analyzer_node(analysis_ready_state)

            # Merge results
            merged_state = {**analysis_ready_state, **analyzer_result}
            merged_state["analysis_revision_count"] = 0

            # Run validator
            with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
                validator_result = comparison_validator_node(merged_state)

            # Verify revision triggered
            assert validator_result["comparison_verdict"] == "needs_revision"
            assert validator_result["analysis_revision_count"] == 1

    def test_state_propagation_through_flow(self, analysis_ready_state):
        """Test that state updates propagate correctly through flow."""
        analysis_ready_state["analysis_revision_count"] = 2
        analysis_ready_state["existing_field"] = "preserved"

        with patch("src.agents.analysis.check_context_or_escalate", return_value=None), \
             patch("src.agents.analysis.build_agent_prompt", return_value="prompt"), \
             patch("src.agents.analysis.match_output_file", return_value="output.csv"), \
             patch("src.agents.analysis.load_numeric_series", return_value=(
                 np.array([400, 500, 700]), np.array([0.1, 0.5, 0.9])
             )), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={
                 "peak_position_error_percent": 0.5
             }):

            analyzer_result = results_analyzer_node(analysis_ready_state)
            merged_state = {**analysis_ready_state, **analyzer_result}

            # Verify existing fields preserved
            assert merged_state["existing_field"] == "preserved"
            assert merged_state["analysis_revision_count"] == 2

            with patch("src.agents.analysis.check_context_or_escalate", return_value=None):
                validator_result = comparison_validator_node(merged_state)

            # On approve, revision count should not increment
            assert "analysis_revision_count" not in validator_result or \
                   validator_result.get("analysis_revision_count", 2) == 2
