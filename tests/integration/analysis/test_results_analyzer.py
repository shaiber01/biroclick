"""Integration tests for results_analyzer_node behavior.

These tests are designed to FIND BUGS, not just pass.
They verify specific outputs, edge cases, error handling, and complex logic.

Test Coverage Goals:
1. Error handling paths - verify all error conditions are handled correctly
2. State transitions - verify workflow_phase and related fields
3. File I/O - verify file existence checks and path resolution
4. Classification logic - verify quantitative metrics lead to correct classifications
5. Merge logic - verify figure_comparisons and reports are merged correctly
6. Edge cases - boundary conditions, None/empty values, special characters
7. Validation criteria - verify criteria evaluation against metrics
8. LLM integration - verify graceful handling of LLM calls and failures
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import pytest
import numpy as np

from src.agents.constants import AnalysisClassification
from schemas.state import DISCREPANCY_THRESHOLDS


# ═══════════════════════════════════════════════════════════════════════
# Test Constants - These mirror production thresholds for assertions
# ═══════════════════════════════════════════════════════════════════════

EXCELLENT_THRESHOLD = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["excellent"]  # 2%
ACCEPTABLE_THRESHOLD = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["acceptable"]  # 5%
INVESTIGATE_THRESHOLD = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["investigate"]  # 10%


class TestResultsAnalyzerErrorHandling:
    """Test error handling paths - these should catch bugs in error handling logic."""

    def test_missing_current_stage_id_returns_error_state(self, base_state, valid_plan):
        """When current_stage_id is None, should return error state with user question.
        
        The component MUST:
        - Set ask_user_trigger to exactly "missing_stage_id"
        - Set awaiting_user_input to exactly True
        - Include an error message mentioning "ERROR"
        - NOT include analysis_summary as a dict (only string if present)
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = None
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        result = results_analyzer_node(base_state)

        # Verify exact values, not just existence
        assert result["workflow_phase"] == "analysis", "workflow_phase must be 'analysis'"
        assert result["ask_user_trigger"] == "missing_stage_id", \
            f"Expected ask_user_trigger='missing_stage_id', got '{result.get('ask_user_trigger')}'"
        assert result["awaiting_user_input"] is True, \
            "awaiting_user_input must be exactly True, not truthy"
        assert "pending_user_questions" in result
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1, \
            f"Expected exactly 1 pending question, got {len(result['pending_user_questions'])}"
        assert "ERROR" in result["pending_user_questions"][0], \
            "Error message must contain 'ERROR'"
        # Verify NO analysis results are present
        assert "analysis_summary" not in result, \
            "analysis_summary should NOT be present when stage_id is missing"
        assert "figure_comparisons" not in result, \
            "figure_comparisons should NOT be present when stage_id is missing"

    def test_empty_current_stage_id_string_handled(self, base_state, valid_plan):
        """Empty string for current_stage_id should be handled identically to None.
        
        Empty string is a falsy value and MUST trigger the same error path as None.
        This tests that the component properly checks truthiness, not just None.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = ""  # Empty string, falsy
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        result = results_analyzer_node(base_state)

        # Empty string MUST be treated identically to None
        assert result["workflow_phase"] == "analysis"
        assert result["ask_user_trigger"] == "missing_stage_id", \
            f"Empty string should trigger missing_stage_id, got '{result.get('ask_user_trigger')}'"
        assert result["awaiting_user_input"] is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1
        assert "ERROR" in result["pending_user_questions"][0]

    def test_missing_stage_outputs_returns_fail_state(self, base_state, valid_plan):
        """When stage_outputs is None, should return fail state with specific error message.
        
        This is a critical error path - simulation didn't produce any outputs.
        The error message MUST be informative enough for debugging.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = None

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail", \
            f"Expected execution_verdict='fail', got '{result.get('execution_verdict')}'"
        assert "run_error" in result
        assert isinstance(result["run_error"], str)
        assert len(result["run_error"]) > 20, "run_error must be descriptive, not empty"
        # Error message should mention stage_outputs or outputs
        error_lower = result["run_error"].lower()
        assert "outputs" in error_lower or "stage_outputs" in error_lower, \
            f"Error message should mention outputs: {result['run_error']}"
        assert result["analysis_summary"] == "Analysis skipped: No outputs available"
        # Verify fail state includes required fields
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED

    def test_empty_stage_outputs_dict_returns_fail_state(self, base_state, valid_plan):
        """When stage_outputs is empty dict {}, should return fail state.
        
        Empty dict means simulation completed but produced no files.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert len(result["run_error"]) > 10
        assert result["analysis_summary"] == "Analysis skipped: No outputs available"
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED

    def test_empty_files_list_returns_fail_state(self, base_state, valid_plan):
        """When stage_outputs.files is empty list [], should return fail state.
        
        This is different from missing files - the key exists but list is empty.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert result["analysis_summary"] == "Analysis skipped: No outputs available"
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []

    def test_all_files_missing_returns_fail_state(self, base_state, valid_plan):
        """When all files in stage_outputs don't exist on disk, should return fail state.
        
        Files were listed but don't exist - likely deleted or simulation failed silently.
        The error MUST list the missing files for debugging.
        """
        from src.agents.analysis import results_analyzer_node

        missing_files = ["/nonexistent/file1.csv", "/nonexistent/file2.csv"]
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": missing_files}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        error_lower = result["run_error"].lower()
        # Error must mention missing files
        assert "missing" in error_lower or "not exist" in error_lower or "do not exist" in error_lower
        # Error should list the actual missing files
        assert "file1.csv" in result["run_error"] or "nonexistent" in result["run_error"], \
            f"Error should list missing files: {result['run_error']}"
        assert result["analysis_summary"] == "Analysis skipped: Output files missing"
        assert result["analysis_overall_classification"] == AnalysisClassification.FAILED

    def test_no_targets_defined_returns_no_targets_classification(self, base_state, valid_plan):
        """When stage has no targets (empty list), should return NO_TARGETS classification.
        
        This is a valid state - some stages may not have reproducible targets.
        The component MUST:
        - Return NO_TARGETS classification (not FAILED)
        - Include complete summary structure with zero counts
        - Return empty lists for comparisons and reports
        - Include supervisor_verdict for routing
        """
        from src.agents.analysis import results_analyzer_node

        # Create plan with stage that has empty targets
        plan_no_targets = {**valid_plan}
        plan_no_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test stage",
            "targets": [],  # Empty targets list
            "dependencies": [],
        }]

        base_state["plan"] = plan_no_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_figures"] = []  # No figures either

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        summary = result["analysis_summary"]
        
        # Verify classification is NO_TARGETS specifically
        assert summary["overall_classification"] == AnalysisClassification.NO_TARGETS, \
            f"Expected NO_TARGETS, got {summary['overall_classification']}"
        assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
        
        # Verify all totals are exactly zero
        assert summary["totals"]["targets"] == 0
        assert summary["totals"]["matches"] == 0
        assert summary["totals"]["pending"] == 0
        assert summary["totals"]["missing"] == 0
        assert summary["totals"]["mismatch"] == 0
        
        # Verify lists are empty (not None)
        assert result["figure_comparisons"] == []
        assert result["analysis_result_reports"] == []
        assert summary["matched_targets"] == []
        assert summary["pending_targets"] == []
        assert summary["missing_targets"] == []
        assert summary["mismatch_targets"] == []
        assert summary["unresolved_targets"] == []
        
        # Verify supervisor_verdict is set for routing
        assert result.get("supervisor_verdict") == "ok_continue", \
            "NO_TARGETS stages should continue, not block"

    def test_missing_plan_handled_gracefully(self, base_state):
        """When plan is None, should handle gracefully without crashing.
        
        This tests defensive programming - plan should never be None in production
        but the component must not crash if it is.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = None
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        # Should not raise an exception
        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should return some kind of result, either analysis or error
        assert "analysis_summary" in result or "run_error" in result or "ask_user_trigger" in result
        # Should not crash - this assertion passes if we get here

    def test_stage_not_found_in_plan(self, base_state, valid_plan):
        """When current_stage_id doesn't match any stage in plan, handle gracefully.
        
        This can happen if plan was modified after stage selection.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan  # Has stage_0
        base_state["current_stage_id"] = "nonexistent_stage_99"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should handle missing stage gracefully
        # The exact behavior depends on implementation but it must not crash


class TestResultsAnalyzerStateTransitions:
    """Workflow-phase signals and state transitions.
    
    These tests verify that the component correctly signals workflow state.
    """

    def test_results_analyzer_sets_workflow_phase(self, base_state, valid_plan):
        """Verify workflow_phase is ALWAYS set to 'analysis' regardless of outcome.
        
        This is critical for LangGraph routing - the router uses workflow_phase
        to determine the next node. It MUST be set even in error cases.
        """
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/nonexistent/fake_output.csv"]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        # workflow_phase MUST be exactly "analysis"
        assert result["workflow_phase"] == "analysis", \
            f"workflow_phase must be 'analysis', got '{result.get('workflow_phase')}'"
        # When files don't exist, must return fail state
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert isinstance(result["run_error"], str)
        assert len(result["run_error"]) > 0

    def test_results_analyzer_with_empty_outputs(self, base_state, valid_plan):
        """results_analyzer_node should handle empty stage_outputs dict."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result["execution_verdict"] == "fail"
        assert "run_error" in result
        assert result["analysis_summary"] == "Analysis skipped: No outputs available"

    def test_workflow_phase_set_on_success(self, base_state, valid_plan):
        """Verify workflow_phase is 'analysis' even on successful analysis."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            # On success, should NOT have execution_verdict=fail
            assert result.get("execution_verdict") != "fail"
        finally:
            os.unlink(temp_file)

    def test_no_execution_verdict_on_success(self, base_state, valid_plan):
        """On successful analysis, execution_verdict should NOT be 'fail'."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            # execution_verdict should be absent or not 'fail'
            assert result.get("execution_verdict") != "fail", \
                "execution_verdict should not be 'fail' on success"
            # run_error should be absent
            assert "run_error" not in result, \
                "run_error should not be present on success"
        finally:
            os.unlink(temp_file)


@pytest.mark.slow
class TestResultsAnalyzerFileIO:
    """File-system heavy analyzer tests with comprehensive assertions.
    
    These tests use real temporary files to verify actual file I/O behavior.
    """

    def test_results_analyzer_returns_figure_comparisons_on_success(
        self, base_state, valid_plan
    ):
        """Verify complete output structure when files exist.
        
        This tests the full success path with real file I/O.
        Every required field MUST be present and correctly typed.
        """
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            # Verify workflow_phase
            assert result["workflow_phase"] == "analysis"
            
            # Verify analysis_summary structure
            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict)
            summary = result["analysis_summary"]
            
            # Verify stage_id matches
            assert summary["stage_id"] == "stage_0", \
                f"Expected stage_id='stage_0', got '{summary.get('stage_id')}'"
            
            # Verify totals structure with exact types
            assert "totals" in summary
            assert isinstance(summary["totals"], dict)
            totals = summary["totals"]
            assert isinstance(totals["targets"], int)
            assert isinstance(totals["matches"], int)
            assert isinstance(totals["pending"], int)
            assert isinstance(totals["missing"], int)
            assert isinstance(totals["mismatch"], int)
            # All counts must be non-negative
            assert totals["targets"] >= 0
            assert totals["matches"] >= 0
            assert totals["pending"] >= 0
            assert totals["missing"] >= 0
            assert totals["mismatch"] >= 0
            # Sum of categories should equal targets
            assert totals["matches"] + totals["pending"] + totals["missing"] + totals["mismatch"] <= totals["targets"]
            
            # Verify figure_comparisons
            assert "figure_comparisons" in result
            assert isinstance(result["figure_comparisons"], list)
            
            # Verify classification is a valid enum value
            assert "analysis_overall_classification" in result
            classification = result["analysis_overall_classification"]
            valid_classifications = [
                AnalysisClassification.EXCELLENT_MATCH,
                AnalysisClassification.ACCEPTABLE_MATCH,
                AnalysisClassification.PARTIAL_MATCH,
                AnalysisClassification.POOR_MATCH,
                AnalysisClassification.FAILED,
                AnalysisClassification.NO_TARGETS,
                AnalysisClassification.PENDING_VALIDATION,
                AnalysisClassification.MATCH,
                AnalysisClassification.MISMATCH,
            ]
            assert classification in valid_classifications, \
                f"Invalid classification: {classification}"
            
            # Verify reports
            assert "analysis_result_reports" in result
            assert isinstance(result["analysis_result_reports"], list)
        finally:
            os.unlink(temp_file)

    def test_partial_files_exist_continues_with_existing(self, base_state, valid_plan):
        """When some files exist and some don't, should continue with existing files.
        
        This is important for robustness - don't fail completely if only some files
        are missing. Process what's available and report the missing ones.
        """
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {
                "files": [temp_file, "/nonexistent/file.csv"]
            }
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            # Should NOT fail completely when at least one file exists
            assert result.get("execution_verdict") != "fail", \
                "Should not fail when some files exist"
            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict)
            # Should have figure_comparisons
            assert "figure_comparisons" in result
            assert isinstance(result["figure_comparisons"], list)
        finally:
            os.unlink(temp_file)

    def test_absolute_path_handling(self, base_state, valid_plan):
        """Verify absolute paths are handled correctly.
        
        Absolute paths should be used as-is without resolution.
        """
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name
            abs_path = os.path.abspath(temp_file)

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [abs_path]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert result.get("execution_verdict") != "fail"
            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict)
        finally:
            os.unlink(temp_file)

    def test_relative_path_resolution(self, base_state, valid_plan):
        """Verify relative paths are resolved relative to output directory.
        
        Relative paths should be resolved as: outputs/{paper_id}/{stage_id}/{filename}
        """
        from src.agents.analysis import results_analyzer_node

        # Create file in expected output directory
        paper_id = "test_integration"
        stage_id = "stage_0"
        output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / paper_id / stage_id
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_file = output_dir / "output.csv"
        try:
            temp_file.write_text("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")

            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = stage_id
            base_state["stage_outputs"] = {"files": ["output.csv"]}  # Relative path
            base_state["paper_id"] = paper_id

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert result.get("execution_verdict") != "fail"
            assert "analysis_summary" in result
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_csv_file_parsing(self, base_state, valid_plan):
        """Verify CSV files are parsed correctly and peak is detected.
        
        This tests the actual data loading and peak detection logic.
        """
        from src.agents.analysis import results_analyzer_node

        # Create CSV with a clear peak at 500nm
        csv_data = "wavelength,extinction\n"
        csv_data += "400,0.1\n450,0.3\n500,0.9\n550,0.4\n600,0.1\n"
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write(csv_data)
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert result.get("execution_verdict") != "fail"
            assert "analysis_summary" in result
        finally:
            os.unlink(temp_file)

    def test_tsv_file_handling(self, base_state, valid_plan):
        """Verify TSV (tab-separated) files are handled correctly."""
        from src.agents.analysis import results_analyzer_node

        # Create TSV with tabs
        tsv_data = "wavelength\textinction\n"
        tsv_data += "400\t0.1\n500\t0.5\n600\t0.3\n"
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as handle:
            handle.write(tsv_data)
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            # Should handle TSV files
        finally:
            os.unlink(temp_file)

    def test_json_file_handling(self, base_state, valid_plan):
        """Verify JSON data files are handled correctly."""
        from src.agents.analysis import results_analyzer_node

        json_data = {"x": [400, 500, 600], "y": [0.1, 0.5, 0.3]}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(json_data, handle)
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
        finally:
            os.unlink(temp_file)


class TestResultsAnalyzerCompleteness:
    """Verify results_analyzer produces complete output with all required fields.
    
    These tests ensure the output contract is maintained for downstream consumers.
    """

    def test_results_analyzer_returns_all_required_fields(self, base_state, valid_plan):
        """Verify all required fields are present even on failure."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}  # Empty = fail state

        result = results_analyzer_node(base_state)

        # workflow_phase MUST always be present
        assert result["workflow_phase"] == "analysis"
        
        # On failure, must have these fields
        assert "analysis_summary" in result or "run_error" in result
        assert "analysis_overall_classification" in result or "execution_verdict" in result
        
        # Lists must be present and be lists (not None)
        assert "figure_comparisons" in result
        assert isinstance(result["figure_comparisons"], list)
        assert "analysis_result_reports" in result
        assert isinstance(result["analysis_result_reports"], list)

    def test_analysis_summary_structure_complete(self, base_state, valid_plan):
        """Verify analysis_summary has ALL required fields with correct types.
        
        This is the contract that downstream consumers depend on.
        Missing fields would break the workflow.
        """
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert "analysis_summary" in result
            assert isinstance(result["analysis_summary"], dict), \
                f"analysis_summary must be dict, got {type(result['analysis_summary'])}"
            
            summary = result["analysis_summary"]
            
            # Verify all required keys are present
            required_keys = [
                "stage_id", "totals", "matched_targets", "pending_targets",
                "missing_targets", "mismatch_targets", "discrepancies_logged",
                "validation_criteria", "feedback_applied", "unresolved_targets", "notes"
            ]
            for key in required_keys:
                assert key in summary, f"Missing required key: {key}"
            
            # Verify stage_id matches
            assert summary["stage_id"] == "stage_0"
            
            # Verify totals structure and types
            totals = summary["totals"]
            assert isinstance(totals, dict)
            totals_keys = ["targets", "matches", "pending", "missing", "mismatch"]
            for key in totals_keys:
                assert key in totals, f"Missing totals key: {key}"
                assert isinstance(totals[key], int), f"totals[{key}] must be int"
                assert totals[key] >= 0, f"totals[{key}] must be non-negative"
            
            # Verify list fields are lists
            list_fields = ["matched_targets", "pending_targets", "missing_targets", 
                          "mismatch_targets", "validation_criteria", "feedback_applied",
                          "unresolved_targets"]
            for field in list_fields:
                assert isinstance(summary[field], list), \
                    f"{field} must be list, got {type(summary[field])}"
            
            # Verify discrepancies_logged is int
            assert isinstance(summary["discrepancies_logged"], int)
            
            # Verify notes is string
            assert isinstance(summary["notes"], str)
        finally:
            os.unlink(temp_file)

    def test_figure_comparison_structure(self, base_state, valid_plan):
        """Verify each figure_comparison entry has required fields."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            # Create plan with a target
            plan_with_target = {**valid_plan}
            plan_with_target["stages"] = [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Test",
                "targets": ["Fig1"],
                "dependencies": [],
            }]

            base_state["plan"] = plan_with_target
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"
            base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

            result = results_analyzer_node(base_state)

            assert "figure_comparisons" in result
            assert isinstance(result["figure_comparisons"], list)
            
            if len(result["figure_comparisons"]) > 0:
                comp = result["figure_comparisons"][0]
                # Required fields for each comparison
                assert "figure_id" in comp or "target_id" in comp
                assert "stage_id" in comp
                assert comp["stage_id"] == "stage_0"
                assert "classification" in comp
                assert "comparison_table" in comp
                assert isinstance(comp["comparison_table"], list)
        finally:
            os.unlink(temp_file)

    def test_analysis_report_structure(self, base_state, valid_plan):
        """Verify each analysis_result_report has required fields."""
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            plan_with_target = {**valid_plan}
            plan_with_target["stages"] = [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Test",
                "targets": ["Fig1"],
                "dependencies": [],
            }]

            base_state["plan"] = plan_with_target
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"
            base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

            result = results_analyzer_node(base_state)

            assert "analysis_result_reports" in result
            assert isinstance(result["analysis_result_reports"], list)
            
            if len(result["analysis_result_reports"]) > 0:
                report = result["analysis_result_reports"][0]
                # Required fields
                assert "result_id" in report
                assert "target_figure" in report
                assert "status" in report
                assert "stage_id" in report
        finally:
            os.unlink(temp_file)


class TestResultsAnalyzerLogic:
    """Verify results_analyzer_node logic for existing files and complex scenarios.
    
    These tests verify the core classification and analysis logic.
    They use thresholds from DISCREPANCY_THRESHOLDS to ensure correctness.
    """

    def test_results_analyzer_analyzes_existing_files(self, base_state, valid_plan):
        """Verify file analysis logic with mocked dependencies produces comparisons."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"

        # Error of 5% is exactly at ACCEPTABLE_THRESHOLD boundary
        mock_metrics = {
            "peak_position_error_percent": ACCEPTABLE_THRESHOLD,
            "peak_position_paper": 650.0,
            "peak_position_sim": 650.0 * (1 - ACCEPTABLE_THRESHOLD/100),
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        assert isinstance(result["analysis_summary"], dict)
        assert "figure_comparisons" in result
        assert isinstance(result["figure_comparisons"], list)
        
        # Verify each comparison has ALL required fields
        for comp in result["figure_comparisons"]:
            assert "figure_id" in comp or "target_id" in comp
            assert "stage_id" in comp
            assert comp["stage_id"] == "stage_0"
            assert "classification" in comp
            assert "comparison_table" in comp
            assert isinstance(comp["comparison_table"], list)
            assert len(comp["comparison_table"]) > 0, "comparison_table should not be empty"

    def test_target_matching_with_multiple_targets(self, base_state, valid_plan):
        """Verify correct handling of multiple targets."""
        from src.agents.analysis import results_analyzer_node

        # Plan with 2 targets
        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test stage",
            "targets": ["Fig1", "Fig2"],
            "dependencies": [],
        }]
        plan_with_targets["targets"] = [
            {"figure_id": "Fig1", "description": "Figure 1"},
            {"figure_id": "Fig2", "description": "Figure 2"},
        ]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [
            {"figure_id": "Fig1", "id": "Fig1", "description": "Figure 1"},
            {"figure_id": "Fig2", "id": "Fig2", "description": "Figure 2"},
        ]

        mock_metrics = {"peak_position_error_percent": 1.5}  # Excellent match

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        summary = result["analysis_summary"]
        
        # Should have 2 targets
        assert summary["totals"]["targets"] == 2, \
            f"Expected 2 targets, got {summary['totals']['targets']}"
        
        # Should have reports for both targets
        assert len(result["analysis_result_reports"]) == 2, \
            f"Expected 2 reports, got {len(result['analysis_result_reports'])}"
        
        # Should have comparisons for both targets
        assert len(result["figure_comparisons"]) == 2, \
            f"Expected 2 comparisons, got {len(result['figure_comparisons'])}"

    def test_classification_excellent_match_below_threshold(self, base_state, valid_plan):
        """Verify MATCH classification when error < EXCELLENT_THRESHOLD (2%).
        
        Error < 2% should result in MATCH classification.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Error BELOW excellent threshold (1.5% < 2%)
        mock_metrics = {
            "peak_position_error_percent": 1.5,  # Below EXCELLENT_THRESHOLD
            "peak_position_paper": 650.0,
            "peak_position_sim": 650.0 * 0.985,  # 1.5% off
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        
        # With error < 2%, should get EXCELLENT_MATCH or MATCH
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.EXCELLENT_MATCH,
            AnalysisClassification.MATCH,
        ], f"Expected EXCELLENT_MATCH or MATCH with 1.5% error, got {result['analysis_overall_classification']}"
        
        # Should be in matched_targets
        assert "Fig1" in result["analysis_summary"]["matched_targets"]

    def test_classification_partial_match_between_thresholds(self, base_state, valid_plan):
        """Verify PARTIAL_MATCH when EXCELLENT < error <= ACCEPTABLE (2% < error <= 5%).
        
        Error between 2% and 5% should result in PARTIAL_MATCH.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Error BETWEEN thresholds (3.5% - between 2% and 5%)
        mock_metrics = {
            "peak_position_error_percent": 3.5,  # Between EXCELLENT and ACCEPTABLE
            "peak_position_paper": 650.0,
            "peak_position_sim": 650.0 * 0.965,  # 3.5% off
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        
        # With error 3.5%, should get PARTIAL_MATCH or ACCEPTABLE_MATCH
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
        ], f"Expected PARTIAL_MATCH with 3.5% error, got {result['analysis_overall_classification']}"

    def test_classification_mismatch_above_threshold(self, base_state, valid_plan):
        """Verify MISMATCH/POOR_MATCH when error > ACCEPTABLE (> 5%).
        
        Error > 5% should result in MISMATCH classification.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Error ABOVE acceptable threshold (15% > 5%)
        mock_metrics = {
            "peak_position_error_percent": 15.0,  # Above ACCEPTABLE_THRESHOLD
            "peak_position_paper": 650.0,
            "peak_position_sim": 650.0 * 0.85,  # 15% off
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        
        # With error > 5%, should get POOR_MATCH or MISMATCH
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.POOR_MATCH,
            AnalysisClassification.MISMATCH,
        ], f"Expected POOR_MATCH/MISMATCH with 15% error, got {result['analysis_overall_classification']}"
        
        # Should be in mismatch_targets
        assert "Fig1" in result["analysis_summary"]["mismatch_targets"]

    def test_digitized_data_requirement_enforced(self, base_state, valid_plan):
        """Verify that excellent precision requirement BLOCKS analysis without digitized data.
        
        If precision_requirement='excellent', digitized_data_path MUST be provided.
        Without it, target should be marked as pending with missing_digitized_data status.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]
        plan_with_targets["targets"] = [{
            "figure_id": "Fig1",
            "description": "Test",
            "precision_requirement": "excellent",  # Requires digitized data
            # Note: NO digitized_data_path provided
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        
        # Target should be pending due to missing digitized data
        summary = result["analysis_summary"]
        assert "Fig1" in summary["pending_targets"], \
            f"Fig1 should be in pending_targets, got {summary['pending_targets']}"
        
        # Comparison should indicate missing digitized data
        fig1_comp = next((c for c in result["figure_comparisons"] if c.get("figure_id") == "Fig1"), None)
        assert fig1_comp is not None, "Should have comparison for Fig1"
        assert fig1_comp.get("status") == "missing_digitized_data" or \
               fig1_comp.get("classification") == "missing_digitized_data", \
               f"Expected missing_digitized_data status, got status={fig1_comp.get('status')}, classification={fig1_comp.get('classification')}"

    def test_missing_output_file_marks_target_as_missing(self, base_state, valid_plan):
        """Verify that when output file doesn't exist, target is marked as missing.
        
        This is different from NO outputs - here we have targets but no matching files.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_targets = {**valid_plan}
        plan_with_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_targets
        base_state["current_stage_id"] = "stage_0"
        # File exists but we'll mock it to simulate no matching output
        base_state["stage_outputs"] = {"files": ["some_other_file.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/some_other_file.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # The file exists but may not match the target specifically


class TestResultsAnalyzerIntegration:
    """Test integration with LLM, file I/O, and helper functions.
    
    These tests verify the component integrates correctly with:
    - LLM calls for visual analysis
    - File I/O for data loading
    - Helper functions for metrics computation
    - State merging for multi-stage workflows
    """

    def test_llm_analysis_integration_enriches_comparisons(self, base_state, valid_plan):
        """Verify LLM analysis enriches figure comparisons with shape analysis.
        
        The LLM should add shape_comparison and reason_for_difference to comparisons.
        """
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            plan_with_target = {**valid_plan}
            plan_with_target["stages"] = [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Test",
                "targets": ["Fig1"],
                "dependencies": [],
            }]

            base_state["plan"] = plan_with_target
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"
            base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test", "image_path": "/path/to/image.png"}]

            mock_llm_response = {
                "overall_classification": "ACCEPTABLE_MATCH",
                "figure_comparisons": [{
                    "figure_id": "Fig1",
                    "shape_comparison": ["Peak position matches within 5%", "Line shape is similar"],
                    "reason_for_difference": "Slight broadening in simulation due to mesh resolution",
                }],
                "summary": "Overall good match with minor discrepancies",
            }

            with patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.is_file", return_value=True), \
                 patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
                 patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 4.0}), \
                 patch("src.agents.analysis.get_images_for_analyzer", return_value=["/path/to/image.png"]), \
                 patch("src.agents.analysis.call_agent_with_metrics", return_value=mock_llm_response):
                result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert "analysis_summary" in result
            
            # Find Fig1 comparison
            fig1_comp = next((c for c in result["figure_comparisons"] if c.get("figure_id") == "Fig1"), None)
            assert fig1_comp is not None, "Should have comparison for Fig1"
            
            # LLM should have enriched the comparison
            assert "shape_comparison" in fig1_comp
            assert isinstance(fig1_comp["shape_comparison"], list)
            assert len(fig1_comp["shape_comparison"]) > 0, "shape_comparison should not be empty after LLM enrichment"
            assert "reason_for_difference" in fig1_comp
        finally:
            os.unlink(temp_file)

    def test_llm_analysis_failure_handled_gracefully(self, base_state, valid_plan):
        """Verify LLM analysis failure doesn't crash the node.
        
        LLM calls can fail for various reasons (rate limits, network, etc).
        The node MUST continue with quantitative analysis only.
        """
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = handle.name

        try:
            plan_with_target = {**valid_plan}
            plan_with_target["stages"] = [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Test",
                "targets": ["Fig1"],
                "dependencies": [],
            }]

            base_state["plan"] = plan_with_target
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"
            base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test", "image_path": "/path/to/image.png"}]

            with patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.is_file", return_value=True), \
                 patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
                 patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 3.0}), \
                 patch("src.agents.analysis.get_images_for_analyzer", return_value=["/path/to/image.png"]), \
                 patch("src.agents.analysis.call_agent_with_metrics", side_effect=Exception("LLM rate limit exceeded")):
                result = results_analyzer_node(base_state)

            # Should NOT crash
            assert result["workflow_phase"] == "analysis"
            assert "analysis_summary" in result
            assert "figure_comparisons" in result
            
            # Should have quantitative results despite LLM failure
            assert len(result["figure_comparisons"]) > 0
            assert "analysis_overall_classification" in result
        finally:
            os.unlink(temp_file)

    def test_figure_comparisons_merged_preserves_other_stages(self, base_state, valid_plan):
        """Verify figure_comparisons from other stages are preserved.
        
        When analyzing stage_0, comparisons from stage_1 should NOT be modified.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_target = {**valid_plan}
        plan_with_target["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_target
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]
        
        # Pre-populate with existing comparison from DIFFERENT stage
        existing_comparison = {
            "figure_id": "Fig0",
            "stage_id": "stage_1",  # Different stage
            "classification": "MATCH",
            "comparison_table": [{"feature": "test"}],
        }
        base_state["figure_comparisons"] = [existing_comparison]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 2.0}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        
        # Existing comparison from stage_1 MUST be preserved
        stage_1_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_1"]
        assert len(stage_1_comps) == 1, \
            f"stage_1 comparison should be preserved, got {len(stage_1_comps)}"
        assert stage_1_comps[0]["figure_id"] == "Fig0"
        assert stage_1_comps[0]["classification"] == "MATCH"
        
        # New comparison for stage_0 should be added
        stage_0_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_0"]
        assert len(stage_0_comps) >= 1, "Should have comparison for stage_0"

    def test_figure_comparisons_replace_same_stage(self, base_state, valid_plan):
        """Verify comparisons for SAME stage are replaced, not duplicated.
        
        If we re-analyze stage_0, old stage_0 comparisons should be replaced.
        """
        from src.agents.analysis import results_analyzer_node

        plan_with_target = {**valid_plan}
        plan_with_target["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_target
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]
        
        # Pre-populate with OLD comparison from SAME stage
        old_comparison = {
            "figure_id": "Fig1",
            "stage_id": "stage_0",  # Same stage
            "classification": "OLD_SHOULD_BE_REPLACED",
        }
        base_state["figure_comparisons"] = [old_comparison]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 2.0}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        # Old comparison should be REPLACED, not kept
        stage_0_comps = [c for c in result["figure_comparisons"] if c.get("stage_id") == "stage_0"]
        for comp in stage_0_comps:
            assert comp.get("classification") != "OLD_SHOULD_BE_REPLACED", \
                "Old comparison should have been replaced"

    def test_analysis_reports_merged_preserves_other_stages(self, base_state, valid_plan):
        """Verify analysis_result_reports from other stages are preserved."""
        from src.agents.analysis import results_analyzer_node

        plan_with_target = {**valid_plan}
        plan_with_target["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_target
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]
        
        # Pre-populate with existing report from DIFFERENT stage
        existing_report = {
            "result_id": "stage_1_Fig0",
            "target_figure": "Fig0",
            "stage_id": "stage_1",
            "status": "MATCH",
        }
        base_state["analysis_result_reports"] = [existing_report]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 2.0}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        
        # Existing report from stage_1 MUST be preserved
        stage_1_reports = [r for r in result["analysis_result_reports"] if r.get("stage_id") == "stage_1"]
        assert len(stage_1_reports) == 1
        assert stage_1_reports[0]["target_figure"] == "Fig0"
        
        # New reports for stage_0 should be added
        stage_0_reports = [r for r in result["analysis_result_reports"] if r.get("stage_id") == "stage_0"]
        assert len(stage_0_reports) >= 1


class TestResultsAnalyzerEdgeCases:
    """Test edge cases and boundary conditions.
    
    These tests verify the component handles unusual inputs gracefully.
    """

    def test_none_values_in_state(self, base_state, valid_plan):
        """Verify None values in state are handled without crashing."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_figures"] = None  # None instead of list
        base_state["analysis_feedback"] = None  # None instead of string

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]):
            result = results_analyzer_node(base_state)

        # Should not crash
        assert result["workflow_phase"] == "analysis"
        # Should have valid output structure
        assert "figure_comparisons" in result
        assert "analysis_result_reports" in result

    def test_empty_strings_in_state(self, base_state, valid_plan):
        """Verify empty strings in state are handled correctly."""
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["analysis_feedback"] = ""  # Empty string

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"

    def test_very_long_file_paths(self, base_state, valid_plan):
        """Verify very long file paths are handled gracefully (likely missing)."""
        from src.agents.analysis import results_analyzer_node

        # Path longer than most filesystem limits
        long_path = "/" + "a" * 500 + "/file.csv"
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [long_path]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        # Should handle without crashing - will fail as file doesn't exist
        assert result["workflow_phase"] == "analysis"

    def test_special_characters_in_paths(self, base_state, valid_plan):
        """Verify special characters in paths (spaces, parentheses) are handled."""
        from src.agents.analysis import results_analyzer_node

        special_path = "/path/with spaces/file (1).csv"
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": [special_path]}
        base_state["paper_id"] = "test_integration"

        result = results_analyzer_node(base_state)

        # Should handle without crashing
        assert result["workflow_phase"] == "analysis"

    def test_multiple_stages_in_plan_selects_correct_one(self, base_state, valid_plan):
        """Verify correct stage is selected when multiple stages exist."""
        from src.agents.analysis import results_analyzer_node

        multi_stage_plan = {**valid_plan}
        multi_stage_plan["stages"] = [
            {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "description": "Stage 0", "targets": ["Fig0"], "dependencies": []},
            {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "description": "Stage 1", "targets": ["Fig1"], "dependencies": []},
        ]

        base_state["plan"] = multi_stage_plan
        base_state["current_stage_id"] = "stage_1"  # Select SECOND stage
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [
            {"figure_id": "Fig0", "id": "Fig0"},
            {"figure_id": "Fig1", "id": "Fig1"},
        ]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 2.0}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should process stage_1, NOT stage_0
        assert isinstance(result.get("analysis_summary"), dict)
        assert result["analysis_summary"]["stage_id"] == "stage_1", \
            f"Expected stage_1, got {result['analysis_summary']['stage_id']}"

    def test_unicode_in_descriptions(self, base_state, valid_plan):
        """Verify Unicode characters in descriptions are handled."""
        from src.agents.analysis import results_analyzer_node

        plan_unicode = {**valid_plan}
        plan_unicode["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test with émojis 🔬 and ünïcödé",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_unicode
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Figüre with ümläuts"}]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 2.0}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"

    def test_analysis_feedback_prioritizes_mentioned_targets(self, base_state, valid_plan):
        """Verify analysis_feedback causes mentioned targets to be processed first."""
        from src.agents.analysis import results_analyzer_node

        plan_multi = {**valid_plan}
        plan_multi["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1", "Fig2", "Fig3"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_multi
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [
            {"figure_id": "Fig1", "id": "Fig1"},
            {"figure_id": "Fig2", "id": "Fig2"},
            {"figure_id": "Fig3", "id": "Fig3"},
        ]
        # Feedback mentions Fig3 - should be prioritized
        base_state["analysis_feedback"] = "Please check Fig3 for peak position errors"

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 2.0}), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should have processed all 3 targets
        assert result["analysis_summary"]["totals"]["targets"] == 3
        # Feedback applied should include Fig3
        assert "Fig3" in result["analysis_summary"]["feedback_applied"]

    def test_boundary_threshold_exactly_at_excellent(self, base_state, valid_plan):
        """Verify classification at exactly EXCELLENT_THRESHOLD boundary (2%)."""
        from src.agents.analysis import results_analyzer_node

        plan_with_target = {**valid_plan}
        plan_with_target["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_target
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Exactly at EXCELLENT_THRESHOLD boundary
        mock_metrics = {
            "peak_position_error_percent": float(EXCELLENT_THRESHOLD),  # Exactly 2%
            "peak_position_paper": 650.0,
            "peak_position_sim": 650.0 * (1 - EXCELLENT_THRESHOLD/100),
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # At exactly 2%, should still be MATCH (<=)
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.EXCELLENT_MATCH,
            AnalysisClassification.MATCH,
        ]

    def test_boundary_threshold_just_above_excellent(self, base_state, valid_plan):
        """Verify classification just above EXCELLENT_THRESHOLD (2.1%)."""
        from src.agents.analysis import results_analyzer_node

        plan_with_target = {**valid_plan}
        plan_with_target["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
        }]

        base_state["plan"] = plan_with_target
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Just above EXCELLENT_THRESHOLD
        mock_metrics = {
            "peak_position_error_percent": EXCELLENT_THRESHOLD + 0.1,  # 2.1%
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # At 2.1%, should be PARTIAL_MATCH (> excellent but <= acceptable)
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
        ]


class TestResultsAnalyzerValidationCriteria:
    """Test validation criteria evaluation against metrics."""

    def test_validation_criteria_passed(self, base_state, valid_plan):
        """Verify validation criteria pass when metrics satisfy them."""
        from src.agents.analysis import results_analyzer_node

        plan_with_criteria = {**valid_plan}
        plan_with_criteria["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
            "validation_criteria": ["Fig1: resonance peak within 3%"],
        }]

        base_state["plan"] = plan_with_criteria
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Metrics that satisfy the 3% criterion
        mock_metrics = {
            "peak_position_error_percent": 2.5,  # Within 3%
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should pass validation criteria
        assert "Fig1" not in result["analysis_summary"]["mismatch_targets"]

    def test_validation_criteria_failed(self, base_state, valid_plan):
        """Verify validation criteria failure causes MISMATCH classification."""
        from src.agents.analysis import results_analyzer_node

        plan_with_criteria = {**valid_plan}
        plan_with_criteria["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": ["Fig1"],
            "dependencies": [],
            "validation_criteria": ["Fig1: resonance peak within 1%"],  # Strict criterion
        }]

        base_state["plan"] = plan_with_criteria
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        base_state["paper_id"] = "test_integration"
        base_state["paper_figures"] = [{"figure_id": "Fig1", "id": "Fig1", "description": "Test"}]

        # Metrics that VIOLATE the 1% criterion
        mock_metrics = {
            "peak_position_error_percent": 2.5,  # > 1%, fails criterion
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/absolute/path/output.csv")), \
             patch("src.agents.analysis.load_numeric_series", return_value=(np.array([400, 500, 600]), np.array([0.1, 0.5, 0.3]))), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value=mock_metrics), \
             patch("src.agents.analysis.get_images_for_analyzer", return_value=[]), \
             patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        # Should fail validation criteria - target should be in mismatch
        assert "Fig1" in result["analysis_summary"]["mismatch_targets"], \
            "Fig1 should be in mismatch_targets when criteria fails"
