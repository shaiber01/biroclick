"""Unit tests for src/agents/helpers/validation.py"""

import pytest
from src.agents.constants import AnalysisClassification

from src.agents.helpers.validation import (
    classify_percent_error,
    classification_from_metrics,
    evaluate_validation_criteria,
    extract_targets_from_feedback,
    match_output_file,
    normalize_output_file_entry,
    collect_expected_outputs,
    collect_expected_columns,
    match_expected_files,
    stage_comparisons_for_stage,
    analysis_reports_for_stage,
    validate_analysis_reports,
    breakdown_comparison_classifications,
)


class TestClassifyPercentError:
    """Tests for classify_percent_error function."""

    def test_excellent_match(self):
        """Should classify small errors as match."""
        assert classify_percent_error(0.5) == AnalysisClassification.MATCH
        assert classify_percent_error(1.0) == AnalysisClassification.MATCH

    def test_acceptable_partial_match(self):
        """Should classify moderate errors as partial_match."""
        assert classify_percent_error(3.0) == AnalysisClassification.PARTIAL_MATCH
        assert classify_percent_error(5.0) == AnalysisClassification.PARTIAL_MATCH

    def test_large_error_mismatch(self):
        """Should classify large errors as mismatch."""
        assert classify_percent_error(15.0) == AnalysisClassification.MISMATCH
        assert classify_percent_error(50.0) == AnalysisClassification.MISMATCH


class TestClassificationFromMetrics:
    """Tests for classification_from_metrics function."""

    def test_no_reference_returns_pending(self):
        """Should return pending_validation without reference data."""
        result = classification_from_metrics({}, "excellent", has_reference=False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_qualitative_always_matches(self):
        """Should return match for qualitative precision requirement."""
        result = classification_from_metrics({}, "qualitative", has_reference=True)
        assert result == AnalysisClassification.MATCH

    def test_uses_peak_position_error(self):
        """Should classify based on peak position error."""
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        metrics = {"peak_position_error_percent": 20.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH

    def test_falls_back_to_rmse(self):
        """Should use RMSE when peak error not available."""
        metrics = {"normalized_rmse_percent": 3.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        metrics = {"normalized_rmse_percent": 10.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH
        
        metrics = {"normalized_rmse_percent": 20.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH

    def test_returns_pending_without_metrics(self):
        """Should return pending when no metrics available."""
        result = classification_from_metrics({}, "excellent", has_reference=True)
        assert result == AnalysisClassification.PENDING_VALIDATION


class TestEvaluateValidationCriteria:
    """Tests for evaluate_validation_criteria function."""

    def test_empty_criteria_passes(self):
        """Should pass with no criteria."""
        passed, failures = evaluate_validation_criteria({}, [])
        assert passed is True
        assert failures == []

    def test_resonance_within_percent(self):
        """Should evaluate resonance within percent criteria."""
        criteria = ["resonance within 5%"]
        
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 3.0}, criteria
        )
        assert passed is True
        
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 7.0}, criteria
        )
        assert passed is False
        assert len(failures) == 1

    def test_peak_within_percent(self):
        """Should evaluate peak within percent criteria."""
        criteria = ["peak within 10%"]
        
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 8.0}, criteria
        )
        assert passed is True

    def test_normalized_rmse_max(self):
        """Should evaluate normalized RMSE max criteria."""
        criteria = ["normalized rmse <= 5%"]
        
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 3.0}, criteria
        )
        assert passed is True
        
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 8.0}, criteria
        )
        assert passed is False

    def test_correlation_min(self):
        """Should evaluate correlation min criteria."""
        criteria = ["correlation >= 0.9"]
        
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.95}, criteria
        )
        assert passed is True
        
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.85}, criteria
        )
        assert passed is False

    def test_missing_metric_reports_failure(self):
        """Should report failure when metric is missing."""
        criteria = ["resonance within 5%"]
        
        passed, failures = evaluate_validation_criteria({}, criteria)
        assert passed is False
        assert "missing metric" in failures[0]


class TestExtractTargetsFromFeedback:
    """Tests for extract_targets_from_feedback function."""

    def test_empty_feedback_returns_empty(self):
        """Should return empty list for empty feedback."""
        assert extract_targets_from_feedback(None, ["Fig1"]) == []
        assert extract_targets_from_feedback("", ["Fig1"]) == []

    def test_extracts_figure_references(self):
        """Should extract figure references from feedback."""
        feedback = "Please check Fig1 and Fig 2a for discrepancies."
        known = ["Fig1", "Fig2a", "Fig3"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert "Fig1" in result
        assert "Fig2a" in result

    def test_ignores_unknown_figures(self):
        """Should ignore figures not in known list."""
        feedback = "Check Fig99 for issues"
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert result == []

    def test_case_insensitive(self):
        """Should match case-insensitively."""
        feedback = "See fig1 and FIG2"
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2


class TestMatchOutputFile:
    """Tests for match_output_file function."""

    def test_empty_list_returns_none(self):
        """Should return None for empty file list."""
        assert match_output_file([], "Fig1") is None

    def test_matches_by_target_id(self):
        """Should match files containing target ID."""
        files = ["fig1_spectrum.csv", "fig2_absorption.csv"]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"

    def test_returns_first_file_as_fallback(self):
        """Should return first file when no match found."""
        files = ["output.csv", "results.csv"]
        
        result = match_output_file(files, "fig99")
        
        assert result == "output.csv"

    def test_handles_dict_entries(self):
        """Should handle dict file entries."""
        files = [
            {"path": "fig1_spectrum.csv", "format": "csv"},
            {"file": "fig2_absorption.csv"}
        ]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"


class TestNormalizeOutputFileEntry:
    """Tests for normalize_output_file_entry function."""

    def test_string_entry(self):
        """Should return string as-is."""
        assert normalize_output_file_entry("test.csv") == "test.csv"

    def test_dict_with_path(self):
        """Should extract path from dict."""
        assert normalize_output_file_entry({"path": "test.csv"}) == "test.csv"

    def test_dict_with_file(self):
        """Should extract file from dict."""
        assert normalize_output_file_entry({"file": "test.csv"}) == "test.csv"

    def test_dict_with_filename(self):
        """Should extract filename from dict."""
        assert normalize_output_file_entry({"filename": "test.csv"}) == "test.csv"

    def test_none_returns_none(self):
        """Should return None for None input."""
        assert normalize_output_file_entry(None) is None


class TestCollectExpectedOutputs:
    """Tests for collect_expected_outputs function."""

    def test_none_stage_returns_empty(self):
        """Should return empty dict for None stage."""
        result = collect_expected_outputs(None, "paper1", "stage1")
        assert result == {}

    def test_collects_expected_outputs(self):
        """Should collect expected outputs from stage."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{paper_id}_{stage_id}_{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "Fig1" in result
        assert "paper1_stage1_fig1.csv" in result["Fig1"]


class TestCollectExpectedColumns:
    """Tests for collect_expected_columns function."""

    def test_none_stage_returns_empty(self):
        """Should return empty dict for None stage."""
        result = collect_expected_columns(None)
        assert result == {}

    def test_collects_columns(self):
        """Should collect column names from expected outputs."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "columns": ["wavelength", "transmission"]}
            ]
        }
        
        result = collect_expected_columns(stage)
        
        assert result["Fig1"] == ["wavelength", "transmission"]


class TestMatchExpectedFiles:
    """Tests for match_expected_files function."""

    def test_empty_expected_returns_none(self):
        """Should return None for empty expected list."""
        assert match_expected_files([], ["output.csv"]) is None

    def test_exact_match(self):
        """Should match exact filename."""
        expected = ["spectrum.csv"]
        outputs = ["/path/to/spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_substring_match(self):
        """Should fall back to substring match."""
        expected = ["spectrum"]
        outputs = ["/path/to/full_spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result is not None


class TestStageComparisonsForStage:
    """Tests for stage_comparisons_for_stage function."""

    def test_empty_stage_returns_empty(self):
        """Should return empty list for None stage."""
        assert stage_comparisons_for_stage({}, None) == []

    def test_filters_by_stage_id(self):
        """Should filter comparisons by stage ID."""
        state = {
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "Fig1"},
                {"stage_id": "stage2", "figure_id": "Fig2"},
                {"stage_id": "stage1", "figure_id": "Fig3"},
            ]
        }
        
        result = stage_comparisons_for_stage(state, "stage1")
        
        assert len(result) == 2
        assert all(c["stage_id"] == "stage1" for c in result)


class TestAnalysisReportsForStage:
    """Tests for analysis_reports_for_stage function."""

    def test_empty_stage_returns_empty(self):
        """Should return empty list for None stage."""
        assert analysis_reports_for_stage({}, None) == []

    def test_filters_by_stage_id(self):
        """Should filter reports by stage ID."""
        state = {
            "analysis_result_reports": [
                {"stage_id": "stage1", "target_figure": "Fig1"},
                {"stage_id": "stage2", "target_figure": "Fig2"},
            ]
        }
        
        result = analysis_reports_for_stage(state, "stage1")
        
        assert len(result) == 1


class TestValidateAnalysisReports:
    """Tests for validate_analysis_reports function."""

    def test_empty_reports_returns_empty(self):
        """Should return empty list for empty reports."""
        assert validate_analysis_reports([]) == []

    def test_detects_inconsistent_classification(self):
        """Should detect when classification doesn't match metrics."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 20.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) > 0
        assert "Fig1" in issues[0]


class TestBreakdownComparisonClassifications:
    """Tests for breakdown_comparison_classifications function."""

    def test_empty_returns_empty_buckets(self):
        """Should return empty buckets for empty list."""
        result = breakdown_comparison_classifications([])
        assert result == {"missing": [], "pending": [], "matches": []}

    def test_categorizes_classifications(self):
        """Should categorize classifications correctly."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.MATCH},
            {"figure_id": "Fig2", "classification": AnalysisClassification.PARTIAL_MATCH},
            {"figure_id": "Fig3", "classification": AnalysisClassification.MISMATCH},
            {"figure_id": "Fig4", "classification": AnalysisClassification.PENDING_VALIDATION},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig2" in result["pending"]
        assert "Fig3" in result["missing"]
        assert "Fig4" in result["pending"]



