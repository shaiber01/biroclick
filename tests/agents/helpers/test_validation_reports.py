"""Report aggregation validation helper tests."""

import pytest

from src.agents.constants import AnalysisClassification
from src.agents.helpers.validation import (
    analysis_reports_for_stage,
    breakdown_comparison_classifications,
    stage_comparisons_for_stage,
    validate_analysis_reports,
)


class TestStageComparisonsForStage:
    """Tests for stage_comparisons_for_stage function."""

    def test_empty_state_returns_empty(self):
        """Should return empty list for empty state."""
        assert stage_comparisons_for_stage({}, "stage1") == []

    def test_none_stage_id_returns_empty(self):
        """Should return empty list for None stage_id."""
        state = {
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "Fig1"},
            ]
        }
        assert stage_comparisons_for_stage(state, None) == []

    def test_missing_figure_comparisons_key_returns_empty(self):
        """Should return empty list when figure_comparisons key is missing."""
        assert stage_comparisons_for_stage({"other_key": []}, "stage1") == []

    def test_empty_figure_comparisons_returns_empty(self):
        """Should return empty list when figure_comparisons is empty."""
        state = {"figure_comparisons": []}
        assert stage_comparisons_for_stage(state, "stage1") == []

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
        assert result[0]["figure_id"] == "Fig1"
        assert result[1]["figure_id"] == "Fig3"

    def test_excludes_comparisons_with_different_stage_id(self):
        """Should exclude comparisons with different stage_id."""
        state = {
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "Fig1"},
                {"stage_id": "stage2", "figure_id": "Fig2"},
                {"stage_id": "stage3", "figure_id": "Fig3"},
            ]
        }
        
        result = stage_comparisons_for_stage(state, "stage1")
        
        assert len(result) == 1
        assert result[0]["figure_id"] == "Fig1"

    def test_handles_comparisons_with_missing_stage_id(self):
        """Should exclude comparisons missing stage_id field."""
        state = {
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "Fig1"},
                {"figure_id": "Fig2"},  # Missing stage_id
                {"stage_id": "stage1", "figure_id": "Fig3"},
            ]
        }
        
        result = stage_comparisons_for_stage(state, "stage1")
        
        assert len(result) == 2
        assert all(c["stage_id"] == "stage1" for c in result)
        assert "Fig2" not in [c.get("figure_id") for c in result]

    def test_handles_none_stage_id_in_comparisons(self):
        """Should exclude comparisons with None stage_id."""
        state = {
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "Fig1"},
                {"stage_id": None, "figure_id": "Fig2"},
                {"stage_id": "stage1", "figure_id": "Fig3"},
            ]
        }
        
        result = stage_comparisons_for_stage(state, "stage1")
        
        assert len(result) == 2
        assert all(c["stage_id"] == "stage1" for c in result)


class TestAnalysisReportsForStage:
    """Tests for analysis_reports_for_stage function."""

    def test_empty_state_returns_empty(self):
        """Should return empty list for empty state."""
        assert analysis_reports_for_stage({}, "stage1") == []

    def test_none_stage_id_returns_empty(self):
        """Should return empty list for None stage_id."""
        state = {
            "analysis_result_reports": [
                {"stage_id": "stage1", "target_figure": "Fig1"},
            ]
        }
        assert analysis_reports_for_stage(state, None) == []

    def test_missing_analysis_result_reports_key_returns_empty(self):
        """Should return empty list when analysis_result_reports key is missing."""
        assert analysis_reports_for_stage({"other_key": []}, "stage1") == []

    def test_empty_analysis_result_reports_returns_empty(self):
        """Should return empty list when analysis_result_reports is empty."""
        state = {"analysis_result_reports": []}
        assert analysis_reports_for_stage(state, "stage1") == []

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
        assert result[0]["target_figure"] == "Fig1"
        assert result[0]["stage_id"] == "stage1"

    def test_excludes_reports_with_different_stage_id(self):
        """Should exclude reports with different stage_id."""
        state = {
            "analysis_result_reports": [
                {"stage_id": "stage1", "target_figure": "Fig1"},
                {"stage_id": "stage2", "target_figure": "Fig2"},
                {"stage_id": "stage3", "target_figure": "Fig3"},
            ]
        }
        
        result = analysis_reports_for_stage(state, "stage1")
        
        assert len(result) == 1
        assert result[0]["target_figure"] == "Fig1"

    def test_handles_reports_with_missing_stage_id(self):
        """Should exclude reports missing stage_id field."""
        state = {
            "analysis_result_reports": [
                {"stage_id": "stage1", "target_figure": "Fig1"},
                {"target_figure": "Fig2"},  # Missing stage_id
                {"stage_id": "stage1", "target_figure": "Fig3"},
            ]
        }
        
        result = analysis_reports_for_stage(state, "stage1")
        
        assert len(result) == 2
        assert all(r["stage_id"] == "stage1" for r in result)
        assert "Fig2" not in [r.get("target_figure") for r in result]

    def test_handles_none_stage_id_in_reports(self):
        """Should exclude reports with None stage_id."""
        state = {
            "analysis_result_reports": [
                {"stage_id": "stage1", "target_figure": "Fig1"},
                {"stage_id": None, "target_figure": "Fig2"},
                {"stage_id": "stage1", "target_figure": "Fig3"},
            ]
        }
        
        result = analysis_reports_for_stage(state, "stage1")
        
        assert len(result) == 2
        assert all(r["stage_id"] == "stage1" for r in result)


class TestValidateAnalysisReports:
    """Tests for validate_analysis_reports function."""

    def test_empty_reports_returns_empty(self):
        """Should return empty list for empty reports."""
        assert validate_analysis_reports([]) == []

    def test_excellent_precision_without_metrics_detected(self):
        """Should detect excellent precision requirement without metrics."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "excellent precision requires quantitative metrics" in issues[0].lower()

    def test_excellent_precision_with_missing_metrics_key_detected(self):
        """Should detect excellent precision requirement when metrics key is missing."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "excellent precision requires quantitative metrics" in issues[0].lower()

    def test_match_status_with_error_above_acceptable_threshold_detected(self):
        """Should detect match status when error exceeds acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 6.0},  # > 5% acceptable
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "match" in issues[0].lower()
        assert "6.00" in issues[0] or "6.0" in issues[0]
        assert "acceptable" in issues[0].lower()

    def test_match_status_string_with_error_above_acceptable_threshold_detected(self):
        """Should detect match status (string) when error exceeds acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": "match",  # String instead of enum
            "quantitative_metrics": {"peak_position_error_percent": 6.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "match" in issues[0].lower()

    def test_match_status_at_acceptable_threshold_boundary(self):
        """Should not flag match status when error is exactly at acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 5.0},  # Exactly at acceptable
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag this - 5.0 is acceptable threshold, so match is valid
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower()]
        assert len(match_issues) == 0

    def test_match_status_below_acceptable_threshold_no_issue(self):
        """Should not flag match status when error is below acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 2.0},  # Below acceptable
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower()]
        assert len(match_issues) == 0

    def test_pending_status_with_error_above_investigate_threshold_detected(self):
        """Should detect pending status when error exceeds investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PENDING_VALIDATION,
            "quantitative_metrics": {"peak_position_error_percent": 11.0},  # > 10% investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "pending" in issues[0].lower() or "error" in issues[0].lower()
        assert "11.00" in issues[0] or "11.0" in issues[0]
        assert "investigate" in issues[0].lower() or "mismatch" in issues[0].lower()

    def test_partial_match_status_with_error_above_investigate_threshold_detected(self):
        """Should detect partial_match status when error exceeds investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PARTIAL_MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 11.0},  # > 10% investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "11.00" in issues[0] or "11.0" in issues[0]
        assert "investigate" in issues[0].lower() or "mismatch" in issues[0].lower()

    def test_pending_status_string_with_error_above_investigate_threshold_detected(self):
        """Should detect pending status (string) when error exceeds investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": "pending_validation",  # String instead of enum
            "quantitative_metrics": {"peak_position_error_percent": 11.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 1
        assert "Fig1" in issues[0]

    def test_pending_status_at_investigate_threshold_boundary(self):
        """Should not flag pending status when error is exactly at investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PENDING_VALIDATION,
            "quantitative_metrics": {"peak_position_error_percent": 10.0},  # Exactly at investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag this - 10.0 is investigate threshold, so pending is valid
        investigate_issues = [i for i in issues if "investigate" in i.lower() and "mismatch" in i.lower()]
        assert len(investigate_issues) == 0

    def test_criteria_failures_included(self):
        """Should include criteria failures in issues."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 1.0},
            "precision_requirement": "excellent",
            "criteria_failures": ["resonance within 1%", "correlation >= 0.9"],
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) == 2
        assert all("Fig1" in issue for issue in issues)
        assert any("resonance within 1%" in issue for issue in issues)
        assert any("correlation >= 0.9" in issue for issue in issues)

    def test_empty_criteria_failures_no_issue(self):
        """Should not add issues for empty criteria failures."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 1.0},
            "precision_requirement": "excellent",
            "criteria_failures": [],
        }]
        
        issues = validate_analysis_reports(reports)
        
        criteria_issues = [i for i in issues if "criteria" in i.lower() or any(f in i for f in [])]
        assert len(criteria_issues) == 0

    def test_missing_criteria_failures_key_no_issue(self):
        """Should not add issues when criteria_failures key is missing."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 1.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        criteria_issues = [i for i in issues if "criteria" in i.lower()]
        assert len(criteria_issues) == 0

    def test_multiple_issues_in_single_report(self):
        """Should detect multiple issues in a single report."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 6.0},  # Above acceptable
            "precision_requirement": "excellent",
            "criteria_failures": ["resonance within 1%"],
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) >= 2
        assert all("Fig1" in issue for issue in issues)

    def test_multiple_reports_with_issues(self):
        """Should detect issues across multiple reports."""
        reports = [
            {
                "target_figure": "Fig1",
                "status": AnalysisClassification.MATCH,
                "quantitative_metrics": {"peak_position_error_percent": 6.0},
                "precision_requirement": "excellent",
            },
            {
                "target_figure": "Fig2",
                "status": AnalysisClassification.PENDING_VALIDATION,
                "quantitative_metrics": {"peak_position_error_percent": 11.0},
                "precision_requirement": "excellent",
            },
        ]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) >= 2
        assert any("Fig1" in issue for issue in issues)
        assert any("Fig2" in issue for issue in issues)

    def test_missing_target_figure_uses_unknown(self):
        """Should use 'unknown' when target_figure is missing."""
        reports = [{
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 6.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        assert len(issues) > 0
        assert "unknown" in issues[0].lower()

    def test_missing_status_no_error(self):
        """Should handle missing status gracefully."""
        reports = [{
            "target_figure": "Fig1",
            "quantitative_metrics": {"peak_position_error_percent": 1.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not crash, may or may not detect issues depending on implementation
        assert isinstance(issues, list)

    def test_missing_quantitative_metrics_no_error(self):
        """Should handle missing quantitative_metrics gracefully."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect excellent precision without metrics
        assert len(issues) > 0
        assert "excellent precision requires quantitative metrics" in issues[0].lower()

    def test_acceptable_precision_without_metrics_no_issue(self):
        """Should not require metrics for acceptable precision."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {},
            "precision_requirement": "acceptable",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag missing metrics for acceptable precision
        metrics_issues = [i for i in issues if "excellent precision requires" in i.lower()]
        assert len(metrics_issues) == 0

    def test_qualitative_precision_without_metrics_no_issue(self):
        """Should not require metrics for qualitative precision."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {},
            "precision_requirement": "qualitative",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag missing metrics for qualitative precision
        metrics_issues = [i for i in issues if "excellent precision requires" in i.lower()]
        assert len(metrics_issues) == 0

    def test_zero_error_no_issue(self):
        """Should handle zero error correctly."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 0.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Zero error should be fine for match
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower()]
        assert len(match_issues) == 0

    def test_negative_error_handled(self):
        """Should handle negative error values."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": -1.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Negative error should not trigger match threshold check (it's below acceptable)
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower() and "acceptable" in i.lower()]
        assert len(match_issues) == 0

    def test_enum_status_handled_correctly(self):
        """Should handle enum status values correctly after lowercase conversion."""
        # This test checks if the bug where enum comparison fails after lowercase conversion exists
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,  # Enum value "MATCH"
            "quantitative_metrics": {"peak_position_error_percent": 6.0},  # > 5% acceptable
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect the inconsistency even with enum status
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "match" in issues[0].lower()

    def test_enum_pending_status_handled_correctly(self):
        """Should handle enum pending status values correctly after lowercase conversion."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PENDING_VALIDATION,  # Enum value "PENDING_VALIDATION"
            "quantitative_metrics": {"peak_position_error_percent": 11.0},  # > 10% investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect the inconsistency even with enum status
        assert len(issues) == 1
        assert "Fig1" in issues[0]

    def test_enum_partial_match_status_handled_correctly(self):
        """Should handle enum partial_match status values correctly after lowercase conversion."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PARTIAL_MATCH,  # Enum value "PARTIAL_MATCH"
            "quantitative_metrics": {"peak_position_error_percent": 11.0},  # > 10% investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect the inconsistency even with enum status
        assert len(issues) == 1
        assert "Fig1" in issues[0]

    def test_none_error_value_handled(self):
        """Should handle None error value in metrics."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": None},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not crash, and should not flag match error issues
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower() and "acceptable" in i.lower()]
        assert len(match_issues) == 0

    def test_missing_error_key_no_crash(self):
        """Should handle missing peak_position_error_percent key."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"other_metric": 5.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not crash
        assert isinstance(issues, list)
        # Should not flag match error issues (no error metric)
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower() and "acceptable" in i.lower()]
        assert len(match_issues) == 0

    def test_very_large_error_value_handled(self):
        """Should handle very large error values."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 1000.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect the inconsistency
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "match" in issues[0].lower()

    def test_error_at_excellent_threshold_boundary(self):
        """Should handle error exactly at excellent threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 2.0},  # Exactly at excellent threshold
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag this - 2.0 is excellent threshold, so match is valid
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower() and "acceptable" in i.lower()]
        assert len(match_issues) == 0

    def test_error_just_above_excellent_threshold(self):
        """Should handle error just above excellent threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 2.01},  # Just above excellent
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag this - 2.01 is still within acceptable (5%), so match is valid
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower() and "acceptable" in i.lower()]
        assert len(match_issues) == 0

    def test_error_just_below_acceptable_threshold(self):
        """Should handle error just below acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 4.99},  # Just below acceptable
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag this - 4.99 is within acceptable (5%), so match is valid
        match_issues = [i for i in issues if "match" in i.lower() and "error" in i.lower() and "acceptable" in i.lower()]
        assert len(match_issues) == 0

    def test_error_just_above_acceptable_threshold(self):
        """Should handle error just above acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 5.01},  # Just above acceptable
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should flag this - 5.01 is above acceptable (5%), so match is invalid
        assert len(issues) == 1
        assert "Fig1" in issues[0]
        assert "match" in issues[0].lower()

    def test_error_just_below_investigate_threshold(self):
        """Should handle error just below investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PENDING_VALIDATION,
            "quantitative_metrics": {"peak_position_error_percent": 9.99},  # Just below investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should not flag this - 9.99 is within investigate (10%), so pending is valid
        investigate_issues = [i for i in issues if "investigate" in i.lower() and "mismatch" in i.lower()]
        assert len(investigate_issues) == 0

    def test_error_just_above_investigate_threshold(self):
        """Should handle error just above investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.PENDING_VALIDATION,
            "quantitative_metrics": {"peak_position_error_percent": 10.01},  # Just above investigate
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should flag this - 10.01 is above investigate (10%), so pending is invalid
        assert len(issues) == 1
        assert "Fig1" in issues[0]

    def test_invalid_error_type_handled(self):
        """Should handle invalid error type gracefully without crashing.
        
        This test reveals a bug: the function crashes with TypeError when
        peak_position_error_percent is not numeric. The function should validate
        input types and handle invalid data gracefully.
        """
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": "not_a_number"},
            "precision_requirement": "excellent",
        }]
        
        # This should NOT raise an error - invalid data should be handled gracefully
        # If this test fails, it reveals a bug in the component under test
        issues = validate_analysis_reports(reports)
        assert isinstance(issues, list)
        # Should either skip validation for invalid error or include it in issues

    def test_uppercase_status_string_handled(self):
        """Should handle uppercase status strings."""
        reports = [{
            "target_figure": "Fig1",
            "status": "MATCH",  # Uppercase string
            "quantitative_metrics": {"peak_position_error_percent": 6.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect the inconsistency (status is converted to lowercase)
        assert len(issues) == 1
        assert "Fig1" in issues[0]

    def test_mixed_case_status_string_handled(self):
        """Should handle mixed case status strings."""
        reports = [{
            "target_figure": "Fig1",
            "status": "MaTcH",  # Mixed case string
            "quantitative_metrics": {"peak_position_error_percent": 6.0},
            "precision_requirement": "excellent",
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect the inconsistency (status is converted to lowercase)
        assert len(issues) == 1
        assert "Fig1" in issues[0]

    def test_uppercase_precision_requirement_handled(self):
        """Should handle uppercase precision requirement."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {},
            "precision_requirement": "EXCELLENT",  # Uppercase
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should detect missing metrics (precision is converted to lowercase)
        assert len(issues) == 1
        assert "excellent precision requires quantitative metrics" in issues[0].lower()

    def test_non_string_criteria_failures_handled(self):
        """Should handle non-string criteria failures."""
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": 1.0},
            "precision_requirement": "excellent",
            "criteria_failures": [123, None, "valid_string"],
        }]
        
        issues = validate_analysis_reports(reports)
        
        # Should handle gracefully - at least the string should be included
        assert len(issues) >= 1
        assert any("valid_string" in issue for issue in issues)

    def test_list_error_type_crashes(self):
        """Should handle list error type gracefully without crashing.
        
        This test reveals a bug: the function crashes with TypeError when
        peak_position_error_percent is a list. The function should validate
        input types and handle invalid data gracefully.
        """
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": [1.0, 2.0]},
            "precision_requirement": "excellent",
        }]
        
        # This should NOT raise an error - invalid data should be handled gracefully
        # If this test fails, it reveals a bug in the component under test
        issues = validate_analysis_reports(reports)
        assert isinstance(issues, list)

    def test_dict_error_type_crashes(self):
        """Should handle dict error type gracefully without crashing.
        
        This test reveals a bug: the function crashes with TypeError when
        peak_position_error_percent is a dict. The function should validate
        input types and handle invalid data gracefully.
        """
        reports = [{
            "target_figure": "Fig1",
            "status": AnalysisClassification.MATCH,
            "quantitative_metrics": {"peak_position_error_percent": {"value": 5.0}},
            "precision_requirement": "excellent",
        }]
        
        # This should NOT raise an error - invalid data should be handled gracefully
        # If this test fails, it reveals a bug in the component under test
        issues = validate_analysis_reports(reports)
        assert isinstance(issues, list)


class TestBreakdownComparisonClassifications:
    """Tests for breakdown_comparison_classifications function."""

    def test_empty_returns_empty_buckets(self):
        """Should return empty buckets for empty list."""
        result = breakdown_comparison_classifications([])
        assert result == {"missing": [], "pending": [], "matches": []}
        assert isinstance(result["missing"], list)
        assert isinstance(result["pending"], list)
        assert isinstance(result["matches"], list)

    def test_match_classification_goes_to_matches(self):
        """Should categorize MATCH as matches."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_partial_match_classification_goes_to_pending(self):
        """Should categorize PARTIAL_MATCH as pending."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.PARTIAL_MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["pending"]
        assert "Fig1" not in result["matches"]
        assert "Fig1" not in result["missing"]

    def test_mismatch_classification_goes_to_missing(self):
        """Should categorize MISMATCH as missing."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.MISMATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["missing"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["matches"]

    def test_pending_validation_classification_goes_to_pending(self):
        """Should categorize PENDING_VALIDATION as pending."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.PENDING_VALIDATION},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["pending"]
        assert "Fig1" not in result["matches"]
        assert "Fig1" not in result["missing"]

    def test_failed_classification_goes_to_missing(self):
        """Should categorize FAILED as missing."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.FAILED},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["missing"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["matches"]

    def test_poor_match_classification_goes_to_missing(self):
        """Should categorize POOR_MATCH as missing."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.POOR_MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["missing"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["matches"]

    def test_no_targets_classification_goes_to_missing(self):
        """Should categorize NO_TARGETS as missing."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.NO_TARGETS},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["missing"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["matches"]

    def test_excellent_match_classification_goes_to_matches(self):
        """Should categorize EXCELLENT_MATCH as matches."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.EXCELLENT_MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_acceptable_match_classification_goes_to_matches(self):
        """Should categorize ACCEPTABLE_MATCH as matches."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.ACCEPTABLE_MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_string_classifications_handled(self):
        """Should handle string classifications (case-insensitive)."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "match"},
            {"figure_id": "Fig2", "classification": "MATCH"},
            {"figure_id": "Fig3", "classification": "Match"},
            {"figure_id": "Fig4", "classification": "mismatch"},
            {"figure_id": "Fig5", "classification": "MISMATCH"},
            {"figure_id": "Fig6", "classification": "partial_match"},
            {"figure_id": "Fig7", "classification": "PARTIAL_MATCH"},
            {"figure_id": "Fig8", "classification": "pending_validation"},
            {"figure_id": "Fig9", "classification": "PENDING_VALIDATION"},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig2" in result["matches"]
        assert "Fig3" in result["matches"]
        assert "Fig4" in result["missing"]
        assert "Fig5" in result["missing"]
        assert "Fig6" in result["pending"]
        assert "Fig7" in result["pending"]
        assert "Fig8" in result["pending"]
        assert "Fig9" in result["pending"]

    def test_legacy_string_classifications_handled(self):
        """Should handle legacy string classification formats."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "missing_output"},
            {"figure_id": "Fig2", "classification": "fail"},
            {"figure_id": "Fig3", "classification": "not_reproduced"},
            {"figure_id": "Fig4", "classification": "poor_match"},
            {"figure_id": "Fig5", "classification": "match_pending"},
            {"figure_id": "Fig6", "classification": "partial"},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["missing"]
        assert "Fig2" in result["missing"]
        assert "Fig3" in result["missing"]
        assert "Fig4" in result["missing"]
        assert "Fig5" in result["pending"]
        assert "Fig6" in result["pending"]

    def test_missing_classification_goes_to_matches(self):
        """Should categorize missing classification as matches (default)."""
        comparisons = [
            {"figure_id": "Fig1"},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_none_classification_goes_to_matches(self):
        """Should categorize None classification as matches (default)."""
        comparisons = [
            {"figure_id": "Fig1", "classification": None},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_empty_string_classification_goes_to_matches(self):
        """Should categorize empty string classification as matches (default)."""
        comparisons = [
            {"figure_id": "Fig1", "classification": ""},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_missing_figure_id_uses_unknown(self):
        """Should use 'unknown' when figure_id is missing."""
        comparisons = [
            {"classification": AnalysisClassification.MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "unknown" in result["matches"]
        assert len(result["matches"]) == 1

    def test_none_figure_id_uses_unknown(self):
        """Should use 'unknown' when figure_id is None."""
        comparisons = [
            {"figure_id": None, "classification": AnalysisClassification.MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "unknown" in result["matches"]
        assert len(result["matches"]) == 1

    def test_empty_string_figure_id_uses_unknown(self):
        """Should use 'unknown' when figure_id is empty string."""
        comparisons = [
            {"figure_id": "", "classification": AnalysisClassification.MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "unknown" in result["matches"]
        assert len(result["matches"]) == 1

    def test_multiple_classifications_categorized_correctly(self):
        """Should categorize multiple classifications correctly."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.MATCH},
            {"figure_id": "Fig2", "classification": AnalysisClassification.PARTIAL_MATCH},
            {"figure_id": "Fig3", "classification": AnalysisClassification.MISMATCH},
            {"figure_id": "Fig4", "classification": AnalysisClassification.PENDING_VALIDATION},
            {"figure_id": "Fig5", "classification": AnalysisClassification.FAILED},
            {"figure_id": "Fig6", "classification": AnalysisClassification.POOR_MATCH},
            {"figure_id": "Fig7", "classification": AnalysisClassification.NO_TARGETS},
            {"figure_id": "Fig8", "classification": AnalysisClassification.EXCELLENT_MATCH},
            {"figure_id": "Fig9", "classification": AnalysisClassification.ACCEPTABLE_MATCH},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert len(result["matches"]) == 3
        assert "Fig1" in result["matches"]
        assert "Fig8" in result["matches"]
        assert "Fig9" in result["matches"]
        
        assert len(result["pending"]) == 2
        assert "Fig2" in result["pending"]
        assert "Fig4" in result["pending"]
        
        assert len(result["missing"]) == 4
        assert "Fig3" in result["missing"]
        assert "Fig5" in result["missing"]
        assert "Fig6" in result["missing"]
        assert "Fig7" in result["missing"]

    def test_unknown_classification_goes_to_matches(self):
        """Should categorize unknown classification as matches (default)."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "unknown_classification"},
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert "Fig1" in result["matches"]
        assert "Fig1" not in result["pending"]
        assert "Fig1" not in result["missing"]

    def test_all_missing_types_categorized(self):
        """Should categorize all missing types correctly."""
        missing_types = [
            AnalysisClassification.MISMATCH,
            AnalysisClassification.FAILED,
            AnalysisClassification.POOR_MATCH,
            AnalysisClassification.NO_TARGETS,
            "mismatch",
            "fail",
            "not_reproduced",
            "poor_match",
            "missing_output",
        ]
        
        comparisons = [
            {"figure_id": f"Fig{i+1}", "classification": cls}
            for i, cls in enumerate(missing_types)
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert len(result["missing"]) == len(missing_types)
        assert all(f"Fig{i+1}" in result["missing"] for i in range(len(missing_types)))

    def test_all_pending_types_categorized(self):
        """Should categorize all pending types correctly."""
        pending_types = [
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.PENDING_VALIDATION,
            "partial_match",
            "pending_validation",
            "match_pending",
            "partial",
        ]
        
        comparisons = [
            {"figure_id": f"Fig{i+1}", "classification": cls}
            for i, cls in enumerate(pending_types)
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert len(result["pending"]) == len(pending_types)
        assert all(f"Fig{i+1}" in result["pending"] for i in range(len(pending_types)))

    def test_all_match_types_categorized(self):
        """Should categorize all match types correctly."""
        match_types = [
            AnalysisClassification.MATCH,
            AnalysisClassification.EXCELLENT_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
        ]
        
        comparisons = [
            {"figure_id": f"Fig{i+1}", "classification": cls}
            for i, cls in enumerate(match_types)
        ]
        
        result = breakdown_comparison_classifications(comparisons)
        
        assert len(result["matches"]) == len(match_types)
        assert all(f"Fig{i+1}" in result["matches"] for i in range(len(match_types)))
