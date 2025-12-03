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
