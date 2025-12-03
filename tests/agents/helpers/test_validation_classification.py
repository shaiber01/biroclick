"""Classification-oriented validation helper tests."""

import pytest

from src.agents.constants import AnalysisClassification
from src.agents.helpers.validation import (
    classify_percent_error,
    classification_from_metrics,
    evaluate_validation_criteria,
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
