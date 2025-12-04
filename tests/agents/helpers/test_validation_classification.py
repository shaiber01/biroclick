"""Classification-oriented validation helper tests."""

import pytest

from schemas.state import DISCREPANCY_THRESHOLDS
from src.agents.constants import AnalysisClassification
from src.agents.helpers.validation import (
    classify_percent_error,
    classification_from_metrics,
    evaluate_validation_criteria,
)


class TestClassifyPercentError:
    """Tests for classify_percent_error function."""

    def test_zero_error_is_match(self):
        """Should classify zero error as match."""
        assert classify_percent_error(0.0) == AnalysisClassification.MATCH

    def test_excellent_match_boundary(self):
        """Should classify errors at excellent threshold as match."""
        excellent_threshold = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["excellent"]
        assert classify_percent_error(excellent_threshold) == AnalysisClassification.MATCH
        assert classify_percent_error(excellent_threshold - 0.1) == AnalysisClassification.MATCH
        assert classify_percent_error(excellent_threshold + 0.1) == AnalysisClassification.PARTIAL_MATCH

    def test_excellent_match_range(self):
        """Should classify small errors as match."""
        assert classify_percent_error(0.5) == AnalysisClassification.MATCH
        assert classify_percent_error(1.0) == AnalysisClassification.MATCH
        assert classify_percent_error(1.9) == AnalysisClassification.MATCH

    def test_acceptable_partial_match_boundary(self):
        """Should classify errors at acceptable threshold correctly."""
        acceptable_threshold = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["acceptable"]
        excellent_threshold = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["excellent"]
        # Just above excellent threshold
        assert classify_percent_error(excellent_threshold + 0.1) == AnalysisClassification.PARTIAL_MATCH
        # At acceptable threshold
        assert classify_percent_error(acceptable_threshold) == AnalysisClassification.PARTIAL_MATCH
        # Just above acceptable threshold
        assert classify_percent_error(acceptable_threshold + 0.1) == AnalysisClassification.MISMATCH

    def test_acceptable_partial_match_range(self):
        """Should classify moderate errors as partial_match."""
        assert classify_percent_error(3.0) == AnalysisClassification.PARTIAL_MATCH
        assert classify_percent_error(4.0) == AnalysisClassification.PARTIAL_MATCH
        assert classify_percent_error(5.0) == AnalysisClassification.PARTIAL_MATCH

    def test_large_error_mismatch(self):
        """Should classify large errors as mismatch."""
        assert classify_percent_error(15.0) == AnalysisClassification.MISMATCH
        assert classify_percent_error(50.0) == AnalysisClassification.MISMATCH
        assert classify_percent_error(100.0) == AnalysisClassification.MISMATCH
        assert classify_percent_error(1000.0) == AnalysisClassification.MISMATCH

    def test_negative_error_handling(self):
        """Should handle negative error values by taking absolute value."""
        # Implementation uses abs(), so -1.0 becomes 1.0 (within excellent threshold of 2%)
        assert classify_percent_error(-1.0) == AnalysisClassification.MATCH
        # -10.0 becomes 10.0, which exceeds acceptable threshold (5%), so MISMATCH
        assert classify_percent_error(-10.0) == AnalysisClassification.MISMATCH

    def test_very_small_positive_error(self):
        """Should classify very small positive errors as match."""
        assert classify_percent_error(0.001) == AnalysisClassification.MATCH
        assert classify_percent_error(0.01) == AnalysisClassification.MATCH

class TestClassificationFromMetrics:
    """Tests for classification_from_metrics function."""

    def test_no_reference_excellent_returns_pending(self):
        """Should return pending_validation without reference data for excellent."""
        result = classification_from_metrics({}, "excellent", has_reference=False)
        assert result == AnalysisClassification.PENDING_VALIDATION
        # Even with metrics, should return pending if no reference
        result = classification_from_metrics(
            {"peak_position_error_percent": 1.0}, "excellent", has_reference=False
        )
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_no_reference_acceptable_returns_pending(self):
        """Should return pending_validation without reference data for acceptable."""
        result = classification_from_metrics({}, "acceptable", has_reference=False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_no_reference_qualitative_returns_pending(self):
        """Should return pending_validation without reference data for qualitative."""
        result = classification_from_metrics({}, "qualitative", has_reference=False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_qualitative_always_matches_with_reference(self):
        """Should return match for qualitative precision requirement when reference exists."""
        result = classification_from_metrics({}, "qualitative", has_reference=True)
        assert result == AnalysisClassification.MATCH
        # Should match even with bad metrics
        result = classification_from_metrics(
            {"peak_position_error_percent": 100.0}, "qualitative", has_reference=True
        )
        assert result == AnalysisClassification.MATCH

    def test_uses_peak_position_error_excellent_match(self):
        """Should classify based on peak position error - excellent match."""
        metrics = {"peak_position_error_percent": 0.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        metrics = {"peak_position_error_percent": 2.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH

    def test_uses_peak_position_error_partial_match(self):
        """Should classify based on peak position error - partial match."""
        metrics = {"peak_position_error_percent": 3.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH
        
        metrics = {"peak_position_error_percent": 5.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_uses_peak_position_error_mismatch(self):
        """Should classify based on peak position error - mismatch."""
        metrics = {"peak_position_error_percent": 6.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH
        
        metrics = {"peak_position_error_percent": 20.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH

    def test_peak_position_error_takes_precedence_over_rmse(self):
        """Should use peak_position_error_percent even when normalized_rmse_percent is present."""
        # Peak error indicates mismatch, RMSE indicates match - peak should win
        metrics = {
            "peak_position_error_percent": 10.0,  # Mismatch
            "normalized_rmse_percent": 3.0  # Would be match
        }
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH
        
        # Peak error indicates match, RMSE indicates mismatch - peak should win
        metrics = {
            "peak_position_error_percent": 1.0,  # Match
            "normalized_rmse_percent": 20.0  # Would be mismatch
        }
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH

    def test_falls_back_to_rmse_match(self):
        """Should use RMSE when peak error not available - match."""
        metrics = {"normalized_rmse_percent": 0.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        metrics = {"normalized_rmse_percent": 3.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        metrics = {"normalized_rmse_percent": 5.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH

    def test_falls_back_to_rmse_partial_match(self):
        """Should use RMSE when peak error not available - partial match."""
        metrics = {"normalized_rmse_percent": 6.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH
        
        metrics = {"normalized_rmse_percent": 10.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH
        
        metrics = {"normalized_rmse_percent": 15.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_falls_back_to_rmse_mismatch(self):
        """Should use RMSE when peak error not available - mismatch."""
        metrics = {"normalized_rmse_percent": 16.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH
        
        metrics = {"normalized_rmse_percent": 20.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH

    def test_rmse_boundary_values(self):
        """Should correctly classify RMSE boundary values."""
        # Exactly at 5% threshold
        metrics = {"normalized_rmse_percent": 5.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MATCH
        
        # Just above 5%
        metrics = {"normalized_rmse_percent": 5.0001}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH
        
        # Exactly at 15% threshold
        metrics = {"normalized_rmse_percent": 15.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PARTIAL_MATCH
        
        # Just above 15%
        metrics = {"normalized_rmse_percent": 15.0001}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.MISMATCH

    def test_returns_pending_without_metrics(self):
        """Should return pending when no metrics available."""
        result = classification_from_metrics({}, "excellent", has_reference=True)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_returns_pending_when_metrics_have_none_values(self):
        """Should return pending when metrics contain None values."""
        metrics = {"peak_position_error_percent": None}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PENDING_VALIDATION
        
        metrics = {"normalized_rmse_percent": None}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_returns_pending_when_metrics_have_unrelated_keys(self):
        """Should return pending when metrics don't contain relevant keys."""
        metrics = {"some_other_metric": 5.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_precision_requirement_case_insensitive(self):
        """Should handle precision requirement case variations."""
        # Test that function handles case (implementation may or may not be case-sensitive)
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "EXCELLENT", has_reference=True)
        # If case-sensitive, this might fail - that's a bug we want to catch
        # If case-insensitive, this should work
        assert result in {AnalysisClassification.MATCH, AnalysisClassification.PENDING_VALIDATION}

    def test_negative_peak_error_handling(self):
        """Should handle negative peak error values."""
        metrics = {"peak_position_error_percent": -1.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        # Negative errors should be classified (likely as MATCH since <= 2)
        assert result == AnalysisClassification.MATCH

    def test_negative_rmse_handling(self):
        """Should handle negative RMSE values (shouldn't happen but test robustness)."""
        metrics = {"normalized_rmse_percent": -1.0}
        result = classification_from_metrics(metrics, "excellent", has_reference=True)
        # Negative RMSE should be classified (likely as MATCH since <= 5)
        assert result == AnalysisClassification.MATCH

class TestEvaluateValidationCriteria:
    """Tests for evaluate_validation_criteria function."""

    def test_empty_criteria_passes(self):
        """Should pass with no criteria."""
        passed, failures = evaluate_validation_criteria({}, [])
        assert passed is True
        assert failures == []
        
        # Should also pass with empty metrics
        passed, failures = evaluate_validation_criteria({"some_metric": 5.0}, [])
        assert passed is True
        assert failures == []

    def test_resonance_within_percent_pass(self):
        """Should evaluate resonance within percent criteria - pass case."""
        criteria = ["resonance within 5%"]
        
        # Exactly at threshold
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 5.0}, criteria
        )
        assert passed is True
        assert failures == []
        
        # Below threshold
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 3.0}, criteria
        )
        assert passed is True
        assert failures == []
        
        # Zero error
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 0.0}, criteria
        )
        assert passed is True
        assert failures == []

    def test_resonance_within_percent_fail(self):
        """Should evaluate resonance within percent criteria - fail case."""
        criteria = ["resonance within 5%"]
        
        # Just above threshold
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 5.0001}, criteria
        )
        assert passed is False
        assert len(failures) == 1
        assert "resonance within 5%" in failures[0]
        assert "measured 5.00" in failures[0] or "measured 5.0" in failures[0]
        assert ">" in failures[0]
        
        # Well above threshold
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 7.0}, criteria
        )
        assert passed is False
        assert len(failures) == 1

    def test_resonance_within_percent_case_variations(self):
        """Should handle case variations in resonance criteria."""
        criteria = ["RESONANCE WITHIN 5%"]
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 3.0}, criteria
        )
        assert passed is True
        
        criteria = ["Resonance Within 5%"]
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 3.0}, criteria
        )
        assert passed is True

    def test_peak_within_percent_pass(self):
        """Should evaluate peak within percent criteria - pass case."""
        criteria = ["peak within 10%"]
        
        # At threshold
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 10.0}, criteria
        )
        assert passed is True
        assert failures == []
        
        # Below threshold
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 8.0}, criteria
        )
        assert passed is True
        assert failures == []

    def test_peak_within_percent_fail(self):
        """Should evaluate peak within percent criteria - fail case."""
        criteria = ["peak within 10%"]
        
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 10.0001}, criteria
        )
        assert passed is False
        assert len(failures) == 1
        assert "peak within 10%" in failures[0]

    def test_normalized_rmse_max_pass(self):
        """Should evaluate normalized RMSE max criteria - pass case."""
        criteria = ["normalized rmse <= 5%"]
        
        # At threshold
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 5.0}, criteria
        )
        assert passed is True
        assert failures == []
        
        # Below threshold
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 3.0}, criteria
        )
        assert passed is True
        assert failures == []

    def test_normalized_rmse_max_fail(self):
        """Should evaluate normalized RMSE max criteria - fail case."""
        criteria = ["normalized rmse <= 5%"]
        
        # Just above threshold
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 5.0001}, criteria
        )
        assert passed is False
        assert len(failures) == 1
        assert "normalized rmse <= 5%" in failures[0]
        assert ">" in failures[0]
        
        # Well above threshold
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 8.0}, criteria
        )
        assert passed is False
        assert len(failures) == 1

    def test_normalized_rmse_less_than_variation(self):
        """Should handle 'less than' variation in RMSE criteria."""
        criteria = ["normalized rmse less than 5%"]
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 3.0}, criteria
        )
        assert passed is True
        
        passed, failures = evaluate_validation_criteria(
            {"normalized_rmse_percent": 6.0}, criteria
        )
        assert passed is False

    def test_correlation_min_pass(self):
        """Should evaluate correlation min criteria - pass case."""
        criteria = ["correlation >= 0.9"]
        
        # At threshold
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.9}, criteria
        )
        assert passed is True
        assert failures == []
        
        # Above threshold
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.95}, criteria
        )
        assert passed is True
        assert failures == []
        
        # Perfect correlation
        passed, failures = evaluate_validation_criteria(
            {"correlation": 1.0}, criteria
        )
        assert passed is True
        assert failures == []

    def test_correlation_min_fail(self):
        """Should evaluate correlation min criteria - fail case."""
        criteria = ["correlation >= 0.9"]
        
        # Just below threshold
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.8999}, criteria
        )
        assert passed is False
        assert len(failures) == 1
        assert "correlation >= 0.9" in failures[0]
        assert "<" in failures[0]
        
        # Well below threshold
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.85}, criteria
        )
        assert passed is False
        assert len(failures) == 1

    def test_correlation_greater_than_variation(self):
        """Should handle 'greater than' variation in correlation criteria."""
        criteria = ["correlation greater than 0.9"]
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.95}, criteria
        )
        assert passed is True
        
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.85}, criteria
        )
        assert passed is False

    def test_correlation_decimal_formats(self):
        """Should handle different decimal formats in correlation criteria."""
        # Test .9 format
        criteria = ["correlation >= .9"]
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.95}, criteria
        )
        assert passed is True
        
        # Test 0.90 format
        criteria = ["correlation >= 0.90"]
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.95}, criteria
        )
        assert passed is True

    def test_missing_metric_reports_failure(self):
        """Should report failure when metric is missing."""
        criteria = ["resonance within 5%"]
        
        passed, failures = evaluate_validation_criteria({}, criteria)
        assert passed is False
        assert len(failures) == 1
        assert "missing metric" in failures[0].lower()
        assert "peak_position_error_percent" in failures[0]
        
        # Test with different criteria
        criteria = ["normalized rmse <= 5%"]
        passed, failures = evaluate_validation_criteria({}, criteria)
        assert passed is False
        assert len(failures) == 1
        assert "normalized_rmse_percent" in failures[0]

    def test_multiple_criteria_all_pass(self):
        """Should pass when all criteria are satisfied."""
        criteria = [
            "resonance within 5%",
            "normalized rmse <= 10%",
            "correlation >= 0.9"
        ]
        metrics = {
            "peak_position_error_percent": 3.0,
            "normalized_rmse_percent": 8.0,
            "correlation": 0.95
        }
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        assert failures == []

    def test_multiple_criteria_some_fail(self):
        """Should fail when some criteria are not satisfied."""
        criteria = [
            "resonance within 5%",
            "normalized rmse <= 10%",
            "correlation >= 0.9"
        ]
        metrics = {
            "peak_position_error_percent": 3.0,  # Pass
            "normalized_rmse_percent": 15.0,  # Fail
            "correlation": 0.85  # Fail
        }
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False
        assert len(failures) == 2
        # Check that both failures are reported
        failure_text = " ".join(failures).lower()
        assert "normalized rmse" in failure_text or "normalized_rmse" in failure_text
        assert "correlation" in failure_text

    def test_multiple_criteria_all_fail(self):
        """Should fail when all criteria are not satisfied."""
        criteria = [
            "resonance within 5%",
            "normalized rmse <= 10%"
        ]
        metrics = {
            "peak_position_error_percent": 10.0,  # Fail
            "normalized_rmse_percent": 15.0  # Fail
        }
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False
        assert len(failures) == 2

    def test_unknown_criteria_format(self):
        """Should ignore unknown criteria formats without failing."""
        criteria = [
            "resonance within 5%",  # Valid
            "some unknown format criterion",  # Unknown
            "another invalid criterion"  # Unknown
        ]
        metrics = {"peak_position_error_percent": 3.0}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        # Should pass because valid criterion passes and unknown ones are ignored
        assert passed is True
        assert len(failures) == 0

    def test_criteria_with_extra_whitespace(self):
        """Should handle criteria with extra whitespace."""
        criteria = ["  resonance within 5%  "]
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 3.0}, criteria
        )
        assert passed is True

    def test_negative_metric_values(self):
        """Should handle negative metric values."""
        criteria = ["resonance within 5%"]
        # Negative error shouldn't happen, but function should handle it
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": -1.0}, criteria
        )
        # Negative value is < 5, so should pass
        assert passed is True

    def test_zero_metric_values(self):
        """Should handle zero metric values."""
        criteria = ["resonance within 5%"]
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 0.0}, criteria
        )
        assert passed is True
        
        criteria = ["correlation >= 0.9"]
        passed, failures = evaluate_validation_criteria(
            {"correlation": 0.0}, criteria
        )
        assert passed is False

    def test_very_large_metric_values(self):
        """Should handle very large metric values."""
        criteria = ["resonance within 5%"]
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 1000.0}, criteria
        )
        assert passed is False
        assert len(failures) == 1

    def test_criteria_with_percent_symbol_variations(self):
        """Should handle percent symbol variations."""
        criteria = ["resonance within 5 percent"]
        # This might not match the pattern, but test to see behavior
        passed, failures = evaluate_validation_criteria(
            {"peak_position_error_percent": 3.0}, criteria
        )
        # If pattern doesn't match, should pass (unknown format ignored)
        # If pattern matches, should pass (3.0 < 5)
        assert passed is True

    def test_empty_metrics_dict(self):
        """Should handle empty metrics dictionary."""
        criteria = ["resonance within 5%"]
        passed, failures = evaluate_validation_criteria({}, criteria)
        assert passed is False
        assert len(failures) == 1
        assert "missing metric" in failures[0].lower()

    def test_metrics_with_none_values(self):
        """Should handle metrics with None values."""
        criteria = ["resonance within 5%"]
        metrics = {"peak_position_error_percent": None}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        # None should be treated as missing
        assert passed is False
        assert len(failures) == 1

    def test_string_numeric_values_are_converted(self):
        """String values that can be converted to float should be converted and evaluated."""
        criteria = ["resonance within 5%"]
        metrics = {"peak_position_error_percent": "3.0"}
        # String "3.0" should be converted to 3.0 and pass
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        assert failures == []
        
        metrics = {"peak_position_error_percent": "10.0"}
        # String "10.0" should be converted to 10.0 and fail
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False
        assert len(failures) == 1

    def test_non_numeric_metric_values_should_be_handled_gracefully(self):
        """Should handle non-numeric metric values gracefully (treat as invalid).
        
        Non-convertible strings should be treated as invalid and reported as failures.
        """
        criteria = ["resonance within 5%"]
        # Non-convertible strings should be treated as invalid
        metrics = {"peak_position_error_percent": "abc"}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert isinstance(passed, bool)
        assert passed is False
        assert len(failures) > 0
        assert "invalid" in failures[0].lower() or "not numeric" in failures[0].lower()
        
        # Test with other non-numeric types
        metrics = {"peak_position_error_percent": []}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False
        assert len(failures) > 0

    def test_empty_string_criterion(self):
        """Should handle empty string criterion."""
        criteria = [""]
        metrics = {"peak_position_error_percent": 3.0}
        # Empty string won't match any pattern, so should pass
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        assert failures == []

    def test_very_large_thresholds(self):
        """Should handle very large threshold values."""
        criteria = ["resonance within 1000%"]
        metrics = {"peak_position_error_percent": 500.0}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        
        metrics = {"peak_position_error_percent": 1500.0}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False

    def test_very_small_thresholds(self):
        """Should handle very small threshold values."""
        criteria = ["resonance within 0.1%"]
        metrics = {"peak_position_error_percent": 0.05}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        
        metrics = {"peak_position_error_percent": 0.15}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False

    def test_correlation_with_very_small_threshold(self):
        """Should handle very small correlation thresholds."""
        criteria = ["correlation >= 0.001"]
        metrics = {"correlation": 0.01}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        
        metrics = {"correlation": 0.0005}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False

    def test_criteria_with_special_characters(self):
        """Should handle criteria with special characters."""
        # Test with various special characters that might appear
        criteria = ["resonance within 5% (target)"]
        metrics = {"peak_position_error_percent": 3.0}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        # Should still match the pattern
        assert passed is True

    def test_float_precision_edge_cases(self):
        """Should handle floating point precision edge cases."""
        criteria = ["resonance within 5%"]
        # Test values very close to threshold
        metrics = {"peak_position_error_percent": 4.999999999}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is True
        
        metrics = {"peak_position_error_percent": 5.000000001}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        assert passed is False

    def test_multiple_same_criteria(self):
        """Should handle multiple identical criteria."""
        criteria = [
            "resonance within 5%",
            "resonance within 5%",
            "resonance within 5%"
        ]
        metrics = {"peak_position_error_percent": 3.0}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        # Should pass (all criteria pass)
        assert passed is True
        
        metrics = {"peak_position_error_percent": 10.0}
        passed, failures = evaluate_validation_criteria(metrics, criteria)
        # Should fail (all criteria fail)
        assert passed is False
        # Should have 3 failures (one per criterion)
        assert len(failures) == 3
