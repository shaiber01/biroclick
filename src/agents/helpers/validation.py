"""
Validation and classification utilities for analysis results.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from schemas.state import ReproState, DISCREPANCY_THRESHOLDS


# Regex patterns for parsing validation criteria
CRITERIA_PATTERNS = [
    ("resonance_within_percent", re.compile(r"resonance.*within\s+(\d+)%"), "peak_position_error_percent", "max"),
    ("peak_within_percent", re.compile(r"peak.*within\s+(\d+)%"), "peak_position_error_percent", "max"),
    ("normalized_rmse_max", re.compile(r"normalized\s*rmse.*(?:<=|less than)\s*(\d+)%"), "normalized_rmse_percent", "max"),
    ("correlation_min", re.compile(r"correlation.*(?:>=|greater than)\s*(0\.\d+|\d+(\.\d+)?)"), "correlation", "min"),
]


def classify_percent_error(error_percent: float) -> str:
    """Classify error percentage into match/partial_match/mismatch."""
    thresholds = DISCREPANCY_THRESHOLDS["resonance_wavelength"]
    if error_percent <= thresholds["excellent"]:
        return "match"
    if error_percent <= thresholds["acceptable"]:
        return "partial_match"
    return "mismatch"


def classification_from_metrics(
    metrics: Dict[str, float],
    precision_requirement: str,
    has_reference: bool,
) -> str:
    """
    Determine classification based on quantitative metrics.
    
    Args:
        metrics: Dict of computed metrics
        precision_requirement: "excellent", "acceptable", or "qualitative"
        has_reference: Whether reference data was available
        
    Returns:
        Classification string: "match", "partial_match", "mismatch", or "pending_validation"
    """
    if not has_reference:
        if precision_requirement == "excellent":
            return "pending_validation"
        # Without reference data, treat as qualitative-only pending further review
        return "pending_validation"
    if precision_requirement == "qualitative":
        return "match"
    
    error_percent = metrics.get("peak_position_error_percent")
    if error_percent is not None:
        return classify_percent_error(error_percent)
    
    rmse = metrics.get("normalized_rmse_percent")
    if rmse is not None:
        if rmse <= 5:
            return "match"
        if rmse <= 15:
            return "partial_match"
        return "mismatch"
    
    return "pending_validation"


def evaluate_validation_criteria(metrics: Dict[str, Any], criteria: List[str]) -> Tuple[bool, List[str]]:
    """
    Evaluate validation criteria against computed metrics.
    
    Args:
        metrics: Dict of computed metrics
        criteria: List of validation criteria strings
        
    Returns:
        Tuple of (all_passed, list_of_failures)
    """
    failures: List[str] = []
    if not criteria:
        return True, failures
    for criterion in criteria:
        normalized = criterion.lower()
        matched_pattern = False
        for pattern_name, pattern, metric_key, mode in CRITERIA_PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue
            matched_pattern = True
            threshold = float(match.group(1))
            metric_value = metrics.get(metric_key)
            if metric_value is None:
                failures.append(f"{criterion.strip()} (missing metric '{metric_key}')")
            else:
                if mode == "max" and metric_value > threshold:
                    failures.append(f"{criterion.strip()} (measured {metric_value:.2f} > {threshold})")
                if mode == "min" and metric_value < threshold:
                    failures.append(f"{criterion.strip()} (measured {metric_value:.2f} < {threshold})")
            break
        if not matched_pattern:
            # Unknown format; treat as informational
            continue
    return (len(failures) == 0), failures


def extract_targets_from_feedback(feedback: Optional[str], known_targets: List[str]) -> List[str]:
    """Extract figure IDs mentioned in feedback text."""
    if not feedback:
        return []
    pattern = re.compile(r"(Fig\s?[0-9]+[a-zA-Z]?)", re.IGNORECASE)
    matches = [m.strip().replace(" ", "") for m in pattern.findall(feedback)]
    ordered = []
    seen = set()
    normalized_known = {t.lower(): t for t in known_targets}
    for match in matches:
        normalized = match.lower()
        if normalized in normalized_known and normalized not in seen:
            ordered.append(normalized_known[normalized])
            seen.add(normalized)
    return ordered


def match_output_file(file_entries: List[Any], target_id: str) -> Optional[str]:
    """Best-effort match simulation output files to a target figure."""
    normalized_id = (target_id or "").lower()
    for entry in file_entries:
        path_str = entry if isinstance(entry, str) else entry.get("path") or entry.get("file") or str(entry)
        name = Path(path_str).name.lower()
        if normalized_id and normalized_id in name:
            return path_str
    if file_entries:
        entry = file_entries[0]
        return entry if isinstance(entry, str) else entry.get("path") or entry.get("file")
    return None


def normalize_output_file_entry(entry: Any) -> Optional[str]:
    """Return a string path for a stage output entry."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("file") or entry.get("filename")
    return str(entry) if entry is not None else None


def collect_expected_outputs(plan_stage: Optional[Dict[str, Any]], paper_id: str, stage_id: Optional[str]) -> Dict[str, List[str]]:
    """Build a mapping of target_id -> list of expected filenames for the stage."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    if not plan_stage:
        return mapping
    expected = plan_stage.get("expected_outputs") or []
    for spec in expected:
        target = spec.get("target_figure")
        pattern = spec.get("filename_pattern")
        if not target or not pattern:
            continue
        resolved = pattern.replace("{paper_id}", paper_id or "")
        if stage_id:
            resolved = resolved.replace("{stage_id}", stage_id)
        resolved = resolved.replace("{target_id}", target.lower())
        mapping[target].append(resolved)
    return mapping


def collect_expected_columns(plan_stage: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Map target_id -> ordered column names, when provided."""
    mapping: Dict[str, List[str]] = {}
    if not plan_stage:
        return mapping
    expected = plan_stage.get("expected_outputs") or []
    for spec in expected:
        target = spec.get("target_figure")
        columns = spec.get("columns")
        if target and columns:
            mapping[target] = columns
    return mapping


def match_expected_files(expected_filenames: List[str], output_files: List[Any]) -> Optional[str]:
    """Match actual files against the plan's expected filenames."""
    if not expected_filenames:
        return None
    normalized_outputs = [
        Path(normalize_output_file_entry(entry) or "").name
        for entry in output_files
    ]
    path_lookup = {
        Path(normalize_output_file_entry(entry) or "").name: normalize_output_file_entry(entry)
        for entry in output_files
    }
    for expected in expected_filenames:
        expected_name = Path(expected).name
        if expected_name in path_lookup:
            return path_lookup[expected_name]
    # Allow substring match if exact name missing
    for expected in expected_filenames:
        expected_name = Path(expected).name
        for actual in normalized_outputs:
            if expected_name in actual:
                return path_lookup.get(actual)
    return None


def stage_comparisons_for_stage(state: ReproState, stage_id: Optional[str]) -> List[Dict[str, Any]]:
    """Return figure comparison entries associated with a specific stage."""
    if not stage_id:
        return []
    return [
        comp for comp in state.get("figure_comparisons", [])
        if comp.get("stage_id") == stage_id
    ]


def analysis_reports_for_stage(state: ReproState, stage_id: Optional[str]) -> List[Dict[str, Any]]:
    """Return analysis reports for a specific stage."""
    if not stage_id:
        return []
    return [
        report for report in state.get("analysis_result_reports", [])
        if report.get("stage_id") == stage_id
    ]


def validate_analysis_reports(reports: List[Dict[str, Any]]) -> List[str]:
    """Validate analysis reports for consistency issues."""
    issues: List[str] = []
    for report in reports:
        status = (report.get("status") or "").lower()
        metrics = report.get("quantitative_metrics") or {}
        precision = (report.get("precision_requirement") or "").lower()
        target = report.get("target_figure", "unknown")
        error = metrics.get("peak_position_error_percent")
        if precision == "excellent" and not metrics:
            issues.append(f"{target}: excellent precision requires quantitative metrics.")
            continue
        if error is not None:
            thresholds = DISCREPANCY_THRESHOLDS["resonance_wavelength"]
            if status == "match" and error > thresholds["acceptable"]:
                issues.append(f"{target}: classified as match but peak error {error:.2f}% > acceptable {thresholds['acceptable']}%.")
            if status in {"pending_validation", "partial_match"} and error > thresholds["investigate"]:
                issues.append(f"{target}: error {error:.2f}% exceeds investigate threshold; classification should be 'mismatch'.")
        failures = report.get("criteria_failures") or []
        for failure in failures:
            issues.append(f"{target}: {failure}")
    return issues


def breakdown_comparison_classifications(comparisons: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Summarize comparison classifications into missing/pending/match buckets."""
    breakdown: Dict[str, List[str]] = {"missing": [], "pending": [], "matches": []}
    for comp in comparisons:
        classification = (comp.get("classification") or "").lower()
        figure_id = comp.get("figure_id") or "unknown"
        if classification in {"missing_output", "fail", "not_reproduced", "mismatch", "poor_match"}:
            breakdown["missing"].append(figure_id)
        elif classification in {"pending_validation", "partial_match", "match_pending", "partial"}:
            breakdown["pending"].append(figure_id)
        else:
            breakdown["matches"].append(figure_id)
    return breakdown



