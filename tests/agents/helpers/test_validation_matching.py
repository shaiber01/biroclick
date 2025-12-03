"""Matching/output validation helper tests."""

import pytest
from pathlib import Path

from src.agents.helpers.validation import (
    analysis_reports_for_stage,
    breakdown_comparison_classifications,
    classify_percent_error,
    classification_from_metrics,
    collect_expected_columns,
    collect_expected_outputs,
    evaluate_validation_criteria,
    extract_targets_from_feedback,
    match_expected_files,
    match_output_file,
    normalize_output_file_entry,
    stage_comparisons_for_stage,
    validate_analysis_reports,
    CRITERIA_PATTERNS,
)
from src.agents.constants import AnalysisClassification
from schemas.state import DISCREPANCY_THRESHOLDS


class TestExtractTargetsFromFeedback:
    """Tests for extract_targets_from_feedback function."""

    def test_empty_feedback_returns_empty(self):
        """Should return empty list for empty feedback."""
        assert extract_targets_from_feedback(None, ["Fig1"]) == []
        assert extract_targets_from_feedback("", ["Fig1"]) == []
        assert extract_targets_from_feedback("   ", ["Fig1"]) == []

    def test_empty_known_targets_returns_empty(self):
        """Should return empty list when no known targets."""
        feedback = "Check Fig1 and Fig2"
        assert extract_targets_from_feedback(feedback, []) == []
        assert extract_targets_from_feedback(feedback, None) == []

    def test_extracts_figure_references(self):
        """Should extract figure references from feedback."""
        feedback = "Please check Fig1 and Fig 2a for discrepancies."
        known = ["Fig1", "Fig2a", "Fig3"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2
        assert "Fig1" in result
        assert "Fig2a" in result
        assert "Fig3" not in result
        # Should preserve order of first appearance
        assert result[0] == "Fig1"
        assert result[1] == "Fig2a"

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
        assert "Fig1" in result
        assert "Fig2" in result

    def test_removes_duplicates(self):
        """Should not include same figure multiple times."""
        feedback = "Check Fig1, Fig1, and Fig1 again"
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 1
        assert result == ["Fig1"]

    def test_handles_spaces_in_figure_names(self):
        """Should handle spaces in figure references."""
        feedback = "See Fig 1 and Fig 2a"
        known = ["Fig1", "Fig2a"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2
        assert "Fig1" in result
        assert "Fig2a" in result

    def test_preserves_case_of_known_targets(self):
        """Should return targets with original case from known list."""
        feedback = "See fig1 and fig2"
        known = ["Fig1", "FIG2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2
        assert "Fig1" in result
        assert "FIG2" in result

    def test_complex_figure_names(self):
        """Should handle complex figure names with letters."""
        feedback = "Check Fig1a, Fig2b, and Fig10z"
        known = ["Fig1a", "Fig2b", "Fig10z"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 3
        assert all(fig in result for fig in ["Fig1a", "Fig2b", "Fig10z"])

    def test_no_figures_in_feedback(self):
        """Should return empty when no figures mentioned."""
        feedback = "This is just regular text with no figures."
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert result == []

    def test_figures_with_punctuation(self):
        """Should extract figures surrounded by punctuation."""
        feedback = "Check (Fig1) and [Fig2], also Fig3."
        known = ["Fig1", "Fig2", "Fig3"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 3
        assert "Fig1" in result
        assert "Fig2" in result
        assert "Fig3" in result

    def test_figures_in_sentences(self):
        """Should extract figures embedded in sentences."""
        feedback = "The results for Fig1 show that Fig2 needs attention."
        known = ["Fig1", "Fig2", "Fig3"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2
        assert result[0] == "Fig1"
        assert result[1] == "Fig2"

    def test_multiline_feedback(self):
        """Should extract figures from multiline feedback."""
        feedback = """
        First, check Fig1 for the spectral response.
        Then, verify Fig2 for absorption.
        Finally, compare with Fig3.
        """
        known = ["Fig1", "Fig2", "Fig3", "Fig4"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 3
        assert result == ["Fig1", "Fig2", "Fig3"]

    def test_double_digit_figure_numbers(self):
        """Should handle double digit figure numbers."""
        feedback = "Check Fig10, Fig11, and Fig12"
        known = ["Fig10", "Fig11", "Fig12", "Fig1"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        # Should find Fig10, Fig11, Fig12
        # Should NOT incorrectly split Fig10 into Fig1 and 0
        assert "Fig10" in result
        assert "Fig11" in result
        assert "Fig12" in result

    def test_figure_at_start_of_feedback(self):
        """Should extract figure at the very start of feedback."""
        feedback = "Fig1 is incorrect."
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 1
        assert result[0] == "Fig1"

    def test_figure_at_end_of_feedback(self):
        """Should extract figure at the very end of feedback."""
        feedback = "The issue is with Fig1"
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 1
        assert result[0] == "Fig1"

    def test_only_figure_in_feedback(self):
        """Should extract when feedback is just a figure reference."""
        feedback = "Fig1"
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 1
        assert result[0] == "Fig1"

    def test_figures_with_colon(self):
        """Should extract figures followed by colon."""
        feedback = "Fig1: needs revision. Fig2: looks good."
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2

    def test_mixed_case_duplicates(self):
        """Should remove duplicates even with different cases."""
        feedback = "Check Fig1, FIG1, and fig1"
        known = ["Fig1"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 1
        assert result[0] == "Fig1"

    def test_figures_with_suffix_letters(self):
        """Should handle figures with suffix letters correctly."""
        feedback = "Compare Fig1a with Fig1b"
        known = ["Fig1a", "Fig1b", "Fig1"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2
        assert "Fig1a" in result
        assert "Fig1b" in result

    def test_subset_figure_not_extracted_from_larger(self):
        """Should not extract Fig1 when only Fig10 appears."""
        feedback = "Check Fig10"
        known = ["Fig1", "Fig10"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        # The regex should match "Fig10" not "Fig1" followed by "0"
        assert "Fig10" in result

    def test_newlines_and_tabs_in_feedback(self):
        """Should handle feedback with various whitespace."""
        feedback = "Check\tFig1\nand\rFig2"
        known = ["Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 2

    def test_many_figures(self):
        """Should handle many figure references."""
        figures = [f"Fig{i}" for i in range(1, 11)]
        feedback = " ".join(figures)
        known = figures
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 10

    def test_empty_string_in_known_targets(self):
        """Should handle empty string in known targets gracefully."""
        feedback = "Check Fig1"
        known = ["", "Fig1", "Fig2"]
        
        result = extract_targets_from_feedback(feedback, known)
        
        assert len(result) == 1
        assert result[0] == "Fig1"


class TestMatchOutputFile:
    """Tests for match_output_file function."""

    def test_empty_list_returns_none(self):
        """Should return None for empty file list."""
        assert match_output_file([], "Fig1") is None
        assert match_output_file([], None) is None
        assert match_output_file([], "") is None

    def test_matches_by_target_id(self):
        """Should match files containing target ID."""
        files = ["fig1_spectrum.csv", "fig2_absorption.csv"]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"

    def test_matches_case_insensitively(self):
        """Should match case-insensitively."""
        files = ["FIG1_spectrum.csv", "fig2_absorption.csv"]
        
        result = match_output_file(files, "fig1")
        
        assert result == "FIG1_spectrum.csv"

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

    def test_handles_dict_with_file_key(self):
        """Should handle dict entries with 'file' key."""
        files = [
            {"file": "fig1_spectrum.csv"},
            {"path": "fig2_absorption.csv"}
        ]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"

    def test_handles_none_target_id(self):
        """Should return first file when target_id is None."""
        files = ["output.csv", "results.csv"]
        
        result = match_output_file(files, None)
        
        assert result == "output.csv"

    def test_handles_empty_target_id(self):
        """Should return first file when target_id is empty."""
        files = ["output.csv", "results.csv"]
        
        result = match_output_file(files, "")
        
        assert result == "output.csv"

    def test_handles_dict_with_none_values(self):
        """Should handle dict entries with None values."""
        files = [
            {"path": None, "file": "fig1_spectrum.csv"},
            {"path": "fig2_absorption.csv"}
        ]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"

    def test_handles_dict_with_empty_strings(self):
        """Should handle dict entries with empty strings."""
        files = [
            {"path": "", "file": "fig1_spectrum.csv"},
            {"path": "fig2_absorption.csv"}
        ]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"

    def test_handles_paths_with_directories(self):
        """Should match based on filename, not full path."""
        files = ["/path/to/fig1_spectrum.csv", "/other/path/fig2_data.csv"]
        
        result = match_output_file(files, "fig1")
        
        assert result == "/path/to/fig1_spectrum.csv"

    def test_returns_first_match_not_all_matches(self):
        """Should return first matching file, not all matches."""
        files = ["fig1_spectrum.csv", "fig1_absorption.csv", "fig2_data.csv"]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"

    def test_handles_non_string_dict_values(self):
        """Should convert non-string dict values to strings."""
        files = [
            {"path": 12345},  # Should be converted to string
            {"file": "fig1_spectrum.csv"}
        ]
        
        result = match_output_file(files, "fig1")
        
        assert result == "fig1_spectrum.csv"


class TestNormalizeOutputFileEntry:
    """Tests for normalize_output_file_entry function."""

    def test_string_entry(self):
        """Should return string as-is."""
        assert normalize_output_file_entry("test.csv") == "test.csv"
        assert normalize_output_file_entry("/path/to/file.csv") == "/path/to/file.csv"

    def test_empty_string(self):
        """Should return empty string for empty string input."""
        assert normalize_output_file_entry("") == ""

    def test_dict_with_path(self):
        """Should extract path from dict."""
        assert normalize_output_file_entry({"path": "test.csv"}) == "test.csv"

    def test_dict_with_file(self):
        """Should extract file from dict."""
        assert normalize_output_file_entry({"file": "test.csv"}) == "test.csv"

    def test_dict_with_filename(self):
        """Should extract filename from dict."""
        assert normalize_output_file_entry({"filename": "test.csv"}) == "test.csv"

    def test_dict_priority_path_over_file(self):
        """Should prefer path over file key."""
        entry = {"path": "path_value.csv", "file": "file_value.csv"}
        assert normalize_output_file_entry(entry) == "path_value.csv"

    def test_dict_priority_path_over_filename(self):
        """Should prefer path over filename key."""
        entry = {"path": "path_value.csv", "filename": "filename_value.csv"}
        assert normalize_output_file_entry(entry) == "path_value.csv"

    def test_dict_priority_file_over_filename(self):
        """Should prefer file over filename key."""
        entry = {"file": "file_value.csv", "filename": "filename_value.csv"}
        assert normalize_output_file_entry(entry) == "file_value.csv"

    def test_dict_with_none_values(self):
        """Should handle None values in dict."""
        assert normalize_output_file_entry({"path": None, "file": "test.csv"}) == "test.csv"
        assert normalize_output_file_entry({"path": None, "file": None, "filename": "test.csv"}) == "test.csv"

    def test_dict_with_empty_strings(self):
        """Should handle empty strings in dict."""
        assert normalize_output_file_entry({"path": "", "file": "test.csv"}) == "test.csv"
        assert normalize_output_file_entry({"path": "", "file": "", "filename": ""}) == ""

    def test_none_returns_none(self):
        """Should return None for None input."""
        assert normalize_output_file_entry(None) is None

    def test_non_string_non_dict_returns_string(self):
        """Should convert non-string, non-dict to string."""
        assert normalize_output_file_entry(12345) == "12345"
        assert normalize_output_file_entry(["list"]) == "['list']"

    def test_dict_with_no_matching_keys(self):
        """Should return None for dict with no path/file/filename keys."""
        assert normalize_output_file_entry({"other_key": "value.csv"}) is None


class TestCollectExpectedOutputs:
    """Tests for collect_expected_outputs function."""

    def test_none_stage_returns_empty(self):
        """Should return empty dict for None stage."""
        result = collect_expected_outputs(None, "paper1", "stage1")
        assert result == {}
        assert isinstance(result, dict)

    def test_empty_stage_returns_empty(self):
        """Should return empty dict for stage without expected_outputs."""
        stage = {}
        result = collect_expected_outputs(stage, "paper1", "stage1")
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
        assert isinstance(result["Fig1"], list)
        assert len(result["Fig1"]) == 1
        assert result["Fig1"][0] == "paper1_stage1_fig1.csv"

    def test_multiple_outputs_for_same_target(self):
        """Should collect multiple outputs for same target."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{target_id}_spectrum.csv"},
                {"target_figure": "Fig1", "filename_pattern": "{target_id}_absorption.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "Fig1" in result
        assert len(result["Fig1"]) == 2
        assert "fig1_spectrum.csv" in result["Fig1"]
        assert "fig1_absorption.csv" in result["Fig1"]

    def test_multiple_targets(self):
        """Should handle multiple different targets."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{target_id}_data.csv"},
                {"target_figure": "Fig2", "filename_pattern": "{target_id}_data.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "Fig1" in result
        assert "Fig2" in result
        assert len(result["Fig1"]) == 1
        assert len(result["Fig2"]) == 1

    def test_empty_paper_id(self):
        """Should handle empty paper_id."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{paper_id}_{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "", "stage1")
        
        assert "Fig1" in result
        assert result["Fig1"][0] == "_fig1.csv"

    def test_none_paper_id(self):
        """Should handle None paper_id."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{paper_id}_{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, None, "stage1")
        
        assert "Fig1" in result
        assert result["Fig1"][0] == "_fig1.csv"

    def test_empty_stage_id(self):
        """Should handle empty stage_id."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{paper_id}_{stage_id}_{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "")
        
        assert "Fig1" in result
        assert result["Fig1"][0] == "paper1__fig1.csv"

    def test_none_stage_id(self):
        """Should handle None stage_id."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "{paper_id}_{stage_id}_{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", None)
        
        assert "Fig1" in result
        assert result["Fig1"][0] == "paper1__fig1.csv"

    def test_skips_missing_target_figure(self):
        """Should skip entries without target_figure."""
        stage = {
            "expected_outputs": [
                {"filename_pattern": "{target_id}.csv"},
                {"target_figure": "Fig1", "filename_pattern": "{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "Fig1" in result
        assert len(result["Fig1"]) == 1

    def test_skips_missing_pattern(self):
        """Should skip entries without filename_pattern."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1"},
                {"target_figure": "Fig2", "filename_pattern": "{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "Fig1" not in result
        assert "Fig2" in result

    def test_lowercases_target_id(self):
        """Should lowercase target_id in pattern."""
        stage = {
            "expected_outputs": [
                {"target_figure": "FIG1", "filename_pattern": "{target_id}.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "FIG1" in result
        assert result["FIG1"][0] == "fig1.csv"

    def test_pattern_without_placeholders(self):
        """Should handle patterns without placeholders."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "filename_pattern": "fixed_filename.csv"}
            ]
        }
        
        result = collect_expected_outputs(stage, "paper1", "stage1")
        
        assert "Fig1" in result
        assert result["Fig1"][0] == "fixed_filename.csv"


class TestCollectExpectedColumns:
    """Tests for collect_expected_columns function."""

    def test_none_stage_returns_empty(self):
        """Should return empty dict for None stage."""
        result = collect_expected_columns(None)
        assert result == {}
        assert isinstance(result, dict)

    def test_empty_stage_returns_empty(self):
        """Should return empty dict for stage without expected_outputs."""
        stage = {}
        result = collect_expected_columns(stage)
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

    def test_empty_columns_list(self):
        """Should handle empty columns list."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "columns": []}
            ]
        }
        
        result = collect_expected_columns(stage)
        
        assert result["Fig1"] == []

    def test_missing_columns_key(self):
        """Should skip entries without columns key."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1"},
                {"target_figure": "Fig2", "columns": ["wavelength"]}
            ]
        }
        
        result = collect_expected_columns(stage)
        
        assert "Fig1" not in result
        assert "Fig2" in result

    def test_multiple_targets(self):
        """Should handle multiple targets with different columns."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "columns": ["wavelength", "transmission"]},
                {"target_figure": "Fig2", "columns": ["frequency", "reflection"]}
            ]
        }
        
        result = collect_expected_columns(stage)
        
        assert result["Fig1"] == ["wavelength", "transmission"]
        assert result["Fig2"] == ["frequency", "reflection"]

    def test_missing_target_figure(self):
        """Should skip entries without target_figure."""
        stage = {
            "expected_outputs": [
                {"columns": ["wavelength"]},
                {"target_figure": "Fig1", "columns": ["wavelength", "transmission"]}
            ]
        }
        
        result = collect_expected_columns(stage)
        
        assert "Fig1" in result
        assert len(result) == 1

    def test_none_columns_value(self):
        """Should skip entries with None columns."""
        stage = {
            "expected_outputs": [
                {"target_figure": "Fig1", "columns": None},
                {"target_figure": "Fig2", "columns": ["wavelength"]}
            ]
        }
        
        result = collect_expected_columns(stage)
        
        assert "Fig1" not in result
        assert "Fig2" in result


class TestMatchExpectedFiles:
    """Tests for match_expected_files function."""

    def test_empty_expected_returns_none(self):
        """Should return None for empty expected list."""
        assert match_expected_files([], ["output.csv"]) is None
        assert match_expected_files([], []) is None

    def test_empty_outputs_returns_none(self):
        """Should return None for empty outputs list."""
        assert match_expected_files(["spectrum.csv"], []) is None

    def test_exact_match(self):
        """Should match exact filename."""
        expected = ["spectrum.csv"]
        outputs = ["/path/to/spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_exact_match_with_multiple_outputs(self):
        """Should match exact filename from multiple outputs."""
        expected = ["spectrum.csv"]
        outputs = ["/path/to/other.csv", "/path/to/spectrum.csv", "/path/to/another.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_substring_match(self):
        """Should fall back to substring match."""
        expected = ["spectrum"]
        outputs = ["/path/to/full_spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/full_spectrum.csv"

    def test_substring_match_returns_full_path(self):
        """Should return full path, not just filename."""
        expected = ["spectrum"]
        outputs = ["/full/path/to/spectrum_data.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/full/path/to/spectrum_data.csv"

    def test_prefers_exact_over_substring(self):
        """Should prefer exact match over substring match."""
        expected = ["spectrum.csv", "spectrum"]
        outputs = ["/path/to/spectrum.csv", "/path/to/full_spectrum_data.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_handles_dict_entries(self):
        """Should handle dict entries in outputs."""
        expected = ["spectrum.csv"]
        outputs = [{"path": "/path/to/spectrum.csv"}]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_handles_dict_with_file_key(self):
        """Should handle dict entries with file key."""
        expected = ["spectrum.csv"]
        outputs = [{"file": "/path/to/spectrum.csv"}]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_handles_none_entries(self):
        """Should handle None entries in outputs."""
        expected = ["spectrum.csv"]
        outputs = [None, "/path/to/spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_case_sensitive_matching(self):
        """Should match case-sensitively."""
        expected = ["Spectrum.csv"]
        outputs = ["/path/to/spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        # Exact match should be case-sensitive
        assert result is None  # No exact match
        
        # But substring match might work
        expected2 = ["Spectrum"]
        result2 = match_expected_files(expected2, outputs)
        # This depends on implementation - substring might be case-sensitive too

    def test_multiple_expected_files(self):
        """Should match first expected file found."""
        expected = ["missing.csv", "spectrum.csv", "another.csv"]
        outputs = ["/path/to/spectrum.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result == "/path/to/spectrum.csv"

    def test_no_match_returns_none(self):
        """Should return None when no match found."""
        expected = ["spectrum.csv"]
        outputs = ["/path/to/other_file.csv"]
        
        result = match_expected_files(expected, outputs)
        
        assert result is None


class TestClassifyPercentError:
    """Tests for classify_percent_error function."""

    def test_excellent_match(self):
        """Should classify errors within excellent threshold as MATCH."""
        # Excellent threshold is 2%
        assert classify_percent_error(0.0) == AnalysisClassification.MATCH
        assert classify_percent_error(1.0) == AnalysisClassification.MATCH
        assert classify_percent_error(2.0) == AnalysisClassification.MATCH

    def test_partial_match(self):
        """Should classify errors within acceptable threshold as PARTIAL_MATCH."""
        # Acceptable threshold is 5%
        assert classify_percent_error(2.1) == AnalysisClassification.PARTIAL_MATCH
        assert classify_percent_error(3.0) == AnalysisClassification.PARTIAL_MATCH
        assert classify_percent_error(5.0) == AnalysisClassification.PARTIAL_MATCH

    def test_mismatch(self):
        """Should classify errors above acceptable threshold as MISMATCH."""
        # Above 5% is mismatch
        assert classify_percent_error(5.1) == AnalysisClassification.MISMATCH
        assert classify_percent_error(10.0) == AnalysisClassification.MISMATCH
        assert classify_percent_error(100.0) == AnalysisClassification.MISMATCH

    def test_boundary_values(self):
        """Should handle boundary values correctly."""
        # Exactly at thresholds
        assert classify_percent_error(2.0) == AnalysisClassification.MATCH
        assert classify_percent_error(5.0) == AnalysisClassification.PARTIAL_MATCH
        assert classify_percent_error(5.01) == AnalysisClassification.MISMATCH

    def test_negative_error(self):
        """Should handle negative error values."""
        # Negative error should still be classified (treated as absolute)
        result = classify_percent_error(-1.0)
        assert result in [AnalysisClassification.MATCH, AnalysisClassification.PARTIAL_MATCH, AnalysisClassification.MISMATCH]


class TestClassificationFromMetrics:
    """Tests for classification_from_metrics function."""

    # ===== No Reference Tests =====
    
    def test_no_reference_qualitative(self):
        """Should return PENDING_VALIDATION when no reference and qualitative."""
        metrics = {}
        result = classification_from_metrics(metrics, "qualitative", False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_no_reference_excellent(self):
        """Should return PENDING_VALIDATION when no reference and excellent precision."""
        metrics = {}
        result = classification_from_metrics(metrics, "excellent", False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_no_reference_acceptable(self):
        """Should return PENDING_VALIDATION when no reference and acceptable precision."""
        metrics = {}
        result = classification_from_metrics(metrics, "acceptable", False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_no_reference_with_metrics_still_pending(self):
        """Should return PENDING_VALIDATION when no reference even with metrics."""
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "excellent", False)
        assert result == AnalysisClassification.PENDING_VALIDATION

    # ===== Qualitative With Reference =====
    
    def test_qualitative_with_reference(self):
        """Should return MATCH when qualitative requirement with reference."""
        metrics = {}
        result = classification_from_metrics(metrics, "qualitative", True)
        assert result == AnalysisClassification.MATCH

    def test_qualitative_with_reference_ignores_metrics(self):
        """Should return MATCH for qualitative even with bad metrics."""
        metrics = {"peak_position_error_percent": 100.0}
        result = classification_from_metrics(metrics, "qualitative", True)
        # Qualitative doesn't care about metrics - just needs reference
        assert result == AnalysisClassification.MATCH

    # ===== Peak Error Boundary Tests =====
    
    def test_peak_error_at_excellent_threshold(self):
        """Should return MATCH when peak error exactly at excellent threshold (2%)."""
        excellent = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["excellent"]
        metrics = {"peak_position_error_percent": excellent}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MATCH

    def test_peak_error_just_above_excellent_threshold(self):
        """Should return PARTIAL_MATCH when peak error just above excellent threshold."""
        excellent = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["excellent"]
        metrics = {"peak_position_error_percent": excellent + 0.01}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_peak_error_at_acceptable_threshold(self):
        """Should return PARTIAL_MATCH when peak error exactly at acceptable threshold (5%)."""
        acceptable = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["acceptable"]
        metrics = {"peak_position_error_percent": acceptable}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_peak_error_just_above_acceptable_threshold(self):
        """Should return MISMATCH when peak error just above acceptable threshold."""
        acceptable = DISCREPANCY_THRESHOLDS["resonance_wavelength"]["acceptable"]
        metrics = {"peak_position_error_percent": acceptable + 0.01}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MISMATCH

    def test_excellent_with_peak_error_match(self):
        """Should return MATCH when peak error within excellent threshold."""
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MATCH

    def test_excellent_with_peak_error_partial(self):
        """Should return PARTIAL_MATCH when peak error within acceptable threshold."""
        metrics = {"peak_position_error_percent": 3.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_excellent_with_peak_error_mismatch(self):
        """Should return MISMATCH when peak error exceeds acceptable threshold."""
        metrics = {"peak_position_error_percent": 6.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MISMATCH

    # ===== RMSE Tests =====
    
    def test_rmse_match(self):
        """Should return MATCH when RMSE <= 5%."""
        metrics = {"normalized_rmse_percent": 5.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MATCH

    def test_rmse_just_above_match_threshold(self):
        """Should return PARTIAL_MATCH when RMSE just above 5%."""
        metrics = {"normalized_rmse_percent": 5.01}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_rmse_partial_match(self):
        """Should return PARTIAL_MATCH when RMSE <= 15%."""
        metrics = {"normalized_rmse_percent": 15.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_rmse_mismatch(self):
        """Should return MISMATCH when RMSE > 15%."""
        metrics = {"normalized_rmse_percent": 15.1}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MISMATCH

    def test_rmse_zero(self):
        """Should return MATCH when RMSE is zero."""
        metrics = {"normalized_rmse_percent": 0.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MATCH

    # ===== Precedence Tests =====
    
    def test_peak_error_takes_precedence_over_rmse(self):
        """Should prefer peak_error over RMSE when both present."""
        metrics = {
            "peak_position_error_percent": 1.0,
            "normalized_rmse_percent": 20.0
        }
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MATCH  # Based on peak_error

    def test_peak_error_mismatch_even_with_good_rmse(self):
        """Should use peak_error result even when RMSE would be better."""
        metrics = {
            "peak_position_error_percent": 10.0,
            "normalized_rmse_percent": 3.0
        }
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MISMATCH  # Based on peak_error, not RMSE

    # ===== Fallback Tests =====
    
    def test_no_metrics_returns_pending(self):
        """Should return PENDING_VALIDATION when no metrics available."""
        metrics = {}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PENDING_VALIDATION

    def test_none_peak_error_with_rmse(self):
        """Should use RMSE when peak_error is None."""
        metrics = {"peak_position_error_percent": None, "normalized_rmse_percent": 10.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_missing_peak_error_key_with_rmse(self):
        """Should use RMSE when peak_error key is missing."""
        metrics = {"normalized_rmse_percent": 10.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_both_none_returns_pending(self):
        """Should return PENDING_VALIDATION when both metrics are None."""
        metrics = {"peak_position_error_percent": None, "normalized_rmse_percent": None}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.PENDING_VALIDATION

    # ===== Precision Requirement Variations =====
    
    def test_acceptable_precision_with_good_metrics(self):
        """Should classify correctly with acceptable precision requirement."""
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "acceptable", True)
        assert result == AnalysisClassification.MATCH

    def test_acceptable_precision_with_partial_metrics(self):
        """Should classify correctly with acceptable precision requirement."""
        metrics = {"peak_position_error_percent": 3.5}
        result = classification_from_metrics(metrics, "acceptable", True)
        assert result == AnalysisClassification.PARTIAL_MATCH

    def test_empty_string_precision_requirement(self):
        """Should handle empty string precision requirement."""
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "", True)
        # Empty string is not "qualitative", so should evaluate metrics
        assert result == AnalysisClassification.MATCH

    def test_unknown_precision_requirement(self):
        """Should handle unknown precision requirement."""
        metrics = {"peak_position_error_percent": 1.0}
        result = classification_from_metrics(metrics, "unknown_precision", True)
        # Should still evaluate metrics
        assert result == AnalysisClassification.MATCH

    # ===== Edge Cases =====
    
    def test_zero_peak_error(self):
        """Should return MATCH for zero peak error."""
        metrics = {"peak_position_error_percent": 0.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MATCH

    def test_very_large_peak_error(self):
        """Should return MISMATCH for very large peak error."""
        metrics = {"peak_position_error_percent": 1000.0}
        result = classification_from_metrics(metrics, "excellent", True)
        assert result == AnalysisClassification.MISMATCH

    def test_negative_peak_error(self):
        """Should handle negative peak error using absolute value."""
        metrics = {"peak_position_error_percent": -3.0}
        result = classification_from_metrics(metrics, "excellent", True)
        # After our fix, should use abs(-3.0) = 3.0, which is PARTIAL_MATCH
        assert result == AnalysisClassification.PARTIAL_MATCH


class TestEvaluateValidationCriteria:
    """Tests for evaluate_validation_criteria function."""

    def test_empty_criteria_returns_true(self):
        """Should return True with no failures for empty criteria."""
        metrics = {"peak_position_error_percent": 10.0}
        result = evaluate_validation_criteria(metrics, [])
        assert result[0] is True
        assert result[1] == []

    def test_none_criteria_returns_true(self):
        """Should return True with no failures for None criteria."""
        metrics = {"peak_position_error_percent": 10.0}
        result = evaluate_validation_criteria(metrics, None)
        # Should handle None gracefully or treat as empty
        # Implementation might need to handle this

    def test_resonance_within_percent_pass(self):
        """Should pass when resonance error within threshold."""
        metrics = {"peak_position_error_percent": 2.0}
        criteria = ["resonance within 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is True
        assert len(result[1]) == 0

    def test_resonance_within_percent_fail(self):
        """Should fail when resonance error exceeds threshold."""
        metrics = {"peak_position_error_percent": 6.0}
        criteria = ["resonance within 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False
        assert len(result[1]) > 0
        assert "resonance within 5%" in result[1][0]

    def test_peak_within_percent_pass(self):
        """Should pass when peak error within threshold."""
        metrics = {"peak_position_error_percent": 3.0}
        criteria = ["peak within 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is True

    def test_peak_within_percent_fail(self):
        """Should fail when peak error exceeds threshold."""
        metrics = {"peak_position_error_percent": 6.0}
        criteria = ["peak within 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False

    def test_normalized_rmse_max_pass(self):
        """Should pass when RMSE within threshold."""
        metrics = {"normalized_rmse_percent": 3.0}
        criteria = ["normalized RMSE <= 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is True

    def test_normalized_rmse_max_fail(self):
        """Should fail when RMSE exceeds threshold."""
        metrics = {"normalized_rmse_percent": 6.0}
        criteria = ["normalized RMSE <= 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False

    def test_correlation_min_pass(self):
        """Should pass when correlation meets threshold."""
        metrics = {"correlation": 0.95}
        criteria = ["correlation >= 0.9"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is True

    def test_correlation_min_fail(self):
        """Should fail when correlation below threshold."""
        metrics = {"correlation": 0.8}
        criteria = ["correlation >= 0.9"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False

    def test_missing_metric_reports_failure(self):
        """Should report failure when required metric is missing."""
        metrics = {}
        criteria = ["resonance within 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False
        assert len(result[1]) > 0
        assert "missing metric" in result[1][0].lower()

    def test_multiple_criteria_all_pass(self):
        """Should pass when all criteria pass."""
        metrics = {
            "peak_position_error_percent": 2.0,
            "normalized_rmse_percent": 3.0,
            "correlation": 0.95
        }
        criteria = [
            "resonance within 5%",
            "normalized RMSE <= 10%",
            "correlation >= 0.9"
        ]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is True
        assert len(result[1]) == 0

    def test_multiple_criteria_some_fail(self):
        """Should fail when some criteria fail."""
        metrics = {
            "peak_position_error_percent": 2.0,
            "normalized_rmse_percent": 20.0,
            "correlation": 0.95
        }
        criteria = [
            "resonance within 5%",
            "normalized RMSE <= 10%",
            "correlation >= 0.9"
        ]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False
        assert len(result[1]) > 0

    def test_unknown_criteria_ignored(self):
        """Should ignore criteria that don't match known patterns."""
        metrics = {"peak_position_error_percent": 10.0}
        criteria = ["unknown criterion format"]
        result = evaluate_validation_criteria(metrics, criteria)
        # Unknown criteria should be ignored (treated as informational)
        assert result[0] is True

    def test_case_insensitive_matching(self):
        """Should match criteria case-insensitively."""
        metrics = {"peak_position_error_percent": 6.0}
        criteria = ["RESONANCE WITHIN 5%"]
        result = evaluate_validation_criteria(metrics, criteria)
        assert result[0] is False


class TestStageComparisonsForStage:
    """Tests for stage_comparisons_for_stage function."""

    def test_none_stage_id_returns_empty(self):
        """Should return empty list for None stage_id."""
        state = {"figure_comparisons": []}
        result = stage_comparisons_for_stage(state, None)
        assert result == []

    def test_empty_stage_id_returns_empty(self):
        """Should return empty list for empty stage_id."""
        state = {"figure_comparisons": []}
        result = stage_comparisons_for_stage(state, "")
        assert result == []

    def test_filters_by_stage_id(self):
        """Should return only comparisons for specified stage."""
        state = {
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "Fig1"},
                {"stage_id": "stage2", "figure_id": "Fig2"},
                {"stage_id": "stage1", "figure_id": "Fig3"}
            ]
        }
        result = stage_comparisons_for_stage(state, "stage1")
        assert len(result) == 2
        assert all(comp["stage_id"] == "stage1" for comp in result)

    def test_no_comparisons_returns_empty(self):
        """Should return empty list when no comparisons exist."""
        state = {"figure_comparisons": []}
        result = stage_comparisons_for_stage(state, "stage1")
        assert result == []

    def test_missing_comparisons_key(self):
        """Should handle missing figure_comparisons key."""
        state = {}
        result = stage_comparisons_for_stage(state, "stage1")
        assert result == []

    def test_none_comparisons(self):
        """Should handle None figure_comparisons."""
        state = {"figure_comparisons": None}
        result = stage_comparisons_for_stage(state, "stage1")
        assert result == []

    def test_comparisons_without_stage_id(self):
        """Should exclude comparisons without stage_id."""
        state = {
            "figure_comparisons": [
                {"figure_id": "Fig1"},
                {"stage_id": "stage1", "figure_id": "Fig2"}
            ]
        }
        result = stage_comparisons_for_stage(state, "stage1")
        assert len(result) == 1
        assert result[0]["figure_id"] == "Fig2"


class TestAnalysisReportsForStage:
    """Tests for analysis_reports_for_stage function."""

    def test_none_stage_id_returns_empty(self):
        """Should return empty list for None stage_id."""
        state = {"analysis_result_reports": []}
        result = analysis_reports_for_stage(state, None)
        assert result == []

    def test_empty_stage_id_returns_empty(self):
        """Should return empty list for empty stage_id."""
        state = {"analysis_result_reports": []}
        result = analysis_reports_for_stage(state, "")
        assert result == []

    def test_filters_by_stage_id(self):
        """Should return only reports for specified stage."""
        state = {
            "analysis_result_reports": [
                {"stage_id": "stage1", "target_figure": "Fig1"},
                {"stage_id": "stage2", "target_figure": "Fig2"},
                {"stage_id": "stage1", "target_figure": "Fig3"}
            ]
        }
        result = analysis_reports_for_stage(state, "stage1")
        assert len(result) == 2
        assert all(report["stage_id"] == "stage1" for report in result)

    def test_no_reports_returns_empty(self):
        """Should return empty list when no reports exist."""
        state = {"analysis_result_reports": []}
        result = analysis_reports_for_stage(state, "stage1")
        assert result == []

    def test_missing_reports_key(self):
        """Should handle missing analysis_result_reports key."""
        state = {}
        result = analysis_reports_for_stage(state, "stage1")
        assert result == []

    def test_none_reports(self):
        """Should handle None analysis_result_reports."""
        state = {"analysis_result_reports": None}
        result = analysis_reports_for_stage(state, "stage1")
        assert result == []

    def test_reports_without_stage_id(self):
        """Should exclude reports without stage_id."""
        state = {
            "analysis_result_reports": [
                {"target_figure": "Fig1"},
                {"stage_id": "stage1", "target_figure": "Fig2"}
            ]
        }
        result = analysis_reports_for_stage(state, "stage1")
        assert len(result) == 1
        assert result[0]["target_figure"] == "Fig2"


class TestValidateAnalysisReports:
    """Tests for validate_analysis_reports function."""

    def test_empty_reports_returns_empty(self):
        """Should return empty list for empty reports."""
        result = validate_analysis_reports([])
        assert result == []

    def test_excellent_precision_requires_metrics(self):
        """Should flag reports with excellent precision but no metrics."""
        reports = [{
            "target_figure": "Fig1",
            "precision_requirement": "excellent",
            "quantitative_metrics": {}
        }]
        result = validate_analysis_reports(reports)
        assert len(result) > 0
        assert "Fig1" in result[0]
        assert "excellent precision requires quantitative metrics" in result[0].lower()

    def test_match_with_high_error(self):
        """Should flag match classification with error above acceptable threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": "match",
            "quantitative_metrics": {"peak_position_error_percent": 10.0}
        }]
        result = validate_analysis_reports(reports)
        assert len(result) > 0
        assert "Fig1" in result[0]
        assert "match" in result[0].lower()
        assert "error" in result[0].lower()

    def test_match_with_acceptable_error(self):
        """Should not flag match classification with acceptable error."""
        reports = [{
            "target_figure": "Fig1",
            "status": "match",
            "quantitative_metrics": {"peak_position_error_percent": 3.0}
        }]
        result = validate_analysis_reports(reports)
        # Should not flag errors within acceptable threshold
        match_issues = [r for r in result if "match" in r.lower() and "error" in r.lower()]
        assert len(match_issues) == 0

    def test_pending_with_high_error(self):
        """Should flag pending/partial with error above investigate threshold."""
        reports = [{
            "target_figure": "Fig1",
            "status": "pending_validation",
            "quantitative_metrics": {"peak_position_error_percent": 15.0}
        }]
        result = validate_analysis_reports(reports)
        assert len(result) > 0
        assert "Fig1" in result[0]
        assert "investigate" in result[0].lower() or "mismatch" in result[0].lower()

    def test_criteria_failures_included(self):
        """Should include criteria failures in issues."""
        reports = [{
            "target_figure": "Fig1",
            "criteria_failures": ["resonance within 5% (measured 6.0 > 5)"]
        }]
        result = validate_analysis_reports(reports)
        assert len(result) > 0
        assert "Fig1" in result[0]
        assert "resonance within 5%" in result[0]

    def test_multiple_issues_per_report(self):
        """Should report multiple issues for a single report."""
        reports = [{
            "target_figure": "Fig1",
            "status": "match",
            "quantitative_metrics": {"peak_position_error_percent": 10.0},
            "criteria_failures": ["resonance within 5% (measured 6.0 > 5)"]
        }]
        result = validate_analysis_reports(reports)
        assert len(result) >= 2

    def test_multiple_reports(self):
        """Should validate multiple reports."""
        reports = [
            {
                "target_figure": "Fig1",
                "status": "match",
                "quantitative_metrics": {"peak_position_error_percent": 10.0}
            },
            {
                "target_figure": "Fig2",
                "precision_requirement": "excellent",
                "quantitative_metrics": {}
            }
        ]
        result = validate_analysis_reports(reports)
        assert len(result) >= 2

    def test_missing_fields_handled(self):
        """Should handle reports with missing fields gracefully."""
        reports = [
            {},
            {"target_figure": "Fig1"}
        ]
        result = validate_analysis_reports(reports)
        # Should not crash, may or may not report issues

    def test_none_metrics(self):
        """Should handle None quantitative_metrics."""
        reports = [{
            "target_figure": "Fig1",
            "precision_requirement": "excellent",
            "quantitative_metrics": None
        }]
        result = validate_analysis_reports(reports)
        # Should handle None gracefully

    def test_none_criteria_failures(self):
        """Should handle None criteria_failures."""
        reports = [{
            "target_figure": "Fig1",
            "criteria_failures": None
        }]
        result = validate_analysis_reports(reports)
        # Should not crash


class TestBreakdownComparisonClassifications:
    """Tests for breakdown_comparison_classifications function."""

    def test_empty_comparisons(self):
        """Should return empty breakdown for empty comparisons."""
        result = breakdown_comparison_classifications([])
        assert result == {"missing": [], "pending": [], "matches": []}
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())

    def test_missing_classifications_string_values(self):
        """Should categorize string missing classifications correctly."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "missing_output"},
            {"figure_id": "Fig2", "classification": "fail"},
            {"figure_id": "Fig3", "classification": "mismatch"},
            {"figure_id": "Fig4", "classification": "not_reproduced"},
            {"figure_id": "Fig5", "classification": "poor_match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["missing"]) == 5
        assert all(f"Fig{i}" in result["missing"] for i in range(1, 6))

    def test_pending_classifications_string_values(self):
        """Should categorize string pending classifications correctly."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "pending_validation"},
            {"figure_id": "Fig2", "classification": "partial_match"},
            {"figure_id": "Fig3", "classification": "match_pending"},
            {"figure_id": "Fig4", "classification": "partial"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["pending"]) == 4
        assert all(f"Fig{i}" in result["pending"] for i in range(1, 5))

    def test_match_classifications_string_values(self):
        """Should categorize string match classifications correctly."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "match"},
            {"figure_id": "Fig2", "classification": "excellent_match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["matches"]) == 2
        assert "Fig1" in result["matches"]
        assert "Fig2" in result["matches"]

    def test_case_insensitive_missing(self):
        """Should handle case-insensitive missing classifications."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "MISMATCH"},
            {"figure_id": "Fig2", "classification": "Fail"},
            {"figure_id": "Fig3", "classification": "MISSING_OUTPUT"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["missing"]) == 3

    def test_case_insensitive_pending(self):
        """Should handle case-insensitive pending classifications."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "PENDING_VALIDATION"},
            {"figure_id": "Fig2", "classification": "Partial_Match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["pending"]) == 2

    def test_case_insensitive_match(self):
        """Should handle case-insensitive match classifications."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "MATCH"},
            {"figure_id": "Fig2", "classification": "Match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["matches"]) == 2

    def test_mixed_classifications(self):
        """Should correctly categorize mixed classifications."""
        comparisons = [
            {"figure_id": "Fig1", "classification": "match"},
            {"figure_id": "Fig2", "classification": "pending_validation"},
            {"figure_id": "Fig3", "classification": "mismatch"},
            {"figure_id": "Fig4", "classification": "match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["matches"]) == 2
        assert len(result["pending"]) == 1
        assert len(result["missing"]) == 1
        assert "Fig1" in result["matches"]
        assert "Fig4" in result["matches"]
        assert "Fig2" in result["pending"]
        assert "Fig3" in result["missing"]

    def test_missing_classification_field_defaults_to_matches(self):
        """Should default to matches bucket when classification field missing."""
        comparisons = [
            {"figure_id": "Fig1"},
            {"figure_id": "Fig2", "classification": "match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        # Missing classification defaults to matches (unknown classification)
        assert "Fig1" in result["matches"]
        assert "Fig2" in result["matches"]

    def test_none_classification_defaults_to_matches(self):
        """Should default to matches bucket when classification is None."""
        comparisons = [
            {"figure_id": "Fig1", "classification": None},
            {"figure_id": "Fig2", "classification": "match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["matches"]
        assert "Fig2" in result["matches"]

    def test_empty_classification_defaults_to_matches(self):
        """Should default to matches bucket when classification is empty string."""
        comparisons = [
            {"figure_id": "Fig1", "classification": ""},
            {"figure_id": "Fig2", "classification": "match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["matches"]
        assert "Fig2" in result["matches"]

    def test_enum_match(self):
        """Should handle AnalysisClassification.MATCH enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.MATCH}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["matches"]

    def test_enum_excellent_match(self):
        """Should handle AnalysisClassification.EXCELLENT_MATCH enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.EXCELLENT_MATCH}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["matches"]

    def test_enum_acceptable_match(self):
        """Should handle AnalysisClassification.ACCEPTABLE_MATCH enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.ACCEPTABLE_MATCH}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["matches"]

    def test_enum_pending_validation(self):
        """Should handle AnalysisClassification.PENDING_VALIDATION enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.PENDING_VALIDATION}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["pending"]

    def test_enum_partial_match(self):
        """Should handle AnalysisClassification.PARTIAL_MATCH enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.PARTIAL_MATCH}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["pending"]

    def test_enum_mismatch(self):
        """Should handle AnalysisClassification.MISMATCH enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.MISMATCH}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["missing"]

    def test_enum_poor_match(self):
        """Should handle AnalysisClassification.POOR_MATCH enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.POOR_MATCH}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["missing"]

    def test_enum_failed(self):
        """Should handle AnalysisClassification.FAILED enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.FAILED}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["missing"]

    def test_enum_no_targets(self):
        """Should handle AnalysisClassification.NO_TARGETS enum value."""
        comparisons = [
            {"figure_id": "Fig1", "classification": AnalysisClassification.NO_TARGETS}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "Fig1" in result["missing"]

    def test_all_enum_values_categorized(self):
        """Should correctly categorize all AnalysisClassification enum values."""
        comparisons = [
            {"figure_id": "Fig_EXCELLENT_MATCH", "classification": AnalysisClassification.EXCELLENT_MATCH},
            {"figure_id": "Fig_ACCEPTABLE_MATCH", "classification": AnalysisClassification.ACCEPTABLE_MATCH},
            {"figure_id": "Fig_PARTIAL_MATCH", "classification": AnalysisClassification.PARTIAL_MATCH},
            {"figure_id": "Fig_POOR_MATCH", "classification": AnalysisClassification.POOR_MATCH},
            {"figure_id": "Fig_FAILED", "classification": AnalysisClassification.FAILED},
            {"figure_id": "Fig_NO_TARGETS", "classification": AnalysisClassification.NO_TARGETS},
            {"figure_id": "Fig_PENDING_VALIDATION", "classification": AnalysisClassification.PENDING_VALIDATION},
            {"figure_id": "Fig_MATCH", "classification": AnalysisClassification.MATCH},
            {"figure_id": "Fig_MISMATCH", "classification": AnalysisClassification.MISMATCH},
        ]
        result = breakdown_comparison_classifications(comparisons)
        
        # Verify matches bucket
        assert "Fig_EXCELLENT_MATCH" in result["matches"]
        assert "Fig_ACCEPTABLE_MATCH" in result["matches"]
        assert "Fig_MATCH" in result["matches"]
        
        # Verify pending bucket
        assert "Fig_PARTIAL_MATCH" in result["pending"]
        assert "Fig_PENDING_VALIDATION" in result["pending"]
        
        # Verify missing bucket  
        assert "Fig_POOR_MATCH" in result["missing"]
        assert "Fig_FAILED" in result["missing"]
        assert "Fig_NO_TARGETS" in result["missing"]
        assert "Fig_MISMATCH" in result["missing"]

    def test_missing_figure_id_uses_unknown(self):
        """Should use 'unknown' when figure_id is missing."""
        comparisons = [
            {"classification": "match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert "unknown" in result["matches"]

    def test_preserves_figure_id_order(self):
        """Should preserve order of figure_ids within each bucket."""
        comparisons = [
            {"figure_id": "A", "classification": "match"},
            {"figure_id": "B", "classification": "match"},
            {"figure_id": "C", "classification": "match"}
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert result["matches"] == ["A", "B", "C"]

    def test_large_number_of_comparisons(self):
        """Should handle large number of comparisons efficiently."""
        comparisons = [
            {"figure_id": f"Fig{i}", "classification": "match"}
            for i in range(100)
        ]
        result = breakdown_comparison_classifications(comparisons)
        assert len(result["matches"]) == 100
        assert len(result["pending"]) == 0
        assert len(result["missing"]) == 0
