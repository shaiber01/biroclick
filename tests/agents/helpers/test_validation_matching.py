"""Matching/output validation helper tests."""

import pytest

from src.agents.helpers.validation import (
    collect_expected_columns,
    collect_expected_outputs,
    extract_targets_from_feedback,
    match_expected_files,
    match_output_file,
    normalize_output_file_entry,
)


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
