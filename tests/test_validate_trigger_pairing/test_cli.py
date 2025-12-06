"""
Tests for the validate_trigger_pairing CLI interface.

These tests verify that the CLI:
1. Returns correct exit codes
2. Produces valid JSON output
3. Handles file paths correctly
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
TOOL_PATH = PROJECT_ROOT / "tools" / "validate_trigger_pairing.py"


def run_cli(*args, input_text=None):
    """Run the CLI tool and return the result."""
    cmd = [sys.executable, str(TOOL_PATH)] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_text,
        cwd=str(PROJECT_ROOT),
    )
    return result


class TestCLIExitCodes:
    """Tests for CLI exit codes."""
    
    def test_returns_zero_on_valid_file(self, tmp_path, valid_dict_literal):
        """Should return 0 when file has no violations."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli(str(test_file))
        
        assert result.returncode == 0
    
    def test_returns_nonzero_on_violation(self, tmp_path, invalid_dict_literal):
        """Should return non-zero when file has violations."""
        test_file = tmp_path / "invalid.py"
        test_file.write_text(invalid_dict_literal)
        
        result = run_cli(str(test_file))
        
        assert result.returncode != 0
    
    def test_returns_zero_on_warnings_only(self, tmp_path, suspicious_variable):
        """Should return 0 when file has only warnings (not errors)."""
        test_file = tmp_path / "warning.py"
        test_file.write_text(suspicious_variable)
        
        result = run_cli(str(test_file))
        
        # Warnings alone don't cause failure
        assert result.returncode == 0
    
    def test_warnings_as_errors_flag(self, tmp_path, suspicious_variable):
        """Should return non-zero with --warnings-as-errors flag."""
        test_file = tmp_path / "warning.py"
        test_file.write_text(suspicious_variable)
        
        result = run_cli("--warnings-as-errors", str(test_file))
        
        assert result.returncode != 0
    
    def test_returns_zero_on_empty_file(self, tmp_path, empty_file):
        """Should return 0 for empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text(empty_file)
        
        result = run_cli(str(test_file))
        
        assert result.returncode == 0
    
    def test_returns_zero_on_no_triggers(self, tmp_path, no_triggers):
        """Should return 0 for file with no triggers."""
        test_file = tmp_path / "no_triggers.py"
        test_file.write_text(no_triggers)
        
        result = run_cli(str(test_file))
        
        assert result.returncode == 0


class TestCLIJsonOutput:
    """Tests for JSON output mode."""
    
    def test_json_output_is_valid(self, tmp_path, valid_dict_literal):
        """JSON output should be valid JSON."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli("--json", str(test_file))
        
        data = json.loads(result.stdout)
        assert isinstance(data, dict)
    
    def test_json_contains_required_fields(self, tmp_path, valid_dict_literal):
        """JSON output should contain required fields."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli("--json", str(test_file))
        data = json.loads(result.stdout)
        
        assert "files_analyzed" in data
        assert "trigger_count" in data
        assert "questions_count" in data
        assert "violations" in data
        assert "parse_errors" in data
    
    def test_json_violations_structure(self, tmp_path, invalid_dict_literal):
        """JSON violations should have correct structure."""
        test_file = tmp_path / "invalid.py"
        test_file.write_text(invalid_dict_literal)
        
        result = run_cli("--json", str(test_file))
        data = json.loads(result.stdout)
        
        assert len(data["violations"]) > 0
        violation = data["violations"][0]
        
        assert "type" in violation
        assert "filepath" in violation
        assert "line" in violation
        assert "message" in violation
        assert "severity" in violation
    
    def test_json_counts_are_correct(self, tmp_path, valid_dict_literal):
        """JSON counts should reflect actual analysis."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli("--json", str(test_file))
        data = json.loads(result.stdout)
        
        assert data["files_analyzed"] == 1
        assert data["trigger_count"] == 1
        assert data["questions_count"] == 1


class TestCLIFilePaths:
    """Tests for file path handling."""
    
    def test_handles_single_file(self, tmp_path, valid_dict_literal):
        """Should handle a single file path."""
        test_file = tmp_path / "single.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli("--json", str(test_file))
        data = json.loads(result.stdout)
        
        assert data["files_analyzed"] == 1
    
    def test_handles_multiple_files(self, tmp_path, valid_dict_literal, valid_subscript):
        """Should handle multiple file paths."""
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text(valid_dict_literal)
        file2.write_text(valid_subscript)
        
        result = run_cli("--json", str(file1), str(file2))
        data = json.loads(result.stdout)
        
        assert data["files_analyzed"] == 2
    
    def test_handles_nonexistent_file(self, tmp_path):
        """Should handle nonexistent file gracefully."""
        nonexistent = tmp_path / "does_not_exist.py"
        
        result = run_cli("--json", str(nonexistent))
        data = json.loads(result.stdout)
        
        assert len(data["parse_errors"]) > 0
        assert "not found" in data["parse_errors"][0].lower()
    
    def test_handles_directory_scan(self):
        """Running without args should scan src/ directory."""
        result = run_cli("--json")
        data = json.loads(result.stdout)
        
        # Should analyze multiple files in src/
        assert data["files_analyzed"] > 1


class TestCLIOutput:
    """Tests for human-readable output."""
    
    def test_shows_stats_in_output(self, tmp_path, valid_dict_literal):
        """Should show file stats in output."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli(str(test_file))
        
        assert "Files analyzed:" in result.stdout
        assert "Trigger assignments" in result.stdout
        assert "Questions assignments" in result.stdout
    
    def test_shows_success_message(self, tmp_path, valid_dict_literal):
        """Should show success message when no violations."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli(str(test_file))
        
        assert "No violations found" in result.stdout or "No errors" in result.stdout
    
    def test_shows_error_details(self, tmp_path, invalid_dict_literal):
        """Should show error details when violations found."""
        test_file = tmp_path / "invalid.py"
        test_file.write_text(invalid_dict_literal)
        
        result = run_cli(str(test_file))
        
        assert "ERROR" in result.stdout
        assert "test_trigger" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI with real codebase."""
    
    def test_full_scan_completes(self):
        """Full src/ scan should complete without crashing."""
        result = run_cli()
        
        # Should complete (exit code might be 0 or 1 depending on state)
        assert result.returncode in (0, 1)
    
    def test_full_scan_analyzes_files(self):
        """Full src/ scan should analyze files."""
        result = run_cli("--json")
        data = json.loads(result.stdout)
        
        assert data["files_analyzed"] > 0
    
    def test_verbose_flag_accepted(self, tmp_path, valid_dict_literal):
        """Verbose flag should be accepted."""
        test_file = tmp_path / "valid.py"
        test_file.write_text(valid_dict_literal)
        
        result = run_cli("--verbose", str(test_file))
        
        # Should complete without error
        assert result.returncode == 0
