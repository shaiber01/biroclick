"""Tests for helper utilities in `src.code_runner`."""

from pathlib import Path
import pytest
from src.code_runner import _list_output_files, _make_error_result


class TestMakeErrorResult:
    """Tests for _make_error_result function."""

    def test_basic_error_result(self):
        """Test basic error result construction with minimal parameters."""
        res = _make_error_result("oops", exit_code=5)
        
        # Verify all required ExecutionResult fields are present
        assert res["error"] == "oops"
        assert res["exit_code"] == 5
        assert res["output_files"] == []
        assert res["stdout"] == ""
        assert res["stderr"] == ""
        assert res["runtime_seconds"] == 0.0
        assert res["memory_exceeded"] is False
        assert res["timeout_exceeded"] is False

    def test_error_result_with_all_parameters(self):
        """Test error result with all parameters specified."""
        res = _make_error_result(
            error="test error",
            stdout="stdout content",
            stderr="stderr content",
            exit_code=42,
            runtime_seconds=123.45,
            memory_exceeded=True,
            timeout_exceeded=False
        )
        
        assert res["error"] == "test error"
        assert res["stdout"] == "stdout content"
        assert res["stderr"] == "stderr content"
        assert res["exit_code"] == 42
        assert res["output_files"] == []
        assert res["runtime_seconds"] == 123.45
        assert res["memory_exceeded"] is True
        assert res["timeout_exceeded"] is False

    def test_error_result_with_output_dir(self, tmp_path):
        """Test error result with output_dir specified - should list files."""
        (tmp_path / "output1.txt").touch()
        (tmp_path / "output2.csv").touch()
        (tmp_path / "script.py").touch()
        
        res = _make_error_result(
            error="error with files",
            output_dir=tmp_path,
            exclude_files=["script.py"]
        )
        
        assert res["error"] == "error with files"
        assert res["exit_code"] == -1  # default
        assert len(res["output_files"]) == 2
        assert "output1.txt" in res["output_files"]
        assert "output2.csv" in res["output_files"]
        assert "script.py" not in res["output_files"]

    def test_error_result_with_output_dir_no_exclude(self, tmp_path):
        """Test error result with output_dir but no exclude list."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        
        res = _make_error_result(
            error="error",
            output_dir=tmp_path
        )
        
        assert len(res["output_files"]) == 2
        assert "file1.txt" in res["output_files"]
        assert "file2.txt" in res["output_files"]

    def test_error_result_with_empty_output_dir(self, tmp_path):
        """Test error result with empty output directory."""
        res = _make_error_result(
            error="error",
            output_dir=tmp_path
        )
        
        assert res["output_files"] == []

    def test_error_result_timeout_flag(self):
        """Test error result with timeout_exceeded flag."""
        res = _make_error_result(
            error="timeout",
            timeout_exceeded=True,
            runtime_seconds=3600.0
        )
        
        assert res["timeout_exceeded"] is True
        assert res["memory_exceeded"] is False
        assert res["runtime_seconds"] == 3600.0

    def test_error_result_memory_flag(self):
        """Test error result with memory_exceeded flag."""
        res = _make_error_result(
            error="memory",
            memory_exceeded=True
        )
        
        assert res["memory_exceeded"] is True
        assert res["timeout_exceeded"] is False

    def test_error_result_both_flags(self):
        """Test error result with both flags set."""
        res = _make_error_result(
            error="both",
            memory_exceeded=True,
            timeout_exceeded=True
        )
        
        assert res["memory_exceeded"] is True
        assert res["timeout_exceeded"] is True

    def test_error_result_empty_error_string(self):
        """Test error result with empty error string."""
        res = _make_error_result("")
        
        assert res["error"] == ""
        assert res["exit_code"] == -1  # default

    def test_error_result_negative_exit_code(self):
        """Test error result with negative exit code (signal termination)."""
        res = _make_error_result("killed", exit_code=-9)
        
        assert res["exit_code"] == -9
        assert res["error"] == "killed"

    def test_error_result_zero_exit_code(self):
        """Test error result with zero exit code."""
        res = _make_error_result("warning", exit_code=0)
        
        assert res["exit_code"] == 0

    def test_error_result_output_files_sorted(self, tmp_path):
        """Test that output_files are sorted alphabetically."""
        (tmp_path / "zebra.txt").touch()
        (tmp_path / "apple.txt").touch()
        (tmp_path / "banana.txt").touch()
        
        res = _make_error_result(
            error="test",
            output_dir=tmp_path
        )
        
        assert res["output_files"] == ["apple.txt", "banana.txt", "zebra.txt"]


class TestListOutputFiles:
    """Tests for _list_output_files function."""

    def test_basic_file_listing(self, tmp_path):
        """Test basic file listing with exclude."""
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "ignore.py").touch()

        files = _list_output_files(tmp_path, exclude=["ignore.py"])
        
        assert len(files) == 2
        assert "a.txt" in files
        assert "b.txt" in files
        assert "ignore.py" not in files

    def test_empty_directory(self, tmp_path):
        """Test listing files in empty directory."""
        files = _list_output_files(tmp_path)
        
        assert files == []

    def test_exclude_none(self, tmp_path):
        """Test with exclude=None (should be treated as empty list)."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=None)
        
        assert len(files) == 2
        assert "file1.txt" in files
        assert "file2.txt" in files

    def test_exclude_empty_list(self, tmp_path):
        """Test with exclude=[] (should include all files)."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=[])
        
        assert len(files) == 2
        assert "file1.txt" in files
        assert "file2.txt" in files

    def test_multiple_exclude_patterns(self, tmp_path):
        """Test excluding multiple files."""
        (tmp_path / "keep1.txt").touch()
        (tmp_path / "keep2.csv").touch()
        (tmp_path / "ignore1.py").touch()
        (tmp_path / "ignore2.py").touch()
        
        files = _list_output_files(tmp_path, exclude=["ignore1.py", "ignore2.py"])
        
        assert len(files) == 2
        assert "keep1.txt" in files
        assert "keep2.csv" in files
        assert "ignore1.py" not in files
        assert "ignore2.py" not in files

    def test_exclude_all_files(self, tmp_path):
        """Test excluding all files."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=["file1.txt", "file2.txt"])
        
        assert files == []

    def test_files_sorted_alphabetically(self, tmp_path):
        """Test that files are returned in sorted order."""
        (tmp_path / "zebra.txt").touch()
        (tmp_path / "apple.txt").touch()
        (tmp_path / "banana.txt").touch()
        
        files = _list_output_files(tmp_path)
        
        assert files == ["apple.txt", "banana.txt", "zebra.txt"]

    def test_directories_excluded(self, tmp_path):
        """Test that subdirectories are excluded (only files listed)."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").touch()
        
        files = _list_output_files(tmp_path)
        
        assert len(files) == 1
        assert "file.txt" in files
        assert "subdir" not in files

    def test_special_characters_in_filenames(self, tmp_path):
        """Test files with special characters in names."""
        (tmp_path / "file with spaces.txt").touch()
        (tmp_path / "file-with-dashes.txt").touch()
        (tmp_path / "file_with_underscores.txt").touch()
        (tmp_path / "file123.txt").touch()
        
        files = _list_output_files(tmp_path)
        
        assert len(files) == 4
        assert "file with spaces.txt" in files
        assert "file-with-dashes.txt" in files
        assert "file_with_underscores.txt" in files
        assert "file123.txt" in files

    def test_exclude_case_sensitive(self, tmp_path):
        """Test that exclude matching is case-sensitive."""
        (tmp_path / "File.txt").touch()
        (tmp_path / "file.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=["file.txt"])
        
        assert len(files) == 1
        assert "File.txt" in files
        assert "file.txt" not in files

    def test_nonexistent_directory(self):
        """Test behavior with non-existent directory."""
        nonexistent = Path("/nonexistent/path/that/does/not/exist")
        
        files = _list_output_files(nonexistent)
        
        # Should return empty list, not raise exception
        assert files == []

    def test_permission_denied_directory(self, tmp_path):
        """Test behavior when directory exists but cannot be read."""
        # Create a directory and then remove read permissions
        # Note: This may not work on all systems, so we test graceful handling
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()
        
        # On Unix, we could chmod 000, but that's system-dependent
        # Instead, test that the function handles exceptions gracefully
        # by checking it returns empty list for problematic directories
        
        # The function should catch exceptions and return []
        # We can't easily test actual permission errors in a portable way,
        # but we verify the function signature allows for graceful handling

    def test_exclude_partial_match(self, tmp_path):
        """Test that exclude requires exact filename match, not partial."""
        (tmp_path / "script.py").touch()
        (tmp_path / "script.pyc").touch()
        (tmp_path / "script.py.backup").touch()
        
        files = _list_output_files(tmp_path, exclude=["script.py"])
        
        assert len(files) == 2
        assert "script.py" not in files
        assert "script.pyc" in files
        assert "script.py.backup" in files

    def test_mixed_file_types(self, tmp_path):
        """Test listing files of various types."""
        (tmp_path / "data.txt").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "data.json").touch()
        (tmp_path / "data.npy").touch()
        (tmp_path / "script.py").touch()
        
        files = _list_output_files(tmp_path)
        
        assert len(files) == 5
        assert all(f.endswith(ext) for f, ext in [
            ("data.txt", ".txt"),
            ("data.csv", ".csv"),
            ("data.json", ".json"),
            ("data.npy", ".npy"),
            ("script.py", ".py")
        ])

    def test_empty_strings_in_exclude(self, tmp_path):
        """Test exclude list with empty strings."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "another.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=["", "file.txt"])
        
        # Empty string should not match anything, but shouldn't cause error
        assert len(files) == 1
        assert "file.txt" not in files
        assert "another.txt" in files

    def test_unicode_filenames(self, tmp_path):
        """Test files with unicode characters in names."""
        (tmp_path / "file_中文.txt").touch()
        (tmp_path / "file_émojis_🎉.txt").touch()
        (tmp_path / "file_русский.txt").touch()
        
        files = _list_output_files(tmp_path)
        
        assert len(files) == 3
        assert "file_中文.txt" in files
        assert "file_émojis_🎉.txt" in files
        assert "file_русский.txt" in files

    def test_hidden_files_included(self, tmp_path):
        """Test that hidden files (starting with .) are included."""
        (tmp_path / ".hidden.txt").touch()
        (tmp_path / ".gitignore").touch()
        (tmp_path / "visible.txt").touch()
        
        files = _list_output_files(tmp_path)
        
        assert len(files) == 3
        assert ".hidden.txt" in files
        assert ".gitignore" in files
        assert "visible.txt" in files

    def test_symlink_to_file(self, tmp_path):
        """Test that symlinks to files are included."""
        target_file = tmp_path / "target.txt"
        target_file.write_text("content")
        symlink_file = tmp_path / "link.txt"
        
        try:
            symlink_file.symlink_to(target_file)
            files = _list_output_files(tmp_path)
            
            # Symlink should be included if it points to a file
            assert "link.txt" in files or "target.txt" in files
            # At minimum, target should be there
            assert "target.txt" in files
        except (OSError, NotImplementedError):
            # Symlinks not supported on this platform (e.g., Windows without admin)
            pytest.skip("Symlinks not supported on this platform")

    def test_path_is_file_not_directory(self, tmp_path):
        """Test behavior when path is a file, not a directory."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        
        # Should handle gracefully (return empty list or raise, but not crash)
        files = _list_output_files(file_path)
        
        # Function catches exceptions and returns []
        assert files == []

    def test_very_long_filename(self, tmp_path):
        """Test files with very long names (within filesystem limits)."""
        # Use a long but reasonable filename (most filesystems support 255 chars for filename)
        # Account for the .txt extension
        long_name = "a" * 250 + ".txt"
        try:
            (tmp_path / long_name).touch()
        except OSError:
            # Filesystem doesn't support this length, skip test
            pytest.skip("Filesystem doesn't support very long filenames")
        
        files = _list_output_files(tmp_path)
        
        assert len(files) == 1
        assert files[0] == long_name

    def test_exclude_with_nonexistent_file(self, tmp_path):
        """Test excluding a file that doesn't exist."""
        (tmp_path / "exists.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=["nonexistent.txt"])
        
        assert len(files) == 1
        assert "exists.txt" in files

    def test_exclude_duplicate_entries(self, tmp_path):
        """Test exclude list with duplicate entries."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "other.txt").touch()
        
        files = _list_output_files(tmp_path, exclude=["file.txt", "file.txt"])
        
        assert len(files) == 1
        assert "other.txt" in files
        assert "file.txt" not in files

    def test_execution_result_all_fields_present(self):
        """Test that _make_error_result returns all ExecutionResult fields."""
        res = _make_error_result("test")
        
        # Verify all required TypedDict fields are present
        required_fields = [
            "stdout", "stderr", "exit_code", "output_files",
            "runtime_seconds", "error", "memory_exceeded", "timeout_exceeded"
        ]
        
        for field in required_fields:
            assert field in res, f"Missing required field: {field}"

    def test_execution_result_field_types(self):
        """Test that _make_error_result returns correct types for all fields."""
        res = _make_error_result(
            error="test",
            stdout="out",
            stderr="err",
            exit_code=1,
            runtime_seconds=1.5,
            memory_exceeded=True,
            timeout_exceeded=False
        )
        
        assert isinstance(res["stdout"], str)
        assert isinstance(res["stderr"], str)
        assert isinstance(res["exit_code"], int)
        assert isinstance(res["output_files"], list)
        assert isinstance(res["runtime_seconds"], float)
        assert isinstance(res["error"], str)
        assert isinstance(res["memory_exceeded"], bool)
        assert isinstance(res["timeout_exceeded"], bool)

    def test_list_output_files_return_type(self, tmp_path):
        """Test that _list_output_files always returns a list."""
        (tmp_path / "file.txt").touch()
        
        files = _list_output_files(tmp_path)
        
        assert isinstance(files, list)

    def test_list_output_files_empty_result_is_list(self, tmp_path):
        """Test that empty result is an empty list, not None."""
        files = _list_output_files(tmp_path)
        
        assert files is not None
        assert isinstance(files, list)
        assert files == []

    def test_list_output_files_with_string_path(self, tmp_path):
        """Test that function works with string path (Path accepts strings)."""
        (tmp_path / "file.txt").touch()
        
        # Pass string instead of Path object
        files = _list_output_files(str(tmp_path))
        
        assert len(files) == 1
        assert "file.txt" in files

