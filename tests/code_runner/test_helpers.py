"""Tests for helper utilities in `src.code_runner`."""

from src.code_runner import _list_output_files, _make_error_result


class TestHelpers:
    """Tests for helper functions."""

    def test_make_error_result(self):
        """Test error result construction."""
        res = _make_error_result("oops", exit_code=5)
        assert res["error"] == "oops"
        assert res["exit_code"] == 5
        assert res["output_files"] == []

    def test_list_output_files(self, tmp_path):
        """Test listing files."""
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "ignore.py").touch()

        files = _list_output_files(tmp_path, exclude=["ignore.py"])
        assert "a.txt" in files
        assert "b.txt" in files
        assert "ignore.py" not in files

