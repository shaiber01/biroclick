"""Integration tests that exercise file validation logic via real nodes."""

from pathlib import Path
from unittest.mock import patch
import pytest


class TestFileValidation:
    """
    Test file handling in analysis nodes.

    These tests exercise REAL file validation code, not mocked.
    They use temporary files to test actual file handling logic.
    """

    def test_results_analyzer_handles_missing_files(self, analysis_state):
        """results_analyzer should detect missing files and FAIL execution."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = {
            "files": ["/nonexistent/path/spectrum.csv"],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Analysis should mark execution as failed when output files are missing"
        )
        assert result.get("run_error") is not None, "Should provide a run_error explanation"
        error_msg = result["run_error"].lower()
        assert "exist on disk" in error_msg or "missing" in error_msg or "do not exist" in error_msg, (
            f"Error message should mention missing files, got: {error_msg}"
        )
        # Verify specific error message content
        assert "/nonexistent/path/spectrum.csv" in result["run_error"] or "spectrum.csv" in result["run_error"], (
            f"Error message should mention the missing file path, got: {result['run_error']}"
        )
        # Verify analysis_summary is set correctly
        assert result.get("analysis_summary") == "Analysis skipped: Output files missing" or isinstance(result.get("analysis_summary"), dict), (
            "Should set analysis_summary when files are missing"
        )
        assert result.get("analysis_overall_classification") == "FAILED", (
            f"Should set overall classification to FAILED, got: {result.get('analysis_overall_classification')}"
        )
        assert result.get("figure_comparisons") == [], (
            "Should have empty figure_comparisons when files are missing"
        )
        assert result.get("analysis_result_reports") == [], (
            "Should have empty analysis_result_reports when files are missing"
        )
        mock_llm.assert_not_called()

    def test_results_analyzer_with_real_csv_file(self, analysis_state, tmp_path):
        """results_analyzer should successfully process real CSV files."""
        from src.agents.analysis import results_analyzer_node

        csv_file = tmp_path / "extinction_spectrum.csv"
        csv_file.write_text(
            "wavelength_nm,extinction\n"
            "400,0.1\n"
            "500,0.3\n"
            "600,0.8\n"
            "700,1.0\n"
            "800,0.5\n",
            encoding="utf-8",
        )
        analysis_state["stage_outputs"] = {
            "files": [str(csv_file)],
            "stdout": "Simulation completed",
            "stderr": "",
            "runtime_seconds": 10,
        }
        mock_response = {
            "overall_classification": "ACCEPTABLE_MATCH",
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "classification": "partial_match",
                    "shape_comparison": ["Peak shape matches"],
                    "reason_for_difference": "Minor numerical differences",
                }
            ],
            "summary": "Results analyzed successfully",
        }

        with patch(
            "src.agents.analysis.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = results_analyzer_node(analysis_state)

        assert result is not None, "Result should not be None"
        assert result.get("workflow_phase") == "analysis", (
            f"Should set workflow_phase to 'analysis', got: {result.get('workflow_phase')}"
        )
        assert "analysis_summary" in result, "Should include analysis_summary"
        assert isinstance(result["analysis_summary"], dict), (
            f"analysis_summary should be a dict, got: {type(result['analysis_summary'])}"
        )
        assert result["analysis_summary"]["totals"]["targets"] > 0, (
            f"Should have at least one target, got: {result['analysis_summary']['totals']['targets']}"
        )
        assert "analysis_overall_classification" in result, (
            "Should include analysis_overall_classification"
        )
        assert "analysis_result_reports" in result, (
            "Should include analysis_result_reports"
        )
        assert isinstance(result["analysis_result_reports"], list), (
            f"analysis_result_reports should be a list, got: {type(result['analysis_result_reports'])}"
        )
        assert "figure_comparisons" in result, (
            "Should include figure_comparisons"
        )
        assert isinstance(result["figure_comparisons"], list), (
            f"figure_comparisons should be a list, got: {type(result['figure_comparisons'])}"
        )

    def test_results_analyzer_empty_stage_outputs(self, analysis_state):
        """results_analyzer should handle empty stage_outputs by failing."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = {}
        result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail if stage_outputs is empty"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation when stage_outputs is empty"
        )
        assert isinstance(result["run_error"], str), (
            f"run_error should be a string, got: {type(result['run_error'])}"
        )
        assert len(result["run_error"]) > 0, (
            "run_error should not be empty"
        )
        assert "stage outputs" in result["run_error"].lower() or "missing" in result["run_error"].lower(), (
            f"Error message should mention stage outputs or missing, got: {result['run_error']}"
        )
        assert result.get("workflow_phase") == "analysis", (
            f"Should set workflow_phase to 'analysis', got: {result.get('workflow_phase')}"
        )
        assert result.get("analysis_summary") == "Analysis skipped: No outputs available" or isinstance(result.get("analysis_summary"), dict), (
            "Should set analysis_summary appropriately"
        )
        assert result.get("analysis_overall_classification") == "FAILED", (
            f"Should set overall classification to FAILED, got: {result.get('analysis_overall_classification')}"
        )

    def test_results_analyzer_stage_outputs_none(self, analysis_state):
        """results_analyzer should handle None stage_outputs by failing."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = None
        result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail if stage_outputs is None"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation when stage_outputs is None"
        )

    def test_results_analyzer_stage_outputs_missing_key(self, analysis_state):
        """results_analyzer should handle missing stage_outputs key by failing."""
        from src.agents.analysis import results_analyzer_node

        if "stage_outputs" in analysis_state:
            del analysis_state["stage_outputs"]
        result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail if stage_outputs key is missing"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation when stage_outputs key is missing"
        )

    def test_results_analyzer_stage_outputs_empty_files_list(self, analysis_state):
        """results_analyzer should handle empty files list by failing."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = {
            "files": [],
            "stdout": "Simulation completed",
            "stderr": "",
        }
        result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail if files list is empty"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation when files list is empty"
        )
        assert "missing" in result["run_error"].lower() or "empty" in result["run_error"].lower(), (
            f"Error message should mention missing or empty, got: {result['run_error']}"
        )

    def test_results_analyzer_stage_outputs_files_none(self, analysis_state):
        """results_analyzer should handle None files list by failing."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = {
            "files": None,
            "stdout": "Simulation completed",
            "stderr": "",
        }
        result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail if files is None"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation when files is None"
        )

    def test_results_analyzer_partial_files_missing(self, analysis_state, tmp_path):
        """results_analyzer should handle partial file missing scenarios."""
        from src.agents.analysis import results_analyzer_node

        # Create one real file
        real_file = tmp_path / "real_spectrum.csv"
        real_file.write_text("wavelength,value\n400,0.5\n", encoding="utf-8")

        # Include one missing file
        analysis_state["stage_outputs"] = {
            "files": [str(real_file), "/nonexistent/path/missing.csv"],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(analysis_state)

        # Should proceed with available files but log warning
        assert result.get("workflow_phase") == "analysis", (
            "Should proceed to analysis phase when some files exist"
        )
        # Should not fail completely if at least one file exists
        assert result.get("execution_verdict") != "fail", (
            "Should not fail completely when at least one file exists"
        )
        assert "analysis_summary" in result, (
            "Should include analysis_summary when some files exist"
        )

    def test_results_analyzer_all_files_missing(self, analysis_state):
        """results_analyzer should fail when all files in list are missing."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = {
            "files": [
                "/nonexistent/path/file1.csv",
                "/another/missing/file2.csv",
                "/third/missing/file3.csv",
            ],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail when all files are missing"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation"
        )
        # Error message should mention missing files
        error_msg = result["run_error"].lower()
        assert "exist on disk" in error_msg or "missing" in error_msg or "do not exist" in error_msg, (
            f"Error message should mention missing files, got: {error_msg}"
        )
        # Should list the missing files
        assert "file1.csv" in result["run_error"] or "file2.csv" in result["run_error"] or "file3.csv" in result["run_error"], (
            f"Error message should mention at least one missing file, got: {result['run_error']}"
        )
        mock_llm.assert_not_called()

    def test_results_analyzer_relative_path_resolution(self, analysis_state, tmp_path):
        """results_analyzer should resolve relative paths correctly."""
        from src.agents.analysis import results_analyzer_node

        # Create output directory structure matching expected layout
        paper_id = analysis_state.get("paper_id", "test_file_validation")
        stage_id = analysis_state.get("current_stage_id", "stage_0")
        base_output_dir = tmp_path / "outputs" / paper_id / stage_id
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create file in the expected location
        csv_file = base_output_dir / "spectrum.csv"
        csv_file.write_text("wavelength,value\n400,0.5\n", encoding="utf-8")

        # Use relative path
        analysis_state["stage_outputs"] = {
            "files": ["spectrum.csv"],  # Relative path
            "stdout": "Simulation completed",
            "stderr": "",
        }

        # Mock PROJECT_ROOT to point to tmp_path
        with patch("src.agents.analysis.PROJECT_ROOT", tmp_path):
            with patch("src.agents.analysis.call_agent_with_metrics"):
                result = results_analyzer_node(analysis_state)

        # Should successfully resolve relative path
        assert result.get("workflow_phase") == "analysis", (
            "Should proceed to analysis phase when relative path resolves"
        )
        assert result.get("execution_verdict") != "fail", (
            "Should not fail when relative path resolves correctly"
        )

    def test_results_analyzer_absolute_path(self, analysis_state, tmp_path):
        """results_analyzer should handle absolute paths correctly."""
        from src.agents.analysis import results_analyzer_node

        csv_file = tmp_path / "absolute_spectrum.csv"
        csv_file.write_text("wavelength,value\n400,0.5\n", encoding="utf-8")

        analysis_state["stage_outputs"] = {
            "files": [str(csv_file.resolve())],  # Absolute path
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(analysis_state)

        assert result.get("workflow_phase") == "analysis", (
            "Should proceed to analysis phase with absolute path"
        )
        assert result.get("execution_verdict") != "fail", (
            "Should not fail with valid absolute path"
        )

    def test_results_analyzer_path_points_to_directory(self, analysis_state, tmp_path):
        """results_analyzer should handle paths that point to directories, not files."""
        from src.agents.analysis import results_analyzer_node

        # Create a directory instead of a file
        dir_path = tmp_path / "output_dir"
        dir_path.mkdir()

        analysis_state["stage_outputs"] = {
            "files": [str(dir_path)],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            result = results_analyzer_node(analysis_state)

        # Should fail because path points to directory, not file
        assert result.get("execution_verdict") == "fail", (
            "Should fail when path points to directory instead of file"
        )
        assert result.get("run_error") is not None, (
            "Should provide a run_error explanation"
        )
        mock_llm.assert_not_called()

    def test_results_analyzer_invalid_path_characters(self, analysis_state):
        """results_analyzer should handle invalid path characters gracefully."""
        from src.agents.analysis import results_analyzer_node

        # Use path with invalid characters (null byte)
        invalid_path = "/path/with/null\x00byte.csv"

        analysis_state["stage_outputs"] = {
            "files": [invalid_path],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            result = results_analyzer_node(analysis_state)

        # Should handle gracefully (either fail or skip the invalid path)
        # The code catches ValueError/OSError, so it should not crash
        assert result is not None, (
            "Should not crash on invalid path characters"
        )
        # Should either fail or proceed (depending on implementation)
        assert result.get("execution_verdict") in ("fail", None) or result.get("workflow_phase") == "analysis", (
            f"Should handle invalid path gracefully, got execution_verdict={result.get('execution_verdict')}"
        )

    def test_results_analyzer_very_long_path(self, analysis_state):
        """results_analyzer should handle very long paths gracefully."""
        from src.agents.analysis import results_analyzer_node

        # Create a very long path (may exceed filesystem limits)
        very_long_path = "/" + "a" * 1000 + "/file.csv"

        analysis_state["stage_outputs"] = {
            "files": [very_long_path],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            result = results_analyzer_node(analysis_state)

        # Should handle gracefully without crashing
        assert result is not None, (
            "Should not crash on very long path"
        )
        # Should either fail or handle gracefully
        assert result.get("execution_verdict") in ("fail", None) or result.get("workflow_phase") == "analysis", (
            f"Should handle very long path gracefully"
        )

    def test_results_analyzer_multiple_files_all_exist(self, analysis_state, tmp_path):
        """results_analyzer should handle multiple files when all exist."""
        from src.agents.analysis import results_analyzer_node

        file1 = tmp_path / "spectrum1.csv"
        file1.write_text("wavelength,value\n400,0.5\n", encoding="utf-8")
        file2 = tmp_path / "spectrum2.csv"
        file2.write_text("wavelength,value\n500,0.6\n", encoding="utf-8")
        file3 = tmp_path / "spectrum3.csv"
        file3.write_text("wavelength,value\n600,0.7\n", encoding="utf-8")

        analysis_state["stage_outputs"] = {
            "files": [str(file1), str(file2), str(file3)],
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(analysis_state)

        assert result.get("workflow_phase") == "analysis", (
            "Should proceed to analysis phase with multiple files"
        )
        assert result.get("execution_verdict") != "fail", (
            "Should not fail when all files exist"
        )
        assert "analysis_summary" in result, (
            "Should include analysis_summary"
        )

    def test_results_analyzer_file_path_as_path_object(self, analysis_state, tmp_path):
        """results_analyzer should handle Path objects in files list."""
        from src.agents.analysis import results_analyzer_node

        csv_file = tmp_path / "path_object.csv"
        csv_file.write_text("wavelength,value\n400,0.5\n", encoding="utf-8")

        # Pass Path object directly instead of string
        analysis_state["stage_outputs"] = {
            "files": [csv_file],  # Path object, not string
            "stdout": "Simulation completed",
            "stderr": "",
        }

        with patch("src.agents.analysis.call_agent_with_metrics"):
            result = results_analyzer_node(analysis_state)

        assert result.get("workflow_phase") == "analysis", (
            "Should handle Path objects in files list"
        )
        assert result.get("execution_verdict") != "fail", (
            "Should not fail with Path object"
        )


    def test_results_analyzer_missing_current_stage_id(self, analysis_state):
        """results_analyzer should handle missing current_stage_id gracefully."""
        from src.agents.analysis import results_analyzer_node

        if "current_stage_id" in analysis_state:
            del analysis_state["current_stage_id"]

        result = results_analyzer_node(analysis_state)

        # Should handle missing stage_id gracefully
        assert result is not None, (
            "Should not crash when current_stage_id is missing"
        )
        assert result.get("workflow_phase") == "analysis", (
            "Should set workflow_phase to analysis"
        )
        # Should either ask user or fail gracefully
        assert result.get("ask_user_trigger") is not None or result.get("execution_verdict") == "fail", (
            "Should either ask user or fail when stage_id is missing"
        )

    def test_results_analyzer_none_current_stage_id(self, analysis_state):
        """results_analyzer should handle None current_stage_id gracefully."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["current_stage_id"] = None

        result = results_analyzer_node(analysis_state)

        assert result is not None, (
            "Should not crash when current_stage_id is None"
        )
        assert result.get("workflow_phase") == "analysis", (
            "Should set workflow_phase to analysis"
        )


class TestFileMatching:
    """Test file matching logic used by results_analyzer."""

    def test_match_output_file_with_target_id_in_filename(self):
        """match_output_file should match files containing target_id in filename."""
        from src.agents.helpers.validation import match_output_file

        files = ["spectrum_fig1.csv", "other_data.csv", "fig1_results.csv"]
        matched = match_output_file(files, "Fig1")

        assert matched is not None, (
            "Should match file containing target_id"
        )
        assert "fig1" in matched.lower(), (
            f"Should match file with target_id, got: {matched}"
        )

    def test_match_output_file_without_target_id_falls_back(self):
        """match_output_file should fall back to first file if no target_id match."""
        from src.agents.helpers.validation import match_output_file

        files = ["spectrum.csv", "other_data.csv"]
        matched = match_output_file(files, "Fig99")

        assert matched is not None, (
            "Should fall back to first file when no target_id match"
        )
        assert matched == "spectrum.csv", (
            f"Should return first file, got: {matched}"
        )

    def test_match_output_file_empty_list(self):
        """match_output_file should return None for empty file list."""
        from src.agents.helpers.validation import match_output_file

        matched = match_output_file([], "Fig1")

        assert matched is None, (
            "Should return None for empty file list"
        )

    def test_match_output_file_path_objects(self):
        """match_output_file should handle Path objects in file list."""
        from src.agents.helpers.validation import match_output_file

        files = [Path("/path/to/fig1_spectrum.csv"), Path("/path/to/other.csv")]
        matched = match_output_file(files, "Fig1")

        assert matched is not None, (
            "Should handle Path objects"
        )

    def test_match_output_file_dict_entries(self):
        """match_output_file should handle dict entries with 'path' or 'file' keys."""
        from src.agents.helpers.validation import match_output_file

        files = [
            {"path": "/path/to/fig1_spectrum.csv"},
            {"file": "/path/to/other.csv"},
        ]
        matched = match_output_file(files, "Fig1")

        assert matched is not None, (
            "Should handle dict entries with path/file keys"
        )
        assert "fig1" in matched.lower(), (
            f"Should match dict entry, got: {matched}"
        )

    def test_match_expected_files_exact_match(self):
        """match_expected_files should match exact filename."""
        from src.agents.helpers.validation import match_expected_files

        expected = ["spectrum.csv", "results.csv"]
        actual = ["/path/to/spectrum.csv", "/other/path/results.csv"]

        matched = match_expected_files(expected, actual)

        assert matched is not None, (
            "Should match exact filename"
        )
        assert "spectrum.csv" in matched, (
            f"Should match expected filename, got: {matched}"
        )

    def test_match_expected_files_substring_match(self):
        """match_expected_files should fall back to substring match."""
        from src.agents.helpers.validation import match_expected_files

        expected = ["spectrum"]
        actual = ["/path/to/spectrum_data.csv"]

        matched = match_expected_files(expected, actual)

        assert matched is not None, (
            "Should match substring when exact match fails"
        )

    def test_match_expected_files_no_match(self):
        """match_expected_files should return None when no match found."""
        from src.agents.helpers.validation import match_expected_files

        expected = ["spectrum.csv"]
        actual = ["/path/to/completely_different.csv"]

        matched = match_expected_files(expected, actual)

        assert matched is None, (
            "Should return None when no match found"
        )

    def test_match_expected_files_empty_expected(self):
        """match_expected_files should return None for empty expected list."""
        from src.agents.helpers.validation import match_expected_files

        matched = match_expected_files([], ["/path/to/file.csv"])

        assert matched is None, (
            "Should return None for empty expected list"
        )

    def test_collect_expected_outputs_with_patterns(self):
        """collect_expected_outputs should resolve filename patterns correctly."""
        from src.agents.helpers.validation import collect_expected_outputs

        stage_info = {
            "expected_outputs": [
                {
                    "target_figure": "Fig1",
                    "filename_pattern": "{paper_id}_{stage_id}_{target_id}_spectrum.csv",
                }
            ]
        }

        mapping = collect_expected_outputs(stage_info, "test_paper", "stage_0")

        assert "Fig1" in mapping, (
            "Should include target_figure in mapping"
        )
        assert len(mapping["Fig1"]) > 0, (
            "Should resolve filename pattern"
        )
        assert "test_paper" in mapping["Fig1"][0], (
            "Should replace {paper_id} in pattern"
        )
        assert "stage_0" in mapping["Fig1"][0], (
            "Should replace {stage_id} in pattern"
        )
        assert "fig1" in mapping["Fig1"][0].lower(), (
            "Should replace {target_id} in pattern"
        )

    def test_collect_expected_outputs_empty_stage_info(self):
        """collect_expected_outputs should return empty dict for None stage_info."""
        from src.agents.helpers.validation import collect_expected_outputs

        mapping = collect_expected_outputs(None, "test_paper", "stage_0")

        assert isinstance(mapping, dict), (
            "Should return a dict"
        )
        assert len(mapping) == 0, (
            "Should return empty dict for None stage_info"
        )

    def test_collect_expected_outputs_missing_pattern(self):
        """collect_expected_outputs should skip entries without filename_pattern."""
        from src.agents.helpers.validation import collect_expected_outputs

        stage_info = {
            "expected_outputs": [
                {
                    "target_figure": "Fig1",
                    # Missing filename_pattern
                }
            ]
        }

        mapping = collect_expected_outputs(stage_info, "test_paper", "stage_0")

        assert "Fig1" not in mapping or len(mapping["Fig1"]) == 0, (
            "Should skip entries without filename_pattern"
        )


class TestMaterialValidation:
    """Test material validation with real file paths."""

    MATERIALS_DIR = Path(__file__).resolve().parents[3] / "materials"

    def test_material_file_resolution(self):
        """Material files should resolve correctly from the materials directory."""
        expected_materials = [
            "palik_gold.csv",
            "palik_silver.csv",
            "palik_silicon.csv",
            "johnson_christy_chromium.csv",
            "malitson_sio2.csv",
        ]
        existing = []
        for material in expected_materials:
            material_file = self.MATERIALS_DIR / material
            if material_file.exists():
                existing.append(material)
                content = material_file.read_text(encoding="utf-8")
                assert "," in content, f"{material} doesn't look like a CSV"
                lines = content.strip().split("\n")
                assert len(lines) > 1, f"{material} has no data rows"
                # Skip comment lines (starting with #) to find actual CSV header
                header_line = None
                for line in lines:
                    if line.strip() and not line.strip().startswith("#"):
                        header_line = line
                        break
                assert header_line is not None, (
                    f"{material} should have a non-comment header line"
                )
                # Verify header has multiple columns
                header_cols = header_line.split(",")
                assert len(header_cols) > 1, (
                    f"{material} header doesn't look like CSV with multiple columns: {header_line}"
                )
                # Verify it has data rows (non-comment lines after header)
                data_lines = [
                    line for line in lines
                    if line.strip() and not line.strip().startswith("#")
                ]
                assert len(data_lines) >= 2, (
                    f"{material} should have at least header + 1 data row, found {len(data_lines)} non-comment lines"
                )

        assert len(existing) > 0, (
            f"No material files found in {self.MATERIALS_DIR}. "
            f"Expected at least one of: {expected_materials}"
        )
        assert len(existing) >= 2, (
            f"Expected at least 2 material files, found: {existing}"
        )

    def test_material_file_content_structure(self):
        """Material files should have valid CSV structure."""
        material_file = self.MATERIALS_DIR / "palik_gold.csv"
        if not material_file.exists():
            pytest.skip("palik_gold.csv not found")

        content = material_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        assert len(lines) > 1, (
            "Material file should have multiple lines"
        )
        # Skip comment lines (starting with #) to find actual CSV header
        non_comment_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(non_comment_lines) > 0, (
            "Material file should have at least one non-comment line"
        )
        # Check header (first non-comment line)
        header = non_comment_lines[0]
        assert "," in header, (
            f"Header should contain comma separator, got: {header}"
        )
        header_cols = header.split(",")
        assert len(header_cols) >= 2, (
            f"Header should have at least 2 columns, got: {len(header_cols)}"
        )
        # Check data rows (non-comment lines after header)
        data_rows = non_comment_lines[1:]
        assert len(data_rows) > 0, (
            "Material file should have at least one data row"
        )
        for i, line in enumerate(data_rows, start=2):
            cols = line.split(",")
            assert len(cols) == len(header_cols), (
                f"Row {i} should have {len(header_cols)} columns matching header, got {len(cols)}. "
                f"Row content: {line[:100]}"
            )
            # Verify numeric values in data columns (skip first column which might be wavelength)
            for col_idx, col_val in enumerate(cols[1:], start=1):
                col_val = col_val.strip()
                if col_val:  # Allow empty values
                    try:
                        float(col_val)
                    except ValueError:
                        pytest.fail(
                            f"Row {i}, column {col_idx} should be numeric, got: {col_val}"
                        )

    def test_material_directory_exists(self):
        """Materials directory should exist and be accessible."""
        assert self.MATERIALS_DIR.exists(), (
            f"Materials directory should exist: {self.MATERIALS_DIR}"
        )
        assert self.MATERIALS_DIR.is_dir(), (
            f"Materials path should be a directory: {self.MATERIALS_DIR}"
        )

    def test_material_files_readable(self):
        """Material files should be readable."""
        material_files = list(self.MATERIALS_DIR.glob("*.csv"))
        assert len(material_files) > 0, (
            f"Should find at least one CSV file in {self.MATERIALS_DIR}"
        )

        for material_file in material_files[:5]:  # Test first 5 files
            assert material_file.is_file(), (
                f"Material file should be a file: {material_file}"
            )
            assert material_file.stat().st_size > 0, (
                f"Material file should not be empty: {material_file}"
            )
            # Should be readable
            try:
                content = material_file.read_text(encoding="utf-8")
                assert len(content) > 0, (
                    f"Material file content should not be empty: {material_file}"
                )
            except Exception as e:
                pytest.fail(
                    f"Material file should be readable: {material_file}, error: {e}"
                )


