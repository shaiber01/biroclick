"""Integration tests that exercise file validation logic via real nodes."""

from pathlib import Path
from unittest.mock import patch


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
        assert "exist on disk" in error_msg or "missing" in error_msg, (
            f"Error message should mention missing files, got: {error_msg}"
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

        assert result is not None
        assert result.get("workflow_phase") == "analysis"
        assert "analysis_summary" in result
        assert result["analysis_summary"]["totals"]["targets"] > 0

    def test_results_analyzer_empty_stage_outputs(self, analysis_state):
        """results_analyzer should handle empty stage_outputs by failing."""
        from src.agents.analysis import results_analyzer_node

        analysis_state["stage_outputs"] = {}
        result = results_analyzer_node(analysis_state)

        assert result.get("execution_verdict") == "fail", (
            "Should fail if stage_outputs is empty"
        )
        assert result.get("run_error") is not None


class TestMaterialValidation:
    """Test material validation with real file paths."""

    MATERIALS_DIR = Path(__file__).resolve().parents[3] / "materials"

    def test_material_file_resolution(self):
        """Material files should resolve correctly from the materials directory."""
        expected_materials = [
            "palik_gold.csv",
            "palik_silver.csv",
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

        assert existing, (
            f"No material files found in {self.MATERIALS_DIR}. "
            f"Expected at least one of: {expected_materials}"
        )


