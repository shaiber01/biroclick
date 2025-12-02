"""Unit tests for src/agents/helpers/stubs.py"""

import pytest

from src.agents.helpers.stubs import (
    ensure_stub_figures,
    build_stub_targets,
    build_stub_expected_outputs,
    build_stub_stages,
    build_stub_planned_materials,
    build_stub_assumptions,
    build_stub_plan,
)


class TestEnsureStubFigures:
    """Tests for ensure_stub_figures function."""

    def test_returns_existing_figures(self):
        """Should return existing paper figures."""
        state = {
            "paper_figures": [
                {"id": "Fig1", "description": "Test figure"}
            ]
        }
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "Fig1"

    def test_returns_stub_for_empty_figures(self):
        """Should return stub figure when none exist."""
        state = {"paper_figures": []}
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "FigStub"

    def test_returns_stub_for_none_figures(self):
        """Should return stub figure when figures is None."""
        state = {}
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "FigStub"


class TestBuildStubTargets:
    """Tests for build_stub_targets function."""

    def test_builds_targets_from_figures(self):
        """Should build targets from figure list."""
        figures = [
            {"id": "Fig1", "description": "First figure"},
            {"id": "Fig2", "description": "Second figure"},
        ]
        
        result = build_stub_targets(figures)
        
        assert len(result) == 2
        assert result[0]["figure_id"] == "Fig1"
        assert result[0]["type"] == "spectrum"
        assert result[0]["precision_requirement"] == "acceptable"

    def test_builds_stub_for_empty_figures(self):
        """Should build stub target for empty figures."""
        result = build_stub_targets([])
        
        assert len(result) == 1
        assert result[0]["figure_id"] == "FigStub"

    def test_includes_digitized_data_path(self):
        """Should include digitized_data_path from figures."""
        figures = [
            {"id": "Fig1", "digitized_data_path": "/path/to/data.csv"}
        ]
        
        result = build_stub_targets(figures)
        
        assert result[0]["digitized_data_path"] == "/path/to/data.csv"


class TestBuildStubExpectedOutputs:
    """Tests for build_stub_expected_outputs function."""

    def test_builds_expected_outputs(self):
        """Should build expected output specs."""
        result = build_stub_expected_outputs(
            "paper1", "stage1", ["Fig1", "Fig2"], ["wavelength", "transmission"]
        )
        
        assert len(result) == 2
        assert result[0]["artifact_type"] == "spectrum_csv"
        assert "paper1" in result[0]["filename_pattern"]
        assert "stage1" in result[0]["filename_pattern"]
        assert result[0]["target_figure"] == "Fig1"
        assert result[0]["columns"] == ["wavelength", "transmission"]


class TestBuildStubStages:
    """Tests for build_stub_stages function."""

    def test_builds_stage0_and_stage1(self):
        """Should build Stage 0 and Stage 1."""
        targets = [
            {"figure_id": "Fig1"},
            {"figure_id": "Fig2"},
        ]
        
        result = build_stub_stages("paper1", targets)
        
        assert len(result) == 2
        assert result[0]["stage_id"] == "stage0_material_validation"
        assert result[0]["stage_type"] == "MATERIAL_VALIDATION"
        assert result[1]["stage_id"] == "stage1_primary_structure"
        assert result[1]["stage_type"] == "SINGLE_STRUCTURE"

    def test_stage1_depends_on_stage0(self):
        """Stage 1 should depend on Stage 0."""
        targets = [{"figure_id": "Fig1"}]
        
        result = build_stub_stages("paper1", targets)
        
        assert "stage0_material_validation" in result[1]["dependencies"]

    def test_handles_empty_targets(self):
        """Should handle empty targets list."""
        result = build_stub_stages("paper1", [])
        
        assert len(result) == 2
        assert result[0]["targets"] == ["FigStub"]


class TestBuildStubPlannedMaterials:
    """Tests for build_stub_planned_materials function."""

    def test_builds_placeholder_material(self):
        """Should build placeholder material entry."""
        state = {"paper_domain": "plasmonics"}
        
        result = build_stub_planned_materials(state)
        
        assert len(result) == 1
        assert "plasmonics" in result[0]["material_id"]
        assert result[0]["source"] == "stub"

    def test_uses_generic_for_missing_domain(self):
        """Should use generic domain if not specified."""
        state = {}
        
        result = build_stub_planned_materials(state)
        
        assert "generic" in result[0]["material_id"]


class TestBuildStubAssumptions:
    """Tests for build_stub_assumptions function."""

    def test_builds_empty_assumptions(self):
        """Should build empty assumptions structure."""
        result = build_stub_assumptions()
        
        assert "global_assumptions" in result
        assert "stage_specific" in result
        assert result["global_assumptions"]["materials"] == []
        assert result["global_assumptions"]["geometry"] == []
        assert result["global_assumptions"]["sources"] == []


class TestBuildStubPlan:
    """Tests for build_stub_plan function."""

    def test_builds_complete_plan(self):
        """Should build a complete stub plan."""
        state = {
            "paper_id": "test_paper",
            "paper_title": "Test Paper Title",
            "paper_figures": [
                {"id": "Fig1", "description": "First figure"},
            ],
            "paper_domain": "plasmonics",
        }
        
        result = build_stub_plan(state)
        
        assert result["paper_id"] == "test_paper"
        assert result["title"] == "Test Paper Title"
        assert len(result["stages"]) == 2
        assert len(result["targets"]) >= 1
        assert "reproduction_scope" in result
        assert "extracted_parameters" in result

    def test_uses_defaults_for_missing_fields(self):
        """Should use defaults when state fields are missing."""
        state = {}
        
        result = build_stub_plan(state)
        
        assert result["paper_id"] == "paper_stub"
        assert "stages" in result
        assert "targets" in result

    def test_calculates_coverage(self):
        """Should calculate reproduction coverage."""
        state = {
            "paper_id": "test",
            "paper_figures": [
                {"id": "Fig1"},
                {"id": "Fig2"},
            ],
        }
        
        result = build_stub_plan(state)
        
        scope = result["reproduction_scope"]
        assert scope["total_figures"] == 2
        assert scope["reproducible_figures"] == 2
        assert scope["coverage_percent"] == 100.0


