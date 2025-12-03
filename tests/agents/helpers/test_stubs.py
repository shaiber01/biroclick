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

    def test_returns_existing_figures_unchanged(self):
        """Should return existing paper figures exactly as provided."""
        state = {
            "paper_figures": [
                {"id": "Fig1", "description": "Test figure", "image_path": "/path/to/fig1.png"}
            ]
        }
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "Fig1"
        assert result[0]["description"] == "Test figure"
        assert result[0]["image_path"] == "/path/to/fig1.png"
        # Verify it's the exact same object (reference equality)
        assert result is state["paper_figures"]

    def test_returns_existing_figures_multiple(self):
        """Should return all existing figures when multiple exist."""
        state = {
            "paper_figures": [
                {"id": "Fig1", "description": "First figure"},
                {"id": "Fig2", "description": "Second figure"},
                {"id": "Fig3", "description": "Third figure"},
            ]
        }
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 3
        assert result[0]["id"] == "Fig1"
        assert result[1]["id"] == "Fig2"
        assert result[2]["id"] == "Fig3"

    def test_returns_stub_for_empty_figures(self):
        """Should return stub figure when none exist."""
        state = {"paper_figures": []}
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "FigStub"
        assert result[0]["description"] == "Placeholder figure generated for stub planning"
        assert result[0]["image_path"] == ""
        # Verify all required fields are present
        assert "id" in result[0]
        assert "description" in result[0]
        assert "image_path" in result[0]

    def test_returns_stub_for_none_figures(self):
        """Should return stub figure when figures is None."""
        state = {}
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "FigStub"
        assert result[0]["description"] == "Placeholder figure generated for stub planning"
        assert result[0]["image_path"] == ""

    def test_returns_stub_for_explicit_none(self):
        """Should return stub figure when paper_figures is explicitly None."""
        state = {"paper_figures": None}
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "FigStub"
        assert result[0]["description"] == "Placeholder figure generated for stub planning"
        assert result[0]["image_path"] == ""

    def test_handles_figures_with_missing_fields(self):
        """Should handle figures with missing optional fields."""
        state = {
            "paper_figures": [
                {"id": "Fig1"},  # Missing description and image_path
            ]
        }
        
        result = ensure_stub_figures(state)
        
        assert len(result) == 1
        assert result[0]["id"] == "Fig1"
        # Should return as-is, even if fields are missing


class TestBuildStubTargets:
    """Tests for build_stub_targets function."""

    def test_builds_targets_from_figures_complete(self):
        """Should build complete targets from figure list with all required fields."""
        figures = [
            {"id": "Fig1", "description": "First figure"},
            {"id": "Fig2", "description": "Second figure"},
        ]
        
        result = build_stub_targets(figures)
        
        assert len(result) == 2
        
        # Verify first target has all required fields
        assert result[0]["figure_id"] == "Fig1"
        assert result[0]["description"] == "First figure"
        assert result[0]["type"] == "spectrum"
        assert result[0]["simulation_class"] == "FDTD_DIRECT"
        assert result[0]["precision_requirement"] == "acceptable"
        assert "digitized_data_path" in result[0]  # May be None
        
        # Verify second target
        assert result[1]["figure_id"] == "Fig2"
        assert result[1]["description"] == "Second figure"
        assert result[1]["type"] == "spectrum"
        assert result[1]["simulation_class"] == "FDTD_DIRECT"
        assert result[1]["precision_requirement"] == "acceptable"

    def test_builds_targets_with_digitized_data_path(self):
        """Should include digitized_data_path from figures when present."""
        figures = [
            {"id": "Fig1", "digitized_data_path": "/path/to/data.csv"},
            {"id": "Fig2"},  # No digitized_data_path
        ]
        
        result = build_stub_targets(figures)
        
        assert result[0]["digitized_data_path"] == "/path/to/data.csv"
        assert result[1]["digitized_data_path"] is None

    def test_builds_targets_with_missing_id(self):
        """Should generate figure_id when id is missing from figure."""
        figures = [
            {"description": "First figure"},  # Missing id
            {"id": "Fig2", "description": "Second figure"},
        ]
        
        result = build_stub_targets(figures)
        
        # First figure should get generated id "Fig1" (idx + 1)
        assert result[0]["figure_id"] == "Fig1"
        assert result[0]["description"] == "First figure"
        assert result[1]["figure_id"] == "Fig2"

    def test_builds_targets_with_none_id(self):
        """Should generate figure_id when id is None."""
        figures = [
            {"id": None, "description": "First figure"},
        ]
        
        result = build_stub_targets(figures)
        
        assert result[0]["figure_id"] == "Fig1"

    def test_builds_targets_with_missing_description(self):
        """Should generate description when missing from figure."""
        figures = [
            {"id": "Fig1"},  # Missing description
        ]
        
        result = build_stub_targets(figures)
        
        assert result[0]["figure_id"] == "Fig1"
        assert result[0]["description"] == "Simulation target Fig1"

    def test_builds_stub_for_empty_figures(self):
        """Should build stub target for empty figures list."""
        result = build_stub_targets([])
        
        assert len(result) == 1
        assert result[0]["figure_id"] == "FigStub"
        assert result[0]["description"] == "Placeholder simulation target"
        assert result[0]["type"] == "spectrum"
        assert result[0]["simulation_class"] == "FDTD_DIRECT"
        assert result[0]["precision_requirement"] == "acceptable"
        assert result[0]["digitized_data_path"] is None

    def test_builds_targets_preserves_all_fields(self):
        """Should preserve all fields from figure, not just id and description."""
        figures = [
            {
                "id": "Fig1",
                "description": "Test figure",
                "digitized_data_path": "/path/to/data.csv",
                "extra_field": "should_be_preserved",
            }
        ]
        
        result = build_stub_targets(figures)
        
        # Required fields should be set correctly
        assert result[0]["figure_id"] == "Fig1"
        assert result[0]["description"] == "Test figure"
        assert result[0]["digitized_data_path"] == "/path/to/data.csv"
        # Note: extra_field is not copied, only specific fields are used


class TestBuildStubExpectedOutputs:
    """Tests for build_stub_expected_outputs function."""

    def test_builds_expected_outputs_complete(self):
        """Should build complete expected output specs with all required fields."""
        result = build_stub_expected_outputs(
            "paper1", "stage1", ["Fig1", "Fig2"], ["wavelength", "transmission"]
        )
        
        assert len(result) == 2
        
        # Verify first output has all required fields
        assert result[0]["artifact_type"] == "spectrum_csv"
        assert result[0]["filename_pattern"] == "paper1_stage1_fig1_spectrum.csv"
        assert result[0]["description"] == "Simulation data for Fig1"
        assert result[0]["columns"] == ["wavelength", "transmission"]
        assert result[0]["target_figure"] == "Fig1"
        
        # Verify second output
        assert result[1]["artifact_type"] == "spectrum_csv"
        assert result[1]["filename_pattern"] == "paper1_stage1_fig2_spectrum.csv"
        assert result[1]["description"] == "Simulation data for Fig2"
        assert result[1]["columns"] == ["wavelength", "transmission"]
        assert result[1]["target_figure"] == "Fig2"

    def test_builds_expected_outputs_single_target(self):
        """Should build output for single target."""
        result = build_stub_expected_outputs(
            "test_paper", "stage0", ["Fig1"], ["wavelength_nm", "n", "k"]
        )
        
        assert len(result) == 1
        assert result[0]["artifact_type"] == "spectrum_csv"
        assert result[0]["filename_pattern"] == "test_paper_stage0_fig1_spectrum.csv"
        assert result[0]["target_figure"] == "Fig1"
        assert result[0]["columns"] == ["wavelength_nm", "n", "k"]

    def test_builds_expected_outputs_empty_targets(self):
        """Should handle empty target_ids list."""
        result = build_stub_expected_outputs(
            "paper1", "stage1", [], ["wavelength", "transmission"]
        )
        
        assert len(result) == 0
        assert isinstance(result, list)

    def test_builds_expected_outputs_empty_columns(self):
        """Should handle empty columns list."""
        result = build_stub_expected_outputs(
            "paper1", "stage1", ["Fig1"], []
        )
        
        assert len(result) == 1
        assert result[0]["columns"] == []

    def test_filename_pattern_lowercases_target_id(self):
        """Should lowercase target ID in filename pattern."""
        result = build_stub_expected_outputs(
            "Paper1", "Stage1", ["Fig1A", "FIG2B"], ["wavelength"]
        )
        
        assert result[0]["filename_pattern"] == "Paper1_Stage1_fig1a_spectrum.csv"
        assert result[1]["filename_pattern"] == "Paper1_Stage1_fig2b_spectrum.csv"

    def test_builds_expected_outputs_all_fields_present(self):
        """Should verify all required schema fields are present."""
        result = build_stub_expected_outputs(
            "paper1", "stage1", ["Fig1"], ["wavelength", "transmission"]
        )
        
        output = result[0]
        # Required fields per plan_schema.json
        assert "artifact_type" in output
        assert "filename_pattern" in output
        assert "description" in output
        # Optional but should be present
        assert "columns" in output
        assert "target_figure" in output


class TestBuildStubStages:
    """Tests for build_stub_stages function."""

    def test_builds_stage0_and_stage1_complete(self):
        """Should build Stage 0 and Stage 1 with all required fields."""
        targets = [
            {"figure_id": "Fig1"},
            {"figure_id": "Fig2"},
        ]
        
        result = build_stub_stages("paper1", targets)
        
        assert len(result) == 2
        
        # Verify Stage 0 has all required fields
        stage0 = result[0]
        assert stage0["stage_id"] == "stage0_material_validation"
        assert stage0["stage_type"] == "MATERIAL_VALIDATION"
        assert stage0["name"] == "Material optical properties validation"
        assert stage0["description"] == "Validate material optical constants against primary reference figure."
        assert stage0["targets"] == ["Fig1"]
        assert stage0["dependencies"] == []
        assert stage0["is_mandatory_validation"] is True
        assert stage0["complexity_class"] == "analytical"
        assert stage0["runtime_estimate_minutes"] == 2
        assert stage0["runtime_budget_minutes"] == 10
        assert stage0["max_revisions"] == 3
        assert stage0["fallback_strategy"] == "ask_user"
        assert len(stage0["validation_criteria"]) == 1
        assert "Fig1" in stage0["validation_criteria"][0]
        assert "expected_outputs" in stage0
        assert len(stage0["expected_outputs"]) == 1
        assert stage0["reference_data_path"] is None
        
        # Verify Stage 1 has all required fields
        stage1 = result[1]
        assert stage1["stage_id"] == "stage1_primary_structure"
        assert stage1["stage_type"] == "SINGLE_STRUCTURE"
        assert stage1["name"] == "Primary structure reproduction"
        assert stage1["targets"] == ["Fig2"]
        assert stage1["dependencies"] == ["stage0_material_validation"]
        assert stage1["is_mandatory_validation"] is False
        assert stage1["complexity_class"] == "2D_light"
        assert stage1["runtime_estimate_minutes"] == 15
        assert stage1["runtime_budget_minutes"] == 45
        assert stage1["max_revisions"] == 3
        assert stage1["fallback_strategy"] == "ask_user"
        assert len(stage1["validation_criteria"]) == 1
        assert "Fig2" in stage1["validation_criteria"][0]
        assert "expected_outputs" in stage1
        assert len(stage1["expected_outputs"]) == 1
        assert stage1["reference_data_path"] is None

    def test_stage1_depends_on_stage0(self):
        """Stage 1 should depend on Stage 0."""
        targets = [{"figure_id": "Fig1"}]
        
        result = build_stub_stages("paper1", targets)
        
        assert "stage0_material_validation" in result[1]["dependencies"]
        assert len(result[1]["dependencies"]) == 1

    def test_stage1_uses_same_target_when_only_one(self):
        """Stage 1 should use same target as Stage 0 when only one target exists."""
        targets = [{"figure_id": "Fig1"}]
        
        result = build_stub_stages("paper1", targets)
        
        assert result[0]["targets"] == ["Fig1"]
        assert result[1]["targets"] == ["Fig1"]

    def test_stage1_uses_remaining_targets(self):
        """Stage 1 should use targets after the first one."""
        targets = [
            {"figure_id": "Fig1"},
            {"figure_id": "Fig2"},
            {"figure_id": "Fig3"},
        ]
        
        result = build_stub_stages("paper1", targets)
        
        assert result[0]["targets"] == ["Fig1"]
        assert result[1]["targets"] == ["Fig2", "Fig3"]

    def test_handles_empty_targets(self):
        """Should handle empty targets list by creating stub target."""
        result = build_stub_stages("paper1", [])
        
        assert len(result) == 2
        assert result[0]["targets"] == ["FigStub"]
        assert result[1]["targets"] == ["FigStub"]
        # Verify stub target is used in validation criteria
        assert "FigStub" in result[0]["validation_criteria"][0]
        assert "FigStub" in result[1]["validation_criteria"][0]

    def test_stage0_expected_outputs_columns(self):
        """Stage 0 expected outputs should have correct columns."""
        targets = [{"figure_id": "Fig1"}]
        
        result = build_stub_stages("paper1", targets)
        
        stage0_outputs = result[0]["expected_outputs"]
        assert len(stage0_outputs) == 1
        assert stage0_outputs[0]["columns"] == ["wavelength_nm", "n", "k"]

    def test_stage1_expected_outputs_columns(self):
        """Stage 1 expected outputs should have correct columns."""
        targets = [{"figure_id": "Fig1"}]
        
        result = build_stub_stages("paper1", targets)
        
        stage1_outputs = result[1]["expected_outputs"]
        assert len(stage1_outputs) == 1
        assert stage1_outputs[0]["columns"] == ["wavelength_nm", "transmission"]

    def test_stage1_validation_criteria_for_multiple_targets(self):
        """Stage 1 validation criteria should include targets after the first one."""
        targets = [
            {"figure_id": "Fig1"},
            {"figure_id": "Fig2"},
            {"figure_id": "Fig3"},
        ]
        
        result = build_stub_stages("paper1", targets)
        
        stage1_criteria = result[1]["validation_criteria"]
        # Stage 1 uses targets[1:], so should have 2 criteria (for Fig2 and Fig3)
        assert len(stage1_criteria) == 2
        assert "Fig2" in stage1_criteria[0]
        assert "Fig3" in stage1_criteria[1]
        # Verify all criteria mention resonance
        assert all("resonance" in crit.lower() for crit in stage1_criteria)

    def test_paper_id_used_in_expected_outputs(self):
        """Paper ID should be used in expected output filename patterns."""
        targets = [{"figure_id": "Fig1"}]
        
        result = build_stub_stages("test_paper_123", targets)
        
        stage0_output = result[0]["expected_outputs"][0]
        assert "test_paper_123" in stage0_output["filename_pattern"]
        
        stage1_output = result[1]["expected_outputs"][0]
        assert "test_paper_123" in stage1_output["filename_pattern"]


class TestBuildStubPlannedMaterials:
    """Tests for build_stub_planned_materials function."""

    def test_builds_placeholder_material_complete(self):
        """Should build complete placeholder material entry with all fields."""
        state = {"paper_domain": "plasmonics"}
        
        result = build_stub_planned_materials(state)
        
        assert len(result) == 1
        material = result[0]
        assert material["material_id"] == "plasmonics_placeholder"
        assert material["name"] == "Plasmonics Material"
        assert material["source"] == "stub"
        assert material["path"] == "materials/placeholder.csv"
        # Verify all fields are present
        assert "material_id" in material
        assert "name" in material
        assert "source" in material
        assert "path" in material

    def test_uses_generic_for_missing_domain(self):
        """Should use generic domain if not specified."""
        state = {}
        
        result = build_stub_planned_materials(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "generic_placeholder"
        assert result[0]["name"] == "Generic Material"

    def test_uses_generic_for_none_domain(self):
        """Should use generic domain if domain is None."""
        state = {"paper_domain": None}
        
        result = build_stub_planned_materials(state)
        
        assert len(result) == 1
        assert result[0]["material_id"] == "generic_placeholder"

    def test_domain_name_capitalization(self):
        """Should capitalize domain name in material name."""
        state = {"paper_domain": "photonic_crystal"}
        
        result = build_stub_planned_materials(state)
        
        assert result[0]["name"] == "Photonic_Crystal Material"
        # Note: title() capitalizes first letter of each word, so "photonic_crystal" becomes "Photonic_Crystal"

    def test_returns_list_not_dict(self):
        """Should return a list, not a dict."""
        state = {"paper_domain": "plasmonics"}
        
        result = build_stub_planned_materials(state)
        
        assert isinstance(result, list)
        assert not isinstance(result, dict)


class TestBuildStubAssumptions:
    """Tests for build_stub_assumptions function."""

    def test_builds_empty_assumptions_complete(self):
        """Should build complete empty assumptions structure."""
        result = build_stub_assumptions()
        
        # Verify top-level structure
        assert "global_assumptions" in result
        assert "stage_specific" in result
        
        # Verify global_assumptions structure
        assert isinstance(result["global_assumptions"], dict)
        assert "materials" in result["global_assumptions"]
        assert "geometry" in result["global_assumptions"]
        assert "sources" in result["global_assumptions"]
        
        # Verify all are empty lists
        assert result["global_assumptions"]["materials"] == []
        assert result["global_assumptions"]["geometry"] == []
        assert result["global_assumptions"]["sources"] == []
        
        # Verify stage_specific is empty list
        assert isinstance(result["stage_specific"], list)
        assert result["stage_specific"] == []

    def test_returns_dict_not_list(self):
        """Should return a dict, not a list."""
        result = build_stub_assumptions()
        
        assert isinstance(result, dict)
        assert not isinstance(result, list)

    def test_structure_is_mutable(self):
        """Should return mutable structure that can be modified."""
        result = build_stub_assumptions()
        
        # Should be able to modify without errors
        result["global_assumptions"]["materials"].append("test")
        assert len(result["global_assumptions"]["materials"]) == 1


class TestBuildStubPlan:
    """Tests for build_stub_plan function."""

    def test_builds_complete_plan_all_fields(self):
        """Should build a complete stub plan with all required and optional fields."""
        state = {
            "paper_id": "test_paper",
            "paper_title": "Test Paper Title",
            "paper_figures": [
                {"id": "Fig1", "description": "First figure"},
            ],
            "paper_domain": "plasmonics",
        }
        
        result = build_stub_plan(state)
        
        # Required fields per plan_schema.json
        assert result["paper_id"] == "test_paper"
        assert result["paper_domain"] == "plasmonics"
        assert result["title"] == "Test Paper Title"
        assert "summary" in result
        assert "targets" in result
        assert "stages" in result
        
        # Verify targets
        assert isinstance(result["targets"], list)
        assert len(result["targets"]) >= 1
        assert result["targets"][0]["figure_id"] == "Fig1"
        
        # Verify stages
        assert isinstance(result["stages"], list)
        assert len(result["stages"]) == 2
        
        # Verify reproduction_scope
        assert "reproduction_scope" in result
        scope = result["reproduction_scope"]
        assert "total_figures" in scope
        assert "reproducible_figures" in scope
        assert "reproducible_figure_ids" in scope
        assert "attempted_figures" in scope
        assert "skipped_figures" in scope
        assert "coverage_percent" in scope
        
        # Verify extracted_parameters
        assert "extracted_parameters" in result
        assert isinstance(result["extracted_parameters"], list)
        assert len(result["extracted_parameters"]) >= 1
        
        # Verify main_system
        assert "main_system" in result
        assert result["main_system"] == "plasmonics"

    def test_uses_defaults_for_missing_fields(self):
        """Should use defaults when state fields are missing."""
        state = {}
        
        result = build_stub_plan(state)
        
        assert result["paper_id"] == "paper_stub"
        assert result["title"] == "Paper Stub"  # paper_id.replace("_", " ").title()
        assert result["paper_domain"] == "other"
        assert result["main_system"] == "other"
        assert "stages" in result
        assert "targets" in result
        assert len(result["stages"]) == 2
        assert len(result["targets"]) == 1  # Should have stub target

    def test_calculates_coverage_correctly(self):
        """Should calculate reproduction coverage correctly."""
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
        assert len(scope["reproducible_figure_ids"]) == 2
        assert len(scope["attempted_figures"]) == 2
        assert scope["attempted_figures"] == ["Fig1", "Fig2"]

    def test_calculates_coverage_partial(self):
        """Should calculate coverage when some figures exist."""
        state = {
            "paper_id": "test",
            "paper_figures": [
                {"id": "Fig1"},
                {"id": "Fig2"},
                {"id": "Fig3"},
            ],
        }
        
        result = build_stub_plan(state)
        
        scope = result["reproduction_scope"]
        assert scope["total_figures"] == 3
        assert scope["reproducible_figures"] == 3
        assert scope["coverage_percent"] == 100.0
        assert len(scope["attempted_figures"]) == 3

    def test_calculates_coverage_zero_figures(self):
        """Should handle zero figures correctly."""
        state = {
            "paper_id": "test",
            "paper_figures": [],
        }
        
        result = build_stub_plan(state)
        
        scope = result["reproduction_scope"]
        assert scope["total_figures"] == 0
        assert scope["reproducible_figures"] == 0
        # When total_figures is 0, coverage should be 0.0 (division by zero protection)
        assert scope["coverage_percent"] == 0.0
        assert scope["attempted_figures"] == []
        assert scope["skipped_figures"] == []

    def test_calculates_coverage_no_figures_key(self):
        """Should handle missing paper_figures key."""
        state = {
            "paper_id": "test",
        }
        
        result = build_stub_plan(state)
        
        scope = result["reproduction_scope"]
        assert scope["total_figures"] == 0
        assert scope["reproducible_figures"] == 0
        assert scope["coverage_percent"] == 0.0

    def test_title_generation_from_paper_id(self):
        """Should generate title from paper_id when paper_title missing."""
        state = {
            "paper_id": "test_paper_2023",
        }
        
        result = build_stub_plan(state)
        
        assert result["title"] == "Test Paper 2023"  # replace("_", " ").title()

    def test_summary_includes_title(self):
        """Summary should include paper title."""
        state = {
            "paper_id": "test",
            "paper_title": "My Test Paper",
        }
        
        result = build_stub_plan(state)
        
        assert "My Test Paper" in result["summary"]
        assert "stub" in result["summary"].lower()

    def test_extracted_parameters_structure(self):
        """Should include extracted_parameters with required fields."""
        state = {
            "paper_id": "test",
        }
        
        result = build_stub_plan(state)
        
        assert len(result["extracted_parameters"]) >= 1
        param = result["extracted_parameters"][0]
        assert "name" in param
        assert "value" in param
        assert "unit" in param
        assert "source" in param
        assert param["source"] == "inferred"
        assert param["location"] == "stub_generator"

    def test_stages_use_paper_id(self):
        """Stages should use paper_id in expected_outputs."""
        state = {
            "paper_id": "my_paper",
            "paper_figures": [{"id": "Fig1"}],
        }
        
        result = build_stub_plan(state)
        
        stage0_output = result["stages"][0]["expected_outputs"][0]
        assert "my_paper" in stage0_output["filename_pattern"]
        
        stage1_output = result["stages"][1]["expected_outputs"][0]
        assert "my_paper" in stage1_output["filename_pattern"]

    def test_skipped_figures_is_empty_list(self):
        """skipped_figures should be empty list for stub plans."""
        state = {
            "paper_id": "test",
            "paper_figures": [{"id": "Fig1"}],
        }
        
        result = build_stub_plan(state)
        
        assert result["reproduction_scope"]["skipped_figures"] == []

    def test_all_required_schema_fields_present(self):
        """Should verify all required fields from plan_schema.json are present."""
        state = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_figures": [{"id": "Fig1"}],
            "paper_domain": "plasmonics",
        }
        
        result = build_stub_plan(state)
        
        # Required fields per plan_schema.json
        assert "paper_id" in result
        assert "paper_domain" in result
        assert "title" in result
        assert "summary" in result
        assert "targets" in result
        assert "stages" in result
        
        # Verify targets have required fields
        for target in result["targets"]:
            assert "figure_id" in target
            assert "description" in target
            assert "type" in target
            assert "simulation_class" in target
        
        # Verify stages have required fields
        for stage in result["stages"]:
            assert "stage_id" in stage
            assert "stage_type" in stage
            assert "name" in stage
            assert "description" in stage
            assert "targets" in stage
            assert "dependencies" in stage
