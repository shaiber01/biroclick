"""Negative/malformed response tests to ensure schema enforcement.

This module tests that JSON schemas correctly REJECT invalid responses.
Each test verifies that a specific type of malformed response is detected
and causes validation to fail with an appropriate error message.

The goal is to ensure the schemas are strict enough to catch bugs in
LLM-generated responses.
"""

import pytest
from jsonschema import ValidationError, validate

from .helpers import AGENT_SCHEMAS, load_schema


# =============================================================================
# TEST: MISSING REQUIRED FIELDS
# =============================================================================


class TestMissingRequiredFields:
    """Test that missing required fields are properly detected for all schemas."""

    # --- Plan Reviewer Schema ---

    def test_plan_reviewer_missing_verdict(self):
        """Plan reviewer without 'verdict' should fail validation."""
        response = {
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test summary",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value)

    def test_plan_reviewer_missing_checklist_results(self):
        """Plan reviewer without 'checklist_results' should fail validation."""
        response = {
            "verdict": "approve",
            "summary": "Test summary",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "checklist_results" in str(exc_info.value)

    def test_plan_reviewer_missing_summary(self):
        """Plan reviewer without 'summary' should fail validation."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "summary" in str(exc_info.value)

    def test_plan_reviewer_missing_required_checklist_category(self):
        """Plan reviewer checklist missing a required category should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                # Missing: parameter_extraction, assumptions, performance
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        # Should fail because required checklist categories are missing
        error_str = str(exc_info.value)
        assert any(
            cat in error_str
            for cat in ["parameter_extraction", "assumptions", "performance", "required"]
        )

    # --- Code Generator Schema ---

    def test_code_generator_missing_stage_id(self):
        """Code generator without 'stage_id' should fail validation."""
        response = {
            "code": "print('hello')",
            "expected_outputs": [],
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stage_id" in str(exc_info.value)

    def test_code_generator_missing_code(self):
        """Code generator without 'code' should fail validation."""
        response = {
            "stage_id": "stage1",
            "expected_outputs": [],
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "code" in str(exc_info.value)

    def test_code_generator_missing_expected_outputs(self):
        """Code generator without 'expected_outputs' should fail validation."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "expected_outputs" in str(exc_info.value)

    def test_code_generator_missing_runtime_estimate(self):
        """Code generator without 'estimated_runtime_minutes' should fail."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "expected_outputs": [],
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "estimated_runtime_minutes" in str(exc_info.value)

    # --- Planner Schema ---

    def test_planner_missing_paper_id(self):
        """Planner without 'paper_id' should fail validation."""
        response = {
            "paper_domain": "plasmonics",
            "title": "Test Paper",
            "summary": "Test summary",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "paper_id" in str(exc_info.value)

    def test_planner_missing_paper_domain(self):
        """Planner without 'paper_domain' should fail validation."""
        response = {
            "paper_id": "test123",
            "title": "Test Paper",
            "summary": "Test summary",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "paper_domain" in str(exc_info.value)

    def test_planner_missing_stages(self):
        """Planner without 'stages' should fail validation."""
        response = {
            "paper_id": "test123",
            "paper_domain": "plasmonics",
            "title": "Test Paper",
            "summary": "Test summary",
            "extracted_parameters": [],
            "targets": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stages" in str(exc_info.value)

    # --- Simulation Designer Schema ---

    def test_simulation_designer_missing_stage_id(self):
        """Simulation designer without 'stage_id' should fail validation."""
        response = {
            "design_description": "Test design",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stage_id" in str(exc_info.value)

    def test_simulation_designer_missing_unit_system(self):
        """Simulation designer without 'unit_system' should fail validation."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test design",
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "unit_system" in str(exc_info.value)

    # --- Code Reviewer Schema ---

    def test_code_reviewer_missing_checklist_results(self):
        """Code reviewer without 'checklist_results' should fail validation."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "issues": [],
            "summary": "Test summary",
        }
        schema = load_schema("code_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "checklist_results" in str(exc_info.value)

    def test_code_reviewer_missing_issues(self):
        """Code reviewer without 'issues' should fail validation."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "unit_normalization": {"status": "pass"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {"status": "pass", "all_outputs_accounted": True},
            },
            "summary": "Test summary",
        }
        schema = load_schema("code_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "issues" in str(exc_info.value)

    # --- Design Reviewer Schema ---

    def test_design_reviewer_missing_verdict(self):
        """Design reviewer without 'verdict' should fail validation."""
        response = {
            "stage_id": "stage1",
            "checklist_results": {
                "geometry": {"status": "pass"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"},
            },
            "issues": [],
            "summary": "Test summary",
        }
        schema = load_schema("design_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value)

    # --- Execution Validator Schema ---

    def test_execution_validator_missing_execution_status(self):
        """Execution validator without 'execution_status' should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": True,
            },
            "summary": "Test summary",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "execution_status" in str(exc_info.value)

    def test_execution_validator_missing_files_check(self):
        """Execution validator without 'files_check' should fail validation."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": True},
            "summary": "Test summary",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "files_check" in str(exc_info.value)

    # --- Physics Sanity Schema ---

    def test_physics_sanity_missing_conservation_checks(self):
        """Physics sanity without 'conservation_checks' should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "value_range_checks": [],
            "summary": "Test summary",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "conservation_checks" in str(exc_info.value)

    def test_physics_sanity_missing_value_range_checks(self):
        """Physics sanity without 'value_range_checks' should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [],
            "summary": "Test summary",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "value_range_checks" in str(exc_info.value)

    # --- Supervisor Schema ---

    def test_supervisor_missing_validation_hierarchy_status(self):
        """Supervisor without 'validation_hierarchy_status' should fail."""
        response = {
            "verdict": "ok_continue",
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test summary",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "validation_hierarchy_status" in str(exc_info.value)

    def test_supervisor_missing_main_physics_assessment(self):
        """Supervisor without 'main_physics_assessment' should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "summary": "Test summary",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "main_physics_assessment" in str(exc_info.value)

    # --- Results Analyzer Schema ---

    def test_results_analyzer_missing_per_result_reports(self):
        """Results analyzer without 'per_result_reports' should fail."""
        response = {
            "stage_id": "stage1",
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test summary",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "per_result_reports" in str(exc_info.value)

    def test_results_analyzer_missing_overall_classification(self):
        """Results analyzer without 'overall_classification' should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "summary": "Test summary",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "overall_classification" in str(exc_info.value)

    # --- Comparison Validator Schema ---

    def test_comparison_validator_missing_accuracy_check(self):
        """Comparison validator without 'accuracy_check' should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "math_check": {"status": "pass"},
            "summary": "Test summary",
        }
        schema = load_schema("comparison_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "accuracy_check" in str(exc_info.value)

    def test_comparison_validator_missing_math_check(self):
        """Comparison validator without 'math_check' should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "accuracy_check": {"status": "pass"},
            "summary": "Test summary",
        }
        schema = load_schema("comparison_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "math_check" in str(exc_info.value)


# =============================================================================
# TEST: WRONG TYPE DETECTION
# =============================================================================


class TestWrongTypeDetection:
    """Test that wrong field types are properly detected for all schemas."""

    # --- String Fields Given Wrong Types ---

    def test_verdict_as_integer(self):
        """Verdict field given integer instead of string should fail."""
        response = {
            "verdict": 1,  # Should be string
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "verdict" in error_str or "string" in error_str or "type" in error_str

    def test_verdict_as_boolean(self):
        """Verdict field given boolean instead of string should fail."""
        response = {
            "verdict": True,  # Should be string
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "verdict" in error_str or "string" in error_str or "type" in error_str

    def test_summary_as_array(self):
        """Summary field given array instead of string should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": ["line1", "line2"],  # Should be string
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "summary" in error_str or "string" in error_str or "type" in error_str

    def test_stage_id_as_number(self):
        """Stage ID given number instead of string should fail."""
        response = {
            "stage_id": 123,  # Should be string
            "code": "print('hello')",
            "expected_outputs": [],
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "stage_id" in error_str or "string" in error_str or "type" in error_str

    # --- Array Fields Given Wrong Types ---

    def test_issues_as_string(self):
        """Issues field given string instead of array should fail."""
        response = {
            "verdict": "approve",
            "issues": "no issues",  # Should be array
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "issues" in error_str or "array" in error_str or "type" in error_str

    def test_stages_as_object(self):
        """Stages field given object instead of array should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": {"stage1": "data"},  # Should be array
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "stages" in error_str or "array" in error_str or "type" in error_str

    def test_expected_outputs_as_string(self):
        """Expected outputs given string instead of array should fail."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "expected_outputs": "some file",  # Should be array
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "expected_outputs" in error_str or "array" in error_str or "type" in error_str

    def test_conservation_checks_as_object(self):
        """Conservation checks given object instead of array should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": {"energy": "ok"},  # Should be array
            "value_range_checks": [],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "conservation_checks" in error_str or "array" in error_str or "type" in error_str

    # --- Object Fields Given Wrong Types ---

    def test_checklist_results_as_array(self):
        """Checklist results given array instead of object should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": ["pass", "pass"],  # Should be object
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "checklist_results" in error_str or "object" in error_str or "type" in error_str

    def test_unit_system_as_string(self):
        """Unit system given string instead of object should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": "micrometers",  # Should be object
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "unit_system" in error_str or "object" in error_str or "type" in error_str

    def test_execution_status_as_string(self):
        """Execution status given string instead of object should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": "completed",  # Should be object
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": True,
            },
            "summary": "Test",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "execution_status" in error_str or "object" in error_str or "type" in error_str

    def test_validation_hierarchy_status_as_array(self):
        """Validation hierarchy status given array instead of object should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": ["passed", "partial"],  # Should be object
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert (
            "validation_hierarchy_status" in error_str
            or "object" in error_str
            or "type" in error_str
        )

    # --- Number Fields Given Wrong Types ---

    def test_estimated_runtime_as_string(self):
        """Estimated runtime given string instead of number should fail."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "expected_outputs": [],
            "estimated_runtime_minutes": "five minutes",  # Should be number
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert (
            "estimated_runtime_minutes" in error_str
            or "number" in error_str
            or "type" in error_str
        )

    def test_confidence_as_string(self):
        """Confidence given string instead of number should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "confidence": "high",  # Should be number 0-1
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "confidence" in error_str or "number" in error_str or "type" in error_str

    # --- Boolean Fields Given Wrong Types ---

    def test_completed_as_string(self):
        """Completed field given string instead of boolean should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": "yes"},  # Should be boolean
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": True,
            },
            "summary": "Test",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "completed" in error_str or "boolean" in error_str or "type" in error_str

    def test_physics_plausible_as_string(self):
        """Physics plausible given string instead of boolean should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": "yes",  # Should be boolean
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "physics_plausible" in error_str or "boolean" in error_str or "type" in error_str

    def test_all_present_as_integer(self):
        """All present field given integer instead of boolean should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": 1,  # Should be boolean
            },
            "summary": "Test",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "all_present" in error_str or "boolean" in error_str or "type" in error_str


# =============================================================================
# TEST: INVALID ENUM VALUES
# =============================================================================


class TestInvalidEnumValues:
    """Test that invalid enum values are properly detected for all schemas."""

    # --- Verdict Enums ---

    def test_plan_reviewer_invalid_verdict(self):
        """Plan reviewer with invalid verdict should fail."""
        response = {
            "verdict": "maybe",  # Not in enum
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "maybe" in error_str or "enum" in error_str

    def test_plan_reviewer_verdict_wrong_case(self):
        """Plan reviewer verdict with wrong case should fail."""
        response = {
            "verdict": "APPROVE",  # Should be lowercase
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "APPROVE" in error_str or "enum" in error_str

    def test_execution_validator_invalid_verdict(self):
        """Execution validator with invalid verdict should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "success",  # Should be pass/warning/fail
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": True,
            },
            "summary": "Test",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "success" in error_str or "enum" in error_str

    def test_physics_sanity_invalid_verdict(self):
        """Physics sanity with invalid verdict should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "ok",  # Should be pass/warning/fail/design_flaw
            "conservation_checks": [],
            "value_range_checks": [],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "ok" in error_str or "enum" in error_str

    def test_supervisor_invalid_verdict(self):
        """Supervisor with invalid verdict should fail."""
        response = {
            "verdict": "continue",  # Should be ok_continue, replan_needed, etc.
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "continue" in error_str or "enum" in error_str

    # --- Status Enums ---

    def test_checklist_status_invalid(self):
        """Checklist status with invalid value should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "ok"},  # Should be pass/fail/warning
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "ok" in error_str or "enum" in error_str

    def test_validation_hierarchy_invalid_status(self):
        """Validation hierarchy with invalid status should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "done",  # Should be passed/partial/failed/not_done
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "done" in error_str or "enum" in error_str

    # --- Classification Enums ---

    def test_overall_classification_invalid(self):
        """Results analyzer with invalid classification should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "GOOD_MATCH",  # Not in enum
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "GOOD_MATCH" in error_str or "enum" in error_str

    def test_discrepancy_classification_invalid(self):
        """Discrepancy classification with invalid value should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [
                {
                    "result_id": "r1",
                    "target_figure": "fig1",
                    "quantity": "wavelength",
                    "discrepancy": {
                        "absolute": 10,
                        "relative_percent": 5,
                        "classification": "good",  # Should be excellent/acceptable/investigate/unacceptable
                    },
                }
            ],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "good" in error_str or "enum" in error_str

    # --- Severity Enums ---

    def test_issue_severity_invalid(self):
        """Issue with invalid severity should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": [
                {
                    "severity": "critical",  # Should be blocking/major/minor
                    "category": "coverage",
                    "description": "Test issue",
                    "suggested_fix": "Fix it",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "critical" in error_str or "enum" in error_str

    def test_concern_severity_invalid(self):
        """Concern with invalid severity should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "warning",
            "conservation_checks": [],
            "value_range_checks": [],
            "concerns": [
                {
                    "concern": "Something wrong",
                    "severity": "high",  # Should be critical/moderate/minor
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "high" in error_str or "enum" in error_str

    # --- Domain Enums ---

    def test_paper_domain_invalid(self):
        """Planner with invalid paper domain should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "optics",  # Not in enum
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "optics" in error_str or "enum" in error_str

    def test_simulation_class_invalid(self):
        """Target with invalid simulation class should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [
                {
                    "figure_id": "fig1",
                    "description": "Test target",
                    "type": "spectrum",
                    "simulation_class": "SIMPLE",  # Not in enum
                }
            ],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "SIMPLE" in error_str or "enum" in error_str

    def test_stage_type_invalid(self):
        """Stage with invalid stage type should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    "stage_id": "stage1",
                    "stage_type": "BASIC",  # Not in enum
                    "name": "Test Stage",
                    "description": "Test",
                    "targets": [],
                    "dependencies": [],
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "BASIC" in error_str or "enum" in error_str

    # --- Source Enums ---

    def test_parameter_source_invalid(self):
        """Parameter with invalid source should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {
                    "name": "wavelength",
                    "value": 500,
                    "unit": "nm",
                    "source": "paper",  # Should be text/figure_caption/etc.
                }
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "paper" in error_str or "enum" in error_str

    # --- Material Model Type Enum ---

    def test_material_model_type_invalid(self):
        """Material with invalid model type should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [
                {
                    "id": "gold",
                    "name": "Gold",
                    "model_type": "metal",  # Should be constant/tabulated/drude/lorentz/drude_lorentz
                }
            ],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "metal" in error_str or "enum" in error_str

    # --- Artifact Type Enum ---

    def test_artifact_type_invalid(self):
        """Expected output with invalid artifact type should fail."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "expected_outputs": [
                {
                    "artifact_type": "data_file",  # Not in enum
                    "filename": "output.dat",
                    "description": "Some data",
                }
            ],
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "data_file" in error_str or "enum" in error_str

    # --- Category Enums ---

    def test_plan_reviewer_issue_category_invalid(self):
        """Plan reviewer issue with invalid category should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": [
                {
                    "severity": "major",
                    "category": "validation",  # Not in enum for plan_reviewer
                    "description": "Test issue",
                    "suggested_fix": "Fix it",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "validation" in error_str or "enum" in error_str

    def test_design_reviewer_issue_category_invalid(self):
        """Design reviewer issue with invalid category should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "needs_revision",
            "checklist_results": {
                "geometry": {"status": "fail"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"},
            },
            "issues": [
                {
                    "severity": "major",
                    "category": "code_quality",  # Not in enum for design_reviewer
                    "description": "Test issue",
                    "suggested_fix": "Fix it",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("design_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "code_quality" in error_str or "enum" in error_str


# =============================================================================
# TEST: NESTED OBJECT VALIDATION
# =============================================================================


class TestNestedObjectValidation:
    """Test validation of nested objects and their required fields."""

    # --- Unit System Nested Required Fields ---

    def test_unit_system_missing_characteristic_length(self):
        """Unit system without characteristic_length_m should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "length_unit": "µm",
                # Missing: characteristic_length_m
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "characteristic_length_m" in str(exc_info.value)

    def test_unit_system_missing_length_unit(self):
        """Unit system without length_unit should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                # Missing: length_unit
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "length_unit" in str(exc_info.value)

    # --- Performance Estimate Nested Required Fields ---

    def test_performance_estimate_missing_runtime(self):
        """Performance estimate without runtime_estimate_minutes should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "memory_estimate_gb": 2,
                # Missing: runtime_estimate_minutes
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "runtime_estimate_minutes" in str(exc_info.value)

    def test_performance_estimate_missing_memory(self):
        """Performance estimate without memory_estimate_gb should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                # Missing: memory_estimate_gb
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "memory_estimate_gb" in str(exc_info.value)

    # --- Files Check Nested Required Fields ---

    def test_files_check_missing_expected_files(self):
        """Files check without expected_files should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": True},
            "files_check": {
                "found_files": [],
                "missing_files": [],
                "all_present": True,
                # Missing: expected_files
            },
            "summary": "Test",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "expected_files" in str(exc_info.value)

    def test_files_check_missing_all_present(self):
        """Files check without all_present should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                # Missing: all_present
            },
            "summary": "Test",
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "all_present" in str(exc_info.value)

    # --- Main Physics Assessment Nested Required Fields ---

    def test_main_physics_assessment_missing_physics_plausible(self):
        """Main physics assessment without physics_plausible should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
                # Missing: physics_plausible
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "physics_plausible" in str(exc_info.value)

    def test_main_physics_assessment_missing_conservation_satisfied(self):
        """Main physics assessment without conservation_satisfied should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "value_ranges_reasonable": True,
                # Missing: conservation_satisfied
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "conservation_satisfied" in str(exc_info.value)

    # --- Validation Hierarchy Status Nested Required Fields ---

    def test_validation_hierarchy_missing_material_validation(self):
        """Validation hierarchy without material_validation should fail."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
                # Missing: material_validation
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "material_validation" in str(exc_info.value)

    # --- Backtrack Suggestion/Decision Nested Required Fields ---

    def test_supervisor_backtrack_decision_missing_accepted(self):
        """Supervisor backtrack_decision without accepted should fail."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                # Missing: accepted
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
                "reason": "Design issues found",
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "accepted" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_supervisor_backtrack_decision_missing_target_stage_id(self):
        """Supervisor backtrack_decision without target_stage_id should fail."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": True,
                # Missing: target_stage_id
                "stages_to_invalidate": ["stage2"],
                "reason": "Design issues found",
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "target_stage_id" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_supervisor_backtrack_decision_missing_stages_to_invalidate(self):
        """Supervisor backtrack_decision without stages_to_invalidate should fail."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                # Missing: stages_to_invalidate
                "reason": "Design issues found",
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stages_to_invalidate" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_supervisor_backtrack_decision_missing_reason(self):
        """Supervisor backtrack_decision without reason should fail."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage1",
                "stages_to_invalidate": ["stage2"],
                # Missing: reason
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "reason" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_design_reviewer_backtrack_suggestion_missing_suggest_backtrack(self):
        """Design reviewer backtrack_suggestion without suggest_backtrack should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "geometry": {"status": "pass"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"},
            },
            "issues": [],
            "backtrack_suggestion": {
                # Missing: suggest_backtrack (required)
                "target_stage_id": "stage0",
                "reason": "Design issues",
            },
            "summary": "Test",
        }
        schema = load_schema("design_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "suggest_backtrack" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_design_reviewer_backtrack_suggestion_missing_target_stage_id(self):
        """Design reviewer backtrack_suggestion without target_stage_id should fail.
        
        This tests that the schema correctly requires target_stage_id when
        suggesting a backtrack - without knowing WHERE to backtrack to,
        the suggestion is incomplete and will cause runtime errors.
        """
        response = {
            "stage_id": "stage1",
            "verdict": "needs_revision",
            "checklist_results": {
                "geometry": {"status": "fail"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"},
            },
            "issues": [],
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                # Missing: target_stage_id - required to know WHERE to backtrack
                "reason": "Geometry issues require plan revision",
            },
            "summary": "Test",
        }
        schema = load_schema("design_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "target_stage_id" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_design_reviewer_backtrack_suggestion_missing_reason(self):
        """Design reviewer backtrack_suggestion without reason should fail.
        
        A backtrack suggestion without a reason is incomplete - the supervisor
        needs to know WHY backtracking is being suggested.
        """
        response = {
            "stage_id": "stage1",
            "verdict": "needs_revision",
            "checklist_results": {
                "geometry": {"status": "fail"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"},
            },
            "issues": [],
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                "target_stage_id": "stage0",
                # Missing: reason - required to explain WHY backtracking
            },
            "summary": "Test",
        }
        schema = load_schema("design_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "reason" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_code_reviewer_backtrack_suggestion_missing_target_stage_id(self):
        """Code reviewer backtrack_suggestion without target_stage_id should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "needs_revision",
            "checklist_results": {
                "unit_normalization": {"status": "fail"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {"status": "pass", "all_outputs_accounted": True},
            },
            "issues": [],
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                # Missing: target_stage_id
                "reason": "Unit normalization issues require design revision",
            },
            "summary": "Test",
        }
        schema = load_schema("code_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "target_stage_id" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_physics_sanity_backtrack_suggestion_missing_target_stage_id(self):
        """Physics sanity backtrack_suggestion without target_stage_id should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "design_flaw",
            "conservation_checks": [],
            "value_range_checks": [],
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                # Missing: target_stage_id
                "reason": "Physics results indicate design flaw",
            },
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "target_stage_id" in str(exc_info.value) or "required" in str(exc_info.value)

    # --- Checklist Result Nested Required Fields ---

    def test_checklist_result_missing_status(self):
        """Checklist result item without status should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"notes": "No status provided"},  # Missing: status
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "status" in str(exc_info.value)

    def test_code_reviewer_expected_outputs_missing_all_outputs_accounted(self):
        """Code reviewer expected_outputs without all_outputs_accounted should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "unit_normalization": {"status": "pass"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {
                    "status": "pass",
                    # Missing: all_outputs_accounted
                },
            },
            "issues": [],
            "summary": "Test",
        }
        schema = load_schema("code_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "all_outputs_accounted" in str(exc_info.value)


# =============================================================================
# TEST: ARRAY ITEM VALIDATION
# =============================================================================


class TestArrayItemValidation:
    """Test validation of array items and their required fields."""

    # --- Issue Array Items ---

    def test_issue_missing_severity(self):
        """Issue item without severity should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": [
                {
                    # Missing: severity
                    "category": "coverage",
                    "description": "Test issue",
                    "suggested_fix": "Fix it",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "severity" in str(exc_info.value)

    def test_issue_missing_category(self):
        """Issue item without category should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": [
                {
                    "severity": "major",
                    # Missing: category
                    "description": "Test issue",
                    "suggested_fix": "Fix it",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "category" in str(exc_info.value)

    def test_issue_missing_description(self):
        """Issue item without description should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": [
                {
                    "severity": "major",
                    "category": "coverage",
                    # Missing: description
                    "suggested_fix": "Fix it",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "description" in str(exc_info.value)

    def test_issue_missing_suggested_fix(self):
        """Issue item without suggested_fix should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": [
                {
                    "severity": "major",
                    "category": "coverage",
                    "description": "Test issue",
                    # Missing: suggested_fix
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "suggested_fix" in str(exc_info.value)

    # --- Extracted Parameters Array Items ---

    def test_extracted_parameter_missing_name(self):
        """Extracted parameter without name should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {
                    # Missing: name
                    "value": 500,
                    "unit": "nm",
                    "source": "text",
                }
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "name" in str(exc_info.value)

    def test_extracted_parameter_missing_value(self):
        """Extracted parameter without value should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {
                    "name": "wavelength",
                    # Missing: value
                    "unit": "nm",
                    "source": "text",
                }
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "value" in str(exc_info.value)

    # --- Target Array Items ---

    def test_target_missing_figure_id(self):
        """Target without figure_id should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [
                {
                    # Missing: figure_id
                    "description": "Test target",
                    "type": "spectrum",
                    "simulation_class": "FDTD_DIRECT",
                }
            ],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "figure_id" in str(exc_info.value)

    def test_target_missing_simulation_class(self):
        """Target without simulation_class should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [
                {
                    "figure_id": "fig1",
                    "description": "Test target",
                    "type": "spectrum",
                    # Missing: simulation_class
                }
            ],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "simulation_class" in str(exc_info.value)

    # --- Stage Array Items ---

    def test_stage_missing_stage_id(self):
        """Stage without stage_id should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    # Missing: stage_id
                    "stage_type": "SINGLE_STRUCTURE",
                    "name": "Test Stage",
                    "description": "Test",
                    "targets": [],
                    "dependencies": [],
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stage_id" in str(exc_info.value)

    def test_stage_missing_stage_type(self):
        """Stage without stage_type should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    "stage_id": "stage1",
                    # Missing: stage_type
                    "name": "Test Stage",
                    "description": "Test",
                    "targets": [],
                    "dependencies": [],
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stage_type" in str(exc_info.value)

    def test_stage_missing_dependencies(self):
        """Stage without dependencies should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    "stage_id": "stage1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "name": "Test Stage",
                    "description": "Test",
                    "targets": [],
                    # Missing: dependencies
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "dependencies" in str(exc_info.value)

    # --- Expected Output Array Items ---

    def test_expected_output_missing_artifact_type(self):
        """Expected output without artifact_type should fail."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "expected_outputs": [
                {
                    # Missing: artifact_type
                    "filename": "output.csv",
                    "description": "Test output",
                }
            ],
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "artifact_type" in str(exc_info.value)

    def test_expected_output_missing_filename(self):
        """Expected output without filename should fail."""
        response = {
            "stage_id": "stage1",
            "code": "print('hello')",
            "expected_outputs": [
                {
                    "artifact_type": "spectrum_csv",
                    # Missing: filename
                    "description": "Test output",
                }
            ],
            "estimated_runtime_minutes": 5,
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "filename" in str(exc_info.value)

    # --- Material Array Items ---

    def test_material_missing_id(self):
        """Material without id should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [
                {
                    # Missing: id
                    "name": "Gold",
                    "model_type": "tabulated",
                }
            ],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "id" in str(exc_info.value)

    def test_material_missing_model_type(self):
        """Material without model_type should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [
                {
                    "id": "gold",
                    "name": "Gold",
                    # Missing: model_type
                }
            ],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "model_type" in str(exc_info.value)

    # --- Source Array Items ---

    def test_source_missing_type(self):
        """Source without type should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [
                {
                    # Missing: type
                    "center": {"x": 0, "y": 0, "z": 0},
                    "size": {"x": 1, "y": 1, "z": 0},
                }
            ],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "type" in str(exc_info.value)

    def test_source_missing_center(self):
        """Source without center should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [
                {
                    "type": "gaussian",
                    # Missing: center
                    "size": {"x": 1, "y": 1, "z": 0},
                }
            ],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "center" in str(exc_info.value)

    # --- Monitor Array Items ---

    def test_monitor_missing_type(self):
        """Monitor without type should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [
                {
                    # Missing: type
                    "name": "transmission",
                }
            ],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "type" in str(exc_info.value)

    def test_monitor_missing_name(self):
        """Monitor without name should fail."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
            },
            "geometry": {},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [
                {
                    "type": "flux",
                    # Missing: name
                }
            ],
            "performance_estimate": {
                "runtime_estimate_minutes": 10,
                "memory_estimate_gb": 2,
            },
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "name" in str(exc_info.value)

    # --- Per Result Reports Array Items ---

    def test_per_result_report_missing_result_id(self):
        """Per result report without result_id should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [
                {
                    # Missing: result_id
                    "target_figure": "fig1",
                    "quantity": "wavelength",
                    "discrepancy": {
                        "absolute": 10,
                        "relative_percent": 5,
                        "classification": "excellent",
                    },
                }
            ],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "result_id" in str(exc_info.value)

    def test_per_result_report_missing_discrepancy(self):
        """Per result report without discrepancy should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [
                {
                    "result_id": "r1",
                    "target_figure": "fig1",
                    "quantity": "wavelength",
                    # Missing: discrepancy
                }
            ],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "discrepancy" in str(exc_info.value)

    # --- Figure Comparisons Array Items ---

    def test_figure_comparison_missing_paper_figure_id(self):
        """Figure comparison without paper_figure_id should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [
                {
                    # Missing: paper_figure_id
                    "simulated_figure_path": "output.png",
                    "comparison_type": "overlay",
                }
            ],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "paper_figure_id" in str(exc_info.value)

    def test_figure_comparison_missing_comparison_type(self):
        """Figure comparison without comparison_type should fail."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [
                {
                    "paper_figure_id": "fig1",
                    "simulated_figure_path": "output.png",
                    # Missing: comparison_type
                }
            ],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "comparison_type" in str(exc_info.value)

    # --- Conservation Checks Array Items ---

    def test_conservation_check_missing_law(self):
        """Conservation check without law should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [
                {
                    # Missing: law
                    "status": "pass",
                }
            ],
            "value_range_checks": [],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "law" in str(exc_info.value)

    def test_conservation_check_missing_status(self):
        """Conservation check without status should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [
                {
                    "law": "energy: T+R+A=1",
                    # Missing: status
                }
            ],
            "value_range_checks": [],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "status" in str(exc_info.value)

    # --- Value Range Checks Array Items ---

    def test_value_range_check_missing_quantity(self):
        """Value range check without quantity should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [],
            "value_range_checks": [
                {
                    # Missing: quantity
                    "status": "pass",
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "quantity" in str(exc_info.value)

    def test_value_range_check_missing_status(self):
        """Value range check without status should fail."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [],
            "value_range_checks": [
                {
                    "quantity": "transmission",
                    # Missing: status
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "status" in str(exc_info.value)

    # --- Array Item Type Validation ---

    def test_issues_array_contains_non_object(self):
        """Issues array containing non-object should fail."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "issues": ["This is a string, not an object"],  # Should be object
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "type" in error_str or "object" in error_str

    def test_stages_array_contains_non_object(self):
        """Stages array containing non-object should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": ["stage1", "stage2"],  # Should be objects
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "type" in error_str or "object" in error_str


# =============================================================================
# TEST: BOUNDARY VALUES AND SPECIAL CASES
# =============================================================================


class TestBoundaryValuesAndSpecialCases:
    """Test boundary values and special edge cases."""

    # --- Confidence Bounds ---

    def test_confidence_below_minimum(self):
        """Confidence below 0 should fail validation."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "confidence": -0.1,  # Should be >= 0
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "minimum" in error_str or "confidence" in error_str or "-0.1" in error_str

    def test_confidence_above_maximum(self):
        """Confidence above 1 should fail validation."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "confidence": 1.5,  # Should be <= 1
            "summary": "Test",
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "maximum" in error_str or "confidence" in error_str or "1.5" in error_str

    # --- Null vs Missing ---

    def test_null_string_field(self):
        """Null for non-nullable string field should fail."""
        response = {
            "paper_id": None,  # Should be string, not null
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "paper_id" in error_str or "null" in error_str or "type" in error_str

    def test_null_array_field(self):
        """Null for non-nullable array field should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": None,  # Should be array, not null
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert (
            "extracted_parameters" in error_str or "null" in error_str or "type" in error_str
        )

    def test_null_object_field(self):
        """Null for non-nullable object field should fail."""
        response = {
            "paper_id": "test",
            "paper_domain": "plasmonics",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": None,  # Should be object, not null
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "assumptions" in error_str or "null" in error_str or "type" in error_str

    # --- Empty Object Validation ---

    def test_empty_response(self):
        """Completely empty response should fail validation."""
        response = {}
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "required" in error_str or "verdict" in error_str

    def test_empty_response_planner(self):
        """Empty planner response should fail validation."""
        response = {}
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "required" in error_str or "paper_id" in error_str

    def test_empty_response_code_generator(self):
        """Empty code generator response should fail validation."""
        response = {}
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "required" in error_str or "stage_id" in error_str

    def test_empty_response_supervisor(self):
        """Empty supervisor response should fail validation."""
        response = {}
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert "required" in error_str or "verdict" in error_str

    # --- Multiple Violations ---

    def test_multiple_missing_required_fields(self):
        """Response missing multiple required fields should fail with clear error."""
        response = {"summary": "Just a summary"}
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        # Should fail on at least one required field
        error_str = str(exc_info.value)
        assert any(
            field in error_str for field in ["verdict", "checklist_results", "required"]
        )

    def test_multiple_wrong_types(self):
        """Response with multiple wrong types should fail validation."""
        response = {
            "verdict": 123,  # Wrong type
            "checklist_results": "string",  # Wrong type
            "summary": ["array"],  # Wrong type
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        # Should fail on at least one type error
        error_str = str(exc_info.value)
        assert "type" in error_str or "string" in error_str or "object" in error_str

    # --- OneOf Validation (escalate_to_user) ---

    def test_escalate_to_user_valid_boolean(self):
        """escalate_to_user as boolean should be valid."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "escalate_to_user": False,
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        # Should not raise
        validate(instance=response, schema=schema)

    def test_escalate_to_user_valid_string(self):
        """escalate_to_user as string should be valid."""
        response = {
            "verdict": "needs_revision",
            "checklist_results": {
                "coverage": {"status": "fail"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "escalate_to_user": "What parameters should we use?",
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        # Should not raise
        validate(instance=response, schema=schema)

    def test_escalate_to_user_invalid_type(self):
        """escalate_to_user as number should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "escalate_to_user": 42,  # Should be boolean or string
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert (
            "escalate_to_user" in error_str
            or "oneOf" in error_str
            or "valid" in error_str
        )

    def test_escalate_to_user_invalid_array(self):
        """escalate_to_user as array should fail."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"},
            },
            "escalate_to_user": ["question1", "question2"],  # Should be boolean or string
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert (
            "escalate_to_user" in error_str
            or "oneOf" in error_str
            or "valid" in error_str
        )


# =============================================================================
# TEST: PARAMETRIZED TESTS FOR ALL SCHEMAS
# =============================================================================


class TestAllSchemasLoadAndValidate:
    """Ensure all schemas can be loaded and validate basic structure."""

    @pytest.mark.parametrize(
        "schema_name",
        [
            "planner_output_schema.json",
            "plan_reviewer_output_schema.json",
            "simulation_designer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_generator_output_schema.json",
            "code_reviewer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "results_analyzer_output_schema.json",
            "comparison_validator_output_schema.json",
            "supervisor_output_schema.json",
        ],
    )
    def test_schema_loads_without_error(self, schema_name):
        """All schemas should load without error."""
        schema = load_schema(schema_name)
        assert schema is not None
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"

    @pytest.mark.parametrize(
        "schema_name",
        [
            "planner_output_schema.json",
            "plan_reviewer_output_schema.json",
            "simulation_designer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_generator_output_schema.json",
            "code_reviewer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "results_analyzer_output_schema.json",
            "comparison_validator_output_schema.json",
            "supervisor_output_schema.json",
        ],
    )
    def test_empty_object_fails_validation(self, schema_name):
        """Empty object should fail validation for all schemas (have required fields)."""
        schema = load_schema(schema_name)
        with pytest.raises(ValidationError):
            validate(instance={}, schema=schema)

    @pytest.mark.parametrize(
        "schema_name",
        [
            "planner_output_schema.json",
            "plan_reviewer_output_schema.json",
            "simulation_designer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_generator_output_schema.json",
            "code_reviewer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "results_analyzer_output_schema.json",
            "comparison_validator_output_schema.json",
            "supervisor_output_schema.json",
        ],
    )
    def test_schema_has_required_fields(self, schema_name):
        """All schemas should have a 'required' field defining required properties."""
        schema = load_schema(schema_name)
        assert "required" in schema, f"{schema_name} should have 'required' field"
        assert isinstance(schema["required"], list), f"{schema_name} 'required' should be a list"
        assert len(schema["required"]) > 0, f"{schema_name} should have at least one required field"

    @pytest.mark.parametrize(
        "schema_name",
        [
            "planner_output_schema.json",
            "plan_reviewer_output_schema.json",
            "simulation_designer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_generator_output_schema.json",
            "code_reviewer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "results_analyzer_output_schema.json",
            "comparison_validator_output_schema.json",
            "supervisor_output_schema.json",
        ],
    )
    def test_schema_has_properties(self, schema_name):
        """All schemas should have a 'properties' field."""
        schema = load_schema(schema_name)
        assert "properties" in schema, f"{schema_name} should have 'properties' field"
        assert isinstance(schema["properties"], dict), f"{schema_name} 'properties' should be a dict"

    @pytest.mark.parametrize(
        "schema_name,required_fields",
        [
            ("plan_reviewer_output_schema.json", ["verdict", "checklist_results", "summary"]),
            ("code_generator_output_schema.json", ["stage_id", "code", "expected_outputs", "estimated_runtime_minutes"]),
            ("planner_output_schema.json", ["paper_id", "paper_domain", "title", "summary", "extracted_parameters", "targets", "stages", "assumptions", "progress"]),
            ("supervisor_output_schema.json", ["verdict", "validation_hierarchy_status", "main_physics_assessment", "summary"]),
            ("execution_validator_output_schema.json", ["stage_id", "verdict", "execution_status", "files_check", "summary"]),
            ("physics_sanity_output_schema.json", ["stage_id", "verdict", "conservation_checks", "value_range_checks", "summary"]),
            ("results_analyzer_output_schema.json", ["stage_id", "per_result_reports", "figure_comparisons", "overall_classification", "summary"]),
            ("comparison_validator_output_schema.json", ["stage_id", "verdict", "accuracy_check", "math_check", "summary"]),
        ],
    )
    def test_schema_required_fields_match_expected(self, schema_name, required_fields):
        """Schema required fields should match expected list."""
        schema = load_schema(schema_name)
        actual_required = set(schema.get("required", []))
        expected_required = set(required_fields)
        assert expected_required.issubset(actual_required), (
            f"{schema_name}: Expected required fields {expected_required} "
            f"but got {actual_required}. Missing: {expected_required - actual_required}"
        )

