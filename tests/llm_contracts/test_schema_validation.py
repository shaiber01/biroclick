"""Schema validation tests for LLM mock responses.

This module provides comprehensive validation that:
1. All mock response files exist for all defined agents
2. Mock responses fully validate against their schemas
3. Mock responses have complete required fields
4. Field types and values are correct and consistent
5. Cross-field logic is satisfied
6. Mock responses are semantically correct and realistic
"""

import pytest
from jsonschema import validate, ValidationError

from .helpers import AGENT_SCHEMAS, load_mock_response, load_schema, MOCK_RESPONSES_DIR


class TestMockResponseFileExistence:
    """Test that all mock response files exist for all defined agents."""

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_mock_response_file_exists(self, agent_name, schema_file):
        """Each agent in AGENT_SCHEMAS must have a corresponding mock response file.
        
        This test FAILS if a mock response file is missing, rather than skipping.
        Missing mock responses indicate a test fixture gap that must be fixed.
        """
        expected_path = MOCK_RESPONSES_DIR / f"{agent_name}_response.json"
        assert expected_path.exists(), (
            f"Missing mock response file for agent '{agent_name}'. "
            f"Expected file at: {expected_path}. "
            f"Either create the mock response file or remove '{agent_name}' from AGENT_SCHEMAS."
        )

    def test_all_agents_have_schemas(self):
        """All agents in AGENT_SCHEMAS must have valid schema files."""
        for agent_name, schema_file in AGENT_SCHEMAS.items():
            schema = load_schema(schema_file)
            assert isinstance(schema, dict), (
                f"Schema for '{agent_name}' must be a dict, got {type(schema)}"
            )
            assert "type" in schema or "properties" in schema or "$ref" in schema, (
                f"Schema for '{agent_name}' missing required schema elements"
            )


class TestMockResponsesFullSchemaValidation:
    """Strict validation: mock responses MUST fully conform to their schemas."""

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_mock_response_validates_against_schema(self, agent_name, schema_file):
        """Mock response must fully validate against its schema without errors.
        
        This test does NOT skip for missing files - that indicates a bug.
        """
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.fail(
                f"Missing mock response file for '{agent_name}'. "
                f"Create the file or fix AGENT_SCHEMAS."
            )
        
        schema = load_schema(schema_file)
        
        # This should NOT raise ValidationError
        try:
            validate(instance=response, schema=schema)
        except ValidationError as e:
            pytest.fail(
                f"Mock response for '{agent_name}' failed schema validation:\n"
                f"Error: {e.message}\n"
                f"Path: {'.'.join(str(p) for p in e.absolute_path)}\n"
                f"Schema path: {'.'.join(str(p) for p in e.absolute_schema_path)}"
            )
        
        # Additional assertions to verify the response is meaningful
        assert isinstance(response, dict), (
            f"Mock response for '{agent_name}' must be a dict"
        )
        assert len(response) > 0, (
            f"Mock response for '{agent_name}' cannot be empty"
        )


class TestMockResponseRequiredFields:
    """Test that mock responses have all required fields from their schemas."""

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_required_fields_present(self, agent_name, schema_file):
        """All required fields must be present in mock response."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found - covered by existence test")
        
        schema = load_schema(schema_file)
        required_fields = schema.get("required", [])
        
        missing_fields = [field for field in required_fields if field not in response]
        assert not missing_fields, (
            f"Mock response for '{agent_name}' missing required fields: {missing_fields}"
        )

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_required_fields_not_empty(self, agent_name, schema_file):
        """Required fields must have non-empty values."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        schema = load_schema(schema_file)
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            if field in response:
                value = response[field]
                # Check for None
                assert value is not None, (
                    f"Required field '{field}' in '{agent_name}' is None"
                )
                # Check for empty strings (unless schema allows it)
                if isinstance(value, str):
                    field_schema = schema.get("properties", {}).get(field, {})
                    min_length = field_schema.get("minLength", 0)
                    if min_length > 0:
                        assert len(value) >= min_length, (
                            f"Required field '{field}' in '{agent_name}' is too short: "
                            f"length {len(value)} < minLength {min_length}"
                        )


class TestMockResponseFieldTypes:
    """Test that mock response fields have correct types."""

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_field_types_match_schema(self, agent_name, schema_file):
        """Field types must match what the schema expects."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        schema = load_schema(schema_file)
        properties = schema.get("properties", {})
        
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        for field, field_schema in properties.items():
            if field in response:
                value = response[field]
                expected_type = field_schema.get("type")
                
                # Handle null types
                if value is None:
                    nullable_types = field_schema.get("type", [])
                    if isinstance(nullable_types, list):
                        assert "null" in nullable_types, (
                            f"Field '{field}' in '{agent_name}' is None but not nullable"
                        )
                    continue
                
                # Handle union types like ["string", "null"]
                if isinstance(expected_type, list):
                    expected_type = [t for t in expected_type if t != "null"][0]
                
                if expected_type and expected_type in type_map:
                    expected_python_type = type_map[expected_type]
                    assert isinstance(value, expected_python_type), (
                        f"Field '{field}' in '{agent_name}' has wrong type: "
                        f"expected {expected_type}, got {type(value).__name__}"
                    )


class TestMockResponseEnumValues:
    """Test that enum fields have valid values."""

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_enum_values_are_valid(self, agent_name, schema_file):
        """Enum fields must have values from the allowed set."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        schema = load_schema(schema_file)
        properties = schema.get("properties", {})
        
        for field, field_schema in properties.items():
            if field in response and "enum" in field_schema:
                value = response[field]
                valid_values = field_schema["enum"]
                assert value in valid_values, (
                    f"Field '{field}' in '{agent_name}' has invalid enum value: "
                    f"'{value}' not in {valid_values}"
                )


class TestVerdictConsistency:
    """Test verdict field consistency across different agent types."""

    REVIEWER_AGENTS = ["plan_reviewer", "design_reviewer", "code_reviewer"]
    
    @pytest.mark.parametrize("agent_name", REVIEWER_AGENTS)
    def test_reviewer_verdict_is_valid(self, agent_name):
        """Reviewer agents must have valid verdict values."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        verdict = response.get("verdict")
        valid_verdicts = ["approve", "needs_revision"]
        
        assert verdict in valid_verdicts, (
            f"Reviewer '{agent_name}' has invalid verdict: "
            f"'{verdict}' not in {valid_verdicts}"
        )

    def test_supervisor_verdict_is_valid(self):
        """Supervisor agent must have valid verdict value."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        verdict = response.get("verdict")
        valid_verdicts = [
            "ok_continue", "replan_needed", "change_priority",
            "ask_user", "backtrack_to_stage", "all_complete"
        ]
        
        assert verdict in valid_verdicts, (
            f"Supervisor has invalid verdict: '{verdict}' not in {valid_verdicts}"
        )

    def test_execution_validator_verdict_is_valid(self):
        """Execution validator must have valid verdict value."""
        try:
            response = load_mock_response("execution_validator")
        except FileNotFoundError:
            pytest.skip("Execution validator mock response not found")
        
        verdict = response.get("verdict")
        valid_verdicts = ["pass", "warning", "fail"]
        
        assert verdict in valid_verdicts, (
            f"Execution validator has invalid verdict: "
            f"'{verdict}' not in {valid_verdicts}"
        )

    def test_physics_sanity_verdict_is_valid(self):
        """Physics sanity agent must have valid verdict value."""
        try:
            response = load_mock_response("physics_sanity")
        except FileNotFoundError:
            pytest.skip("Physics sanity mock response not found")
        
        verdict = response.get("verdict")
        valid_verdicts = ["pass", "warning", "fail", "design_flaw"]
        
        assert verdict in valid_verdicts, (
            f"Physics sanity has invalid verdict: "
            f"'{verdict}' not in {valid_verdicts}"
        )


class TestReviewerIssuesConsistency:
    """Test that reviewer issues are consistent with verdict."""

    REVIEWER_AGENTS = ["plan_reviewer", "design_reviewer", "code_reviewer"]

    @pytest.mark.parametrize("agent_name", REVIEWER_AGENTS)
    def test_needs_revision_has_issues(self, agent_name):
        """needs_revision verdict must have at least one issue."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        verdict = response.get("verdict")
        issues = response.get("issues", [])
        
        if verdict == "needs_revision":
            assert len(issues) > 0, (
                f"Reviewer '{agent_name}' has 'needs_revision' verdict "
                f"but no issues listed"
            )
            
            # Each issue must have required fields
            for idx, issue in enumerate(issues):
                assert isinstance(issue, dict), (
                    f"Issue {idx} in '{agent_name}' must be a dict"
                )
                assert "severity" in issue, (
                    f"Issue {idx} in '{agent_name}' missing 'severity'"
                )
                assert "category" in issue, (
                    f"Issue {idx} in '{agent_name}' missing 'category'"
                )
                assert "description" in issue, (
                    f"Issue {idx} in '{agent_name}' missing 'description'"
                )
                assert issue["description"], (
                    f"Issue {idx} in '{agent_name}' has empty description"
                )

    @pytest.mark.parametrize("agent_name", REVIEWER_AGENTS)
    def test_approve_has_no_blocking_issues(self, agent_name):
        """approve verdict cannot have blocking issues."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        verdict = response.get("verdict")
        issues = response.get("issues", [])
        
        if verdict == "approve":
            blocking_issues = [
                i for i in issues 
                if isinstance(i, dict) and i.get("severity") == "blocking"
            ]
            assert len(blocking_issues) == 0, (
                f"Reviewer '{agent_name}' has 'approve' verdict "
                f"but {len(blocking_issues)} blocking issue(s)"
            )


class TestSupervisorResponseConsistency:
    """Test supervisor response cross-field consistency."""

    def test_validation_hierarchy_has_all_fields(self):
        """Validation hierarchy must have all required status fields."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        hierarchy = response.get("validation_hierarchy_status", {})
        required_fields = [
            "material_validation", "single_structure",
            "arrays_systems", "parameter_sweeps"
        ]
        
        for field in required_fields:
            assert field in hierarchy, (
                f"Supervisor validation_hierarchy_status missing '{field}'"
            )
            assert hierarchy[field] in ["passed", "partial", "failed", "not_done"], (
                f"Supervisor validation_hierarchy_status.{field} has invalid "
                f"value: '{hierarchy[field]}'"
            )

    def test_physics_assessment_has_all_fields(self):
        """Main physics assessment must have all required boolean fields."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        assessment = response.get("main_physics_assessment", {})
        required_fields = [
            "physics_plausible", "conservation_satisfied",
            "value_ranges_reasonable"
        ]
        
        for field in required_fields:
            assert field in assessment, (
                f"Supervisor main_physics_assessment missing '{field}'"
            )
            assert isinstance(assessment[field], bool), (
                f"Supervisor main_physics_assessment.{field} must be boolean, "
                f"got {type(assessment[field]).__name__}"
            )

    def test_backtrack_verdict_has_backtrack_decision(self):
        """backtrack_to_stage verdict must have backtrack_decision."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        verdict = response.get("verdict")
        
        if verdict == "backtrack_to_stage":
            assert "backtrack_decision" in response, (
                "Supervisor has 'backtrack_to_stage' verdict but no backtrack_decision"
            )
            decision = response["backtrack_decision"]
            assert "accepted" in decision, "backtrack_decision missing 'accepted'"
            assert "target_stage_id" in decision, "backtrack_decision missing 'target_stage_id'"
            assert "stages_to_invalidate" in decision, "backtrack_decision missing 'stages_to_invalidate'"
            assert "reason" in decision, "backtrack_decision missing 'reason'"

    def test_ask_user_verdict_has_user_question(self):
        """ask_user verdict must have user_question field."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        verdict = response.get("verdict")
        
        if verdict == "ask_user":
            assert "user_question" in response, (
                "Supervisor has 'ask_user' verdict but no user_question"
            )
            assert response["user_question"], (
                "Supervisor user_question cannot be empty"
            )

    def test_all_complete_verdict_should_stop(self):
        """all_complete verdict must have should_stop=True."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        verdict = response.get("verdict")
        
        if verdict == "all_complete":
            assert response.get("should_stop") is True, (
                "Supervisor has 'all_complete' verdict but should_stop is not True"
            )

    def test_summary_is_meaningful(self):
        """Supervisor summary must be meaningful (not empty or too short)."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock response not found")
        
        summary = response.get("summary", "")
        assert isinstance(summary, str), "summary must be a string"
        assert len(summary) >= 1, "summary cannot be empty (schema requires minLength: 1)"
        assert len(summary) >= 10, (
            f"summary appears too short to be meaningful: '{summary[:50]}...'"
        )


class TestPlannerResponseConsistency:
    """Test planner response cross-field consistency."""

    def test_stages_have_required_fields(self):
        """Each stage must have all required fields."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock response not found")
        
        stages = response.get("stages", [])
        required_fields = [
            "stage_id", "stage_type", "name",
            "description", "targets", "dependencies"
        ]
        
        for idx, stage in enumerate(stages):
            for field in required_fields:
                assert field in stage, (
                    f"Planner stage {idx} missing required field '{field}'"
                )

    def test_stage_ids_are_unique(self):
        """Stage IDs must be unique within the plan."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock response not found")
        
        stages = response.get("stages", [])
        stage_ids = [s.get("stage_id") for s in stages if isinstance(s, dict)]
        
        assert len(stage_ids) == len(set(stage_ids)), (
            f"Planner has duplicate stage IDs: {stage_ids}"
        )

    def test_dependencies_reference_existing_stages(self):
        """Stage dependencies must reference existing stage IDs."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock response not found")
        
        stages = response.get("stages", [])
        stage_ids = {s.get("stage_id") for s in stages if isinstance(s, dict)}
        
        for stage in stages:
            stage_id = stage.get("stage_id", "unknown")
            dependencies = stage.get("dependencies", [])
            
            for dep in dependencies:
                assert dep in stage_ids, (
                    f"Planner stage '{stage_id}' depends on unknown stage '{dep}'"
                )
                assert dep != stage_id, (
                    f"Planner stage '{stage_id}' cannot depend on itself"
                )

    def test_stage_type_is_valid(self):
        """Stage types must be valid enum values."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock response not found")
        
        stages = response.get("stages", [])
        valid_types = [
            "MATERIAL_VALIDATION", "SINGLE_STRUCTURE", "ARRAY_SYSTEM",
            "PARAMETER_SWEEP", "COMPLEX_PHYSICS"
        ]
        
        for stage in stages:
            stage_type = stage.get("stage_type")
            assert stage_type in valid_types, (
                f"Planner stage '{stage.get('stage_id')}' has invalid "
                f"stage_type: '{stage_type}'"
            )

    def test_extracted_parameters_have_required_fields(self):
        """Extracted parameters must have required fields."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock response not found")
        
        params = response.get("extracted_parameters", [])
        required_fields = ["name", "value", "unit", "source"]
        
        for idx, param in enumerate(params):
            for field in required_fields:
                assert field in param, (
                    f"Planner extracted_parameters[{idx}] missing '{field}'"
                )

    def test_targets_have_required_fields(self):
        """Targets must have required fields."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock response not found")
        
        targets = response.get("targets", [])
        required_fields = ["figure_id", "description", "type", "simulation_class"]
        
        for idx, target in enumerate(targets):
            for field in required_fields:
                assert field in target, (
                    f"Planner targets[{idx}] missing required field '{field}'"
                )


class TestCodeGeneratorConsistency:
    """Test code generator response consistency."""

    def test_code_is_non_empty(self):
        """Generated code must be non-empty and substantial."""
        try:
            response = load_mock_response("code_generator")
        except FileNotFoundError:
            pytest.skip("Code generator mock response not found")
        
        code = response.get("code", "")
        assert isinstance(code, str), "code must be a string"
        assert len(code.strip()) > 0, "code cannot be empty"
        assert len(code) > 100, (
            "code appears too short to be valid simulation code"
        )

    def test_expected_outputs_have_required_fields(self):
        """Expected outputs must have required fields."""
        try:
            response = load_mock_response("code_generator")
        except FileNotFoundError:
            pytest.skip("Code generator mock response not found")
        
        outputs = response.get("expected_outputs", [])
        required_fields = ["artifact_type", "filename", "description"]
        
        assert len(outputs) > 0, "code_generator must have at least one expected_output"
        
        for idx, output in enumerate(outputs):
            for field in required_fields:
                assert field in output, (
                    f"Code generator expected_outputs[{idx}] missing '{field}'"
                )
            assert output.get("filename"), (
                f"Code generator expected_outputs[{idx}] has empty filename"
            )

    def test_safety_checks_all_pass(self):
        """All safety checks must pass in mock response."""
        try:
            response = load_mock_response("code_generator")
        except FileNotFoundError:
            pytest.skip("Code generator mock response not found")
        
        safety = response.get("safety_checks", {})
        required_checks = [
            "no_plt_show", "no_input", "uses_plt_savefig_close",
            "relative_paths_only", "includes_result_json"
        ]
        
        for check in required_checks:
            assert check in safety, (
                f"Code generator safety_checks missing '{check}'"
            )
            assert safety[check] is True, (
                f"Code generator failed safety check '{check}': {safety[check]}"
            )

    def test_runtime_estimate_is_positive(self):
        """Runtime estimate must be positive."""
        try:
            response = load_mock_response("code_generator")
        except FileNotFoundError:
            pytest.skip("Code generator mock response not found")
        
        runtime = response.get("estimated_runtime_minutes", 0)
        assert isinstance(runtime, (int, float)), (
            "estimated_runtime_minutes must be a number"
        )
        assert runtime > 0, (
            f"estimated_runtime_minutes must be positive, got {runtime}"
        )


class TestSimulationDesignerConsistency:
    """Test simulation designer response consistency."""

    def test_unit_system_is_valid(self):
        """Unit system must have valid values."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock response not found")
        
        unit_system = response.get("unit_system", {})
        
        assert "characteristic_length_m" in unit_system, (
            "unit_system missing 'characteristic_length_m'"
        )
        char_length = unit_system["characteristic_length_m"]
        assert isinstance(char_length, (int, float)), (
            "characteristic_length_m must be a number"
        )
        assert char_length > 0, (
            f"characteristic_length_m must be positive, got {char_length}"
        )
        assert char_length < 1, (
            f"characteristic_length_m should be < 1 meter, got {char_length}"
        )

    def test_geometry_has_structures(self):
        """Geometry must have at least one structure."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock response not found")
        
        geometry = response.get("geometry", {})
        structures = geometry.get("structures", [])
        
        assert len(structures) > 0, (
            "Simulation designer geometry must have at least one structure"
        )
        
        for idx, structure in enumerate(structures):
            assert "name" in structure, (
                f"geometry.structures[{idx}] missing 'name'"
            )
            assert "type" in structure, (
                f"geometry.structures[{idx}] missing 'type'"
            )

    def test_materials_are_defined(self):
        """Materials must be defined and valid."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock response not found")
        
        materials = response.get("materials", [])
        
        assert len(materials) > 0, (
            "Simulation designer must have at least one material"
        )
        
        material_ids = set()
        for idx, material in enumerate(materials):
            assert "id" in material, f"materials[{idx}] missing 'id'"
            assert "name" in material, f"materials[{idx}] missing 'name'"
            assert "model_type" in material, f"materials[{idx}] missing 'model_type'"
            
            # Material IDs must be unique
            mat_id = material["id"]
            assert mat_id not in material_ids, (
                f"Duplicate material ID: '{mat_id}'"
            )
            material_ids.add(mat_id)

    def test_sources_are_defined(self):
        """Sources must be defined."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock response not found")
        
        sources = response.get("sources", [])
        
        assert len(sources) > 0, (
            "Simulation designer must have at least one source"
        )
        
        for idx, source in enumerate(sources):
            assert "type" in source, f"sources[{idx}] missing 'type'"

    def test_monitors_are_defined(self):
        """Monitors must be defined."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock response not found")
        
        monitors = response.get("monitors", [])
        
        assert len(monitors) > 0, (
            "Simulation designer must have at least one monitor"
        )
        
        for idx, monitor in enumerate(monitors):
            assert "type" in monitor, f"monitors[{idx}] missing 'type'"
            assert "name" in monitor, f"monitors[{idx}] missing 'name'"


class TestExecutionValidatorConsistency:
    """Test execution validator response consistency."""

    def test_execution_status_is_consistent(self):
        """Execution status must be consistent with verdict."""
        try:
            response = load_mock_response("execution_validator")
        except FileNotFoundError:
            pytest.skip("Execution validator mock response not found")
        
        verdict = response.get("verdict")
        exec_status = response.get("execution_status", {})
        completed = exec_status.get("completed")
        
        # If not completed, verdict must be fail
        if completed is False:
            assert verdict == "fail", (
                f"Verdict should be 'fail' if execution not completed, got '{verdict}'"
            )

    def test_files_check_is_consistent(self):
        """Files check must be internally consistent."""
        try:
            response = load_mock_response("execution_validator")
        except FileNotFoundError:
            pytest.skip("Execution validator mock response not found")
        
        files_check = response.get("files_check", {})
        expected = set(files_check.get("expected_files", []))
        found = set(files_check.get("found_files", []))
        missing = set(files_check.get("missing_files", []))
        all_present = files_check.get("all_present")
        
        # Missing files should be expected - found
        computed_missing = expected - found
        assert missing == computed_missing, (
            f"missing_files ({missing}) doesn't match expected - found ({computed_missing})"
        )
        
        # all_present should match whether missing is empty
        if expected and expected.issubset(found):
            assert all_present is True, (
                "all_present should be True when all expected files are found"
            )
        elif missing:
            assert all_present is False, (
                "all_present should be False when files are missing"
            )


class TestResultsAnalyzerConsistency:
    """Test results analyzer response consistency."""

    def test_per_result_reports_have_required_fields(self):
        """Per-result reports must have required fields."""
        try:
            response = load_mock_response("results_analyzer")
        except FileNotFoundError:
            pytest.skip("Results analyzer mock response not found")
        
        reports = response.get("per_result_reports", [])
        required_fields = ["result_id", "target_figure", "quantity", "discrepancy"]
        
        for idx, report in enumerate(reports):
            for field in required_fields:
                assert field in report, (
                    f"Results analyzer per_result_reports[{idx}] missing '{field}'"
                )

    def test_discrepancy_classification_is_valid(self):
        """Discrepancy classification must be valid."""
        try:
            response = load_mock_response("results_analyzer")
        except FileNotFoundError:
            pytest.skip("Results analyzer mock response not found")
        
        reports = response.get("per_result_reports", [])
        valid_classifications = ["excellent", "acceptable", "investigate", "unacceptable"]
        
        for idx, report in enumerate(reports):
            discrepancy = report.get("discrepancy", {})
            classification = discrepancy.get("classification")
            
            if classification:
                assert classification in valid_classifications, (
                    f"Results analyzer per_result_reports[{idx}] has invalid "
                    f"discrepancy classification: '{classification}'"
                )

    def test_overall_classification_is_valid(self):
        """Overall classification must be valid."""
        try:
            response = load_mock_response("results_analyzer")
        except FileNotFoundError:
            pytest.skip("Results analyzer mock response not found")
        
        classification = response.get("overall_classification")
        valid_classifications = [
            "EXCELLENT_MATCH", "ACCEPTABLE_MATCH", "PARTIAL_MATCH",
            "POOR_MATCH", "FAILED"
        ]
        
        assert classification in valid_classifications, (
            f"Results analyzer has invalid overall_classification: '{classification}'"
        )

    def test_confidence_is_in_range(self):
        """Confidence must be between 0 and 1."""
        try:
            response = load_mock_response("results_analyzer")
        except FileNotFoundError:
            pytest.skip("Results analyzer mock response not found")
        
        confidence = response.get("confidence")
        
        if confidence is not None:
            assert isinstance(confidence, (int, float)), (
                "confidence must be a number"
            )
            assert 0 <= confidence <= 1, (
                f"confidence must be between 0 and 1, got {confidence}"
            )


class TestPhysicsSanityConsistency:
    """Test physics sanity response consistency."""

    def test_conservation_checks_have_required_fields(self):
        """Conservation checks must have required fields."""
        try:
            response = load_mock_response("physics_sanity")
        except FileNotFoundError:
            pytest.skip("Physics sanity mock response not found")
        
        checks = response.get("conservation_checks", [])
        
        for idx, check in enumerate(checks):
            assert "law" in check, (
                f"Physics sanity conservation_checks[{idx}] missing 'law'"
            )
            assert "status" in check, (
                f"Physics sanity conservation_checks[{idx}] missing 'status'"
            )
            assert check["status"] in ["pass", "warning", "fail"], (
                f"Physics sanity conservation_checks[{idx}] has invalid status"
            )

    def test_value_range_checks_have_required_fields(self):
        """Value range checks must have required fields."""
        try:
            response = load_mock_response("physics_sanity")
        except FileNotFoundError:
            pytest.skip("Physics sanity mock response not found")
        
        checks = response.get("value_range_checks", [])
        
        for idx, check in enumerate(checks):
            assert "quantity" in check, (
                f"Physics sanity value_range_checks[{idx}] missing 'quantity'"
            )
            assert "status" in check, (
                f"Physics sanity value_range_checks[{idx}] missing 'status'"
            )
            assert check["status"] in ["pass", "warning", "fail"], (
                f"Physics sanity value_range_checks[{idx}] has invalid status"
            )

    def test_pass_verdict_has_no_failures(self):
        """Pass verdict should not have failed checks."""
        try:
            response = load_mock_response("physics_sanity")
        except FileNotFoundError:
            pytest.skip("Physics sanity mock response not found")
        
        verdict = response.get("verdict")
        
        if verdict == "pass":
            # Check conservation
            conservation = response.get("conservation_checks", [])
            failed_conservation = [
                c for c in conservation 
                if isinstance(c, dict) and c.get("status") == "fail"
            ]
            assert not failed_conservation, (
                f"Pass verdict but {len(failed_conservation)} failed conservation check(s)"
            )
            
            # Check value ranges
            ranges = response.get("value_range_checks", [])
            failed_ranges = [
                r for r in ranges 
                if isinstance(r, dict) and r.get("status") == "fail"
            ]
            assert not failed_ranges, (
                f"Pass verdict but {len(failed_ranges)} failed value range check(s)"
            )


class TestStageIdConsistency:
    """Test stage_id consistency across different agent responses."""

    AGENTS_WITH_STAGE_ID = [
        "simulation_designer", "code_generator", "code_reviewer",
        "design_reviewer", "execution_validator", "physics_sanity",
        "results_analyzer"
    ]

    @pytest.mark.parametrize("agent_name", AGENTS_WITH_STAGE_ID)
    def test_stage_id_is_present_and_valid(self, agent_name):
        """Stage ID must be present and non-empty."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        stage_id = response.get("stage_id")
        
        assert stage_id is not None, (
            f"'{agent_name}' response missing 'stage_id'"
        )
        assert isinstance(stage_id, str), (
            f"'{agent_name}' stage_id must be a string"
        )
        assert len(stage_id.strip()) > 0, (
            f"'{agent_name}' stage_id cannot be empty"
        )


class TestSummaryFieldQuality:
    """Test that summary fields are meaningful across all agents."""

    AGENTS_WITH_SUMMARY = [
        "supervisor", "execution_validator", "physics_sanity",
        "results_analyzer", "plan_reviewer", "design_reviewer", 
        "code_reviewer"
    ]

    @pytest.mark.parametrize("agent_name", AGENTS_WITH_SUMMARY)
    def test_summary_is_meaningful(self, agent_name):
        """Summary fields must be non-empty and meaningful."""
        try:
            response = load_mock_response(agent_name)
        except FileNotFoundError:
            pytest.skip(f"Mock response for '{agent_name}' not found")
        
        summary = response.get("summary")
        
        if summary is not None:
            assert isinstance(summary, str), (
                f"'{agent_name}' summary must be a string"
            )
            assert len(summary.strip()) > 0, (
                f"'{agent_name}' summary cannot be empty"
            )
            # A meaningful summary should be at least a sentence
            assert len(summary) >= 20, (
                f"'{agent_name}' summary appears too short: '{summary[:50]}...'"
            )
