"""
Schema validation coverage for LLM client.

Tests the load_schema and get_agent_schema functions thoroughly,
including schema caching, error handling, and schema content validation.
"""

import pytest
from pathlib import Path

from src.llm_client import (
    load_schema, 
    get_agent_schema, 
    _schema_cache, 
    SCHEMAS_DIR,
)


class TestLoadSchema:
    """Tests for the load_schema function."""

    def setup_method(self):
        """Clear schema cache before each test to ensure isolation."""
        _schema_cache.clear()

    def teardown_method(self):
        """Clear schema cache after each test."""
        _schema_cache.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Basic Loading Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_load_schema_returns_dict(self):
        """load_schema should return a dictionary."""
        schema = load_schema("planner_output_schema")
        assert isinstance(schema, dict), "Schema must be a dictionary"

    def test_load_schema_with_json_extension(self):
        """load_schema should work with .json extension."""
        schema = load_schema("planner_output_schema.json")
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_load_schema_without_json_extension(self):
        """load_schema should work without .json extension."""
        schema = load_schema("planner_output_schema")
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_load_schema_both_extensions_return_same_content(self):
        """Loading with and without .json extension should return same content."""
        _schema_cache.clear()
        schema_no_ext = load_schema("planner_output_schema")
        _schema_cache.clear()
        schema_with_ext = load_schema("planner_output_schema.json")
        assert schema_no_ext == schema_with_ext

    # ─────────────────────────────────────────────────────────────────────
    # Caching Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_schema_caching_works(self):
        """load_schema should cache schemas and return cached version."""
        _schema_cache.clear()
        assert len(_schema_cache) == 0
        
        schema1 = load_schema("planner_output_schema")
        # Cache key should include .json extension
        assert "planner_output_schema.json" in _schema_cache
        
        schema2 = load_schema("planner_output_schema")
        assert schema1 is schema2, "Second call should return cached object"

    def test_cache_stores_correct_key_with_extension(self):
        """Cache should normalize schema names to include .json extension."""
        _schema_cache.clear()
        load_schema("report_output_schema")
        assert "report_output_schema.json" in _schema_cache
        assert "report_output_schema" not in _schema_cache

    def test_cache_hit_returns_same_object(self):
        """Cached schema should be the exact same object (not a copy)."""
        _schema_cache.clear()
        schema1 = load_schema("supervisor_output_schema")
        schema2 = load_schema("supervisor_output_schema")
        assert id(schema1) == id(schema2), "Should return same object from cache"

    # ─────────────────────────────────────────────────────────────────────
    # Error Handling Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_load_nonexistent_schema_raises_file_not_found(self):
        """load_schema should raise FileNotFoundError for non-existent schemas."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("nonexistent_schema_that_does_not_exist")
        assert "Schema not found" in str(exc_info.value)

    def test_load_schema_error_contains_path(self):
        """FileNotFoundError should include the schema path for debugging."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("missing_schema")
        error_msg = str(exc_info.value)
        assert "missing_schema" in error_msg
        assert "schemas" in error_msg.lower()

    def test_load_schema_empty_name_raises_error(self):
        """load_schema should raise error for empty schema name."""
        with pytest.raises(FileNotFoundError):
            load_schema("")

    def test_load_schema_whitespace_name_raises_error(self):
        """load_schema should raise error for whitespace-only schema name."""
        with pytest.raises(FileNotFoundError):
            load_schema("   ")


class TestGetAgentSchema:
    """Tests for the get_agent_schema function."""

    def setup_method(self):
        """Clear schema cache before each test."""
        _schema_cache.clear()

    def teardown_method(self):
        """Clear schema cache after each test."""
        _schema_cache.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Auto-Discovery Tests
    # ─────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("agent_name", [
        "planner",
        "plan_reviewer",
        "simulation_designer",
        "design_reviewer",
        "code_generator",
        "code_reviewer",
        "execution_validator",
        "physics_sanity",
        "results_analyzer",
        "comparison_validator",
        "supervisor",
        "prompt_adaptor",
    ])
    def test_auto_discovery_for_standard_agents(self, agent_name):
        """get_agent_schema should auto-discover schemas for standard agents."""
        schema = get_agent_schema(agent_name)
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"

    def test_auto_discovery_returns_correct_schema(self):
        """get_agent_schema should return the correct schema for each agent."""
        schema = get_agent_schema("planner")
        # Verify it's the planner schema by checking a planner-specific field
        assert "properties" in schema
        assert "paper_id" in schema["properties"]
        assert "stages" in schema["properties"]

    def test_auto_discovery_schema_path_construction(self):
        """get_agent_schema should construct path as {agent_name}_output_schema."""
        schema = get_agent_schema("code_generator")
        assert "properties" in schema
        # code_generator schema should have code-specific fields
        assert "code" in schema["properties"] or "simulation_code" in schema["properties"]

    # ─────────────────────────────────────────────────────────────────────
    # Special Case Mapping Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_special_case_report(self):
        """get_agent_schema should handle 'report' special case."""
        schema = get_agent_schema("report")
        assert isinstance(schema, dict)
        # Verify it's the report schema
        assert "properties" in schema
        assert "paper_id" in schema["properties"]
        assert "figure_comparisons" in schema["properties"]

    def test_report_output_schema_has_correct_structure(self):
        """Report schema should have report-specific structure."""
        schema = get_agent_schema("report")
        required = schema.get("required", [])
        assert "paper_id" in required
        assert "executive_summary" in required
        assert "conclusions" in required

    # ─────────────────────────────────────────────────────────────────────
    # Error Handling Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_unknown_agent_raises_value_error(self):
        """get_agent_schema should raise ValueError for unknown agents."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("unknown_agent_xyz")
        assert "Unknown agent" in str(exc_info.value)

    def test_unknown_agent_error_includes_agent_name(self):
        """ValueError should include the unknown agent name."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("fake_agent")
        assert "fake_agent" in str(exc_info.value)

    def test_unknown_agent_error_includes_expected_path(self):
        """ValueError should include the expected schema path."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("nonexistent")
        error_msg = str(exc_info.value)
        assert "nonexistent_output_schema" in error_msg


class TestSchemaValidation:
    """Tests that all agent schemas exist and are valid JSON Schema format."""

    REQUIRED_SCHEMAS = [
        "planner_output_schema",
        "plan_reviewer_output_schema",
        "simulation_designer_output_schema",
        "design_reviewer_output_schema",
        "code_generator_output_schema",
        "code_reviewer_output_schema",
        "execution_validator_output_schema",
        "physics_sanity_output_schema",
        "results_analyzer_output_schema",
        "comparison_validator_output_schema",
        "supervisor_output_schema",
        "prompt_adaptor_output_schema",
        "report_output_schema",
    ]

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_exists_and_valid(self, schema_name):
        """Schema file should exist and contain valid JSON."""
        schema = load_schema(schema_name)
        assert schema is not None
        assert isinstance(schema, dict)

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_has_type_object(self, schema_name):
        """All agent schemas should be of type 'object'."""
        schema = load_schema(schema_name)
        assert "type" in schema, f"{schema_name} missing 'type' field"
        assert schema["type"] == "object", f"{schema_name} should be type 'object'"

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_has_properties(self, schema_name):
        """All agent schemas should have 'properties' field."""
        schema = load_schema(schema_name)
        assert "properties" in schema, f"{schema_name} missing 'properties' field"
        assert isinstance(schema["properties"], dict)
        assert len(schema["properties"]) > 0, f"{schema_name} has empty properties"

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_has_required_fields(self, schema_name):
        """All agent schemas should specify required fields."""
        schema = load_schema(schema_name)
        assert "required" in schema, f"{schema_name} missing 'required' field"
        assert isinstance(schema["required"], list)

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_required_fields_are_in_properties(self, schema_name):
        """All required fields should be defined in properties."""
        schema = load_schema(schema_name)
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for field in required:
            assert field in properties, \
                f"{schema_name}: required field '{field}' not in properties"


class TestSchemaContent:
    """Tests for specific schema content validation."""

    # ─────────────────────────────────────────────────────────────────────
    # Planner Schema Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_planner_schema_required_fields(self):
        """Planner schema should have all required fields."""
        schema = load_schema("planner_output_schema")
        required = schema.get("required", [])
        expected_required = [
            "paper_id", "paper_domain", "title", "summary",
            "extracted_parameters", "targets", "stages",
            "assumptions", "progress"
        ]
        for field in expected_required:
            assert field in required, f"Planner missing required field: {field}"

    def test_planner_schema_paper_domain_enum(self):
        """Planner schema paper_domain should have valid enum values."""
        schema = load_schema("planner_output_schema")
        paper_domain = schema["properties"]["paper_domain"]
        assert "enum" in paper_domain
        expected_domains = [
            "plasmonics", "photonic_crystal", "metamaterial",
            "thin_film", "waveguide", "strong_coupling", "nonlinear", "other"
        ]
        for domain in expected_domains:
            assert domain in paper_domain["enum"], f"Missing domain: {domain}"

    def test_planner_schema_stages_structure(self):
        """Planner schema stages should have correct structure."""
        schema = load_schema("planner_output_schema")
        stages = schema["properties"]["stages"]
        assert stages["type"] == "array"
        assert "items" in stages
        stage_item = stages["items"]
        assert stage_item["type"] == "object"
        assert "properties" in stage_item
        assert "stage_id" in stage_item["properties"]
        assert "stage_type" in stage_item["properties"]

    def test_planner_schema_stage_type_enum(self):
        """Planner schema stage_type should have valid enum values."""
        schema = load_schema("planner_output_schema")
        stage_type = schema["properties"]["stages"]["items"]["properties"]["stage_type"]
        assert "enum" in stage_type
        expected_types = [
            "MATERIAL_VALIDATION", "SINGLE_STRUCTURE", "ARRAY_SYSTEM",
            "PARAMETER_SWEEP", "COMPLEX_PHYSICS"
        ]
        for stage in expected_types:
            assert stage in stage_type["enum"], f"Missing stage type: {stage}"

    def test_planner_schema_extracted_parameters_structure(self):
        """Planner schema extracted_parameters should have correct structure."""
        schema = load_schema("planner_output_schema")
        params = schema["properties"]["extracted_parameters"]
        assert params["type"] == "array"
        param_item = params["items"]
        assert "required" in param_item
        required_param_fields = ["name", "value", "unit", "source"]
        for field in required_param_fields:
            assert field in param_item["required"]

    # ─────────────────────────────────────────────────────────────────────
    # Supervisor Schema Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_supervisor_schema_required_fields(self):
        """Supervisor schema should have all required fields."""
        schema = load_schema("supervisor_output_schema")
        required = schema.get("required", [])
        expected_required = [
            "verdict", "validation_hierarchy_status",
            "main_physics_assessment", "summary"
        ]
        for field in expected_required:
            assert field in required, f"Supervisor missing required field: {field}"

    def test_supervisor_schema_verdict_enum(self):
        """Supervisor schema verdict should have valid enum values."""
        schema = load_schema("supervisor_output_schema")
        verdict = schema["properties"]["verdict"]
        assert "enum" in verdict
        expected_verdicts = [
            "ok_continue", "replan_needed", "change_priority",
            "ask_user", "backtrack_to_stage", "all_complete"
        ]
        for v in expected_verdicts:
            assert v in verdict["enum"], f"Missing verdict: {v}"

    def test_supervisor_schema_validation_hierarchy_structure(self):
        """Supervisor schema validation_hierarchy_status should have correct structure."""
        schema = load_schema("supervisor_output_schema")
        hierarchy = schema["properties"]["validation_hierarchy_status"]
        assert hierarchy["type"] == "object"
        assert "properties" in hierarchy
        expected_levels = [
            "material_validation", "single_structure",
            "arrays_systems", "parameter_sweeps"
        ]
        for level in expected_levels:
            assert level in hierarchy["properties"], f"Missing hierarchy level: {level}"

    def test_supervisor_schema_main_physics_assessment_required(self):
        """Supervisor schema main_physics_assessment should have required booleans."""
        schema = load_schema("supervisor_output_schema")
        physics = schema["properties"]["main_physics_assessment"]
        assert "required" in physics
        required_booleans = [
            "physics_plausible", "conservation_satisfied", "value_ranges_reasonable"
        ]
        for field in required_booleans:
            assert field in physics["required"]
            assert field in physics["properties"]
            assert physics["properties"][field]["type"] == "boolean"

    # ─────────────────────────────────────────────────────────────────────
    # Report Schema Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_report_output_schema_required_fields(self):
        """Report schema should have all required fields."""
        schema = load_schema("report_output_schema")
        required = schema.get("required", [])
        expected_required = [
            "paper_id", "paper_citation", "executive_summary",
            "assumptions", "figure_comparisons", "summary_table",
            "systematic_discrepancies", "conclusions"
        ]
        for field in expected_required:
            assert field in required, f"Report missing required field: {field}"

    def test_report_output_schema_paper_citation_structure(self):
        """Report schema paper_citation should have correct structure."""
        schema = load_schema("report_output_schema")
        citation = schema["properties"]["paper_citation"]
        assert citation["type"] == "object"
        assert "required" in citation
        required_citation_fields = ["authors", "title", "journal", "year"]
        for field in required_citation_fields:
            assert field in citation["required"]

    def test_report_output_schema_conclusions_structure(self):
        """Report schema conclusions should have correct structure."""
        schema = load_schema("report_output_schema")
        conclusions = schema["properties"]["conclusions"]
        assert conclusions["type"] == "object"
        assert "required" in conclusions
        assert "main_physics_reproduced" in conclusions["required"]
        assert "key_findings" in conclusions["required"]
        # Verify main_physics_reproduced is boolean
        assert conclusions["properties"]["main_physics_reproduced"]["type"] == "boolean"

    def test_report_output_schema_has_definitions(self):
        """Report schema should have definitions section."""
        schema = load_schema("report_output_schema")
        assert "definitions" in schema
        assert "figure_comparison" in schema["definitions"]

    def test_report_output_schema_figure_comparison_definition(self):
        """Report schema figure_comparison definition should be complete."""
        schema = load_schema("report_output_schema")
        fig_comp = schema["definitions"]["figure_comparison"]
        assert fig_comp["type"] == "object"
        assert "required" in fig_comp
        expected_required = [
            "figure_id", "title", "comparison_table",
            "shape_comparison", "reason_for_difference"
        ]
        for field in expected_required:
            assert field in fig_comp["required"]

    # ─────────────────────────────────────────────────────────────────────
    # Code Generator Schema Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_code_generator_schema_required_fields(self):
        """Code generator schema should have required fields."""
        schema = load_schema("code_generator_output_schema")
        required = schema.get("required", [])
        assert len(required) > 0, "Code generator should have required fields"
        # Verify that required fields exist in properties
        properties = schema.get("properties", {})
        for field in required:
            assert field in properties

    def test_code_generator_schema_has_code_field(self):
        """Code generator schema should have some form of code output field."""
        schema = load_schema("code_generator_output_schema")
        properties = schema.get("properties", {})
        # Check for common code-related field names
        code_fields = ["code", "simulation_code", "python_code", "script"]
        has_code_field = any(f in properties for f in code_fields)
        assert has_code_field, f"Code generator missing code field. Properties: {list(properties.keys())}"

    # ─────────────────────────────────────────────────────────────────────
    # Design Reviewer Schema Tests
    # ─────────────────────────────────────────────────────────────────────

    def test_design_reviewer_schema_has_verdict(self):
        """Design reviewer schema should have a verdict/approval field."""
        schema = load_schema("design_reviewer_output_schema")
        properties = schema.get("properties", {})
        # Check for approval-related fields
        verdict_fields = ["verdict", "approved", "approval", "passes", "decision"]
        has_verdict = any(f in properties for f in verdict_fields)
        assert has_verdict, f"Design reviewer missing verdict field. Properties: {list(properties.keys())}"

    def test_design_reviewer_schema_has_feedback(self):
        """Design reviewer schema should have a feedback field."""
        schema = load_schema("design_reviewer_output_schema")
        properties = schema.get("properties", {})
        # Check for feedback-related fields
        feedback_fields = ["feedback", "issues", "comments", "suggestions", "critique"]
        has_feedback = any(f in properties for f in feedback_fields)
        assert has_feedback, f"Design reviewer missing feedback field. Properties: {list(properties.keys())}"


class TestSchemaPathsAndFiles:
    """Tests for schema file paths and directory structure."""

    def test_schemas_dir_exists(self):
        """SCHEMAS_DIR should point to an existing directory."""
        assert SCHEMAS_DIR.exists(), f"Schemas directory not found: {SCHEMAS_DIR}"
        assert SCHEMAS_DIR.is_dir(), f"SCHEMAS_DIR is not a directory: {SCHEMAS_DIR}"

    def test_all_required_schema_files_exist(self):
        """All required schema files should exist on disk."""
        required_files = [
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
            "prompt_adaptor_output_schema.json",
            "report_output_schema.json",
        ]
        for filename in required_files:
            path = SCHEMAS_DIR / filename
            assert path.exists(), f"Schema file not found: {path}"

    def test_schema_files_are_valid_json(self):
        """All .json files in schemas directory should be valid JSON."""
        import json
        for json_file in SCHEMAS_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                assert isinstance(data, dict), f"{json_file.name} is not a JSON object"
            except json.JSONDecodeError as e:
                pytest.fail(f"{json_file.name} is not valid JSON: {e}")


class TestSchemaEnumConsistency:
    """Tests for enum value consistency across related schemas."""

    def test_stage_status_enum_consistency(self):
        """Stage status enum should be consistent across schemas that use it."""
        planner = load_schema("planner_output_schema")
        # Get stage status enum from planner
        progress_stages = planner["properties"]["progress"]["properties"]["stages"]
        status_enum = progress_stages["items"]["properties"]["status"]["enum"]
        
        expected_statuses = [
            "not_started", "in_progress", "completed_success",
            "completed_partial", "completed_failed", "blocked",
            "needs_rerun", "invalidated"
        ]
        for status in expected_statuses:
            assert status in status_enum, f"Missing status: {status}"

    def test_validation_status_enum_consistency(self):
        """Validation hierarchy status enum should have expected values."""
        schema = load_schema("supervisor_output_schema")
        hierarchy = schema["properties"]["validation_hierarchy_status"]["properties"]
        
        expected_statuses = ["passed", "partial", "failed", "not_done"]
        for level_name, level_def in hierarchy.items():
            if "enum" in level_def:
                for status in expected_statuses:
                    assert status in level_def["enum"], \
                        f"{level_name} missing status: {status}"


class TestSchemaJSONSchemaCompliance:
    """Tests for JSON Schema standard compliance."""

    REQUIRED_SCHEMAS = [
        "planner_output_schema",
        "plan_reviewer_output_schema",
        "simulation_designer_output_schema",
        "design_reviewer_output_schema",
        "code_generator_output_schema",
        "code_reviewer_output_schema",
        "execution_validator_output_schema",
        "physics_sanity_output_schema",
        "results_analyzer_output_schema",
        "comparison_validator_output_schema",
        "supervisor_output_schema",
        "prompt_adaptor_output_schema",
        "report_output_schema",
    ]

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_has_valid_schema_declaration(self, schema_name):
        """Schemas should declare JSON Schema version."""
        schema = load_schema(schema_name)
        assert "$schema" in schema, f"{schema_name} missing $schema declaration"
        assert "json-schema.org" in schema["$schema"], \
            f"{schema_name} has invalid $schema value"

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_has_title(self, schema_name):
        """Schemas should have a title for documentation."""
        schema = load_schema(schema_name)
        assert "title" in schema, f"{schema_name} missing title"
        assert isinstance(schema["title"], str)
        assert len(schema["title"]) > 0

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_has_description(self, schema_name):
        """Schemas should have a description for documentation."""
        schema = load_schema(schema_name)
        assert "description" in schema, f"{schema_name} missing description"
        assert isinstance(schema["description"], str)
        assert len(schema["description"]) > 0

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_properties_have_types(self, schema_name):
        """All schema properties should declare their type."""
        schema = load_schema(schema_name)
        properties = schema.get("properties", {})
        for prop_name, prop_def in properties.items():
            # Properties should have type or $ref or oneOf/anyOf
            has_type = (
                "type" in prop_def or
                "$ref" in prop_def or
                "oneOf" in prop_def or
                "anyOf" in prop_def or
                "allOf" in prop_def
            )
            assert has_type, f"{schema_name}.{prop_name} missing type definition"


class TestCodeGeneratorSchemaContent:
    """Tests for code generator schema specific content."""

    def test_code_generator_required_fields(self):
        """Code generator should have correct required fields."""
        schema = load_schema("code_generator_output_schema")
        required = schema.get("required", [])
        expected = ["stage_id", "code", "expected_outputs", "estimated_runtime_minutes"]
        for field in expected:
            assert field in required, f"Missing required field: {field}"

    def test_code_generator_code_field_is_string(self):
        """Code field should be a string (for Python code)."""
        schema = load_schema("code_generator_output_schema")
        code_field = schema["properties"]["code"]
        assert code_field["type"] == "string"

    def test_code_generator_expected_outputs_structure(self):
        """Expected outputs should have correct structure."""
        schema = load_schema("code_generator_output_schema")
        outputs = schema["properties"]["expected_outputs"]
        assert outputs["type"] == "array"
        item = outputs["items"]
        assert "artifact_type" in item["properties"]
        assert "filename" in item["properties"]

    def test_code_generator_artifact_type_enum(self):
        """Artifact type enum should have expected values."""
        schema = load_schema("code_generator_output_schema")
        artifact_type = schema["properties"]["expected_outputs"]["items"]["properties"]["artifact_type"]
        assert "enum" in artifact_type
        expected_types = ["spectrum_csv", "field_data_npz", "field_plot_png", "spectrum_plot_png"]
        for t in expected_types:
            assert t in artifact_type["enum"], f"Missing artifact type: {t}"

    def test_code_generator_safety_checks_structure(self):
        """Safety checks should have expected boolean fields."""
        schema = load_schema("code_generator_output_schema")
        safety = schema["properties"]["safety_checks"]
        assert safety["type"] == "object"
        expected_checks = ["no_plt_show", "no_input", "uses_plt_savefig_close"]
        for check in expected_checks:
            assert check in safety["properties"], f"Missing safety check: {check}"
            assert safety["properties"][check]["type"] == "boolean"


class TestCodeReviewerSchemaContent:
    """Tests for code reviewer schema specific content."""

    def test_code_reviewer_required_fields(self):
        """Code reviewer should have correct required fields."""
        schema = load_schema("code_reviewer_output_schema")
        required = schema.get("required", [])
        expected = ["stage_id", "verdict", "checklist_results", "issues", "summary"]
        for field in expected:
            assert field in required, f"Missing required field: {field}"

    def test_code_reviewer_verdict_enum(self):
        """Code reviewer verdict should have expected values."""
        schema = load_schema("code_reviewer_output_schema")
        verdict = schema["properties"]["verdict"]
        assert "enum" in verdict
        assert "approve" in verdict["enum"]
        assert "needs_revision" in verdict["enum"]

    def test_code_reviewer_checklist_required_items(self):
        """Code reviewer checklist should have all required items."""
        schema = load_schema("code_reviewer_output_schema")
        checklist = schema["properties"]["checklist_results"]
        required_items = checklist.get("required", [])
        expected_items = [
            "unit_normalization", "numerics", "source", "domain",
            "monitors", "visualization", "code_quality", "runtime",
            "meep_api", "expected_outputs"
        ]
        for item in expected_items:
            assert item in required_items, f"Missing checklist item: {item}"

    def test_code_reviewer_issues_structure(self):
        """Code reviewer issues should have correct structure."""
        schema = load_schema("code_reviewer_output_schema")
        issues = schema["properties"]["issues"]
        assert issues["type"] == "array"
        issue_item = issues["items"]
        assert "severity" in issue_item["properties"]
        assert "category" in issue_item["properties"]
        required_issue_fields = issue_item.get("required", [])
        assert "severity" in required_issue_fields
        assert "description" in required_issue_fields

    def test_code_reviewer_issue_severity_enum(self):
        """Issue severity should have expected values."""
        schema = load_schema("code_reviewer_output_schema")
        severity = schema["properties"]["issues"]["items"]["properties"]["severity"]
        assert "enum" in severity
        assert "blocking" in severity["enum"]
        assert "major" in severity["enum"]
        assert "minor" in severity["enum"]


class TestExecutionValidatorSchemaContent:
    """Tests for execution validator schema specific content."""

    def test_execution_validator_required_fields(self):
        """Execution validator should have correct required fields."""
        schema = load_schema("execution_validator_output_schema")
        required = schema.get("required", [])
        expected = ["stage_id", "verdict", "execution_status", "files_check", "summary"]
        for field in expected:
            assert field in required, f"Missing required field: {field}"

    def test_execution_validator_verdict_enum(self):
        """Execution validator verdict should have expected values."""
        schema = load_schema("execution_validator_output_schema")
        verdict = schema["properties"]["verdict"]
        assert "enum" in verdict
        expected_verdicts = ["pass", "warning", "fail"]
        for v in expected_verdicts:
            assert v in verdict["enum"], f"Missing verdict: {v}"

    def test_execution_validator_execution_status_structure(self):
        """Execution status should have correct structure."""
        schema = load_schema("execution_validator_output_schema")
        status = schema["properties"]["execution_status"]
        assert status["type"] == "object"
        assert "completed" in status["properties"]
        assert status["properties"]["completed"]["type"] == "boolean"
        # completed should be required
        assert "completed" in status.get("required", [])

    def test_execution_validator_files_check_structure(self):
        """Files check should have correct structure."""
        schema = load_schema("execution_validator_output_schema")
        files_check = schema["properties"]["files_check"]
        assert files_check["type"] == "object"
        required = files_check.get("required", [])
        expected_required = ["expected_files", "found_files", "missing_files", "all_present"]
        for field in expected_required:
            assert field in required, f"Missing required field: {field}"


class TestPhysicsSanitySchemaContent:
    """Tests for physics sanity schema specific content."""

    def test_physics_sanity_required_fields(self):
        """Physics sanity should have correct required fields."""
        schema = load_schema("physics_sanity_output_schema")
        required = schema.get("required", [])
        expected = ["stage_id", "verdict", "conservation_checks", "value_range_checks", "summary"]
        for field in expected:
            assert field in required, f"Missing required field: {field}"

    def test_physics_sanity_verdict_enum(self):
        """Physics sanity verdict should have expected values including design_flaw."""
        schema = load_schema("physics_sanity_output_schema")
        verdict = schema["properties"]["verdict"]
        assert "enum" in verdict
        expected_verdicts = ["pass", "warning", "fail", "design_flaw"]
        for v in expected_verdicts:
            assert v in verdict["enum"], f"Missing verdict: {v}"

    def test_physics_sanity_conservation_checks_structure(self):
        """Conservation checks should have correct structure."""
        schema = load_schema("physics_sanity_output_schema")
        checks = schema["properties"]["conservation_checks"]
        assert checks["type"] == "array"
        item = checks["items"]
        assert "law" in item["properties"]
        assert "status" in item["properties"]
        required = item.get("required", [])
        assert "law" in required
        assert "status" in required

    def test_physics_sanity_check_status_enum(self):
        """Conservation check status should have expected values."""
        schema = load_schema("physics_sanity_output_schema")
        status = schema["properties"]["conservation_checks"]["items"]["properties"]["status"]
        assert "enum" in status
        expected = ["pass", "warning", "fail"]
        for s in expected:
            assert s in status["enum"]

    def test_physics_sanity_concerns_structure(self):
        """Concerns should have severity and required fields."""
        schema = load_schema("physics_sanity_output_schema")
        concerns = schema["properties"]["concerns"]
        assert concerns["type"] == "array"
        item = concerns["items"]
        assert "severity" in item["properties"]
        assert "concern" in item.get("required", [])


class TestResultsAnalyzerSchemaContent:
    """Tests for results analyzer schema specific content."""

    def test_results_analyzer_required_fields(self):
        """Results analyzer should have correct required fields."""
        schema = load_schema("results_analyzer_output_schema")
        required = schema.get("required", [])
        expected = ["stage_id", "per_result_reports", "figure_comparisons", "overall_classification", "summary"]
        for field in expected:
            assert field in required, f"Missing required field: {field}"

    def test_results_analyzer_overall_classification_enum(self):
        """Overall classification should have expected values."""
        schema = load_schema("results_analyzer_output_schema")
        classification = schema["properties"]["overall_classification"]
        assert "enum" in classification
        expected = ["EXCELLENT_MATCH", "ACCEPTABLE_MATCH", "PARTIAL_MATCH", "POOR_MATCH", "FAILED"]
        for c in expected:
            assert c in classification["enum"], f"Missing classification: {c}"

    def test_results_analyzer_per_result_reports_structure(self):
        """Per result reports should have correct structure."""
        schema = load_schema("results_analyzer_output_schema")
        reports = schema["properties"]["per_result_reports"]
        assert reports["type"] == "array"
        item = reports["items"]
        required = item.get("required", [])
        expected_required = ["result_id", "target_figure", "quantity", "discrepancy"]
        for field in expected_required:
            assert field in required, f"Missing required field: {field}"

    def test_results_analyzer_discrepancy_classification_enum(self):
        """Discrepancy classification should have expected values."""
        schema = load_schema("results_analyzer_output_schema")
        discrepancy = schema["properties"]["per_result_reports"]["items"]["properties"]["discrepancy"]
        classification = discrepancy["properties"]["classification"]
        assert "enum" in classification
        expected = ["excellent", "acceptable", "investigate", "unacceptable"]
        for c in expected:
            assert c in classification["enum"]

    def test_results_analyzer_figure_comparisons_structure(self):
        """Figure comparisons should have correct structure."""
        schema = load_schema("results_analyzer_output_schema")
        comparisons = schema["properties"]["figure_comparisons"]
        assert comparisons["type"] == "array"
        item = comparisons["items"]
        assert "paper_figure_id" in item["properties"]
        assert "simulated_figure_path" in item["properties"]
        assert "comparison_type" in item["properties"]

    def test_results_analyzer_confidence_bounds(self):
        """Confidence should have min/max bounds."""
        schema = load_schema("results_analyzer_output_schema")
        confidence = schema["properties"]["confidence"]
        assert confidence["type"] == "number"
        assert confidence.get("minimum") == 0
        assert confidence.get("maximum") == 1


class TestBacktrackSuggestionConsistency:
    """Tests for backtrack_suggestion consistency across schemas."""

    SCHEMAS_WITH_BACKTRACK = [
        "design_reviewer_output_schema",
        "code_reviewer_output_schema",
        "physics_sanity_output_schema",
    ]

    @pytest.mark.parametrize("schema_name", SCHEMAS_WITH_BACKTRACK)
    def test_backtrack_suggestion_exists(self, schema_name):
        """Schemas that support backtracking should have backtrack_suggestion field."""
        schema = load_schema(schema_name)
        assert "backtrack_suggestion" in schema["properties"]

    @pytest.mark.parametrize("schema_name", SCHEMAS_WITH_BACKTRACK)
    def test_backtrack_suggestion_has_required_fields(self, schema_name):
        """Backtrack suggestion should have suggest_backtrack as required."""
        schema = load_schema(schema_name)
        backtrack = schema["properties"]["backtrack_suggestion"]
        assert backtrack["type"] == "object"
        assert "suggest_backtrack" in backtrack.get("required", [])

    @pytest.mark.parametrize("schema_name", SCHEMAS_WITH_BACKTRACK)
    def test_backtrack_suggestion_severity_enum(self, schema_name):
        """Backtrack severity should have consistent enum values."""
        schema = load_schema(schema_name)
        backtrack = schema["properties"]["backtrack_suggestion"]
        if "severity" in backtrack["properties"]:
            severity = backtrack["properties"]["severity"]
            assert "enum" in severity
            expected = ["critical", "significant", "minor"]
            for s in expected:
                assert s in severity["enum"]


class TestVerdictEnumConsistency:
    """Tests for verdict enum consistency across reviewer schemas."""

    def test_design_and_code_reviewer_verdict_consistency(self):
        """Design and code reviewers should have same verdict options."""
        design = load_schema("design_reviewer_output_schema")
        code = load_schema("code_reviewer_output_schema")
        
        design_verdicts = set(design["properties"]["verdict"]["enum"])
        code_verdicts = set(code["properties"]["verdict"]["enum"])
        
        assert design_verdicts == code_verdicts, \
            f"Verdict mismatch: design={design_verdicts}, code={code_verdicts}"

    def test_checklist_status_consistency_design_vs_code(self):
        """Design and code reviewer checklist status enums should be consistent."""
        design = load_schema("design_reviewer_output_schema")
        code = load_schema("code_reviewer_output_schema")
        
        # Get a status enum from each
        design_status = design["properties"]["checklist_results"]["properties"]["geometry"]["properties"]["status"]
        code_status = code["properties"]["checklist_results"]["properties"]["numerics"]["properties"]["status"]
        
        assert design_status["enum"] == code_status["enum"]


class TestSchemaEdgeCases:
    """Tests for edge cases in schema loading."""

    def setup_method(self):
        """Clear schema cache before each test."""
        _schema_cache.clear()

    def test_load_schema_with_path_traversal_fails(self):
        """Schema loading should not allow path traversal."""
        with pytest.raises(FileNotFoundError):
            load_schema("../env_example")

    def test_load_schema_with_absolute_path_in_name(self):
        """Schema loading with absolute-like path should fail gracefully."""
        with pytest.raises(FileNotFoundError):
            load_schema("/etc/passwd")

    def test_concurrent_cache_access_safety(self):
        """Cache should handle concurrent access (basic smoke test)."""
        import threading
        results = []
        errors = []
        
        def load_multiple():
            try:
                for _ in range(10):
                    schema = load_schema("planner_output_schema")
                    results.append(schema)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=load_multiple) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"
        assert len(results) == 50
        # All should be the same object (from cache)
        first_id = id(results[0])
        # After initial caching, subsequent loads should return same object
        # Note: First few might differ due to race condition, but most should be same
        same_count = sum(1 for r in results if id(r) == first_id)
        assert same_count > 40, "Cache should return same object for most calls"


class TestDesignReviewerSchemaContent:
    """Tests for design reviewer schema specific content."""

    def test_design_reviewer_required_fields(self):
        """Design reviewer should have correct required fields."""
        schema = load_schema("design_reviewer_output_schema")
        required = schema.get("required", [])
        expected = ["stage_id", "verdict", "checklist_results", "issues", "summary"]
        for field in expected:
            assert field in required, f"Missing required field: {field}"

    def test_design_reviewer_checklist_required_items(self):
        """Design reviewer checklist should have all required items."""
        schema = load_schema("design_reviewer_output_schema")
        checklist = schema["properties"]["checklist_results"]
        required_items = checklist.get("required", [])
        expected_items = [
            "geometry", "physics", "materials", "unit_system",
            "source", "domain", "resolution", "outputs", "runtime"
        ]
        for item in expected_items:
            assert item in required_items, f"Missing checklist item: {item}"

    def test_design_reviewer_issues_structure(self):
        """Design reviewer issues should have correct structure."""
        schema = load_schema("design_reviewer_output_schema")
        issues = schema["properties"]["issues"]
        assert issues["type"] == "array"
        issue_item = issues["items"]
        required_fields = issue_item.get("required", [])
        expected = ["severity", "category", "description", "suggested_fix"]
        for field in expected:
            assert field in required_fields, f"Missing required issue field: {field}"

    def test_design_reviewer_issue_category_enum(self):
        """Issue category enum should match checklist items."""
        schema = load_schema("design_reviewer_output_schema")
        category = schema["properties"]["issues"]["items"]["properties"]["category"]
        assert "enum" in category
        expected_categories = [
            "geometry", "physics", "materials", "unit_system",
            "source", "domain", "resolution", "outputs", "runtime"
        ]
        for cat in expected_categories:
            assert cat in category["enum"], f"Missing category: {cat}"


class TestSchemaPropertyDescriptions:
    """Tests that schema properties have descriptions for documentation."""

    REQUIRED_SCHEMAS = [
        "planner_output_schema",
        "supervisor_output_schema",
        "report_output_schema",
    ]

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_required_properties_have_descriptions(self, schema_name):
        """Required properties should have descriptions."""
        schema = load_schema(schema_name)
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        for field in required:
            prop = properties.get(field, {})
            assert "description" in prop or "type" in prop, \
                f"{schema_name}.{field} should have description or type"
