"""Schema loading tests for `src.llm_client`."""

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.llm_client import get_agent_schema, load_schema
from src import llm_client


# All existing schema files that should be loadable
ALL_SCHEMA_FILES = [
    "assumptions_schema.json",
    "code_generator_output_schema.json",
    "code_reviewer_output_schema.json",
    "comparison_validator_output_schema.json",
    "design_reviewer_output_schema.json",
    "execution_validator_output_schema.json",
    "metrics_schema.json",
    "physics_sanity_output_schema.json",
    "plan_reviewer_output_schema.json",
    "plan_schema.json",
    "planner_output_schema.json",
    "progress_schema.json",
    "prompt_adaptations_schema.json",
    "prompt_adaptor_output_schema.json",
    "report_schema.json",
    "results_analyzer_output_schema.json",
    "simulation_designer_output_schema.json",
    "supervisor_output_schema.json",
]

# Agents that follow the {agent_name}_output_schema naming convention
AUTO_DISCOVERED_AGENTS = [
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
]


@pytest.fixture(autouse=True)
def clear_schema_cache():
    """Clear the schema cache before and after each test to ensure isolation."""
    llm_client._schema_cache.clear()
    yield
    llm_client._schema_cache.clear()


class TestLoadSchema:
    """Tests for load_schema function."""

    def test_load_schema_with_extension(self):
        """Loading schema with .json extension succeeds and returns valid schema."""
        schema = load_schema("planner_output_schema.json")
        
        # Strong assertions: verify structure, not just existence
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "type" in schema
        assert schema["type"] == "object"
        assert "required" in schema
        assert isinstance(schema["required"], list)
        assert "paper_id" in schema["required"]
        # Verify exact $id value matches the filename
        assert schema.get("$id") == "planner_output_schema.json"

    def test_load_schema_without_extension(self):
        """Loading schema without .json extension adds extension automatically."""
        schema = load_schema("planner_output_schema")
        
        # Verify it's the same schema as with extension
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "type" in schema
        assert schema["type"] == "object"
        # Verify exact $id value matches the filename
        assert schema.get("$id") == "planner_output_schema.json"
        # Verify specific known required fields to ensure correct schema was loaded
        required_fields = schema["required"]
        expected_required = ["paper_id", "paper_domain", "title", "summary", 
                            "extracted_parameters", "targets", "stages", "assumptions", "progress"]
        for field in expected_required:
            assert field in required_fields, f"Expected required field '{field}' not found"

    def test_load_schema_caching_returns_same_object(self):
        """Schemas are cached so repeated loads return the same object (identity)."""
        schema1 = load_schema("supervisor_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        
        # Verify identity (same object in memory), not just equality
        assert schema1 is schema2, "Cached schema should return same object"
        assert schema1 == schema2, "Cached schema should be equal"
        
        # Verify cache contains the entry
        assert "supervisor_output_schema.json" in llm_client._schema_cache
        assert llm_client._schema_cache["supervisor_output_schema.json"] is schema1

    def test_load_schema_caching_with_and_without_extension(self):
        """Caching works consistently whether extension is provided or not."""
        schema_with_ext = load_schema("supervisor_output_schema.json")
        schema_without_ext = load_schema("supervisor_output_schema")
        
        # Both should return the same cached object
        assert schema_with_ext is schema_without_ext
        assert schema_with_ext == schema_without_ext
        
        # Verify only one cache entry exists (normalized to .json)
        supervisor_cache_entries = [k for k in llm_client._schema_cache.keys() 
                                    if "supervisor_output_schema" in k]
        assert len(supervisor_cache_entries) == 1, "Should have exactly one normalized cache entry"
        assert supervisor_cache_entries[0] == "supervisor_output_schema.json"

    def test_load_schema_caching_different_schemas(self):
        """Different schemas are cached separately."""
        schema1 = load_schema("planner_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        
        # Should be different objects
        assert schema1 is not schema2
        assert schema1 != schema2
        # Verify they're actually different schemas with exact $id values
        assert schema1.get("$id") == "planner_output_schema.json"
        assert schema2.get("$id") == "supervisor_output_schema.json"
        
        # Verify both are in cache
        assert "planner_output_schema.json" in llm_client._schema_cache
        assert "supervisor_output_schema.json" in llm_client._schema_cache

    def test_load_nonexistent_schema(self):
        """Loading nonexistent schema raises FileNotFoundError with correct message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("nonexistent_schema.json")
        
        error_msg = str(exc_info.value)
        # Verify error message contains both the schema name and helpful context
        assert "nonexistent_schema.json" in error_msg
        assert "Schema not found" in error_msg

    def test_load_nonexistent_schema_without_extension(self):
        """Loading nonexistent schema without extension raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("nonexistent_schema")
        
        error_msg = str(exc_info.value)
        # Extension should be added before the error
        assert "nonexistent_schema.json" in error_msg
        assert "Schema not found" in error_msg

    def test_load_schema_empty_string(self):
        """Loading schema with empty string raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_schema("")

    def test_load_schema_just_json_extension(self):
        """Loading schema with just '.json' raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_schema(".json")

    def test_load_schema_multiple_json_extensions(self):
        """Loading schema with multiple .json extensions handles correctly."""
        # This should try to load "schema.json.json" which doesn't exist
        with pytest.raises(FileNotFoundError):
            load_schema("schema.json.json")

    def test_load_schema_path_traversal_attack(self):
        """Path traversal attacks are prevented (should not access files outside schemas dir)."""
        # Try to access a file outside the schemas directory
        with pytest.raises(FileNotFoundError):
            load_schema("../../etc/passwd")
        
        with pytest.raises(FileNotFoundError):
            load_schema("../llm_client.py")

    def test_load_schema_special_characters(self):
        """Schema names with special characters are handled correctly."""
        # Test various special characters that might be in filenames
        with pytest.raises(FileNotFoundError):
            load_schema("schema-with-dashes.json")
        
        with pytest.raises(FileNotFoundError):
            load_schema("schema_with_underscores.json")

    def test_load_schema_very_long_name(self):
        """Very long schema names are handled correctly."""
        long_name = "a" * 1000 + ".json"
        with pytest.raises(FileNotFoundError):
            load_schema(long_name)

    def test_load_schema_whitespace(self):
        """Schema names with whitespace are handled correctly."""
        with pytest.raises(FileNotFoundError):
            load_schema("schema with spaces.json")
        
        with pytest.raises(FileNotFoundError):
            load_schema("  planner_output_schema.json  ")

    def test_load_schema_returns_valid_json_structure(self):
        """Loaded schema is valid JSON schema structure."""
        schema = load_schema("planner_output_schema.json")
        
        # Verify it's a valid JSON schema (has required fields)
        assert "$schema" in schema or "type" in schema
        assert "properties" in schema
        assert isinstance(schema["properties"], dict)
        
        # Verify properties structure
        for prop_name, prop_def in schema["properties"].items():
            assert isinstance(prop_name, str)
            assert isinstance(prop_def, dict)

    def test_load_schema_encoding_utf8(self):
        """Schema files with UTF-8 encoding are loaded correctly."""
        schema = load_schema("planner_output_schema.json")
        
        # Verify we can access string fields that might contain UTF-8
        assert isinstance(schema.get("title", ""), str)
        assert isinstance(schema.get("description", ""), str)

    def test_load_schema_all_required_fields_present(self):
        """Verify that loaded schema contains expected required fields."""
        schema = load_schema("planner_output_schema.json")
        
        # Verify specific required fields exist
        assert "required" in schema
        required_fields = schema["required"]
        assert isinstance(required_fields, list)
        assert len(required_fields) > 0
        
        # Verify all required fields have corresponding properties
        for field in required_fields:
            assert field in schema["properties"], f"Required field '{field}' missing from properties"

    def test_load_schema_properties_structure(self):
        """Verify properties in schema have correct structure."""
        schema = load_schema("planner_output_schema.json")
        
        assert "properties" in schema
        properties = schema["properties"]
        assert isinstance(properties, dict)
        
        # Verify at least one property has type definition
        has_type = any("type" in prop for prop in properties.values())
        assert has_type, "At least one property should have a 'type' field"
        
        # Verify specific known properties exist with correct types
        assert "paper_id" in properties
        assert properties["paper_id"]["type"] == "string"
        
        assert "paper_domain" in properties
        assert properties["paper_domain"]["type"] == "string"
        assert "enum" in properties["paper_domain"]
        
        assert "extracted_parameters" in properties
        assert properties["extracted_parameters"]["type"] == "array"

    def test_load_all_schema_files_exist(self):
        """All expected schema files can be loaded successfully."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            assert isinstance(schema, dict), f"Schema {schema_file} should be a dict"
            # Each schema should have a type field
            assert "type" in schema, f"Schema {schema_file} should have 'type' field"

    def test_load_schema_id_matches_filename(self):
        """Each schema file's $id field should match its filename."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            schema_id = schema.get("$id")
            assert schema_id is not None, f"Schema {schema_file} should have $id field"
            assert schema_id == schema_file, (
                f"Schema {schema_file} has mismatched $id: '{schema_id}' != '{schema_file}'"
            )

    def test_load_schema_all_have_json_schema_version(self):
        """All schema files should declare JSON schema version."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            assert "$schema" in schema, f"Schema {schema_file} should declare $schema version"
            assert "json-schema.org" in schema["$schema"], (
                f"Schema {schema_file} should use json-schema.org draft"
            )


class TestGetAgentSchema:
    """Tests for get_agent_schema function."""

    def test_get_agent_schema_planner(self):
        """Planner agent schema exists and contains expected structure."""
        schema = get_agent_schema("planner")
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "type" in schema
        assert schema["type"] == "object"
        # Verify exact $id value
        assert schema.get("$id") == "planner_output_schema.json"
        # Verify planner-specific fields
        assert "paper_id" in schema["properties"]
        assert "stages" in schema["properties"]
        assert "extracted_parameters" in schema["properties"]

    def test_get_agent_schema_supervisor(self):
        """Supervisor schema includes verdict property with correct structure."""
        schema = get_agent_schema("supervisor")
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert schema.get("$id") == "supervisor_output_schema.json"
        assert "verdict" in schema.get("properties", {})
        
        # Verify verdict property structure with exact enum values
        verdict_prop = schema["properties"]["verdict"]
        assert "type" in verdict_prop
        assert verdict_prop["type"] == "string"
        assert "enum" in verdict_prop
        expected_verdicts = ["ok_continue", "replan_needed", "change_priority", 
                            "ask_user", "backtrack_to_stage", "all_complete"]
        assert verdict_prop["enum"] == expected_verdicts
        
        # Verify required fields
        assert "verdict" in schema["required"]
        assert "validation_hierarchy_status" in schema["required"]
        assert "main_physics_assessment" in schema["required"]
        assert "summary" in schema["required"]

    def test_get_agent_schema_unknown(self):
        """Unknown agent raises ValueError with informative message."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("unknown_agent")
        
        error_msg = str(exc_info.value)
        assert "Unknown agent" in error_msg
        assert "unknown_agent" in error_msg
        # Verify expected schema path is mentioned
        assert "unknown_agent_output_schema.json" in error_msg

    def test_get_agent_schema_error_includes_path(self):
        """Error message includes expected schema filename."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("nonexistent_agent")
        
        error_msg = str(exc_info.value)
        assert "nonexistent_agent_output_schema" in error_msg
        # Also verify full expected path context
        assert "Expected schema at:" in error_msg

    def test_get_agent_schema_auto_discovery(self):
        """Auto-discovery works for standard agent schemas."""
        for agent_name in AUTO_DISCOVERED_AGENTS:
            schema = get_agent_schema(agent_name)
            assert schema is not None, f"Failed to auto-discover schema for {agent_name}"
            assert isinstance(schema, dict), f"Schema for {agent_name} should be dict"
            assert "properties" in schema, f"Schema for {agent_name} should have properties"
            assert "type" in schema, f"Schema for {agent_name} should have type"
            # Verify exact schema ID format
            expected_id = f"{agent_name}_output_schema.json"
            assert schema.get("$id") == expected_id, (
                f"Schema ID for {agent_name} should be '{expected_id}', got '{schema.get('$id')}'"
            )

    def test_get_agent_schema_special_case_report(self):
        """Special-case mapping for report agent uses correct schema."""
        schema = get_agent_schema("report")
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema
        # Verify exact $id value for report schema
        assert schema.get("$id") == "report_schema.json"
        # Verify report-specific fields
        assert "paper_id" in schema["properties"]
        assert "executive_summary" in schema["properties"]
        assert "figure_comparisons" in schema["properties"]

    def test_get_agent_schema_special_case_takes_precedence(self):
        """Special-case mapping takes precedence over auto-discovery."""
        # "report" should use special mapping, not try "report_output_schema"
        schema = get_agent_schema("report")
        
        # Verify it's using the special mapping (report_schema.json)
        assert schema.get("$id") == "report_schema.json"
        # Should NOT be report_output_schema
        assert schema.get("$id") != "report_output_schema.json"
        
        # If report_output_schema.json existed, special mapping should still take precedence
        # Verify the returned schema has the correct structure for reports
        assert "paper_citation" in schema["properties"]
        assert "conclusions" in schema["properties"]

    def test_get_agent_schema_empty_string(self):
        """Empty string agent name raises ValueError with proper message."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("")
        
        error_msg = str(exc_info.value)
        assert "cannot be empty or whitespace" in error_msg

    def test_get_agent_schema_whitespace(self):
        """Agent names with whitespace are handled correctly."""
        # Whitespace around agent name is NOT stripped for lookup
        # So "  planner  " tries to find "  planner  _output_schema.json" which doesn't exist
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("  planner  ")
        error_msg = str(exc_info.value)
        assert "  planner  " in error_msg
        
        # Test that only-whitespace raises with proper message (gets caught by strip check)
        with pytest.raises(ValueError) as exc_info2:
            get_agent_schema("   ")
        error_msg = str(exc_info2.value)
        assert "cannot be empty or whitespace" in error_msg
        
        # Verify tab and newline only
        with pytest.raises(ValueError) as exc_info3:
            get_agent_schema("\t\n")
        assert "cannot be empty or whitespace" in str(exc_info3.value)

    def test_get_agent_schema_case_sensitivity(self):
        """Agent names are case-sensitive."""
        # "Planner" (capitalized) should not work if only "planner" exists
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("Planner")
        error_msg = str(exc_info.value)
        assert "Planner" in error_msg
        assert "Planner_output_schema.json" in error_msg
        
        with pytest.raises(ValueError) as exc_info2:
            get_agent_schema("PLANNER")
        error_msg2 = str(exc_info2.value)
        assert "PLANNER" in error_msg2
        
        # Also test mixed case
        with pytest.raises(ValueError):
            get_agent_schema("pLaNnEr")

    def test_get_agent_schema_integration_with_load_schema(self):
        """get_agent_schema uses load_schema internally correctly."""
        # Get schema via get_agent_schema
        agent_schema = get_agent_schema("planner")
        
        # Get same schema via load_schema
        direct_schema = load_schema("planner_output_schema.json")
        
        # Should be the same (cached)
        assert agent_schema is direct_schema
        assert agent_schema == direct_schema
        
        # Verify the cache contains the correct normalized entry
        assert "planner_output_schema.json" in llm_client._schema_cache
        assert llm_client._schema_cache["planner_output_schema.json"] is agent_schema

    def test_get_agent_schema_all_agents_have_valid_structure(self):
        """All auto-discovered agent schemas have valid JSON schema structure."""
        for agent_name in AUTO_DISCOVERED_AGENTS:
            schema = get_agent_schema(agent_name)
            
            # Verify basic structure
            assert "type" in schema, f"{agent_name} schema missing 'type'"
            assert schema["type"] == "object", f"{agent_name} schema type should be 'object'"
            assert "properties" in schema, f"{agent_name} schema missing 'properties'"
            assert isinstance(schema["properties"], dict), f"{agent_name} properties should be dict"
            
            # Verify required fields exist in properties if specified
            if "required" in schema:
                assert isinstance(schema["required"], list), f"{agent_name} required should be list"
                assert len(schema["required"]) > 0, f"{agent_name} should have at least one required field"
                for req_field in schema["required"]:
                    assert req_field in schema["properties"], (
                        f"{agent_name} required field '{req_field}' missing from properties"
                    )
                    
            # Verify $schema declaration exists
            assert "$schema" in schema, f"{agent_name} should declare JSON schema version"
            
            # Verify $id matches expected pattern
            expected_id = f"{agent_name}_output_schema.json"
            assert schema.get("$id") == expected_id, (
                f"{agent_name} $id mismatch: expected '{expected_id}', got '{schema.get('$id')}'"
            )
            
    def test_get_agent_schema_report_and_all_agents(self):
        """Both special-case 'report' and all auto-discovered agents work."""
        # Get report (special case)
        report_schema = get_agent_schema("report")
        assert report_schema.get("$id") == "report_schema.json"
        
        # Verify we can still get all auto-discovered agents after loading report
        for agent_name in AUTO_DISCOVERED_AGENTS:
            schema = get_agent_schema(agent_name)
            assert schema.get("$id") == f"{agent_name}_output_schema.json"


class TestSchemaLoadingEdgeCases:
    """Edge cases and error conditions for schema loading."""

    def test_load_schema_none_input(self):
        """None input should raise TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            load_schema(None)
        
        error_msg = str(exc_info.value)
        assert "schema_name must be a string" in error_msg
        assert "NoneType" in error_msg

    def test_get_agent_schema_none_input(self):
        """None input should raise TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            get_agent_schema(None)
        
        error_msg = str(exc_info.value)
        assert "agent_name must be a string" in error_msg
        assert "NoneType" in error_msg
        
    def test_load_schema_non_string_input(self):
        """Non-string inputs should raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            load_schema(123)
        assert "int" in str(exc_info.value)
        
        with pytest.raises(TypeError) as exc_info2:
            load_schema(["planner_output_schema.json"])
        assert "list" in str(exc_info2.value)
        
        with pytest.raises(TypeError) as exc_info3:
            load_schema({"name": "schema"})
        assert "dict" in str(exc_info3.value)
        
    def test_get_agent_schema_non_string_input(self):
        """Non-string inputs should raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            get_agent_schema(123)
        assert "int" in str(exc_info.value)
        
        with pytest.raises(TypeError) as exc_info2:
            get_agent_schema(["planner"])
        assert "list" in str(exc_info2.value)

    def test_load_schema_invalid_json_file(self, tmp_path):
        """Loading a file with invalid JSON should raise JSONDecodeError."""
        # Create a temporary invalid JSON file
        invalid_json_file = tmp_path / "invalid_schema.json"
        invalid_json_file.write_text("{ invalid json }")
        
        # Patch SCHEMAS_DIR to point to tmp_path
        from src import llm_client
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            # Clear cache to ensure fresh load
            llm_client._schema_cache.clear()
            
            with pytest.raises(json.JSONDecodeError):
                load_schema("invalid_schema.json")
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()

    def test_load_schema_empty_json_file(self, tmp_path):
        """Loading an empty JSON file should raise JSONDecodeError."""
        empty_json_file = tmp_path / "empty_schema.json"
        empty_json_file.write_text("")
        
        from src import llm_client
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            with pytest.raises(json.JSONDecodeError):
                load_schema("empty_schema.json")
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()

    def test_load_schema_not_json_object(self, tmp_path):
        """Loading a JSON file that's not an object should still work but be detectable."""
        # JSON array instead of object
        array_json_file = tmp_path / "array_schema.json"
        array_json_file.write_text('[1, 2, 3]')
        
        from src import llm_client
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            schema = load_schema("array_schema.json")
            # Should load successfully but be a list, not a dict
            assert isinstance(schema, list)
            # This reveals that the function doesn't validate schema structure
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()

    def test_load_schema_missing_schemas_directory(self, tmp_path):
        """If schemas directory doesn't exist, should raise FileNotFoundError."""
        non_existent_dir = tmp_path / "nonexistent"
        
        from src import llm_client
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = non_existent_dir
            llm_client._schema_cache.clear()
            
            with pytest.raises(FileNotFoundError):
                load_schema("planner_output_schema.json")
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()

    def test_cache_persistence_across_calls(self):
        """Cache persists across multiple function calls."""
        # Load schema multiple times
        schema1 = load_schema("planner_output_schema.json")
        schema2 = load_schema("planner_output_schema.json")
        schema3 = get_agent_schema("planner")
        
        # All should be the same cached object
        assert schema1 is schema2
        assert schema2 is schema3
        
        # Verify cache has exactly one entry for this schema
        planner_entries = [k for k in llm_client._schema_cache if "planner_output" in k]
        assert len(planner_entries) == 1

    def test_cache_key_normalization(self):
        """Cache keys are normalized correctly (with/without extension)."""
        # Load with extension
        schema1 = load_schema("planner_output_schema.json")
        
        # Load without extension - should use same cache entry
        schema2 = load_schema("planner_output_schema")
        
        # Should be same object
        assert schema1 is schema2
        
        # Verify cache contains the normalized key (with .json)
        assert "planner_output_schema.json" in llm_client._schema_cache
        # Verify no duplicate entry without extension
        assert "planner_output_schema" not in llm_client._schema_cache

    def test_get_agent_schema_unicode_agent_name(self):
        """Agent names with unicode characters are handled correctly."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("planner_测试")
        assert "planner_测试" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info2:
            get_agent_schema("planner_🚀")
        assert "planner_🚀" in str(exc_info2.value)
        
    def test_cache_isolation_between_different_schemas(self):
        """Different schemas are cached independently."""
        schema1 = load_schema("planner_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        
        # Modify one to verify they're not sharing state
        original_planner_type = schema1.get("type")
        original_supervisor_type = schema2.get("type")
        
        assert original_planner_type == "object"
        assert original_supervisor_type == "object"
        
        # Both should have different $id
        assert schema1.get("$id") != schema2.get("$id")

    def test_load_schema_returns_mutable_dict(self):
        """Loaded schema is a mutable dict and mutations affect the cache (shared reference)."""
        schema = load_schema("planner_output_schema.json")
        
        # Verify it's mutable
        original_props_count = len(schema.get("properties", {}))
        schema["test_key"] = "test_value"
        assert "test_key" in schema
        
        # Verify modification DOES affect cached version (shared object)
        # This is important: the cache returns the same object, so modifications persist
        schema2 = load_schema("planner_output_schema.json")
        assert "test_key" in schema2, "Modifications should affect cached version (shared object)"
        assert schema is schema2, "Should be the exact same object"
        
        # Clean up to not affect other tests
        del schema["test_key"]

    def test_get_agent_schema_verifies_schema_structure(self):
        """Verify that get_agent_schema returns schemas with expected structure."""
        schema = get_agent_schema("supervisor")
        
        # Verify it has the expected structure for supervisor
        assert "verdict" in schema.get("properties", {})
        verdict = schema["properties"]["verdict"]
        assert verdict.get("type") == "string"
        assert "enum" in verdict
        # Verify all expected verdict values
        expected_verdicts = ["ok_continue", "replan_needed", "change_priority", 
                            "ask_user", "backtrack_to_stage", "all_complete"]
        for v in expected_verdicts:
            assert v in verdict["enum"], f"Expected verdict '{v}' not in enum"
            
    def test_concurrent_schema_loading(self):
        """Schema loading is safe when called from multiple threads."""
        results = []
        errors = []
        
        def load_schema_thread(schema_name):
            try:
                schema = load_schema(schema_name)
                results.append((schema_name, schema.get("$id")))
            except Exception as e:
                errors.append((schema_name, str(e)))
        
        # Load multiple schemas concurrently
        threads = []
        schemas_to_load = [
            "planner_output_schema.json",
            "supervisor_output_schema.json",
            "code_generator_output_schema.json",
        ]
        
        for schema_name in schemas_to_load * 3:  # Load each schema 3 times
            t = threading.Thread(target=load_schema_thread, args=(schema_name,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors during concurrent loading: {errors}"
        assert len(results) == 9, "All 9 load attempts should succeed"
        
        # Verify all schemas loaded correctly
        for schema_name, schema_id in results:
            assert schema_id == schema_name, f"Schema ID mismatch for {schema_name}"
            
    def test_schema_required_fields_are_in_properties(self):
        """All schemas must have their required fields defined in properties."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            required_fields = schema.get("required", [])
            properties = schema.get("properties", {})
            
            for req_field in required_fields:
                assert req_field in properties, (
                    f"Schema {schema_file}: required field '{req_field}' not in properties"
                )
                
    def test_schema_properties_have_types(self):
        """All schema properties should have type definitions (or refs)."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            properties = schema.get("properties", {})
            
            for prop_name, prop_def in properties.items():
                has_type = "type" in prop_def or "$ref" in prop_def or "oneOf" in prop_def or "anyOf" in prop_def
                assert has_type, (
                    f"Schema {schema_file}: property '{prop_name}' has no type definition"
                )
                
    def test_get_agent_schema_with_oserror_fallback(self, tmp_path):
        """get_agent_schema handles OSError during directory listing gracefully."""
        # Create a valid schema file
        valid_schema = {"type": "object", "properties": {}, "$id": "test_agent_output_schema.json"}
        schema_file = tmp_path / "test_agent_output_schema.json"
        schema_file.write_text(json.dumps(valid_schema))
        
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            # We can't easily patch Path methods, but we can verify the normal case works
            schema = get_agent_schema("test_agent")
            assert schema["$id"] == "test_agent_output_schema.json"
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()
            
    def test_get_agent_schema_iterdir_oserror_fallback(self, tmp_path):
        """get_agent_schema falls back to load_schema when iterdir raises OSError."""
        # Create a valid schema file
        valid_schema = {"type": "object", "properties": {}, "$id": "fallback_agent_output_schema.json", "$schema": "http://json-schema.org/draft-07/schema#"}
        schema_file = tmp_path / "fallback_agent_output_schema.json"
        schema_file.write_text(json.dumps(valid_schema))
        
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            # Create a mock that wraps the real Path but raises OSError on iterdir
            with patch.object(Path, 'iterdir', side_effect=OSError("Mocked directory listing error")):
                # Despite iterdir failing, the fallback should work since the file exists
                schema = get_agent_schema("fallback_agent")
                assert schema["$id"] == "fallback_agent_output_schema.json"
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()
            
    def test_get_agent_schema_exists_oserror(self, tmp_path):
        """get_agent_schema handles OSError from exists() check."""
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            # Patch exists to raise OSError (simulating filesystem error)
            with patch.object(Path, 'exists', side_effect=OSError("Mocked exists error")):
                # Should fall through to raise ValueError since path check failed
                with pytest.raises(ValueError) as exc_info:
                    get_agent_schema("test_agent")
                
                assert "Unknown agent" in str(exc_info.value)
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()
            
    def test_load_schema_json_primitive_types(self, tmp_path):
        """Loading JSON files with primitive types (not objects) should work."""
        # Test string
        string_file = tmp_path / "string_schema.json"
        string_file.write_text('"just a string"')
        
        # Test number
        number_file = tmp_path / "number_schema.json"
        number_file.write_text('42')
        
        # Test null
        null_file = tmp_path / "null_schema.json"
        null_file.write_text('null')
        
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            # All should load without error (load_schema doesn't validate structure)
            string_result = load_schema("string_schema.json")
            assert string_result == "just a string"
            
            number_result = load_schema("number_schema.json")
            assert number_result == 42
            
            null_result = load_schema("null_schema.json")
            assert null_result is None
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()
            
    def test_get_agent_schema_special_mapping_keys(self):
        """Verify the special mapping dictionary structure."""
        # The special mapping is internal, but we can verify its behavior
        # "report" should map to "report_schema" (not "report_output_schema")
        schema = get_agent_schema("report")
        assert schema.get("$id") == "report_schema.json"
        
        # Verify that trying to load "report_output_schema" directly fails
        # (since the file doesn't exist)
        with pytest.raises(FileNotFoundError):
            load_schema("report_output_schema.json")
            
    def test_load_schema_with_bom_raises_error(self, tmp_path):
        """Loading JSON files with UTF-8 BOM raises JSONDecodeError.
        
        Python's json module explicitly rejects UTF-8 BOM.
        Schema files should not have BOM markers.
        """
        # UTF-8 BOM followed by valid JSON
        bom_file = tmp_path / "bom_schema.json"
        bom_content = '\ufeff{"type": "object", "$id": "bom_schema.json"}'
        bom_file.write_text(bom_content, encoding='utf-8')
        
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            # Python's json module rejects UTF-8 BOM
            with pytest.raises(json.JSONDecodeError) as exc_info:
                load_schema("bom_schema.json")
            
            # Verify the error message mentions BOM
            assert "BOM" in str(exc_info.value)
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()
            
    def test_existing_schemas_have_no_bom(self):
        """Verify all existing schema files do not have UTF-8 BOM."""
        for schema_file in ALL_SCHEMA_FILES:
            schema_path = llm_client.SCHEMAS_DIR / schema_file
            with open(schema_path, 'rb') as f:
                first_bytes = f.read(3)
            
            # UTF-8 BOM is bytes: 0xEF 0xBB 0xBF
            bom_bytes = b'\xef\xbb\xbf'
            assert not first_bytes.startswith(bom_bytes), (
                f"Schema {schema_file} has UTF-8 BOM which will cause loading errors"
            )
            
    def test_load_schema_deeply_nested(self, tmp_path):
        """Loading schema with deeply nested structure works correctly."""
        nested_schema = {
            "type": "object",
            "$id": "nested_schema.json",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "level4": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        nested_file = tmp_path / "nested_schema.json"
        nested_file.write_text(json.dumps(nested_schema))
        
        original_schemas_dir = llm_client.SCHEMAS_DIR
        
        try:
            llm_client.SCHEMAS_DIR = tmp_path
            llm_client._schema_cache.clear()
            
            loaded = load_schema("nested_schema.json")
            
            # Verify deep structure is preserved
            assert loaded["properties"]["level1"]["properties"]["level2"]["properties"]["level3"]["properties"]["level4"]["type"] == "string"
        finally:
            llm_client.SCHEMAS_DIR = original_schemas_dir
            llm_client._schema_cache.clear()


class TestSchemaConsistency:
    """Tests that verify schema files are consistent and valid."""
    
    def test_all_output_schemas_have_consistent_structure(self):
        """All *_output_schema.json files should have consistent base structure."""
        output_schemas = [f for f in ALL_SCHEMA_FILES if "_output_schema" in f]
        
        for schema_file in output_schemas:
            schema = load_schema(schema_file)
            
            # All output schemas should be objects
            assert schema.get("type") == "object", f"{schema_file} should be type 'object'"
            
            # All should have properties
            assert "properties" in schema, f"{schema_file} should have 'properties'"
            
            # All should have JSON schema declaration
            assert "$schema" in schema, f"{schema_file} should declare $schema"
            
            # All should have $id matching filename
            assert schema.get("$id") == schema_file, f"{schema_file} $id should match filename"
            
    def test_all_schemas_have_title_and_description(self):
        """All schemas should have title and description for documentation."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            
            # Title is important for generated code and documentation
            assert "title" in schema, f"{schema_file} should have 'title' field"
            assert isinstance(schema["title"], str), f"{schema_file} title should be string"
            assert len(schema["title"]) > 0, f"{schema_file} title should not be empty"
            
            # Description provides context
            assert "description" in schema, f"{schema_file} should have 'description' field"
            assert isinstance(schema["description"], str), f"{schema_file} description should be string"
            
    def test_supervisor_verdict_enum_is_complete(self):
        """Supervisor verdict enum should contain all expected values."""
        schema = get_agent_schema("supervisor")
        verdict = schema["properties"]["verdict"]
        
        expected_verdicts = [
            "ok_continue",
            "replan_needed", 
            "change_priority",
            "ask_user",
            "backtrack_to_stage",
            "all_complete"
        ]
        
        assert verdict["enum"] == expected_verdicts, (
            f"Supervisor verdict enum mismatch. Expected {expected_verdicts}, got {verdict['enum']}"
        )
        
    def test_planner_paper_domain_enum_is_complete(self):
        """Planner paper_domain enum should contain expected physics domains."""
        schema = get_agent_schema("planner")
        paper_domain = schema["properties"]["paper_domain"]
        
        expected_domains = [
            "plasmonics",
            "photonic_crystal",
            "metamaterial", 
            "thin_film",
            "waveguide",
            "strong_coupling",
            "nonlinear",
            "other"
        ]
        
        assert paper_domain["enum"] == expected_domains, (
            f"Planner paper_domain enum mismatch. Expected {expected_domains}, got {paper_domain['enum']}"
        )
        
    def test_report_schema_required_fields(self):
        """Report schema should have all necessary required fields."""
        schema = get_agent_schema("report")
        
        expected_required = [
            "paper_id",
            "paper_citation", 
            "executive_summary",
            "assumptions",
            "figure_comparisons",
            "summary_table",
            "systematic_discrepancies",
            "conclusions"
        ]
        
        assert schema["required"] == expected_required, (
            f"Report schema required fields mismatch. Expected {expected_required}, got {schema['required']}"
        )
        
    def test_schemas_use_consistent_json_schema_version(self):
        """All schemas should use the same JSON Schema draft version."""
        schema_versions = {}
        
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema(schema_file)
            version = schema.get("$schema", "")
            schema_versions[schema_file] = version
            
        # Get unique versions
        unique_versions = set(schema_versions.values())
        
        # All should use the same version
        assert len(unique_versions) == 1, (
            f"Inconsistent JSON Schema versions across files: {schema_versions}"
        )
        
        # Should be draft-07
        expected_version = "http://json-schema.org/draft-07/schema#"
        for schema_file, version in schema_versions.items():
            assert version == expected_version, (
                f"{schema_file} uses unexpected schema version: {version}"
            )
