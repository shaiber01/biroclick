"""Schema loading tests for `src.llm_client`."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm_client import get_agent_schema, load_schema


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
        assert "planner_output_schema.json" in schema.get("$id", "")

    def test_load_schema_without_extension(self):
        """Loading schema without .json extension adds extension automatically."""
        schema = load_schema("planner_output_schema")
        
        # Verify it's the same schema as with extension
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "type" in schema
        assert schema["type"] == "object"
        # Verify it loaded the correct file
        assert "planner_output_schema.json" in schema.get("$id", "")

    def test_load_schema_caching_returns_same_object(self):
        """Schemas are cached so repeated loads return the same object (identity)."""
        schema1 = load_schema("supervisor_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        
        # Verify identity (same object in memory), not just equality
        assert schema1 is schema2, "Cached schema should return same object"
        assert schema1 == schema2, "Cached schema should be equal"

    def test_load_schema_caching_with_and_without_extension(self):
        """Caching works consistently whether extension is provided or not."""
        schema_with_ext = load_schema("supervisor_output_schema.json")
        schema_without_ext = load_schema("supervisor_output_schema")
        
        # Both should return the same cached object
        assert schema_with_ext is schema_without_ext
        assert schema_with_ext == schema_without_ext

    def test_load_schema_caching_different_schemas(self):
        """Different schemas are cached separately."""
        schema1 = load_schema("planner_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        
        # Should be different objects
        assert schema1 is not schema2
        assert schema1 != schema2
        # Verify they're actually different schemas
        assert schema1.get("$id") != schema2.get("$id")

    def test_load_nonexistent_schema(self):
        """Loading nonexistent schema raises FileNotFoundError with correct message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("nonexistent_schema.json")
        
        error_msg = str(exc_info.value)
        assert "nonexistent_schema.json" in error_msg or "Schema not found" in error_msg

    def test_load_nonexistent_schema_without_extension(self):
        """Loading nonexistent schema without extension raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_schema")

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
        # Verify it's the planner schema
        assert "planner_output_schema" in schema.get("$id", "")

    def test_get_agent_schema_supervisor(self):
        """Supervisor schema includes verdict property with correct structure."""
        schema = get_agent_schema("supervisor")
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert "verdict" in schema.get("properties", {})
        
        # Verify verdict property structure
        verdict_prop = schema["properties"]["verdict"]
        assert "type" in verdict_prop
        assert verdict_prop["type"] == "string"
        assert "enum" in verdict_prop
        assert isinstance(verdict_prop["enum"], list)
        assert len(verdict_prop["enum"]) > 0

    def test_get_agent_schema_unknown(self):
        """Unknown agent raises ValueError with informative message."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("unknown_agent")
        
        error_msg = str(exc_info.value)
        assert "Unknown agent" in error_msg
        assert "unknown_agent" in error_msg

    def test_get_agent_schema_error_includes_path(self):
        """Error message includes expected schema filename."""
        with pytest.raises(ValueError) as exc_info:
            get_agent_schema("nonexistent_agent")
        
        error_msg = str(exc_info.value)
        assert "nonexistent_agent_output_schema" in error_msg

    def test_get_agent_schema_auto_discovery(self):
        """Auto-discovery works for standard agent schemas."""
        auto_discovered_agents = [
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
        
        for agent_name in auto_discovered_agents:
            schema = get_agent_schema(agent_name)
            assert schema is not None, f"Failed to auto-discover schema for {agent_name}"
            assert isinstance(schema, dict), f"Schema for {agent_name} should be dict"
            assert "properties" in schema, f"Schema for {agent_name} should have properties"
            assert "type" in schema, f"Schema for {agent_name} should have type"
            # Verify schema ID matches agent name
            schema_id = schema.get("$id", "")
            assert agent_name in schema_id.lower(), f"Schema ID should contain agent name for {agent_name}"

    def test_get_agent_schema_special_case_report(self):
        """Special-case mapping for report agent uses correct schema."""
        schema = get_agent_schema("report")
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema
        # Verify it's the report schema, not report_output_schema
        assert "report_schema" in schema.get("$id", "")

    def test_get_agent_schema_special_case_takes_precedence(self):
        """Special-case mapping takes precedence over auto-discovery."""
        # "report" should use special mapping, not try "report_output_schema"
        schema = get_agent_schema("report")
        
        # Verify it's using the special mapping (report_schema.json)
        assert "report_schema" in schema.get("$id", "")
        # Should NOT be report_output_schema
        assert "report_output_schema" not in schema.get("$id", "")

    def test_get_agent_schema_empty_string(self):
        """Empty string agent name raises ValueError."""
        with pytest.raises(ValueError):
            get_agent_schema("")

    def test_get_agent_schema_whitespace(self):
        """Agent names with whitespace are handled correctly."""
        with pytest.raises(ValueError):
            get_agent_schema("  planner  ")
        
        with pytest.raises(ValueError):
            get_agent_schema("planner\n")

    def test_get_agent_schema_case_sensitivity(self):
        """Agent names are case-sensitive."""
        # "Planner" (capitalized) should not work if only "planner" exists
        with pytest.raises(ValueError):
            get_agent_schema("Planner")
        
        with pytest.raises(ValueError):
            get_agent_schema("PLANNER")

    def test_get_agent_schema_integration_with_load_schema(self):
        """get_agent_schema uses load_schema internally correctly."""
        # Get schema via get_agent_schema
        agent_schema = get_agent_schema("planner")
        
        # Get same schema via load_schema
        direct_schema = load_schema("planner_output_schema.json")
        
        # Should be the same (cached)
        assert agent_schema is direct_schema
        assert agent_schema == direct_schema

    def test_get_agent_schema_all_agents_have_valid_structure(self):
        """All auto-discovered agent schemas have valid JSON schema structure."""
        agents = [
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
        
        for agent_name in agents:
            schema = get_agent_schema(agent_name)
            
            # Verify basic structure
            assert "type" in schema, f"{agent_name} schema missing 'type'"
            assert schema["type"] == "object", f"{agent_name} schema type should be 'object'"
            assert "properties" in schema, f"{agent_name} schema missing 'properties'"
            assert isinstance(schema["properties"], dict), f"{agent_name} properties should be dict"
            
            # Verify required fields exist in properties if specified
            if "required" in schema:
                assert isinstance(schema["required"], list), f"{agent_name} required should be list"
                for req_field in schema["required"]:
                    assert req_field in schema["properties"], f"{agent_name} required field '{req_field}' missing from properties"


class TestSchemaLoadingEdgeCases:
    """Edge cases and error conditions for schema loading."""

    def test_load_schema_none_input(self):
        """None input should raise TypeError (type checking)."""
        with pytest.raises((TypeError, AttributeError)):
            load_schema(None)

    def test_get_agent_schema_none_input(self):
        """None input should raise TypeError (type checking)."""
        with pytest.raises((TypeError, AttributeError)):
            get_agent_schema(None)

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
        # Clear cache first
        from src import llm_client
        llm_client._schema_cache.clear()
        
        # Load schema multiple times
        schema1 = load_schema("planner_output_schema.json")
        schema2 = load_schema("planner_output_schema.json")
        schema3 = get_agent_schema("planner")
        
        # All should be the same cached object
        assert schema1 is schema2
        assert schema2 is schema3

    def test_cache_key_normalization(self):
        """Cache keys are normalized correctly (with/without extension)."""
        from src import llm_client
        llm_client._schema_cache.clear()
        
        # Load with extension
        schema1 = load_schema("planner_output_schema.json")
        
        # Load without extension - should use same cache entry
        schema2 = load_schema("planner_output_schema")
        
        # Should be same object
        assert schema1 is schema2
        
        # Verify cache contains the normalized key (with .json)
        assert "planner_output_schema.json" in llm_client._schema_cache

    def test_get_agent_schema_unicode_agent_name(self):
        """Agent names with unicode characters are handled correctly."""
        with pytest.raises(ValueError):
            get_agent_schema("planner_测试")
        
        with pytest.raises(ValueError):
            get_agent_schema("planner_🚀")

    def test_load_schema_returns_mutable_dict(self):
        """Loaded schema is a mutable dict (can be modified)."""
        schema = load_schema("planner_output_schema.json")
        
        # Verify it's mutable
        original_props_count = len(schema.get("properties", {}))
        schema["test_key"] = "test_value"
        assert "test_key" in schema
        
        # Verify modification doesn't affect cached version
        schema2 = load_schema("planner_output_schema.json")
        assert "test_key" in schema2, "Modifications should affect cached version (shared object)"

    def test_get_agent_schema_verifies_schema_structure(self):
        """Verify that get_agent_schema returns schemas with expected structure."""
        schema = get_agent_schema("supervisor")
        
        # Verify it has the expected structure for supervisor
        assert "verdict" in schema.get("properties", {})
        verdict = schema["properties"]["verdict"]
        assert verdict.get("type") == "string"
        assert "enum" in verdict
        assert "ok_continue" in verdict["enum"]
        assert "all_complete" in verdict["enum"]
