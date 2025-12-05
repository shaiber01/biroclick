"""
Tests for schema loading and field validation in the schema access validator.

These tests verify that the tool correctly:
- Loads JSON schema files
- Extracts field names from schemas (top-level and nested)
- Navigates JSON pointers to nested schemas
- Validates fields against schema definitions

IMPORTANT: Schema loading is foundational. If these tests fail,
all schema-based validation will be broken.
"""

import json
import pytest
from pathlib import Path

from tools.validate_schema_access import (
    load_schema,
    extract_schema_fields,
    get_schema_fields_for_pointer,
    get_nested_schema_info,
    SCHEMAS_DIR,
)

from .conftest import MOCK_SCHEMA


class TestSchemaLoading:
    """
    Tests for loading JSON schema files.
    """
    
    def test_load_valid_schema_file(self, tmp_path):
        """Should successfully load a valid JSON schema file."""
        schema_file = tmp_path / "test_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        schema = load_schema(str(schema_file))
        
        assert "properties" in schema
        assert "verdict" in schema["properties"]
        assert "summary" in schema["properties"]
    
    def test_load_missing_schema_raises(self):
        """Should raise FileNotFoundError for missing schema."""
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_schema.json")
    
    def test_load_invalid_json_raises(self, tmp_path):
        """Should raise error for invalid JSON content."""
        bad_file = tmp_path / "bad_schema.json"
        bad_file.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_schema(str(bad_file))
    
    def test_load_real_schema_files(self):
        """Should successfully load actual project schema files."""
        # Test a few known schema files
        known_schemas = [
            "planner_output_schema.json",
            "supervisor_output_schema.json",
            "code_generator_output_schema.json",
        ]
        
        for schema_name in known_schemas:
            schema_path = SCHEMAS_DIR / schema_name
            if schema_path.exists():
                schema = load_schema(schema_name)
                assert "properties" in schema, f"{schema_name} missing 'properties'"


class TestExtractSchemaFields:
    """
    Tests for extracting field names from schemas.
    """
    
    def test_extract_top_level_fields(self):
        """Should extract top-level property names."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        assert "verdict" in fields
        assert "summary" in fields
        assert "optional_field" in fields
        assert "nested_obj" in fields
        assert "array_field" in fields
    
    def test_extract_nested_fields(self):
        """Should extract nested property names."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        # Nested fields should be included
        assert "inner_field" in fields
        assert "inner_number" in fields
        assert "deep_nested" in fields
    
    def test_extract_array_item_fields(self):
        """Should extract fields from array items."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        # Array item fields should be included
        assert "item_id" in fields
        assert "item_value" in fields
    
    def test_extract_deeply_nested_fields(self):
        """Should extract fields from deeply nested objects."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        assert "deepest_field" in fields
        assert "deep_item_field" in fields
    
    def test_extract_from_empty_schema(self):
        """Should handle empty schema gracefully."""
        fields = extract_schema_fields({})
        
        assert isinstance(fields, set)
        assert len(fields) == 0
    
    def test_extract_from_schema_without_properties(self):
        """Should handle schema without properties key."""
        schema = {"type": "string"}
        fields = extract_schema_fields(schema)
        
        assert isinstance(fields, set)
        assert len(fields) == 0
    
    def test_full_path_fields_included(self):
        """Should include full dotted paths for nested fields."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        # Full paths should be included
        assert "nested_obj.inner_field" in fields
        assert "nested_obj.deep_nested" in fields


class TestGetSchemaFieldsForPointer:
    """
    Tests for JSON pointer navigation.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_empty_pointer_returns_root_fields(self, schema_file):
        """Empty pointer should return root-level fields."""
        fields = get_schema_fields_for_pointer(str(schema_file), "")
        
        assert "verdict" in fields
        assert "summary" in fields
        assert "nested_obj" in fields
    
    def test_properties_pointer(self, schema_file):
        """Should navigate to nested object properties."""
        fields = get_schema_fields_for_pointer(
            str(schema_file),
            "/properties/nested_obj"
        )
        
        assert "inner_field" in fields
        assert "inner_number" in fields
        assert "deep_nested" in fields
    
    def test_items_pointer(self, schema_file):
        """Should navigate to array items schema."""
        fields = get_schema_fields_for_pointer(
            str(schema_file),
            "/properties/array_field/items"
        )
        
        assert "item_id" in fields
        assert "item_value" in fields
        assert "nested_in_item" in fields
    
    def test_deep_pointer(self, schema_file):
        """Should navigate deep into schema."""
        fields = get_schema_fields_for_pointer(
            str(schema_file),
            "/properties/nested_obj/properties/deep_nested"
        )
        
        assert "deepest_field" in fields
    
    def test_invalid_pointer_raises(self, schema_file):
        """Should raise ValueError for invalid pointer."""
        with pytest.raises(ValueError):
            get_schema_fields_for_pointer(
                str(schema_file),
                "/properties/nonexistent_path/invalid"
            )
    
    def test_missing_schema_file_raises(self):
        """Should raise FileNotFoundError for missing schema."""
        with pytest.raises(FileNotFoundError):
            get_schema_fields_for_pointer(
                "nonexistent.json",
                "/properties/field"
            )


class TestGetNestedSchemaInfo:
    """
    Tests for get_nested_schema_info function.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return str(schema_file)
    
    def test_nested_object_field(self, schema_file):
        """Should return nested schema info for object field."""
        result = get_nested_schema_info(schema_file, "", "nested_obj")
        
        assert result is not None
        new_pointer, valid_fields, is_array = result
        
        assert "/properties/nested_obj" in new_pointer
        assert "inner_field" in valid_fields
        assert is_array is False
    
    def test_array_field_returns_items_schema(self, schema_file):
        """Should return items schema for array field."""
        result = get_nested_schema_info(schema_file, "", "array_field")
        
        assert result is not None
        new_pointer, valid_fields, is_array = result
        
        assert "items" in new_pointer
        assert "item_id" in valid_fields
        assert is_array is True
    
    def test_nonexistent_field_returns_none(self, schema_file):
        """Should return None for non-existent field."""
        result = get_nested_schema_info(schema_file, "", "nonexistent")
        
        assert result is None
    
    def test_scalar_field(self, schema_file):
        """Should return empty fields for scalar field."""
        result = get_nested_schema_info(schema_file, "", "verdict")
        
        # Scalar fields don't have nested properties
        if result is not None:
            new_pointer, valid_fields, is_array = result
            assert len(valid_fields) == 0 or valid_fields is None
    
    def test_nested_in_nested(self, schema_file):
        """Should navigate nested schema within nested object."""
        result = get_nested_schema_info(
            schema_file,
            "/properties/nested_obj",
            "deep_nested"
        )
        
        assert result is not None
        new_pointer, valid_fields, is_array = result
        assert "deepest_field" in valid_fields
    
    def test_empty_schema_file_handled(self):
        """Should handle empty schema file gracefully."""
        result = get_nested_schema_info("", "", "field")
        
        # Empty schema file should return None gracefully
        assert result is None


class TestFieldValidation:
    """
    Tests for field validation logic.
    """
    
    def test_valid_field_in_set(self):
        """Valid field should be in extracted field set."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        assert "verdict" in fields
        assert "summary" in fields
    
    def test_invalid_field_not_in_set(self):
        """Invalid field should NOT be in extracted field set."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        assert "nonexistent" not in fields
        assert "bad_field" not in fields
    
    def test_case_sensitivity(self):
        """Field matching should be case-sensitive."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        assert "verdict" in fields
        assert "Verdict" not in fields
        assert "VERDICT" not in fields
    
    def test_typo_not_matched(self):
        """Typos should not match valid fields."""
        fields = extract_schema_fields(MOCK_SCHEMA)
        
        assert "verdct" not in fields  # typo
        assert "sumary" not in fields  # typo
        assert "verdictt" not in fields  # extra char


class TestRealSchemaConsistency:
    """
    Tests that verify actual project schemas are consistent.
    """
    
    def test_all_agent_schemas_have_properties(self):
        """All agent schemas should have a properties key."""
        for schema_file in SCHEMAS_DIR.glob("*_output_schema.json"):
            schema = load_schema(schema_file.name)
            assert "properties" in schema, f"{schema_file.name} missing 'properties'"
    
    def test_all_agent_schemas_have_required(self):
        """All agent schemas should have a required key."""
        for schema_file in SCHEMAS_DIR.glob("*_output_schema.json"):
            schema = load_schema(schema_file.name)
            assert "required" in schema, f"{schema_file.name} missing 'required'"
    
    def test_common_fields_exist_in_schemas(self):
        """Common fields like 'summary' should exist in relevant schemas."""
        reviewer_schemas = [
            "code_reviewer_output_schema.json",
            "design_reviewer_output_schema.json",
            "plan_reviewer_output_schema.json",
        ]
        
        for schema_name in reviewer_schemas:
            schema_path = SCHEMAS_DIR / schema_name
            if schema_path.exists():
                fields = extract_schema_fields(load_schema(schema_name))
                assert "summary" in fields, f"{schema_name} missing 'summary'"
                assert "verdict" in fields, f"{schema_name} missing 'verdict'"
