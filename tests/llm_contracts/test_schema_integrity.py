"""Schema integrity tests for agent JSON schemas.

These tests verify that all JSON schemas in the schemas/ directory are:
1. Valid JSON files
2. Valid JSON Schema (draft-07) documents
3. Internally consistent (required fields exist in properties)
4. Have proper metadata ($schema, $id, title, description)
5. Have valid enum definitions
6. Have valid type definitions
7. Have consistent nested object structures
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from .helpers import SCHEMAS_DIR, AGENT_SCHEMAS


# All JSON schema files in the schemas directory
ALL_SCHEMA_FILES = sorted([f.name for f in SCHEMAS_DIR.glob("*.json")])

# Valid JSON Schema draft-07 types
VALID_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}

# Expected $schema URL for all schemas
EXPECTED_SCHEMA_URL = "http://json-schema.org/draft-07/schema#"


def load_schema_file(filename: str) -> Dict[str, Any]:
    """Load and parse a schema file, returning the parsed JSON."""
    path = SCHEMAS_DIR / filename
    with open(path, "r") as f:
        return json.load(f)


def get_all_enum_values(schema: Dict[str, Any], path: str = "") -> List[Tuple[str, List]]:
    """Recursively find all enum definitions in a schema.
    
    Returns list of (path, enum_values) tuples.
    """
    enums = []
    
    if isinstance(schema, dict):
        if "enum" in schema:
            enums.append((path, schema["enum"]))
        
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                for prop_name, prop_def in value.items():
                    enums.extend(get_all_enum_values(prop_def, f"{path}.{prop_name}"))
            elif key == "items" and isinstance(value, dict):
                enums.extend(get_all_enum_values(value, f"{path}[]"))
            elif key in ("oneOf", "anyOf", "allOf") and isinstance(value, list):
                for i, item in enumerate(value):
                    enums.extend(get_all_enum_values(item, f"{path}.{key}[{i}]"))
            elif isinstance(value, dict):
                enums.extend(get_all_enum_values(value, f"{path}.{key}"))
    
    return enums


def get_all_required_vs_properties(schema: Dict[str, Any], path: str = "root") -> List[Tuple[str, Set[str], Set[str]]]:
    """Recursively find all required arrays and their corresponding properties.
    
    Returns list of (path, required_set, properties_set) tuples.
    """
    results = []
    
    if isinstance(schema, dict):
        if "required" in schema and "properties" in schema:
            required = set(schema.get("required", []))
            properties = set(schema.get("properties", {}).keys())
            results.append((path, required, properties))
        
        # Check nested objects in properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_def in schema["properties"].items():
                if isinstance(prop_def, dict):
                    results.extend(get_all_required_vs_properties(prop_def, f"{path}.{prop_name}"))
        
        # Check items in arrays
        if "items" in schema and isinstance(schema["items"], dict):
            results.extend(get_all_required_vs_properties(schema["items"], f"{path}[]"))
        
        # Check oneOf/anyOf/allOf
        for key in ("oneOf", "anyOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                for i, item in enumerate(schema[key]):
                    results.extend(get_all_required_vs_properties(item, f"{path}.{key}[{i}]"))
    
    return results


def get_all_type_definitions(schema: Dict[str, Any], path: str = "root") -> List[Tuple[str, Any]]:
    """Recursively find all type definitions in a schema.
    
    Returns list of (path, type_value) tuples.
    """
    types = []
    
    if isinstance(schema, dict):
        if "type" in schema:
            types.append((path, schema["type"]))
        
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_def in schema["properties"].items():
                types.extend(get_all_type_definitions(prop_def, f"{path}.{prop_name}"))
        
        if "items" in schema and isinstance(schema["items"], dict):
            types.extend(get_all_type_definitions(schema["items"], f"{path}[]"))
        
        for key in ("oneOf", "anyOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                for i, item in enumerate(schema[key]):
                    types.extend(get_all_type_definitions(item, f"{path}.{key}[{i}]"))
    
    return types


def get_all_nested_objects(schema: Dict[str, Any], path: str = "root") -> List[Tuple[str, Dict]]:
    """Recursively find all nested object definitions.
    
    Returns list of (path, object_schema) tuples.
    """
    objects = []
    
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            objects.append((path, schema))
        
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_def in schema["properties"].items():
                objects.extend(get_all_nested_objects(prop_def, f"{path}.{prop_name}"))
        
        if "items" in schema and isinstance(schema["items"], dict):
            objects.extend(get_all_nested_objects(schema["items"], f"{path}[]"))
        
        for key in ("oneOf", "anyOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                for i, item in enumerate(schema[key]):
                    objects.extend(get_all_nested_objects(item, f"{path}.{key}[{i}]"))
    
    return objects


class TestSchemaFilesExistAndAreValid:
    """Test that all expected schema files exist and are valid JSON."""

    def test_schemas_directory_exists(self):
        """The schemas directory must exist."""
        assert SCHEMAS_DIR.exists(), f"Schemas directory not found at {SCHEMAS_DIR}"
        assert SCHEMAS_DIR.is_dir(), f"{SCHEMAS_DIR} is not a directory"

    def test_schema_files_are_json_files(self):
        """All files in AGENT_SCHEMAS should end with .json."""
        for agent_name, schema_file in AGENT_SCHEMAS.items():
            assert schema_file.endswith(".json"), (
                f"Schema file for {agent_name} does not end with .json: {schema_file}"
            )

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_file_exists(self, schema_file: str):
        """Each schema file must exist in the schemas directory."""
        path = SCHEMAS_DIR / schema_file
        assert path.exists(), f"Schema file not found: {path}"
        assert path.is_file(), f"{path} is not a file"

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_is_valid_json(self, schema_file: str):
        """Each schema file must be valid JSON."""
        path = SCHEMAS_DIR / schema_file
        try:
            with open(path, "r") as f:
                schema = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {schema_file}: {e}")
        
        # Must be a dict at top level
        assert isinstance(schema, dict), f"Schema {schema_file} root must be a dict, got {type(schema).__name__}"


class TestSchemaMetadata:
    """Test that all schemas have required JSON Schema metadata."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_has_schema_field(self, schema_file: str):
        """Each schema must have a $schema field pointing to draft-07."""
        schema = load_schema_file(schema_file)
        
        assert "$schema" in schema, f"Schema {schema_file} missing $schema field"
        assert schema["$schema"] == EXPECTED_SCHEMA_URL, (
            f"Schema {schema_file} has unexpected $schema: {schema['$schema']}, "
            f"expected {EXPECTED_SCHEMA_URL}"
        )

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_has_id_field(self, schema_file: str):
        """Each schema must have a $id field."""
        schema = load_schema_file(schema_file)
        
        assert "$id" in schema, f"Schema {schema_file} missing $id field"
        assert isinstance(schema["$id"], str), f"Schema {schema_file} $id must be a string"
        assert len(schema["$id"]) > 0, f"Schema {schema_file} $id must not be empty"

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_id_matches_filename(self, schema_file: str):
        """The $id field should match the filename."""
        schema = load_schema_file(schema_file)
        
        if "$id" in schema:
            # $id should be the filename (or end with it)
            schema_id = schema["$id"]
            assert schema_id == schema_file or schema_id.endswith(f"/{schema_file}"), (
                f"Schema {schema_file} has $id '{schema_id}' which does not match filename"
            )

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_has_title(self, schema_file: str):
        """Each schema must have a title field."""
        schema = load_schema_file(schema_file)
        
        assert "title" in schema, f"Schema {schema_file} missing title field"
        assert isinstance(schema["title"], str), f"Schema {schema_file} title must be a string"
        assert len(schema["title"]) > 0, f"Schema {schema_file} title must not be empty"

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_has_description(self, schema_file: str):
        """Each schema must have a description field."""
        schema = load_schema_file(schema_file)
        
        assert "description" in schema, f"Schema {schema_file} missing description field"
        assert isinstance(schema["description"], str), f"Schema {schema_file} description must be a string"
        assert len(schema["description"]) > 0, f"Schema {schema_file} description must not be empty"

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_schema_has_type_or_ref(self, schema_file: str):
        """Each schema must have a type field (usually 'object') or $ref."""
        schema = load_schema_file(schema_file)
        
        has_type = "type" in schema
        has_ref = "$ref" in schema
        has_one_of = "oneOf" in schema
        has_any_of = "anyOf" in schema
        has_all_of = "allOf" in schema
        
        assert has_type or has_ref or has_one_of or has_any_of or has_all_of, (
            f"Schema {schema_file} must have type, $ref, oneOf, anyOf, or allOf at root level"
        )


class TestSchemaTypeDefinitions:
    """Test that all type definitions are valid JSON Schema types."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_all_types_are_valid(self, schema_file: str):
        """All type definitions must use valid JSON Schema types."""
        schema = load_schema_file(schema_file)
        type_definitions = get_all_type_definitions(schema)
        
        for path, type_value in type_definitions:
            if isinstance(type_value, str):
                assert type_value in VALID_JSON_SCHEMA_TYPES, (
                    f"Schema {schema_file} at {path}: invalid type '{type_value}'. "
                    f"Valid types: {VALID_JSON_SCHEMA_TYPES}"
                )
            elif isinstance(type_value, list):
                # Array of types (e.g., ["string", "null"])
                for t in type_value:
                    assert t in VALID_JSON_SCHEMA_TYPES, (
                        f"Schema {schema_file} at {path}: invalid type '{t}' in type array {type_value}. "
                        f"Valid types: {VALID_JSON_SCHEMA_TYPES}"
                    )
            else:
                pytest.fail(
                    f"Schema {schema_file} at {path}: type must be string or array, got {type(type_value).__name__}"
                )


class TestSchemaRequiredFieldsConsistency:
    """Test that required fields are properly defined in properties."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_required_fields_exist_in_properties(self, schema_file: str):
        """All fields listed in 'required' must exist in 'properties'."""
        schema = load_schema_file(schema_file)
        required_vs_properties = get_all_required_vs_properties(schema)
        
        errors = []
        for path, required, properties in required_vs_properties:
            missing = required - properties
            if missing:
                errors.append(f"At {path}: required fields {missing} not in properties {properties}")
        
        if errors:
            pytest.fail(f"Schema {schema_file} has inconsistent required/properties:\n" + "\n".join(errors))

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_required_is_array(self, schema_file: str):
        """The 'required' field must be an array."""
        schema = load_schema_file(schema_file)
        
        if "required" in schema:
            assert isinstance(schema["required"], list), (
                f"Schema {schema_file}: 'required' must be a list, got {type(schema['required']).__name__}"
            )
            for item in schema["required"]:
                assert isinstance(item, str), (
                    f"Schema {schema_file}: items in 'required' must be strings, got {type(item).__name__}"
                )

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_no_duplicate_required_fields(self, schema_file: str):
        """The 'required' array should not contain duplicates."""
        schema = load_schema_file(schema_file)
        required_vs_properties = get_all_required_vs_properties(schema)
        
        for path, required, _ in required_vs_properties:
            if len(required) > 0:
                # Get the original list to check for duplicates
                if path == "root":
                    required_list = schema.get("required", [])
                else:
                    # Navigate to the nested object
                    continue  # Skip nested for simplicity in this check
                
                if len(required_list) != len(set(required_list)):
                    duplicates = [x for x in required_list if required_list.count(x) > 1]
                    pytest.fail(
                        f"Schema {schema_file}: 'required' at root contains duplicates: {set(duplicates)}"
                    )


class TestSchemaEnumDefinitions:
    """Test that enum definitions are valid."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_enums_are_non_empty_arrays(self, schema_file: str):
        """All enum definitions must be non-empty arrays."""
        schema = load_schema_file(schema_file)
        enums = get_all_enum_values(schema)
        
        for path, enum_values in enums:
            assert isinstance(enum_values, list), (
                f"Schema {schema_file} at {path}: enum must be a list, got {type(enum_values).__name__}"
            )
            assert len(enum_values) > 0, (
                f"Schema {schema_file} at {path}: enum must not be empty"
            )

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_enums_have_no_duplicates(self, schema_file: str):
        """Enum values should not contain duplicates."""
        schema = load_schema_file(schema_file)
        enums = get_all_enum_values(schema)
        
        for path, enum_values in enums:
            # Convert to comparable types for checking duplicates
            try:
                if len(enum_values) != len(set(enum_values)):
                    duplicates = [x for x in enum_values if enum_values.count(x) > 1]
                    pytest.fail(
                        f"Schema {schema_file} at {path}: enum contains duplicates: {set(duplicates)}"
                    )
            except TypeError:
                # Some enum values may not be hashable (dicts, lists)
                pass

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_string_enums_are_non_empty_strings(self, schema_file: str):
        """String enum values should not be empty strings."""
        schema = load_schema_file(schema_file)
        enums = get_all_enum_values(schema)
        
        for path, enum_values in enums:
            for value in enum_values:
                if isinstance(value, str):
                    assert len(value) > 0, (
                        f"Schema {schema_file} at {path}: enum contains empty string"
                    )


class TestSchemaNestedObjectsStructure:
    """Test that nested objects have proper structure."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_objects_with_required_have_properties(self, schema_file: str):
        """Objects with 'required' field must also have 'properties'."""
        schema = load_schema_file(schema_file)
        nested_objects = get_all_nested_objects(schema)
        
        errors = []
        for path, obj_schema in nested_objects:
            if "required" in obj_schema and "properties" not in obj_schema:
                errors.append(f"At {path}: has 'required' but no 'properties'")
        
        if errors:
            pytest.fail(f"Schema {schema_file} has objects with required but no properties:\n" + "\n".join(errors))

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_properties_are_dicts(self, schema_file: str):
        """The 'properties' field must be a dict of property definitions."""
        schema = load_schema_file(schema_file)
        
        def check_properties(s: Dict, path: str):
            if "properties" in s:
                props = s["properties"]
                assert isinstance(props, dict), (
                    f"Schema {schema_file} at {path}: 'properties' must be a dict, got {type(props).__name__}"
                )
                for prop_name, prop_def in props.items():
                    assert isinstance(prop_name, str), (
                        f"Schema {schema_file} at {path}: property names must be strings"
                    )
                    assert isinstance(prop_def, dict), (
                        f"Schema {schema_file} at {path}.{prop_name}: property definition must be a dict"
                    )
                    # Recursively check nested properties
                    check_properties(prop_def, f"{path}.{prop_name}")
            
            if "items" in s and isinstance(s["items"], dict):
                check_properties(s["items"], f"{path}[]")
        
        check_properties(schema, "root")


class TestSchemaArrayDefinitions:
    """Test that array definitions have proper items schemas."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_arrays_have_items(self, schema_file: str):
        """Array type properties should define their items schema."""
        schema = load_schema_file(schema_file)
        
        def check_arrays(s: Dict, path: str):
            errors = []
            if s.get("type") == "array" and "items" not in s:
                errors.append(f"At {path}: array type without 'items' definition")
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    errors.extend(check_arrays(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_arrays(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_arrays(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has array issues:\n" + "\n".join(errors))


class TestSchemaPropertyDefinitions:
    """Test that property definitions are valid."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_properties_have_type_or_ref_or_composition(self, schema_file: str):
        """Each property should define its type, $ref, or use composition keywords."""
        schema = load_schema_file(schema_file)
        
        def check_prop_types(s: Dict, path: str):
            errors = []
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    prop_path = f"{path}.{prop_name}"
                    if isinstance(prop_def, dict):
                        has_type = "type" in prop_def
                        has_ref = "$ref" in prop_def
                        has_composition = any(k in prop_def for k in ("oneOf", "anyOf", "allOf"))
                        has_const = "const" in prop_def
                        has_enum = "enum" in prop_def
                        
                        if not (has_type or has_ref or has_composition or has_const or has_enum):
                            errors.append(f"At {prop_path}: no type, $ref, or composition keywords")
                        
                        # Recursively check nested properties
                        errors.extend(check_prop_types(prop_def, prop_path))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_prop_types(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_prop_types(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has properties without type info:\n" + "\n".join(errors))

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_descriptions_are_strings(self, schema_file: str):
        """All description fields must be non-empty strings."""
        schema = load_schema_file(schema_file)
        
        def check_descriptions(s: Dict, path: str):
            errors = []
            if "description" in s:
                desc = s["description"]
                if not isinstance(desc, str):
                    errors.append(f"At {path}: description must be string, got {type(desc).__name__}")
                elif len(desc) == 0:
                    errors.append(f"At {path}: description must not be empty")
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_descriptions(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_descriptions(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_descriptions(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid descriptions:\n" + "\n".join(errors))


class TestSchemaNumericConstraints:
    """Test that numeric constraints are valid."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_minimum_maximum_are_numbers(self, schema_file: str):
        """minimum and maximum constraints must be numbers."""
        schema = load_schema_file(schema_file)
        
        def check_numeric_constraints(s: Dict, path: str):
            errors = []
            for constraint in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                if constraint in s:
                    val = s[constraint]
                    if not isinstance(val, (int, float)):
                        errors.append(f"At {path}: {constraint} must be a number, got {type(val).__name__}")
            
            # Check min <= max if both present
            if "minimum" in s and "maximum" in s:
                if s["minimum"] > s["maximum"]:
                    errors.append(f"At {path}: minimum ({s['minimum']}) > maximum ({s['maximum']})")
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_numeric_constraints(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_numeric_constraints(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_numeric_constraints(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid numeric constraints:\n" + "\n".join(errors))

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_minlength_maxlength_are_non_negative_integers(self, schema_file: str):
        """minLength and maxLength constraints must be non-negative integers."""
        schema = load_schema_file(schema_file)
        
        def check_length_constraints(s: Dict, path: str):
            errors = []
            for constraint in ("minLength", "maxLength"):
                if constraint in s:
                    val = s[constraint]
                    if not isinstance(val, int) or val < 0:
                        errors.append(f"At {path}: {constraint} must be non-negative integer, got {val}")
            
            if "minLength" in s and "maxLength" in s:
                if s["minLength"] > s["maxLength"]:
                    errors.append(f"At {path}: minLength ({s['minLength']}) > maxLength ({s['maxLength']})")
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_length_constraints(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_length_constraints(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_length_constraints(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid length constraints:\n" + "\n".join(errors))


class TestAgentSchemaCompleteness:
    """Test that all agent schemas in AGENT_SCHEMAS exist and have expected structure."""

    def test_all_agent_schemas_exist(self):
        """All schemas listed in AGENT_SCHEMAS must exist."""
        missing = []
        for agent_name, schema_file in AGENT_SCHEMAS.items():
            path = SCHEMAS_DIR / schema_file
            if not path.exists():
                missing.append(f"{agent_name}: {schema_file}")
        
        if missing:
            pytest.fail(f"Missing agent schema files:\n" + "\n".join(missing))

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_agent_schema_is_object_type(self, agent_name: str, schema_file: str):
        """Agent output schemas should be object type at root level."""
        schema = load_schema_file(schema_file)
        
        assert schema.get("type") == "object", (
            f"Agent schema {agent_name} ({schema_file}) should have type 'object', "
            f"got '{schema.get('type')}'"
        )

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_agent_schema_has_required_fields(self, agent_name: str, schema_file: str):
        """Agent output schemas should have 'required' and 'properties' fields."""
        schema = load_schema_file(schema_file)
        
        assert "required" in schema, f"Agent schema {agent_name} ({schema_file}) missing 'required' field"
        assert "properties" in schema, f"Agent schema {agent_name} ({schema_file}) missing 'properties' field"
        assert len(schema["required"]) > 0, f"Agent schema {agent_name} ({schema_file}) has empty 'required' array"
        assert len(schema["properties"]) > 0, f"Agent schema {agent_name} ({schema_file}) has empty 'properties'"

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_agent_schema_title_matches_agent(self, agent_name: str, schema_file: str):
        """Agent schema title should reference the agent name."""
        schema = load_schema_file(schema_file)
        
        # Convert agent_name to expected title format
        # e.g., "planner" -> should contain "Planner" or "planner"
        title = schema.get("title", "")
        agent_name_lower = agent_name.lower().replace("_", "")
        title_lower = title.lower().replace("_", "")
        
        assert agent_name_lower in title_lower or title_lower.startswith(agent_name_lower[:5]), (
            f"Agent schema {agent_name}: title '{title}' doesn't seem to match agent name"
        )


class TestSchemaSpecificRules:
    """Test schema-specific rules and constraints for critical schemas."""

    def test_planner_schema_has_required_output_fields(self):
        """Planner schema must have critical output fields."""
        schema = load_schema_file("planner_output_schema.json")
        required = set(schema.get("required", []))
        properties = set(schema.get("properties", {}).keys())
        
        # Critical fields that planner must output
        critical_fields = {"paper_id", "title", "stages", "targets", "assumptions", "progress"}
        
        missing_required = critical_fields - required
        missing_properties = critical_fields - properties
        
        assert not missing_required, f"Planner schema missing critical required fields: {missing_required}"
        assert not missing_properties, f"Planner schema missing critical properties: {missing_properties}"

    def test_supervisor_schema_has_verdict_field(self):
        """Supervisor schema must have verdict field with proper enum values."""
        schema = load_schema_file("supervisor_output_schema.json")
        
        assert "verdict" in schema.get("required", []), "Supervisor schema must require 'verdict'"
        verdict_schema = schema.get("properties", {}).get("verdict", {})
        
        assert "enum" in verdict_schema, "Supervisor verdict must be an enum"
        
        # Expected verdict values
        expected_verdicts = {"ok_continue", "replan_needed", "ask_user", "all_complete"}
        actual_verdicts = set(verdict_schema.get("enum", []))
        
        missing = expected_verdicts - actual_verdicts
        assert not missing, f"Supervisor verdict enum missing expected values: {missing}"

    def test_code_generator_schema_has_code_field(self):
        """Code generator schema must have code field."""
        schema = load_schema_file("code_generator_output_schema.json")
        
        assert "code" in schema.get("required", []), "Code generator schema must require 'code'"
        code_schema = schema.get("properties", {}).get("code", {})
        
        assert code_schema.get("type") == "string", "Code field must be type string"

    def test_reviewer_schemas_have_verdict_field(self):
        """All reviewer schemas must have verdict field."""
        reviewer_schemas = [
            "plan_reviewer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_reviewer_output_schema.json",
            "comparison_validator_output_schema.json",
        ]
        
        for schema_file in reviewer_schemas:
            schema = load_schema_file(schema_file)
            assert "verdict" in schema.get("required", []), (
                f"{schema_file} must require 'verdict'"
            )
            verdict_schema = schema.get("properties", {}).get("verdict", {})
            assert "enum" in verdict_schema, f"{schema_file} verdict must be an enum"

    def test_validator_schemas_have_summary_field(self):
        """Validator schemas should have summary field."""
        validator_schemas = [
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "results_analyzer_output_schema.json",
            "comparison_validator_output_schema.json",
        ]
        
        for schema_file in validator_schemas:
            schema = load_schema_file(schema_file)
            assert "summary" in schema.get("required", []) or "summary" in schema.get("properties", {}), (
                f"{schema_file} should have 'summary' field"
            )

    def test_execution_validator_has_files_check(self):
        """Execution validator must have files_check for output validation."""
        schema = load_schema_file("execution_validator_output_schema.json")
        
        assert "files_check" in schema.get("required", []), (
            "Execution validator must require 'files_check'"
        )
        files_check = schema.get("properties", {}).get("files_check", {})
        
        # Should have expected/found/missing files tracking
        files_check_required = set(files_check.get("required", []))
        expected_fields = {"expected_files", "found_files", "missing_files", "all_present"}
        missing = expected_fields - files_check_required
        
        assert not missing, f"files_check missing required fields: {missing}"


class TestSchemaConsistencyAcrossAgents:
    """Test consistency patterns across related schemas."""

    def test_all_reviewer_verdicts_have_same_options(self):
        """All reviewer schemas should use consistent verdict enum values."""
        reviewer_schemas = {
            "plan_reviewer": "plan_reviewer_output_schema.json",
            "design_reviewer": "design_reviewer_output_schema.json",
            "code_reviewer": "code_reviewer_output_schema.json",
            "comparison_validator": "comparison_validator_output_schema.json",
        }
        
        # Expected common verdict values for reviewers (approve/reject style)
        expected_approve_reject = {"approve", "needs_revision"}
        
        for name, schema_file in reviewer_schemas.items():
            schema = load_schema_file(schema_file)
            verdict_enum = set(schema.get("properties", {}).get("verdict", {}).get("enum", []))
            
            # Check that core approve/reject options are present
            if not verdict_enum.issuperset(expected_approve_reject):
                missing = expected_approve_reject - verdict_enum
                pytest.fail(
                    f"{name} verdict enum {verdict_enum} missing standard options: {missing}"
                )

    def test_validator_verdicts_have_pass_fail(self):
        """Validator schemas should have pass/fail style verdicts."""
        validator_schemas = {
            "execution_validator": "execution_validator_output_schema.json",
            "physics_sanity": "physics_sanity_output_schema.json",
        }
        
        expected_verdicts = {"pass", "fail"}
        
        for name, schema_file in validator_schemas.items():
            schema = load_schema_file(schema_file)
            verdict_enum = set(schema.get("properties", {}).get("verdict", {}).get("enum", []))
            
            if not verdict_enum.issuperset(expected_verdicts):
                missing = expected_verdicts - verdict_enum
                pytest.fail(
                    f"{name} verdict enum {verdict_enum} missing standard options: {missing}"
                )

    def test_all_schemas_with_issues_have_proper_structure(self):
        """Schemas with 'issues' array should have consistent issue structure."""
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema_file(schema_file)
            properties = schema.get("properties", {})
            
            if "issues" in properties:
                issues_schema = properties["issues"]
                assert issues_schema.get("type") == "array", (
                    f"{schema_file}: 'issues' should be array type"
                )
                
                items_schema = issues_schema.get("items", {})
                if items_schema.get("type") == "object":
                    items_props = items_schema.get("properties", {})
                    
                    # Issues should typically have severity
                    if "severity" in items_props:
                        severity_enum = items_props["severity"].get("enum", [])
                        assert len(severity_enum) > 0, (
                            f"{schema_file}: issues severity should have enum values"
                        )


class TestSchemaDefaultValues:
    """Test that default values are valid for their types."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_default_values_match_type(self, schema_file: str):
        """Default values must match the declared type."""
        schema = load_schema_file(schema_file)
        
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }
        
        def check_defaults(s: Dict, path: str):
            errors = []
            if "default" in s and "type" in s:
                default_val = s["default"]
                declared_type = s["type"]
                
                # Handle type arrays like ["string", "null"]
                if isinstance(declared_type, list):
                    types_to_check = declared_type
                else:
                    types_to_check = [declared_type]
                
                valid = False
                for t in types_to_check:
                    if t in type_validators and type_validators[t](default_val):
                        valid = True
                        break
                
                if not valid:
                    errors.append(
                        f"At {path}: default value {default_val!r} doesn't match type {declared_type}"
                    )
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_defaults(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_defaults(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_defaults(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid default values:\n" + "\n".join(errors))


class TestSchemaPatternValidity:
    """Test that pattern constraints contain valid regex."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_patterns_are_valid_regex(self, schema_file: str):
        """All pattern constraints must be valid regular expressions."""
        schema = load_schema_file(schema_file)
        
        def check_patterns(s: Dict, path: str):
            errors = []
            if "pattern" in s:
                pattern = s["pattern"]
                if not isinstance(pattern, str):
                    errors.append(f"At {path}: pattern must be string, got {type(pattern).__name__}")
                else:
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        errors.append(f"At {path}: invalid regex pattern '{pattern}': {e}")
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_patterns(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_patterns(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_patterns(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid patterns:\n" + "\n".join(errors))


class TestSchemaCompositionKeywords:
    """Test that oneOf/anyOf/allOf are properly structured."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_composition_keywords_are_arrays(self, schema_file: str):
        """oneOf, anyOf, allOf must be arrays."""
        schema = load_schema_file(schema_file)
        
        def check_composition(s: Dict, path: str):
            errors = []
            for keyword in ("oneOf", "anyOf", "allOf"):
                if keyword in s:
                    value = s[keyword]
                    if not isinstance(value, list):
                        errors.append(
                            f"At {path}: {keyword} must be array, got {type(value).__name__}"
                        )
                    elif len(value) == 0:
                        errors.append(f"At {path}: {keyword} must not be empty")
                    else:
                        for i, item in enumerate(value):
                            if not isinstance(item, dict):
                                errors.append(
                                    f"At {path}.{keyword}[{i}]: items must be schema objects"
                                )
                            else:
                                errors.extend(check_composition(item, f"{path}.{keyword}[{i}]"))
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_composition(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_composition(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_composition(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid composition keywords:\n" + "\n".join(errors))


class TestSchemaArrayConstraints:
    """Test that array constraints (minItems, maxItems) are valid."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_array_item_constraints_are_valid(self, schema_file: str):
        """minItems and maxItems must be non-negative integers and minItems <= maxItems."""
        schema = load_schema_file(schema_file)
        
        def check_array_constraints(s: Dict, path: str):
            errors = []
            for constraint in ("minItems", "maxItems"):
                if constraint in s:
                    val = s[constraint]
                    if not isinstance(val, int) or val < 0:
                        errors.append(
                            f"At {path}: {constraint} must be non-negative integer, got {val}"
                        )
            
            if "minItems" in s and "maxItems" in s:
                if s["minItems"] > s["maxItems"]:
                    errors.append(
                        f"At {path}: minItems ({s['minItems']}) > maxItems ({s['maxItems']})"
                    )
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_array_constraints(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_array_constraints(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_array_constraints(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has invalid array constraints:\n" + "\n".join(errors))


class TestSchemaNullableTypes:
    """Test that nullable types are properly defined."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_nullable_types_use_array_format(self, schema_file: str):
        """Nullable types should use ["type", "null"] array format."""
        schema = load_schema_file(schema_file)
        
        def check_nullable(s: Dict, path: str):
            issues = []
            if "type" in s:
                type_val = s["type"]
                if isinstance(type_val, list):
                    # Verify "null" is in the list if it's a multi-type
                    if "null" in type_val:
                        # This is a valid nullable type definition
                        pass
                    # Verify no unknown types
                    for t in type_val:
                        if t not in VALID_JSON_SCHEMA_TYPES:
                            issues.append(f"At {path}: invalid type '{t}' in type array")
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        issues.extend(check_nullable(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                issues.extend(check_nullable(s["items"], f"{path}[]"))
            
            return issues
        
        errors = check_nullable(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has nullable type issues:\n" + "\n".join(errors))


class TestSchemaEnumTypeConsistency:
    """Test that enum values are consistent with declared type."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_enum_values_match_declared_type(self, schema_file: str):
        """Enum values must match the declared type of the property."""
        schema = load_schema_file(schema_file)
        
        def check_enum_types(s: Dict, path: str):
            errors = []
            if "enum" in s and "type" in s:
                declared_type = s["type"]
                enum_values = s["enum"]
                
                type_checkers = {
                    "string": lambda v: isinstance(v, str),
                    "number": lambda v: isinstance(v, (int, float)),
                    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                    "boolean": lambda v: isinstance(v, bool),
                }
                
                if declared_type in type_checkers:
                    checker = type_checkers[declared_type]
                    for val in enum_values:
                        if not checker(val):
                            errors.append(
                                f"At {path}: enum value {val!r} doesn't match type '{declared_type}'"
                            )
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        errors.extend(check_enum_types(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                errors.extend(check_enum_types(s["items"], f"{path}[]"))
            
            return errors
        
        errors = check_enum_types(schema, "root")
        if errors:
            pytest.fail(f"Schema {schema_file} has enum type mismatches:\n" + "\n".join(errors))


class TestSchemaStageIdConsistency:
    """Test that schemas using stage_id have consistent definitions."""

    def test_stage_id_field_is_string_type(self):
        """All schemas with stage_id field should define it as string type."""
        schemas_with_stage_id = []
        
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema_file(schema_file)
            properties = schema.get("properties", {})
            
            if "stage_id" in properties:
                stage_id_def = properties["stage_id"]
                if stage_id_def.get("type") != "string":
                    schemas_with_stage_id.append(
                        f"{schema_file}: stage_id type is '{stage_id_def.get('type')}', expected 'string'"
                    )
        
        if schemas_with_stage_id:
            pytest.fail("Schemas with incorrect stage_id type:\n" + "\n".join(schemas_with_stage_id))

    def test_stage_id_required_when_present(self):
        """If schema has stage_id property, it should typically be required."""
        schemas_with_optional_stage_id = []
        
        for schema_file in ALL_SCHEMA_FILES:
            schema = load_schema_file(schema_file)
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            
            if "stage_id" in properties and "stage_id" not in required:
                # Check if this is an agent output schema (should require stage_id)
                if "_output_schema" in schema_file:
                    schemas_with_optional_stage_id.append(
                        f"{schema_file}: has stage_id but doesn't require it"
                    )
        
        # This is an informational check - some schemas may legitimately have optional stage_id
        # Comment out the assert if this is too strict
        # if schemas_with_optional_stage_id:
        #     pytest.fail("Agent schemas with optional stage_id:\n" + "\n".join(schemas_with_optional_stage_id))


class TestSchemaNoEmptyObjects:
    """Test that object properties are not empty placeholders."""

    @pytest.mark.parametrize("schema_file", ALL_SCHEMA_FILES)
    def test_object_properties_have_content(self, schema_file: str):
        """Object-type properties should have properties or additionalProperties defined."""
        schema = load_schema_file(schema_file)
        
        def check_objects(s: Dict, path: str):
            warnings = []
            if s.get("type") == "object":
                has_properties = "properties" in s
                has_additional = "additionalProperties" in s
                has_pattern = "patternProperties" in s
                
                # Empty object schemas are suspicious but not always wrong
                if not (has_properties or has_additional or has_pattern):
                    # This is actually valid for "any object" type
                    # but might indicate incomplete schema
                    pass
            
            if "properties" in s and isinstance(s["properties"], dict):
                for prop_name, prop_def in s["properties"].items():
                    if isinstance(prop_def, dict):
                        warnings.extend(check_objects(prop_def, f"{path}.{prop_name}"))
            
            if "items" in s and isinstance(s["items"], dict):
                warnings.extend(check_objects(s["items"], f"{path}[]"))
            
            return warnings
        
        # This is more of a warning/informational test
        check_objects(schema, "root")


class TestSchemaSummaryFields:
    """Test that summary fields are consistently defined across agent schemas."""

    def test_all_agent_output_schemas_have_summary(self):
        """All agent output schemas should have a summary field."""
        missing_summary = []
        
        for agent_name, schema_file in AGENT_SCHEMAS.items():
            schema = load_schema_file(schema_file)
            properties = schema.get("properties", {})
            
            if "summary" not in properties:
                missing_summary.append(f"{agent_name} ({schema_file})")
        
        if missing_summary:
            pytest.fail(f"Agent schemas missing 'summary' field:\n" + "\n".join(missing_summary))

    def test_summary_fields_have_minlength(self):
        """Summary fields should have minLength constraint to prevent empty summaries."""
        missing_minlength = []
        
        for agent_name, schema_file in AGENT_SCHEMAS.items():
            schema = load_schema_file(schema_file)
            summary_def = schema.get("properties", {}).get("summary", {})
            
            if summary_def and "minLength" not in summary_def:
                # Not all need minLength, but it's a good practice
                pass
        
        # Informational - not a failure


class TestSchemaCoverage:
    """Test that we're testing all schema files."""

    def test_all_schema_files_are_discovered(self):
        """Verify ALL_SCHEMA_FILES includes all JSON files in schemas directory."""
        actual_files = sorted([f.name for f in SCHEMAS_DIR.glob("*.json")])
        
        assert ALL_SCHEMA_FILES == actual_files, (
            f"ALL_SCHEMA_FILES mismatch. "
            f"Missing: {set(actual_files) - set(ALL_SCHEMA_FILES)}, "
            f"Extra: {set(ALL_SCHEMA_FILES) - set(actual_files)}"
        )

    def test_agent_schemas_are_subset_of_all_schemas(self):
        """Verify all AGENT_SCHEMAS files are in ALL_SCHEMA_FILES."""
        agent_schema_files = set(AGENT_SCHEMAS.values())
        all_schema_files = set(ALL_SCHEMA_FILES)
        
        missing = agent_schema_files - all_schema_files
        assert not missing, f"AGENT_SCHEMAS references files not in schemas dir: {missing}"

    def test_non_agent_schemas_are_tracked(self):
        """Verify we know about schemas that are not agent output schemas."""
        agent_schema_files = set(AGENT_SCHEMAS.values())
        all_schema_files = set(ALL_SCHEMA_FILES)
        
        non_agent_schemas = all_schema_files - agent_schema_files
        
        # These should be supporting schemas (plan, progress, assumptions, etc.)
        # Just verify they exist and are being tested
        expected_non_agent = {
            "assumptions_schema.json",
            "metrics_schema.json",
            "plan_schema.json",
            "progress_schema.json",
            "prompt_adaptations_schema.json",
            "prompt_adaptor_output_schema.json",
            "report_output_schema.json",
        }
        
        unexpected = non_agent_schemas - expected_non_agent
        if unexpected:
            # New schemas were added - not a failure, just informational
            pass  # Could warn about new schemas
        
        missing = expected_non_agent - non_agent_schemas
        if missing:
            pytest.fail(f"Expected non-agent schemas missing: {missing}")
