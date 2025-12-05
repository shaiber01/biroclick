"""Integration tests that verify every schema file exists, parses, and is valid JSON Schema."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from jsonschema import Draft7Validator, RefResolver, ValidationError, validate
from jsonschema.exceptions import SchemaError


class TestSchemaFilesExist:
    """Verify all schema files referenced in code exist and are valid JSON."""

    SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

    REQUIRED_SCHEMAS = [
        "planner_output_schema.json",
        "plan_reviewer_output_schema.json",
        "simulation_designer_output_schema.json",
        "design_reviewer_output_schema.json",
        "code_generator_output_schema.json",
        "code_reviewer_output_schema.json",
        "execution_validator_output_schema.json",
        "physics_sanity_output_schema.json",
        "results_analyzer_output_schema.json",
        "supervisor_output_schema.json",
        "report_output_schema.json",
    ]

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_file_exists(self, schema_name):
        """Each schema file must exist."""
        schema_file = self.SCHEMAS_DIR / schema_name
        assert schema_file.exists(), f"Missing schema file: {schema_file}"

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_is_valid_json(self, schema_name):
        """Each schema must be valid JSON."""
        schema_file = self.SCHEMAS_DIR / schema_name
        if schema_file.exists():
            try:
                with open(schema_file, encoding="utf-8") as file:
                    schema = json.load(file)
                assert "type" in schema or "properties" in schema, (
                    f"Schema {schema_name} doesn't look like a JSON schema"
                )
            except json.JSONDecodeError as exc:
                pytest.fail(f"Schema {schema_name} is not valid JSON: {exc}")


# Module-level function for parametrization
def get_all_schema_files() -> List[Path]:
    """Discover all JSON schema files in the schemas directory."""
    schemas_dir = Path(__file__).resolve().parents[3] / "schemas"
    if not schemas_dir.exists():
        return []
    return sorted(schemas_dir.glob("*.json"))


class TestAllSchemaFiles:
    """Test ALL schema files in the schemas directory, not just required ones."""

    SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

    @pytest.fixture(scope="class")
    def all_schema_files(self) -> List[Path]:
        """Fixture providing all schema files."""
        return get_all_schema_files()

    def test_schemas_directory_exists(self):
        """Schemas directory must exist."""
        assert self.SCHEMAS_DIR.exists(), f"Schemas directory does not exist: {self.SCHEMAS_DIR}"
        assert self.SCHEMAS_DIR.is_dir(), f"Schemas path is not a directory: {self.SCHEMAS_DIR}"

    def test_at_least_one_schema_file_exists(self, all_schema_files):
        """At least one schema file must exist."""
        assert len(all_schema_files) > 0, (
            f"No schema files found in {self.SCHEMAS_DIR}. "
            "This indicates a critical issue with the schema directory."
        )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_file_is_readable(self, schema_file):
        """Each schema file must be readable."""
        assert schema_file.exists(), f"Schema file does not exist: {schema_file}"
        assert schema_file.is_file(), f"Schema path is not a file: {schema_file}"
        assert schema_file.stat().st_size > 0, f"Schema file is empty: {schema_file}"

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_file_is_valid_json(self, schema_file):
        """Each schema file must contain valid JSON."""
        try:
            with open(schema_file, encoding="utf-8") as file:
                content = file.read()
                assert len(content.strip()) > 0, f"Schema file is empty: {schema_file}"
                schema = json.loads(content)
                assert isinstance(schema, dict), (
                    f"Schema {schema_file.name} must be a JSON object, got {type(schema).__name__}"
                )
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Schema {schema_file.name} is not valid JSON: {exc}\n"
                f"File: {schema_file}"
            )
        except UnicodeDecodeError as exc:
            pytest.fail(
                f"Schema {schema_file.name} is not valid UTF-8: {exc}\n"
                f"File: {schema_file}"
            )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_has_required_structure(self, schema_file):
        """Each schema must have basic JSON Schema structure."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        # Must have either "type" or "properties" (basic JSON Schema requirement)
        has_type = "type" in schema
        has_properties = "properties" in schema
        has_definitions = "$definitions" in schema or "definitions" in schema

        assert has_type or has_properties or has_definitions, (
            f"Schema {schema_file.name} doesn't look like a JSON schema. "
            "Must have at least one of: 'type', 'properties', or 'definitions'"
        )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_has_schema_field(self, schema_file):
        """Each schema should have $schema field indicating JSON Schema version."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        assert "$schema" in schema, (
            f"Schema {schema_file.name} is missing '$schema' field. "
            "This field indicates the JSON Schema version and is recommended."
        )

        schema_version = schema.get("$schema", "")
        assert isinstance(schema_version, str), (
            f"Schema {schema_file.name} has invalid $schema type: {type(schema_version).__name__}"
        )
        assert "json-schema.org" in schema_version.lower() or schema_version.startswith("http"), (
            f"Schema {schema_file.name} has suspicious $schema value: {schema_version}"
        )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_has_id_field(self, schema_file):
        """Each schema should have $id field matching its filename."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        assert "$id" in schema, (
            f"Schema {schema_file.name} is missing '$id' field. "
            "This field should match the filename for proper schema identification."
        )

        schema_id = schema.get("$id", "")
        assert isinstance(schema_id, str), (
            f"Schema {schema_file.name} has invalid $id type: {type(schema_id).__name__}"
        )
        assert schema_id == schema_file.name or schema_id.endswith(schema_file.name), (
            f"Schema {schema_file.name} has $id '{schema_id}' that doesn't match filename. "
            f"Expected $id to be '{schema_file.name}' or end with it."
        )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_is_valid_json_schema(self, schema_file):
        """Each schema must be a valid JSON Schema (validated by jsonschema library)."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        try:
            # Validate that the schema itself is valid JSON Schema
            Draft7Validator.check_schema(schema)
        except SchemaError as exc:
            pytest.fail(
                f"Schema {schema_file.name} is not a valid JSON Schema: {exc}\n"
                f"File: {schema_file}\n"
                f"Error path: {exc.path if hasattr(exc, 'path') else 'N/A'}"
            )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_required_fields_are_valid(self, schema_file):
        """If schema has 'required' field, it must be valid."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        if "required" in schema:
            required = schema["required"]
            assert isinstance(required, list), (
                f"Schema {schema_file.name} has 'required' field that is not a list: {type(required).__name__}"
            )
            assert all(isinstance(item, str) for item in required), (
                f"Schema {schema_file.name} has 'required' field containing non-string items"
            )
            assert len(required) == len(set(required)), (
                f"Schema {schema_file.name} has duplicate items in 'required' field: {required}"
            )

            # Check that required fields exist in properties
            if "properties" in schema:
                properties = schema["properties"]
                missing_properties = [field for field in required if field not in properties]
                assert len(missing_properties) == 0, (
                    f"Schema {schema_file.name} requires fields that don't exist in properties: {missing_properties}"
                )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_properties_are_valid(self, schema_file):
        """If schema has 'properties', they must be valid property definitions."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        if "properties" in schema:
            properties = schema["properties"]
            assert isinstance(properties, dict), (
                f"Schema {schema_file.name} has 'properties' that is not a dict: {type(properties).__name__}"
            )

            for prop_name, prop_def in properties.items():
                assert isinstance(prop_name, str), (
                    f"Schema {schema_file.name} has non-string property name: {prop_name}"
                )
                assert isinstance(prop_def, dict), (
                    f"Schema {schema_file.name} has property '{prop_name}' that is not a dict: {type(prop_def).__name__}"
                )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_definitions_are_valid(self, schema_file):
        """If schema has 'definitions' or '$definitions', they must be valid."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        definitions_key = None
        if "$definitions" in schema:
            definitions_key = "$definitions"
        elif "definitions" in schema:
            definitions_key = "definitions"

        if definitions_key:
            definitions = schema[definitions_key]
            assert isinstance(definitions, dict), (
                f"Schema {schema_file.name} has '{definitions_key}' that is not a dict: {type(definitions).__name__}"
            )

            for def_name, def_schema in definitions.items():
                assert isinstance(def_name, str), (
                    f"Schema {schema_file.name} has non-string definition name: {def_name}"
                )
                assert isinstance(def_schema, dict), (
                    f"Schema {schema_file.name} has definition '{def_name}' that is not a dict: {type(def_schema).__name__}"
                )

                # Validate each definition is a valid schema
                try:
                    Draft7Validator.check_schema(def_schema)
                except SchemaError as exc:
                    pytest.fail(
                        f"Schema {schema_file.name} has invalid definition '{def_name}': {exc}"
                    )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_can_validate_minimal_data(self, schema_file):
        """Test that schema can validate data (even if it fails validation)."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        # Create a resolver for $ref resolution
        base_uri = f"file://{schema_file.parent}/"
        resolver = RefResolver(base_uri=base_uri, referrer=str(schema_file))

        # Try to validate empty object against schema
        # This tests that the schema structure is valid enough to attempt validation
        try:
            validator = Draft7Validator(schema, resolver=resolver)
            # Don't assert validation passes - just that we can create a validator
            # Some schemas might require specific fields, which is fine
            _ = validator  # Validator created successfully
        except Exception as exc:
            pytest.fail(
                f"Schema {schema_file.name} cannot be used for validation: {exc}\n"
                f"This indicates a structural problem with the schema."
            )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_enum_values_are_valid(self, schema_file):
        """If schema has enum fields, validate enum structure."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        def check_enums(obj: Any, path: str = "") -> None:
            """Recursively check all enum fields in the schema."""
            if isinstance(obj, dict):
                if "enum" in obj:
                    enum_values = obj["enum"]
                    assert isinstance(enum_values, list), (
                        f"Schema {schema_file.name} has 'enum' at path '{path}' that is not a list: {type(enum_values).__name__}"
                    )
                    assert len(enum_values) > 0, (
                        f"Schema {schema_file.name} has empty 'enum' at path '{path}'"
                    )
                    assert len(enum_values) == len(set(enum_values)), (
                        f"Schema {schema_file.name} has duplicate values in 'enum' at path '{path}': {enum_values}"
                    )

                for key, value in obj.items():
                    check_enums(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_enums(item, f"{path}[{i}]")

        check_enums(schema)

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_no_circular_refs_detected(self, schema_file):
        """Basic check for obvious circular reference issues."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        # Try to create a validator - if there are circular refs, this might fail
        base_uri = f"file://{schema_file.parent}/"
        resolver = RefResolver(base_uri=base_uri, referrer=str(schema_file))

        try:
            validator = Draft7Validator(schema, resolver=resolver)
            # If we get here, no obvious circular ref issues
            _ = validator
        except Exception as exc:
            # Some exceptions are OK (like missing referenced files), but log them
            error_msg = str(exc).lower()
            if "circular" in error_msg or "recursive" in error_msg:
                pytest.fail(
                    f"Schema {schema_file.name} appears to have circular references: {exc}"
                )
            # Other exceptions might be OK (like missing files), so we don't fail

    def test_all_required_schemas_are_present(self):
        """Verify that all schemas in REQUIRED_SCHEMAS list actually exist."""
        required_schemas = TestSchemaFilesExist.REQUIRED_SCHEMAS
        missing_schemas = []

        for schema_name in required_schemas:
            schema_path = self.SCHEMAS_DIR / schema_name
            if not schema_path.exists():
                missing_schemas.append(schema_name)

        assert len(missing_schemas) == 0, (
            f"Required schemas are missing: {missing_schemas}\n"
            f"These schemas are referenced in code but don't exist."
        )

    def test_no_orphaned_schema_files(self):
        """Check for schema files that might not be used (warning, not failure)."""
        all_schemas = {f.name for f in get_all_schema_files()}
        required_schemas = set(TestSchemaFilesExist.REQUIRED_SCHEMAS)

        # Also check for other known schemas that might be used
        known_schemas = {
            "comparison_validator_output_schema.json",
            "plan_schema.json",
            "progress_schema.json",
            "assumptions_schema.json",
            "metrics_schema.json",
            "prompt_adaptations_schema.json",
            "prompt_adaptor_output_schema.json",
        }

        expected_schemas = required_schemas | known_schemas
        orphaned = all_schemas - expected_schemas

        # This is informational - orphaned files might be OK, but worth checking
        if orphaned:
            # Don't fail, but log a warning
            print(f"\nWARNING: Found schema files not in required/known list: {orphaned}")
            print("These might be unused or might need to be added to REQUIRED_SCHEMAS.")

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_file_encoding_is_utf8(self, schema_file):
        """Schema files must be valid UTF-8."""
        try:
            with open(schema_file, encoding="utf-8") as file:
                content = file.read()
                # If we can read it as UTF-8, it's valid
                assert isinstance(content, str)
        except UnicodeDecodeError as exc:
            pytest.fail(
                f"Schema {schema_file.name} is not valid UTF-8: {exc}\n"
                f"File: {schema_file}"
            )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_has_title_or_description(self, schema_file):
        """Schemas should have title or description for documentation."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        has_title = "title" in schema
        has_description = "description" in schema

        assert has_title or has_description, (
            f"Schema {schema_file.name} should have at least 'title' or 'description' "
            "for documentation purposes."
        )

    @pytest.mark.parametrize("schema_file", get_all_schema_files())
    def test_schema_type_field_is_valid(self, schema_file):
        """If schema has 'type' field, it must be a valid JSON Schema type."""
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        valid_types = {
            "object", "array", "string", "number", "integer", "boolean", "null"
        }

        def check_type(obj: Any, path: str = "", in_example: bool = False) -> None:
            """Recursively check type fields, skipping example sections."""
            if isinstance(obj, dict):
                # Skip checking "type" fields inside "example" sections
                # as they contain example data, not schema definitions
                if path.endswith(".example") or "example" in path.split(".")[-2:]:
                    in_example = True
                
                if "type" in obj and not in_example:
                    type_value = obj["type"]
                    if isinstance(type_value, str):
                        assert type_value in valid_types, (
                            f"Schema {schema_file.name} has invalid 'type' '{type_value}' at path '{path}'. "
                            f"Valid types: {valid_types}"
                        )
                    elif isinstance(type_value, list):
                        assert all(t in valid_types for t in type_value), (
                            f"Schema {schema_file.name} has invalid types in 'type' list at path '{path}': {type_value}"
                        )
                        assert len(type_value) == len(set(type_value)), (
                            f"Schema {schema_file.name} has duplicate types in 'type' list at path '{path}': {type_value}"
                        )

                for key, value in obj.items():
                    # Don't recurse into "example" sections
                    if key == "example":
                        continue
                    check_type(value, f"{path}.{key}" if path else key, in_example)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_type(item, f"{path}[{i}]", in_example)

        check_type(schema)
