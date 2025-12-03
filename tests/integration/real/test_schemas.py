"""Integration tests that verify every schema file exists and parses."""

import json
from pathlib import Path

import pytest


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
        "report_schema.json",
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


