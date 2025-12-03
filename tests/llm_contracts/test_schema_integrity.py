"""Schema integrity tests for agent JSON schemas."""

import json

import pytest

from .helpers import SCHEMAS_DIR


class TestSchemaCompleteness:
    """Test that all agent schemas exist and are valid JSON schemas."""

    @pytest.mark.parametrize(
        "schema_file",
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
    def test_schema_is_valid_json(self, schema_file):
        """Each schema file must be valid JSON and have schema structure."""
        path = SCHEMAS_DIR / schema_file
        assert path.exists()
        with open(path) as file:
            schema = json.load(file)

        assert "type" in schema or "$ref" in schema or "properties" in schema

