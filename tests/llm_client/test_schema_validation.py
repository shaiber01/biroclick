"""Schema validation coverage for LLM client."""

import pytest

from src.llm_client import load_schema


class TestSchemaValidation:
    """Tests that all agent schemas exist and are valid JSON."""

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
        "report_schema",
    ]

    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_exists_and_valid(self, schema_name):
        schema = load_schema(schema_name)
        assert schema is not None
        assert isinstance(schema, dict)
        if "properties" in schema:
            assert isinstance(schema["properties"], dict)


