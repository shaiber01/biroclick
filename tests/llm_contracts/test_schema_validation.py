"""Schema validation tests for LLM mock responses."""

import pytest
from jsonschema import validate

from .helpers import AGENT_SCHEMAS, load_mock_response, load_schema


class TestMockResponsesFullSchemaValidation:
    """Strict validation: mock responses MUST fully conform to their schemas."""

    @pytest.mark.parametrize("agent_name,schema_file", list(AGENT_SCHEMAS.items()))
    def test_mock_response_fully_validates(self, agent_name, schema_file):
        """Mock response must fully validate against its schema."""
        try:
            response = load_mock_response(agent_name)
            schema = load_schema(schema_file)
            validate(instance=response, schema=schema)
        except FileNotFoundError as exc:
            pytest.skip(f"File not found: {exc}")

