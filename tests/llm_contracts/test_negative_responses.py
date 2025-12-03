"""Negative/malformed response tests to ensure schema enforcement."""

import pytest
from jsonschema import ValidationError, validate

from .helpers import load_schema


class TestMalformedResponses:
    """Test handling of malformed LLM responses (Negative Testing)."""

    def test_missing_required_field_detected(self):
        """Missing required fields should be detected."""
        response = {"issues": [], "summary": "Test"}  # Missing verdict
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value)

    def test_wrong_type_detected(self):
        """Wrong field types should be detected."""
        response = {
            "verdict": "approve",
            "issues": "not an array",
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "issues" in str(exc_info.value) or "array" in str(exc_info.value)

    def test_invalid_enum_detected(self):
        """Invalid enum values should be detected."""
        response = {
            "verdict": "maybe",
            "issues": [],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "maybe" in str(exc_info.value) or "enum" in str(exc_info.value)

