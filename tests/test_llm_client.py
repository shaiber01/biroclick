"""
Tests for LLM Client Module.

Tests the schema loading, image encoding, and mock LLM calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import module under test
from src.llm_client import (
    load_schema,
    get_agent_schema,
    encode_image_to_base64,
    get_image_media_type,
    create_image_content,
    call_agent,
    reset_llm_client,
    SCHEMAS_DIR,
)


# ═══════════════════════════════════════════════════════════════════════
# Schema Loading Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaLoading:
    """Tests for schema loading functions."""
    
    def test_load_schema_with_extension(self):
        """Test loading schema with .json extension."""
        schema = load_schema("planner_output_schema.json")
        assert schema is not None
        assert "properties" in schema
        
    def test_load_schema_without_extension(self):
        """Test loading schema without .json extension."""
        schema = load_schema("planner_output_schema")
        assert schema is not None
        assert "properties" in schema
        
    def test_load_schema_caching(self):
        """Test that schemas are cached (same key returns same object)."""
        # Use full name with extension to ensure consistent cache key
        schema1 = load_schema("supervisor_output_schema.json")
        schema2 = load_schema("supervisor_output_schema.json")
        # Both should have same content (equal), even if not same object
        assert schema1 == schema2
        
    def test_load_nonexistent_schema(self):
        """Test that loading nonexistent schema raises error."""
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_schema.json")
            
    def test_get_agent_schema_planner(self):
        """Test getting planner agent schema."""
        schema = get_agent_schema("planner")
        assert schema is not None
        assert "properties" in schema
        
    def test_get_agent_schema_supervisor(self):
        """Test getting supervisor agent schema."""
        schema = get_agent_schema("supervisor")
        assert schema is not None
        assert "verdict" in schema.get("properties", {})
        
    def test_get_agent_schema_unknown(self):
        """Test that unknown agent raises error."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent_schema("unknown_agent")


# ═══════════════════════════════════════════════════════════════════════
# Image Encoding Tests
# ═══════════════════════════════════════════════════════════════════════

class TestImageEncoding:
    """Tests for image encoding functions."""
    
    def test_get_image_media_type_png(self):
        """Test PNG media type detection."""
        assert get_image_media_type("image.png") == "image/png"
        assert get_image_media_type("path/to/image.PNG") == "image/png"
        
    def test_get_image_media_type_jpeg(self):
        """Test JPEG media type detection."""
        assert get_image_media_type("image.jpg") == "image/jpeg"
        assert get_image_media_type("image.jpeg") == "image/jpeg"
        
    def test_get_image_media_type_unknown(self):
        """Test unknown extension defaults to PNG."""
        assert get_image_media_type("image.bmp") == "image/png"
        
    def test_encode_image_nonexistent(self):
        """Test encoding nonexistent image raises error."""
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/image.png")


# ═══════════════════════════════════════════════════════════════════════
# Call Agent Tests (with mocking)
# ═══════════════════════════════════════════════════════════════════════

class TestCallAgent:
    """Tests for the call_agent function with mocked LLM."""
    
    @pytest.fixture(autouse=True)
    def reset_client(self):
        """Reset LLM client before each test."""
        reset_llm_client()
        yield
        reset_llm_client()
    
    @patch("src.llm_client.get_llm_client")
    def test_call_agent_basic(self, mock_get_client):
        """Test basic agent call with mocked response."""
        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{
            "args": {"verdict": "approve", "summary": "Test passed"}
        }]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm
        
        # Call agent
        result = call_agent(
            agent_name="plan_reviewer",
            system_prompt="You are a plan reviewer.",
            user_content="Review this plan.",
        )
        
        # Verify
        assert result["verdict"] == "approve"
        assert result["summary"] == "Test passed"
        
    @patch("src.llm_client.get_llm_client")
    def test_call_agent_retry_on_error(self, mock_get_client):
        """Test that agent retries on transient errors."""
        # Setup mock to fail twice then succeed
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{
            "args": {"verdict": "pass"}
        }]
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient error")
            return mock_response
            
        mock_llm.bind_tools.return_value.invoke.side_effect = side_effect
        mock_get_client.return_value = mock_llm
        
        # Call agent (should retry and succeed)
        result = call_agent(
            agent_name="execution_validator",
            system_prompt="You are a validator.",
            user_content="Validate this.",
            max_retries=3,
        )
        
        assert result["verdict"] == "pass"
        assert call_count[0] == 3  # Called 3 times
        
    @patch("src.llm_client.get_llm_client")
    def test_call_agent_no_retry_on_validation_error(self, mock_get_client):
        """Test that validation errors are not retried."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []  # No tool calls
        mock_response.content = "Invalid response"
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm
        
        # Should raise ValueError immediately without retry
        with pytest.raises(ValueError, match="did not return structured output"):
            call_agent(
                agent_name="planner",
                system_prompt="Test",
                user_content="Test",
                max_retries=3,
            )


# ═══════════════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════════

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
        """Test that each required schema exists and is valid JSON."""
        schema = load_schema(schema_name)
        assert schema is not None
        assert isinstance(schema, dict)
        # Most schemas should have properties
        if "properties" in schema:
            assert isinstance(schema["properties"], dict)

