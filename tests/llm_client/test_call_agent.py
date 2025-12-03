"""Tests for `src.llm_client.call_agent`."""

import pytest
from unittest.mock import MagicMock, patch, call
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_client import (
    call_agent,
    get_agent_schema,
    load_schema,
    SCHEMAS_DIR,
)


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentBasic:
    """Basic functionality tests for call_agent."""

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_returns_tool_call_args(self, mock_get_client):
        """Verify call_agent extracts and returns args from tool_calls."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        expected_output = {"verdict": "approve", "summary": "Test passed", "nested": {"key": "value"}}
        mock_response.tool_calls = [{"args": expected_output}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="plan_reviewer",
            system_prompt="You are a plan reviewer.",
            user_content="Review this plan.",
        )

        # Verify result is exactly the args from tool_calls
        assert result == expected_output
        assert result["verdict"] == "approve"
        assert result["summary"] == "Test passed"
        assert result["nested"]["key"] == "value"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_uses_first_tool_call_when_multiple(self, mock_get_client):
        """When multiple tool_calls are returned, use the first one."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        first_output = {"verdict": "approve", "reason": "first"}
        second_output = {"verdict": "reject", "reason": "second"}
        mock_response.tool_calls = [{"args": first_output}, {"args": second_output}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="plan_reviewer",
            system_prompt="prompt",
            user_content="content",
        )

        # Should use first tool call
        assert result["verdict"] == "approve"
        assert result["reason"] == "first"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_returns_empty_dict_from_tool_call(self, mock_get_client):
        """Verify empty dict args are returned correctly."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="plan_reviewer",
            system_prompt="prompt",
            user_content="content",
        )

        assert result == {}


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentToolBinding:
    """Tests for tool binding configuration."""

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_binds_correct_tool_name(self, mock_get_client):
        """Tool name should be submit_{agent_name}_output."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
        )

        mock_llm.bind_tools.assert_called_once()
        _, kwargs = mock_llm.bind_tools.call_args
        tools = kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "submit_planner_output"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_binds_correct_tool_description(self, mock_get_client):
        """Tool description should mention the agent name and MUST use directive."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="simulation_designer",
            system_prompt="prompt",
            user_content="content",
        )

        _, kwargs = mock_llm.bind_tools.call_args
        tools = kwargs["tools"]
        description = tools[0]["description"]
        assert "simulation_designer" in description
        assert "MUST" in description
        assert "Submit the simulation_designer agent's structured output" in description

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_uses_tool_choice_auto(self, mock_get_client):
        """Tool choice should be auto for compatibility with thinking mode."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
        )

        _, kwargs = mock_llm.bind_tools.call_args
        assert kwargs["tool_choice"] == {"type": "auto"}

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_passes_schema_to_tool(self, mock_get_client):
        """Verify the schema is passed as input_schema in the tool."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        # Load the expected schema for comparison
        expected_schema = get_agent_schema("planner")

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
        )

        _, kwargs = mock_llm.bind_tools.call_args
        tools = kwargs["tools"]
        input_schema = tools[0]["input_schema"]
        assert input_schema == expected_schema


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentSchemaLoading:
    """Tests for schema loading and override functionality."""

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_uses_schema_name_override(self, mock_get_client):
        """When schema_name is provided, use it instead of agent schema."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        # Use a different schema than the agent's default
        override_schema = load_schema("code_generator_output_schema")

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
            schema_name="code_generator_output_schema",
        )

        _, kwargs = mock_llm.bind_tools.call_args
        tools = kwargs["tools"]
        input_schema = tools[0]["input_schema"]
        assert input_schema == override_schema

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_special_mapping_for_report(self, mock_get_client):
        """Agent 'report' should use 'report_schema' (special case)."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        # Load the expected report schema
        expected_schema = load_schema("report_schema")

        call_agent(
            agent_name="report",
            system_prompt="prompt",
            user_content="content",
        )

        _, kwargs = mock_llm.bind_tools.call_args
        tools = kwargs["tools"]
        input_schema = tools[0]["input_schema"]
        assert input_schema == expected_schema
        # Verify it has specific fields from report_schema
        assert "paper_id" in input_schema.get("properties", {})
        assert "executive_summary" in input_schema.get("properties", {})

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_unknown_agent_raises_error(self, mock_get_client):
        """Unknown agent without schema should raise ValueError."""
        mock_llm = MagicMock()
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError, match="Unknown agent: nonexistent_agent"):
            call_agent(
                agent_name="nonexistent_agent",
                system_prompt="prompt",
                user_content="content",
            )


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentMessageConstruction:
    """Tests for message construction."""

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_message_construction_text(self, mock_get_client):
        """Verify messages are constructed correctly for text content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        system_prompt = "System prompt with specific instructions"
        user_content = "User content with specific question"

        call_agent(agent_name="planner", system_prompt=system_prompt, user_content=user_content)

        mock_bound.invoke.assert_called_once()
        messages = mock_bound.invoke.call_args[0][0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == system_prompt
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == user_content

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_message_construction_with_list_content(self, mock_get_client):
        """Verify list content is used as-is when no images."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        system_prompt = "System prompt"
        user_content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]

        call_agent(agent_name="planner", system_prompt=system_prompt, user_content=user_content)

        mock_bound.invoke.assert_called_once()
        messages = mock_bound.invoke.call_args[0][0]

        assert len(messages) == 2
        assert isinstance(messages[1], HumanMessage)
        # Without images, list content is passed as-is
        assert messages[1].content == user_content

    @patch("src.llm_client.get_llm_client")
    @patch("src.llm_client.encode_image_to_base64")
    @patch("src.llm_client.Path")
    def test_call_agent_message_construction_multimodal_single_image(
        self, mock_path, mock_encode, mock_get_client
    ):
        """Verify multimodal message with single image."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.suffix = ".png"
        mock_encode.return_value = "base64_encoded_image_data"

        system_prompt = "System prompt"
        user_content = "Analyze this image"
        image_path = "test_image.png"

        call_agent(
            agent_name="planner",
            system_prompt=system_prompt,
            user_content=user_content,
            images=[image_path],
        )

        mock_bound.invoke.assert_called_once()
        messages = mock_bound.invoke.call_args[0][0]

        assert len(messages) == 2
        assert isinstance(messages[1], HumanMessage)
        content = messages[1].content
        assert isinstance(content, list)
        assert len(content) == 2

        # First element is text
        assert content[0]["type"] == "text"
        assert content[0]["text"] == user_content

        # Second element is image
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,base64_encoded_image_data"
        assert content[1]["image_url"]["detail"] == "auto"

    @patch("src.llm_client.get_llm_client")
    @patch("src.llm_client.encode_image_to_base64")
    @patch("src.llm_client.Path")
    def test_call_agent_message_construction_multimodal_multiple_images(
        self, mock_path, mock_encode, mock_get_client
    ):
        """Verify multimodal message with multiple images."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        # Set up mock to return different suffixes based on path
        path_mock = MagicMock()
        path_mock.exists.return_value = True
        suffixes = {".png": "image/png", ".jpg": "image/jpeg"}
        call_count = [0]

        def path_side_effect(p):
            pm = MagicMock()
            pm.exists.return_value = True
            pm.suffix = ".png" if "png" in str(p) else ".jpg"
            return pm

        mock_path.side_effect = path_side_effect
        mock_encode.side_effect = ["base64_img1", "base64_img2", "base64_img3"]

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="Analyze these images",
            images=["img1.png", "img2.jpg", "img3.png"],
        )

        mock_bound.invoke.assert_called_once()
        messages = mock_bound.invoke.call_args[0][0]
        content = messages[1].content

        # 1 text + 3 images = 4 content parts
        assert len(content) == 4
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"
        assert content[3]["type"] == "image_url"

    @patch("src.llm_client.get_llm_client")
    @patch("src.llm_client.encode_image_to_base64")
    @patch("src.llm_client.Path")
    def test_call_agent_multimodal_with_list_content(self, mock_path, mock_encode, mock_get_client):
        """When user_content is a list and images provided, extend the list."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.suffix = ".png"
        mock_encode.return_value = "base64data"

        user_content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content=user_content,
            images=["image.png"],
        )

        mock_bound.invoke.assert_called_once()
        messages = mock_bound.invoke.call_args[0][0]
        content = messages[1].content

        # Original list extended with image
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Part 1"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Part 2"
        assert content[2]["type"] == "image_url"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_empty_string_content(self, mock_get_client):
        """Empty string content should be passed through."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(agent_name="planner", system_prompt="prompt", user_content="")

        messages = mock_bound.invoke.call_args[0][0]
        assert messages[1].content == ""


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentJsonFallback:
    """Tests for JSON fallback when tool_calls is empty."""

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_valid_code_block(self, mock_get_client):
        """Extract JSON from code block when no tool_calls."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = 'Here is my analysis:\n```json\n{"verdict": "approve", "reason": "good"}\n```'

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result == {"verdict": "approve", "reason": "good"}

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_code_block_without_json_tag(self, mock_get_client):
        """Extract JSON from code block without explicit json tag."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = '```\n{"verdict": "approve"}\n```'

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result["verdict"] == "approve"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_multiple_code_blocks_uses_last_valid(self, mock_get_client):
        """When multiple code blocks, use the last valid one (reversed iteration)."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        # First block is invalid JSON, second is valid - second should be used
        mock_response.content = '''
        First attempt:
        ```json
        {invalid json here}
        ```
        
        Corrected:
        ```json
        {"verdict": "needs_revision", "reason": "corrected"}
        ```
        '''

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        # Should use the last (valid) code block
        assert result["verdict"] == "needs_revision"
        assert result["reason"] == "corrected"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_multiple_valid_code_blocks_uses_last(self, mock_get_client):
        """When multiple valid code blocks, the last one takes precedence."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = '''
        ```json
        {"verdict": "first"}
        ```
        
        Actually, let me revise:
        ```json
        {"verdict": "second"}
        ```
        
        Final answer:
        ```json
        {"verdict": "last"}
        ```
        '''

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        # Should use the last valid code block
        assert result["verdict"] == "last"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_greedy_search_when_no_code_blocks(self, mock_get_client):
        """Fall back to greedy JSON extraction when no code blocks."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        # No code blocks, but valid JSON in text
        mock_response.content = 'My response is {"verdict": "approve", "notes": "found via greedy"}'

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result["verdict"] == "approve"
        assert result["notes"] == "found via greedy"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_greedy_extracts_largest_object(self, mock_get_client):
        """Greedy search should find a complete JSON object."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = 'I see a set {1, 2, 3} in the data. Result: {"verdict": "approve", "data": {"nested": true}}'

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        # Should find the proper JSON object, not the set notation
        assert result["verdict"] == "approve"
        assert result["data"]["nested"] is True

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_with_noise_and_set_notation(self, mock_get_client):
        """Handle content with Python set notation that looks like JSON."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = '''
        Thinking: I need to check if x in {1, 2, 3}.

        Here is the output:
        ```json
        {
            "verdict": "approve"
        }
        ```
        '''

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")
        assert result["verdict"] == "approve"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_content_as_list_thinking_blocks(self, mock_get_client):
        """Handle response content as list (thinking blocks)."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        # Content as list with dict blocks (like Claude's thinking)
        mock_response.content = [
            {"type": "thinking", "thinking": "Let me analyze this..."},
            {"type": "text", "text": '```json\n{"verdict": "approve"}\n```'},
        ]

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result["verdict"] == "approve"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_content_as_list_non_dict_blocks(self, mock_get_client):
        """Handle response content as list with non-dict items."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        # Content as list with string blocks
        mock_response.content = [
            "Some thinking...",
            '{"verdict": "approve"}',
        ]

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result["verdict"] == "approve"

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_invalid_no_json_raises_error(self, mock_get_client):
        """Raise ValueError when no valid JSON found."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Just some text without any JSON at all."

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError) as exc_info:
            call_agent("plan_reviewer", "prompt", "content", max_retries=1)

        assert "did not return structured output" in str(exc_info.value)
        assert "plan_reviewer" in str(exc_info.value)

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_invalid_json_in_code_block(self, mock_get_client):
        """Raise ValueError when code block contains invalid JSON and greedy fails."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = '```json\n{invalid: json}\n```'

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError, match="did not return structured output"):
            call_agent("plan_reviewer", "prompt", "content", max_retries=1)

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_empty_content(self, mock_get_client):
        """Handle empty content string."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = ""

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError, match="did not return structured output"):
            call_agent("plan_reviewer", "prompt", "content", max_retries=1)

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_error_includes_content_preview(self, mock_get_client):
        """Error message should include preview of response content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "No JSON here but some meaningful text"

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError) as exc_info:
            call_agent("plan_reviewer", "prompt", "content", max_retries=1)

        # Should show preview of content in error
        assert "No JSON here" in str(exc_info.value)

    @patch("src.llm_client.get_llm_client")
    def test_json_fallback_none_tool_calls(self, mock_get_client):
        """Handle None tool_calls (falsy check)."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = None
        mock_response.content = '{"verdict": "approve"}'

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result["verdict"] == "approve"


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentRetryMechanism:
    """Tests for retry mechanism."""

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.get_llm_client")
    def test_retry_on_transient_error_succeeds(self, mock_get_client, mock_sleep):
        """Retry on transient errors and succeed eventually."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"verdict": "pass"}}]

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient network error")
            return mock_response

        mock_llm.bind_tools.return_value.invoke.side_effect = side_effect
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="execution_validator",
            system_prompt="You are a validator.",
            user_content="Validate this.",
            max_retries=3,
        )

        assert result["verdict"] == "pass"
        assert call_count[0] == 3
        # Verify sleep was called for exponential backoff
        assert mock_sleep.call_count == 2  # Called before 2nd and 3rd attempts

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.get_llm_client")
    def test_retry_exponential_backoff(self, mock_get_client, mock_sleep):
        """Verify exponential backoff delay on retries."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 4:
                raise TimeoutError("Request timeout")
            return mock_response

        mock_llm.bind_tools.return_value.invoke.side_effect = side_effect
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
            max_retries=4,
        )

        # Verify delays increase: 2*1=2, 2*2=4, 2*3=6
        expected_delays = [2.0, 4.0, 6.0]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("src.llm_client.get_llm_client")
    def test_retry_exhausted_raises_runtime_error(self, mock_get_client):
        """Raise RuntimeError when all retries exhausted."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.side_effect = ConnectionError("Persistent error")
        mock_get_client.return_value = mock_llm

        with pytest.raises(RuntimeError) as exc_info:
            call_agent(
                agent_name="planner",
                system_prompt="prompt",
                user_content="content",
                max_retries=3,
            )

        error_msg = str(exc_info.value)
        assert "failed after 3 attempts" in error_msg
        assert "planner" in error_msg
        assert "Persistent error" in error_msg

    @patch("src.llm_client.get_llm_client")
    def test_zero_retries_raises_immediately(self, mock_get_client):
        """Zero retries should raise RuntimeError immediately."""
        mock_llm = MagicMock()
        mock_get_client.return_value = mock_llm

        with pytest.raises(RuntimeError) as exc_info:
            call_agent(
                agent_name="planner",
                system_prompt="prompt",
                user_content="content",
                max_retries=0,
            )

        assert "failed after 0 attempts" in str(exc_info.value)

    @patch("src.llm_client.get_llm_client")
    def test_no_retry_on_validation_error(self, mock_get_client):
        """ValueError should not trigger retry."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "No JSON here"

        call_count = [0]

        def invoke_side_effect(*args, **kwargs):
            call_count[0] += 1
            return mock_response

        mock_llm.bind_tools.return_value.invoke.side_effect = invoke_side_effect
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError, match="did not return structured output"):
            call_agent(
                agent_name="planner",
                system_prompt="Test",
                user_content="Test",
                max_retries=5,  # High retry count, but should only call once
            )

        # Should only be called once (no retries on ValueError)
        assert call_count[0] == 1

    @patch("src.llm_client.get_llm_client")
    def test_retry_various_transient_errors(self, mock_get_client):
        """Retry on various types of transient errors."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "success"}}]

        errors = [
            ConnectionError("Network error"),
            TimeoutError("Timeout"),
            IOError("IO error"),
        ]
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= len(errors):
                raise errors[call_count[0] - 1]
            return mock_response

        mock_llm.bind_tools.return_value.invoke.side_effect = side_effect
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
            max_retries=5,
        )

        assert result["output"] == "success"
        assert call_count[0] == 4  # 3 errors + 1 success


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentModelOverride:
    """Tests for model override parameter."""

    @patch("src.llm_client.get_llm_client")
    def test_model_override_passed_to_client(self, mock_get_client):
        """Verify model parameter is passed to get_llm_client."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
            model="claude-sonnet-4-20250514",
        )

        mock_get_client.assert_called_once_with("claude-sonnet-4-20250514")

    @patch("src.llm_client.get_llm_client")
    def test_no_model_override_passes_none(self, mock_get_client):
        """When no model specified, None is passed to get_llm_client."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
        )

        mock_get_client.assert_called_once_with(None)


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentEdgeCases:
    """Edge case tests."""

    @patch("src.llm_client.get_llm_client")
    def test_special_characters_in_agent_name(self, mock_get_client):
        """Agent names with underscores work correctly."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="code_generator",
            system_prompt="prompt",
            user_content="content",
        )

        _, kwargs = mock_llm.bind_tools.call_args
        assert kwargs["tools"][0]["name"] == "submit_code_generator_output"

    @patch("src.llm_client.get_llm_client")
    def test_very_long_content(self, mock_get_client):
        """Handle very long content strings."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        long_content = "x" * 100000  # 100k characters

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content=long_content,
        )

        messages = mock_bound.invoke.call_args[0][0]
        assert messages[1].content == long_content

    @patch("src.llm_client.get_llm_client")
    def test_unicode_in_content(self, mock_get_client):
        """Handle unicode characters in content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "日本語"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        unicode_content = "分析这篇论文 📊 θ = π/4"

        result = call_agent(
            agent_name="planner",
            system_prompt="system 日本語",
            user_content=unicode_content,
        )

        messages = mock_bound.invoke.call_args[0][0]
        assert messages[0].content == "system 日本語"
        assert messages[1].content == unicode_content
        assert result["output"] == "日本語"

    @patch("src.llm_client.get_llm_client")
    def test_newlines_in_content(self, mock_get_client):
        """Handle newlines and formatting in content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        multiline_content = """Line 1
        Line 2 with indent
        
        Line 4 after blank"""

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content=multiline_content,
        )

        messages = mock_bound.invoke.call_args[0][0]
        assert messages[1].content == multiline_content

    @patch("src.llm_client.get_llm_client")
    def test_complex_nested_json_in_args(self, mock_get_client):
        """Handle complex nested structures in tool call args."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        complex_output = {
            "verdict": "approve",
            "nested": {
                "level1": {
                    "level2": {
                        "array": [1, 2, {"deep": "value"}],
                        "null_val": None,
                        "bool_val": True,
                    }
                }
            },
            "list_of_objects": [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"},
            ],
        }
        mock_response.tool_calls = [{"args": complex_output}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="plan_reviewer",
            system_prompt="prompt",
            user_content="content",
        )

        assert result == complex_output
        assert result["nested"]["level1"]["level2"]["array"][2]["deep"] == "value"
        assert result["nested"]["level1"]["level2"]["null_val"] is None
        assert result["list_of_objects"][1]["name"] == "second"


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgentInvokeInteraction:
    """Tests verifying correct interaction with LLM invoke."""

    @patch("src.llm_client.get_llm_client")
    def test_invoke_called_exactly_once_on_success(self, mock_get_client):
        """Invoke should be called exactly once on successful response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
        )

        assert mock_bound.invoke.call_count == 1

    @patch("src.llm_client.get_llm_client")
    def test_bind_tools_called_exactly_once(self, mock_get_client):
        """bind_tools should be called exactly once per call_agent."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        call_agent(
            agent_name="planner",
            system_prompt="prompt",
            user_content="content",
        )

        assert mock_llm.bind_tools.call_count == 1
