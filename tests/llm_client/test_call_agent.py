"""Tests for `src.llm_client.call_agent`."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_client import call_agent


@pytest.mark.usefixtures("fresh_llm_client")
class TestCallAgent:
    """Tests for the call_agent function with mocked LLM."""

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_basic(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"verdict": "approve", "summary": "Test passed"}}]
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent(
            agent_name="plan_reviewer",
            system_prompt="You are a plan reviewer.",
            user_content="Review this plan.",
        )

        assert result["verdict"] == "approve"
        assert result["summary"] == "Test passed"

        mock_llm.bind_tools.assert_called_once()
        _, kwargs = mock_llm.bind_tools.call_args
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "submit_plan_reviewer_output"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_message_construction_text(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        system_prompt = "System prompt"
        user_content = "User content"

        call_agent(agent_name="planner", system_prompt=system_prompt, user_content=user_content)

        mock_bound.invoke.assert_called_once()
        call_args = mock_bound.invoke.call_args[0][0]

        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == system_prompt
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == user_content

    @patch("src.llm_client.get_llm_client")
    @patch("src.llm_client.encode_image_to_base64")
    @patch("src.llm_client.Path")
    def test_call_agent_message_construction_multimodal(self, mock_path, mock_encode, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"output": "value"}}]
        mock_bound = mock_llm.bind_tools.return_value
        mock_bound.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.suffix = ".png"
        mock_encode.return_value = "base64string"

        system_prompt = "System prompt"
        user_content = "User content"
        image_path = "test_image.png"

        call_agent(
            agent_name="planner",
            system_prompt=system_prompt,
            user_content=user_content,
            images=[image_path],
        )

        mock_bound.invoke.assert_called_once()
        call_args = mock_bound.invoke.call_args[0][0]

        assert len(call_args) == 2
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[1].content, list)

        content = call_args[1].content
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == user_content
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,base64string"
        assert content[1]["image_url"]["detail"] == "auto"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_json_fallback_valid(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Here is my thought process...\n```json\n{\"verdict\": \"approve\"}\n```"

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")

        assert result["verdict"] == "approve"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_json_fallback_with_noise(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = """
        Thinking: I need to check if x in {1, 2, 3}.

        Here is the output:
        ```json
        {
            "verdict": "approve"
        }
        ```
        """

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        result = call_agent("plan_reviewer", "prompt", "content")
        assert result["verdict"] == "approve"

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_json_fallback_invalid(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Just some text without JSON."

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError, match="did not return structured output"):
            call_agent("plan_reviewer", "prompt", "content", max_retries=1)

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_retry_on_error(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"args": {"verdict": "pass"}}]

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient error")
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

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_zero_retries(self, mock_get_client):
        mock_llm = MagicMock()
        mock_get_client.return_value = mock_llm

        with pytest.raises(RuntimeError, match="failed after 0 attempts"):
            call_agent(agent_name="planner", system_prompt="prompt", user_content="content", max_retries=0)

    @patch("src.llm_client.get_llm_client")
    def test_call_agent_no_retry_on_validation_error(self, mock_get_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Invalid response"
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_client.return_value = mock_llm

        with pytest.raises(ValueError, match="did not return structured output"):
            call_agent(agent_name="planner", system_prompt="Test", user_content="Test", max_retries=3)


