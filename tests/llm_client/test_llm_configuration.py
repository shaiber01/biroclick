"""LLM client configuration tests."""

import pytest
from unittest.mock import patch

from src.llm_client import DEFAULT_MODEL, get_llm_client


@pytest.mark.usefixtures("fresh_llm_client")
class TestLLMConfiguration:
    """Tests for `get_llm_client` factory options."""

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_defaults(self, mock_chat):
        get_llm_client()

        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args

        assert kwargs.get("model") == DEFAULT_MODEL
        assert kwargs.get("max_tokens") == 16384
        assert kwargs.get("temperature") == 1.0
        assert kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 10000}
        assert kwargs.get("timeout") == 300.0

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_override(self, mock_chat):
        get_llm_client(model="claude-3-sonnet")
        _, kwargs = mock_chat.call_args
        assert kwargs.get("model") == "claude-3-sonnet"


