"""LLM client configuration tests."""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.llm_client import DEFAULT_MODEL, get_llm_client


@pytest.mark.usefixtures("fresh_llm_client")
class TestLLMConfiguration:
    """Tests for `get_llm_client` factory options."""

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_defaults(self, mock_chat):
        """Test that default configuration is correct."""
        client = get_llm_client()

        # Verify ChatAnthropic was called exactly once
        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args

        # Verify all default parameters are set correctly
        assert kwargs.get("model") == DEFAULT_MODEL, f"Expected model {DEFAULT_MODEL}, got {kwargs.get('model')}"
        assert kwargs.get("max_tokens") == 16384, f"Expected max_tokens 16384, got {kwargs.get('max_tokens')}"
        assert kwargs.get("temperature") == 1.0, f"Expected temperature 1.0, got {kwargs.get('temperature')}"
        assert kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 10000}, \
            f"Expected thinking config, got {kwargs.get('thinking')}"
        assert kwargs.get("timeout") == 300.0, f"Expected timeout 300.0, got {kwargs.get('timeout')}"
        
        # Verify the function returns the mocked client instance
        assert client is mock_chat.return_value, "get_llm_client should return the ChatAnthropic instance"
        
        # Verify no unexpected parameters were passed
        expected_keys = {"model", "max_tokens", "temperature", "thinking", "timeout"}
        actual_keys = set(kwargs.keys())
        assert actual_keys == expected_keys, \
            f"Unexpected parameters: {actual_keys - expected_keys}, Missing parameters: {expected_keys - actual_keys}"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_override_model(self, mock_chat):
        """Test that model override works correctly."""
        override_model = "claude-3-sonnet"
        client = get_llm_client(model=override_model)
        
        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args
        
        # Verify model override
        assert kwargs.get("model") == override_model, \
            f"Expected model {override_model}, got {kwargs.get('model')}"
        
        # Verify other parameters still have defaults
        assert kwargs.get("max_tokens") == 16384, "max_tokens should remain default"
        assert kwargs.get("temperature") == 1.0, "temperature should remain default"
        assert kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 10000}, \
            "thinking should remain default"
        assert kwargs.get("timeout") == 300.0, "timeout should remain default"
        
        # Verify return value
        assert client is mock_chat.return_value, "get_llm_client should return the ChatAnthropic instance"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_singleton_behavior(self, mock_chat):
        """Test that multiple calls return the same singleton instance."""
        client1 = get_llm_client()
        client2 = get_llm_client()
        client3 = get_llm_client()
        
        # Should only create one instance
        assert mock_chat.call_count == 1, \
            f"Expected 1 call to ChatAnthropic, got {mock_chat.call_count}"
        
        # All calls should return the same instance
        assert client1 is client2, "Second call should return same instance"
        assert client2 is client3, "Third call should return same instance"
        assert client1 is mock_chat.return_value, "Should return the singleton instance"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_environment_variable_override(self, mock_chat):
        """Test that REPROLAB_MODEL environment variable is used when model is None."""
        env_model = "claude-3-opus"
        
        with patch.dict(os.environ, {"REPROLAB_MODEL": env_model}):
            client = get_llm_client()
            
            mock_chat.assert_called_once()
            _, kwargs = mock_chat.call_args
            
            assert kwargs.get("model") == env_model, \
                f"Expected model from env var {env_model}, got {kwargs.get('model')}"
            assert client is mock_chat.return_value

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_parameter_overrides_env_var(self, mock_chat):
        """Test that explicit model parameter takes precedence over environment variable."""
        param_model = "claude-3-sonnet"
        env_model = "claude-3-opus"
        
        with patch.dict(os.environ, {"REPROLAB_MODEL": env_model}):
            client = get_llm_client(model=param_model)
            
            mock_chat.assert_called_once()
            _, kwargs = mock_chat.call_args
            
            assert kwargs.get("model") == param_model, \
                f"Expected model from parameter {param_model}, got {kwargs.get('model')}"
            assert kwargs.get("model") != env_model, \
                "Parameter should override environment variable"
            assert client is mock_chat.return_value

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_model_override_after_singleton_created(self, mock_chat):
        """Test that model override after singleton creation creates new instance.
        
        This test verifies that when a different model is requested after singleton creation,
        a new instance is created with the correct model.
        """
        from unittest.mock import MagicMock
        
        # Use side_effect to return different instances for each call
        instance1 = MagicMock()
        instance2 = MagicMock()
        mock_chat.side_effect = [instance1, instance2]
        
        # Create singleton with default model
        client1 = get_llm_client()
        mock_chat.assert_called_once()
        _, kwargs1 = mock_chat.call_args
        assert kwargs1.get("model") == DEFAULT_MODEL
        assert client1 is instance1, "First call should return first instance"
        
        # Try to override model after singleton is created
        override_model = "claude-3-sonnet"
        client2 = get_llm_client(model=override_model)
        
        # Verify that ChatAnthropic was called again (new instance created)
        assert mock_chat.call_count == 2, (
            f"Expected 2 calls (one per model), got {mock_chat.call_count}. "
            f"If only 1 call, the singleton was reused and model override was ignored."
        )
        
        # Verify the second call used the override model
        _, kwargs2 = mock_chat.call_args_list[1]
        assert kwargs2.get("model") == override_model, \
            f"Expected override model {override_model}, got {kwargs2.get('model')}"
        
        # Verify different instances were returned
        assert client1 is instance1, "First client should be first instance"
        assert client2 is instance2, "Second client should be second instance"
        assert client1 is not client2, "Different models should return different instances"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_none_model_uses_default(self, mock_chat):
        """Test that explicitly passing None uses default model."""
        client = get_llm_client(model=None)
        
        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args
        
        assert kwargs.get("model") == DEFAULT_MODEL, \
            f"Expected default model {DEFAULT_MODEL}, got {kwargs.get('model')}"
        assert client is mock_chat.return_value

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_empty_string_model(self, mock_chat):
        """Test behavior with empty string model (edge case)."""
        # Empty string is falsy, so should fall back to env var or default
        with patch.dict(os.environ, {}, clear=True):
            # No env var set, should use default
            client = get_llm_client(model="")
            
            mock_chat.assert_called_once()
            _, kwargs = mock_chat.call_args
            
            # Empty string is falsy, so should use default
            assert kwargs.get("model") == DEFAULT_MODEL, \
                f"Empty string should fall back to default {DEFAULT_MODEL}, got {kwargs.get('model')}"
            assert client is mock_chat.return_value

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_thinking_config_structure(self, mock_chat):
        """Test that thinking configuration has correct structure and values."""
        client = get_llm_client()
        
        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args
        
        thinking = kwargs.get("thinking")
        assert thinking is not None, "thinking parameter should be set"
        assert isinstance(thinking, dict), f"thinking should be dict, got {type(thinking)}"
        assert thinking.get("type") == "enabled", \
            f"Expected thinking type 'enabled', got {thinking.get('type')}"
        assert thinking.get("budget_tokens") == 10000, \
            f"Expected budget_tokens 10000, got {thinking.get('budget_tokens')}"
        assert len(thinking) == 2, \
            f"thinking dict should have exactly 2 keys, got {len(thinking)}: {thinking.keys()}"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_all_numeric_parameters(self, mock_chat):
        """Test that all numeric parameters have correct types and values."""
        client = get_llm_client()
        
        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args
        
        # Verify types
        assert isinstance(kwargs.get("max_tokens"), int), \
            f"max_tokens should be int, got {type(kwargs.get('max_tokens'))}"
        assert isinstance(kwargs.get("temperature"), float), \
            f"temperature should be float, got {type(kwargs.get('temperature'))}"
        assert isinstance(kwargs.get("timeout"), float), \
            f"timeout should be float, got {type(kwargs.get('timeout'))}"
        
        # Verify values are positive
        assert kwargs.get("max_tokens") > 0, "max_tokens should be positive"
        assert kwargs.get("temperature") >= 0, "temperature should be non-negative"
        assert kwargs.get("timeout") > 0, "timeout should be positive"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_multiple_calls_same_model(self, mock_chat):
        """Test that multiple calls with same model return singleton."""
        model = "claude-3-sonnet"
        
        client1 = get_llm_client(model=model)
        client2 = get_llm_client(model=model)
        client3 = get_llm_client(model=model)
        
        # Should only create one instance for same model
        assert mock_chat.call_count == 1, \
            f"Expected 1 call for same model, got {mock_chat.call_count}"
        
        # All should be same instance
        assert client1 is client2 is client3, \
            "Multiple calls with same model should return same singleton"
        
        # Verify model was set correctly
        _, kwargs = mock_chat.call_args
        assert kwargs.get("model") == model

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_no_unexpected_kwargs(self, mock_chat):
        """Test that no unexpected keyword arguments are passed."""
        client = get_llm_client()
        
        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args
        
        # Verify only expected parameters are present
        allowed_params = {"model", "max_tokens", "temperature", "thinking", "timeout"}
        unexpected = set(kwargs.keys()) - allowed_params
        
        assert not unexpected, \
            f"Unexpected parameters passed to ChatAnthropic: {unexpected}"

    @patch("src.llm_client.ChatAnthropic")
    def test_get_llm_client_returns_chatanthropic_instance(self, mock_chat):
        """Test that the function returns an actual ChatAnthropic instance."""
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        
        client = get_llm_client()
        
        assert client is mock_instance, \
            "get_llm_client should return the ChatAnthropic instance"
        assert client is not None, "Client should not be None"


