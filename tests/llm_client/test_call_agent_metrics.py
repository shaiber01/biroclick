"""Metrics coverage for `call_agent_with_metrics`."""

import pytest
from unittest.mock import patch

from src.llm_client import call_agent_with_metrics


class TestCallAgentMetrics:
    """Tests for call_agent_with_metrics."""

    @patch("src.llm_client.call_agent")
    def test_call_agent_metrics_success(self, mock_call_agent):
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert len(state["metrics"]["agent_calls"]) == 1

        metric = state["metrics"]["agent_calls"][0]
        assert metric["agent"] == "test_agent"
        assert metric["success"] is True
        assert metric["error"] is None
        assert "duration_seconds" in metric
        assert "timestamp" in metric

    @patch("src.llm_client.call_agent")
    def test_call_agent_metrics_failure(self, mock_call_agent):
        mock_call_agent.side_effect = ValueError("Failure")

        state = {}
        with pytest.raises(ValueError):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        assert "metrics" in state
        assert len(state["metrics"]["agent_calls"]) == 1

        metric = state["metrics"]["agent_calls"][0]
        assert metric["agent"] == "test_agent"
        assert metric["success"] is False
        assert metric["error"] == "Failure"


