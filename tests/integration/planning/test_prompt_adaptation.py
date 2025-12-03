"""Tests for adapting planner prompts before the main workflow."""

import json
from unittest.mock import patch


class TestAdaptPromptsNode:
    """Verify prompt adaptation logic."""

    def test_adapt_prompts_success(self, base_state):
        """adapt_prompts_node should call LLM and update state."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": [
                {"agent": "planner", "adaptation": "Focus on materials"},
                {"agent": "designer", "adaptation": "Check boundaries"},
            ],
            "paper_domain": "metamaterials",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert len(result["prompt_adaptations"]) == 2
        assert result["paper_domain"] == "metamaterials"

        # Verify LLM call
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["agent_name"] == "prompt_adaptor"
        assert "Analyze this paper" in call_kwargs["user_content"]

    def test_adapt_prompts_llm_failure(self, base_state):
        """adapt_prompts_node should return empty list on LLM failure."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=Exception("LLM Error"),
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == []

    def test_adapt_prompts_handles_context_escalation(self, base_state):
        """adapt_prompts_node should handle context escalation via decorator."""
        from src.agents.planning import adapt_prompts_node

        escalation_response = {
            "awaiting_user_input": True,
            "reason": "Context too large",
        }

        # We patch in src.agents.base because that's where the decorator is defined/used
        with patch(
            "src.agents.base.check_context_or_escalate", return_value=escalation_response
        ):
            result = adapt_prompts_node(base_state)

        assert result == escalation_response

    def test_adapt_prompts_node_updates_state(self, base_state):
        """Ensure adaptations and domain bubble through the state."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": ["Focus on plasmonics", "Use Johnson-Christy data"],
            "paper_domain": "plasmonics",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_call:
            result = adapt_prompts_node(base_state)

        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("agent_name") == "prompt_adaptor"
        assert result.get("workflow_phase") == "adapting_prompts"
        assert result["prompt_adaptations"] == mock_response["adaptations"]
        assert result.get("paper_domain") == "plasmonics"

