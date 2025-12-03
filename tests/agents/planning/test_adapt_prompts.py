"""Tests for adapt_prompts_node."""

from unittest.mock import patch

import pytest

from src.agents.planning import adapt_prompts_node


class TestAdaptPromptsNode:
    """Tests for adapt_prompts_node."""

    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_success(self, mock_llm):
        """Test successful adaptation."""
        mock_llm.return_value = {
            "adaptations": ["a1"],
            "paper_domain": "domain"
        }
        
        state = {"paper_text": "text", "paper_domain": "old"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == ["a1"]
        assert result["paper_domain"] == "domain"

    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_failure_fallback(self, mock_llm):
        """Test fallback to empty list on failure."""
        mock_llm.side_effect = Exception("fail")
        
        state = {"paper_text": "text"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        
    @patch("src.agents.planning.call_agent_with_metrics")
    def test_adapt_prompts_truncates_text(self, mock_llm):
        """Verify paper text is truncated in user content."""
        mock_llm.return_value = {}
        long_text = "a" * 10000
        state = {"paper_text": long_text}
        
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        # Code uses [:5000] first then slices further to [:3000] in user_content
        assert len(long_text) > 5000
        assert len(user_content) < 5000
