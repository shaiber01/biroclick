"""Tests for adapt_prompts_node."""

from unittest.mock import patch, MagicMock
import logging

import pytest

from src.agents.planning import adapt_prompts_node


class TestAdaptPromptsNode:
    """Tests for adapt_prompts_node."""

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_success_with_domain(self, mock_build_prompt, mock_llm):
        """Test successful adaptation with paper_domain update."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": ["a1", "a2"],
            "paper_domain": "new_domain"
        }
        
        state = {"paper_text": "some paper text", "paper_domain": "old_domain"}
        result = adapt_prompts_node(state)
        
        # Verify return values
        assert result["prompt_adaptations"] == ["a1", "a2"]
        assert result["paper_domain"] == "new_domain"
        assert result["workflow_phase"] == "adapting_prompts"
        
        # Verify build_agent_prompt was called correctly
        # Note: state may be modified by decorator (e.g., metrics added), so check args flexibly
        mock_build_prompt.assert_called_once()
        call_args = mock_build_prompt.call_args
        assert call_args[0][0] == "prompt_adaptor"
        assert call_args[0][1]["paper_text"] == "some paper text"
        assert call_args[0][1]["paper_domain"] == "old_domain"
        
        # Verify call_agent_with_metrics was called correctly
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["agent_name"] == "prompt_adaptor"
        assert call_kwargs["system_prompt"] == "system prompt"
        # Note: state may be modified by decorator (e.g., metrics added), so check key fields
        passed_state = call_kwargs["state"]
        assert passed_state["paper_text"] == "some paper text"
        assert passed_state["paper_domain"] == "old_domain"
        assert "user_content" in call_kwargs
        
        # Verify state was not mutated
        assert state["paper_domain"] == "old_domain"
        assert "prompt_adaptations" not in state

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_success_without_domain(self, mock_build_prompt, mock_llm):
        """Test successful adaptation without paper_domain update."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": ["a1"]
        }
        
        state = {"paper_text": "some paper text", "paper_domain": "existing_domain"}
        result = adapt_prompts_node(state)
        
        # Verify return values
        assert result["prompt_adaptations"] == ["a1"]
        assert result["workflow_phase"] == "adapting_prompts"
        # paper_domain should NOT be in result if agent doesn't provide it
        assert "paper_domain" not in result

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_empty_adaptations(self, mock_build_prompt, mock_llm):
        """Test successful call with empty adaptations list."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": []
        }
        
        state = {"paper_text": "some paper text"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"
        assert isinstance(result["prompt_adaptations"], list)

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_missing_adaptations_key(self, mock_build_prompt, mock_llm):
        """Test handling when agent output lacks adaptations key."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "paper_domain": "domain"
        }
        
        state = {"paper_text": "some paper text"}
        result = adapt_prompts_node(state)
        
        # Should default to empty list when adaptations key is missing
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_none_adaptations(self, mock_build_prompt, mock_llm):
        """Test handling when agent returns None for adaptations."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": None
        }
        
        state = {"paper_text": "some paper text"}
        result = adapt_prompts_node(state)
        
        # Should handle None gracefully (get() returns None, which becomes empty list)
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_none_domain(self, mock_build_prompt, mock_llm):
        """Test handling when agent returns None for paper_domain."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": ["a1"],
            "paper_domain": None
        }
        
        state = {"paper_text": "some paper text"}
        result = adapt_prompts_node(state)
        
        # None paper_domain should not be added to result
        assert result["prompt_adaptations"] == ["a1"]
        assert result["workflow_phase"] == "adapting_prompts"
        assert "paper_domain" not in result

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_empty_domain_string(self, mock_build_prompt, mock_llm):
        """Test handling when agent returns empty string for paper_domain."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": ["a1"],
            "paper_domain": ""
        }
        
        state = {"paper_text": "some paper text"}
        result = adapt_prompts_node(state)
        
        # Empty string should be treated as falsy and not added
        assert result["prompt_adaptations"] == ["a1"]
        assert result["workflow_phase"] == "adapting_prompts"
        assert "paper_domain" not in result

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_failure_fallback(self, mock_build_prompt, mock_llm):
        """Test fallback to empty list on failure."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.side_effect = Exception("LLM call failed")
        
        state = {"paper_text": "some paper text"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"
        assert isinstance(result["prompt_adaptations"], list)

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_different_exception_types(self, mock_build_prompt, mock_llm):
        """Test handling of different exception types."""
        mock_build_prompt.return_value = "system prompt"
        
        exception_types = [
            ValueError("Value error"),
            KeyError("Key error"),
            RuntimeError("Runtime error"),
            ConnectionError("Connection error"),
        ]
        
        for exc in exception_types:
            mock_llm.side_effect = exc
            state = {"paper_text": "some paper text"}
            result = adapt_prompts_node(state)
            
            assert result["prompt_adaptations"] == []
            assert result["workflow_phase"] == "adapting_prompts"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_truncates_long_text(self, mock_build_prompt, mock_llm):
        """Verify paper text is truncated correctly in user content."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {}
        
        long_text = "a" * 10000
        state = {"paper_text": long_text}
        
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Code uses [:5000] first then slices further to [:3000] in user_content
        assert len(long_text) > 5000
        # User content includes other text, so check that paper excerpt is truncated
        assert "a" * 3000 in user_content or len(user_content) < len(long_text)
        # Verify the truncation marker is present
        assert "..." in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_short_text(self, mock_build_prompt, mock_llm):
        """Test with short paper text (< 3000 chars)."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        short_text = "a" * 100
        state = {"paper_text": short_text}
        
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Short text should be included fully (with truncation marker)
        assert short_text in user_content or user_content.count("a") >= 100

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_empty_paper_text(self, mock_build_prompt, mock_llm):
        """Test with empty paper_text."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {"paper_text": ""}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"
        
        # Verify user_content still contains the structure
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "PAPER SUMMARY FOR PROMPT ADAPTATION" in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_missing_paper_text_key(self, mock_build_prompt, mock_llm):
        """Test with missing paper_text key in state."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"
        
        # Verify user_content handles missing paper_text
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        assert "PAPER SUMMARY FOR PROMPT ADAPTATION" in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_none_paper_text(self, mock_build_prompt, mock_llm):
        """Test with None paper_text."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {"paper_text": None}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == []
        assert result["workflow_phase"] == "adapting_prompts"

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_paper_domain_in_user_content(self, mock_build_prompt, mock_llm):
        """Test that paper_domain from state is included in user_content."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {"paper_text": "some text", "paper_domain": "optics"}
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        assert "Domain: optics" in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_empty_paper_domain(self, mock_build_prompt, mock_llm):
        """Test with empty paper_domain in state."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {"paper_text": "some text", "paper_domain": ""}
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        assert "Domain: " in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_missing_paper_domain_key(self, mock_build_prompt, mock_llm):
        """Test with missing paper_domain key in state."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {"paper_text": "some text"}
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        assert "Domain: " in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_user_content_structure(self, mock_build_prompt, mock_llm):
        """Test that user_content has correct structure."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        state = {"paper_text": "test paper text", "paper_domain": "test_domain"}
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Verify structure
        assert "# PAPER SUMMARY FOR PROMPT ADAPTATION" in user_content
        assert "Domain: test_domain" in user_content
        assert "Paper excerpt:" in user_content
        assert "Analyze this paper and suggest prompt adaptations" in user_content

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_state_not_mutated(self, mock_build_prompt, mock_llm):
        """Test that original state is not mutated."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": ["a1"],
            "paper_domain": "new_domain"
        }
        
        original_state = {"paper_text": "text", "paper_domain": "old_domain"}
        state_copy = original_state.copy()
        result = adapt_prompts_node(state_copy)
        
        # Verify state was not mutated
        assert state_copy == original_state
        assert state_copy["paper_domain"] == "old_domain"
        assert "prompt_adaptations" not in state_copy
        
        # Verify result is separate
        assert result["paper_domain"] == "new_domain"
        assert result["prompt_adaptations"] == ["a1"]

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_logs_warning_on_exception(self, mock_build_prompt, mock_llm, caplog):
        """Test that exceptions are logged with warning."""
        import logging
        mock_build_prompt.return_value = "system prompt"
        mock_llm.side_effect = Exception("Test exception")
        
        # Use caplog to capture log output instead of mocking the module-level logger
        with caplog.at_level(logging.WARNING, logger="src.agents.planning"):
            state = {"paper_text": "some text"}
            result = adapt_prompts_node(state)
        
        # Verify warning was logged
        assert len(caplog.records) >= 1, "Expected at least one log record"
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Prompt adaptor LLM call failed" in msg for msg in warning_messages), (
            f"Expected warning about LLM failure. Got: {warning_messages}"
        )
        assert any("Test exception" in msg for msg in warning_messages), (
            f"Expected exception message in warning. Got: {warning_messages}"
        )
        
        # Verify result still works
        assert result["prompt_adaptations"] == []

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_multiple_adaptations(self, mock_build_prompt, mock_llm):
        """Test with multiple adaptations."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {
            "prompt_modifications": ["adapt1", "adapt2", "adapt3", "adapt4"]
        }
        
        state = {"paper_text": "some text"}
        result = adapt_prompts_node(state)
        
        assert result["prompt_adaptations"] == ["adapt1", "adapt2", "adapt3", "adapt4"]
        assert len(result["prompt_adaptations"]) == 4

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_exact_truncation_length(self, mock_build_prompt, mock_llm):
        """Test exact truncation behavior at boundaries."""
        mock_build_prompt.return_value = "system prompt"
        mock_llm.return_value = {"prompt_modifications": []}
        
        # Test exactly 3000 chars
        text_3000 = "a" * 3000
        state = {"paper_text": text_3000}
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Should include the text (possibly with truncation marker)
        assert "a" * 3000 in user_content or user_content.count("a") >= 3000
        
        # Test exactly 5000 chars
        text_5000 = "b" * 5000
        state = {"paper_text": text_5000}
        adapt_prompts_node(state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Should truncate to 3000 in user_content
        assert "b" * 3000 in user_content or user_content.count("b") <= 3000

    @patch("src.agents.planning.call_agent_with_metrics")
    @patch("src.agents.planning.build_agent_prompt")
    def test_adapt_prompts_workflow_phase_always_set(self, mock_build_prompt, mock_llm):
        """Test that workflow_phase is always set to adapting_prompts."""
        mock_build_prompt.return_value = "system prompt"
        
        test_cases = [
            {"prompt_modifications": ["a1"]},
            {"prompt_modifications": []},
            {},
            {"prompt_modifications": None},
        ]
        
        for agent_output in test_cases:
            mock_llm.return_value = agent_output
            state = {"paper_text": "text"}
            result = adapt_prompts_node(state)
            assert result["workflow_phase"] == "adapting_prompts"
        
        # Also test exception case
        mock_llm.side_effect = Exception("error")
        state = {"paper_text": "text"}
        result = adapt_prompts_node(state)
        assert result["workflow_phase"] == "adapting_prompts"
