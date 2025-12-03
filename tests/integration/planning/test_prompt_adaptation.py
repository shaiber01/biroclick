"""Tests for adapting planner prompts before the main workflow."""

import json
from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest


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

        # Verify exact workflow phase
        assert result["workflow_phase"] == "adapting_prompts"
        
        # Verify adaptations are stored exactly as returned
        assert result["prompt_adaptations"] == mock_response["adaptations"]
        assert len(result["prompt_adaptations"]) == 2
        assert result["prompt_adaptations"][0]["agent"] == "planner"
        assert result["prompt_adaptations"][0]["adaptation"] == "Focus on materials"
        assert result["prompt_adaptations"][1]["agent"] == "designer"
        assert result["prompt_adaptations"][1]["adaptation"] == "Check boundaries"
        
        # Verify paper_domain is stored exactly
        assert result["paper_domain"] == "metamaterials"

        # Verify LLM call parameters
        assert mock_llm.called
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["agent_name"] == "prompt_adaptor"
        assert "system_prompt" in call_kwargs
        assert "user_content" in call_kwargs
        assert "state" in call_kwargs
        # State may be mutated by call_agent_with_metrics (adds metrics), so check key fields
        assert call_kwargs["state"].get("paper_id") == base_state.get("paper_id")
        assert call_kwargs["state"].get("paper_text") == base_state.get("paper_text")
        assert "Analyze this paper" in call_kwargs["user_content"]
        assert "PAPER SUMMARY FOR PROMPT ADAPTATION" in call_kwargs["user_content"]

    def test_adapt_prompts_llm_failure(self, base_state):
        """adapt_prompts_node should return empty list on LLM failure."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=Exception("LLM Error"),
        ):
            result = adapt_prompts_node(base_state)

        # Verify workflow phase is still set correctly
        assert result["workflow_phase"] == "adapting_prompts"
        # Verify empty adaptations list on failure
        assert result["prompt_adaptations"] == []
        # Verify paper_domain is not set on failure
        assert "paper_domain" not in result

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

        # Verify exact escalation response is returned
        assert result == escalation_response
        assert result["awaiting_user_input"] is True
        assert result["reason"] == "Context too large"

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
        # Verify exact adaptations list
        assert result["prompt_adaptations"] == mock_response["adaptations"]
        assert result["prompt_adaptations"] == ["Focus on plasmonics", "Use Johnson-Christy data"]
        assert result.get("paper_domain") == "plasmonics"

    def test_adapt_prompts_handles_none_paper_text(self, base_state):
        """adapt_prompts_node should handle None paper_text gracefully."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_text"] = None
        mock_response = {
            "adaptations": [],
            "paper_domain": "unknown",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == []
        
        # Verify user_content is still built (with empty paper_text)
        call_kwargs = mock_llm.call_args.kwargs
        assert "user_content" in call_kwargs
        assert "Domain:" in call_kwargs["user_content"]
        assert call_kwargs["user_content"].count("Paper excerpt:") == 1

    def test_adapt_prompts_handles_empty_paper_text(self, base_state):
        """adapt_prompts_node should handle empty paper_text."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_text"] = ""
        mock_response = {
            "adaptations": [],
            "paper_domain": "unknown",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == []
        
        # Verify user_content is built with empty paper_text
        call_kwargs = mock_llm.call_args.kwargs
        assert "user_content" in call_kwargs
        assert "Paper excerpt:" in call_kwargs["user_content"]

    def test_adapt_prompts_truncates_long_paper_text(self, base_state):
        """adapt_prompts_node should truncate paper_text to 5000 chars for context."""
        from src.agents.planning import adapt_prompts_node

        # Create paper_text longer than 5000 chars
        long_text = "A" * 10000
        base_state["paper_text"] = long_text
        
        mock_response = {
            "adaptations": [],
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        # Verify truncation happened
        call_kwargs = mock_llm.call_args.kwargs
        user_content = call_kwargs["user_content"]
        
        # Extract the paper excerpt part (between "Paper excerpt:\n" and "...")
        excerpt_start = user_content.find("Paper excerpt:\n")
        assert excerpt_start != -1, "Paper excerpt section not found"
        excerpt_end = user_content.find("...", excerpt_start)
        assert excerpt_end != -1, "Truncation marker not found"
        
        excerpt = user_content[excerpt_start + len("Paper excerpt:\n"):excerpt_end]
        
        # The excerpt should be exactly 3000 chars (truncated from 5000)
        assert len(excerpt) == 3000, f"Expected 3000 chars, got {len(excerpt)}"
        assert excerpt == "A" * 3000, "Excerpt should contain exactly 3000 'A' characters"
        assert "..." in user_content or len(user_content) < len(long_text)

    def test_adapt_prompts_handles_missing_paper_text_key(self, base_state):
        """adapt_prompts_node should handle missing paper_text key."""
        from src.agents.planning import adapt_prompts_node

        if "paper_text" in base_state:
            del base_state["paper_text"]
        
        mock_response = {
            "adaptations": [],
            "paper_domain": "unknown",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == []
        
        # Verify user_content is still built
        call_kwargs = mock_llm.call_args.kwargs
        assert "user_content" in call_kwargs

    def test_adapt_prompts_handles_none_adaptations(self, base_state):
        """adapt_prompts_node should handle None adaptations in response."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": None,
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        # Should convert None to empty list
        assert result["prompt_adaptations"] == []
        assert result["paper_domain"] == "test"

    def test_adapt_prompts_handles_missing_adaptations_key(self, base_state):
        """adapt_prompts_node should handle missing adaptations key."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        # Should default to empty list
        assert result["prompt_adaptations"] == []
        assert result["paper_domain"] == "test"

    def test_adapt_prompts_handles_non_list_adaptations(self, base_state):
        """adapt_prompts_node should handle non-list adaptations."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": "not a list",
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        # Should convert non-list to empty list
        assert result["prompt_adaptations"] == []
        assert result["paper_domain"] == "test"

    def test_adapt_prompts_handles_dict_adaptations(self, base_state):
        """adapt_prompts_node should handle dict adaptations (non-list)."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": {"planner": "Focus on materials"},
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        # Should convert dict to empty list
        assert result["prompt_adaptations"] == []

    def test_adapt_prompts_handles_empty_adaptations_list(self, base_state):
        """adapt_prompts_node should handle empty adaptations list."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": [],
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == []
        assert result["paper_domain"] == "test"

    def test_adapt_prompts_handles_none_paper_domain(self, base_state):
        """adapt_prompts_node should handle None paper_domain."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": ["test"],
            "paper_domain": None,
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == ["test"]
        # Should not set paper_domain if None
        assert "paper_domain" not in result

    def test_adapt_prompts_handles_missing_paper_domain_key(self, base_state):
        """adapt_prompts_node should handle missing paper_domain key."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": ["test"],
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == ["test"]
        # Should not set paper_domain if missing
        assert "paper_domain" not in result

    def test_adapt_prompts_handles_empty_paper_domain(self, base_state):
        """adapt_prompts_node should handle empty string paper_domain."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": ["test"],
            "paper_domain": "",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == ["test"]
        # Empty string should be treated as falsy and not set (code checks `if agent_output.get("paper_domain"):`)
        assert "paper_domain" not in result

    def test_adapt_prompts_preserves_existing_paper_domain(self, base_state):
        """adapt_prompts_node should use paper_domain from state if available."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_domain"] = "existing_domain"
        mock_response = {
            "adaptations": ["test"],
            "paper_domain": "new_domain",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        # Should use new domain from LLM response
        assert result["paper_domain"] == "new_domain"
        
        # Verify user_content includes existing domain
        call_kwargs = mock_llm.call_args.kwargs
        assert "existing_domain" in call_kwargs["user_content"]

    def test_adapt_prompts_handles_none_paper_domain_in_state(self, base_state):
        """adapt_prompts_node should handle None paper_domain in state."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_domain"] = None
        mock_response = {
            "adaptations": ["test"],
            "paper_domain": "new_domain",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        assert result["paper_domain"] == "new_domain"
        
        # Verify user_content handles None domain
        call_kwargs = mock_llm.call_args.kwargs
        assert "Domain:" in call_kwargs["user_content"]

    def test_adapt_prompts_does_not_mutate_input_state(self, base_state):
        """adapt_prompts_node should not mutate input state."""
        from src.agents.planning import adapt_prompts_node

        original_state = deepcopy(base_state)
        original_prompt_adaptations = base_state.get("prompt_adaptations")
        original_workflow_phase = base_state.get("workflow_phase")
        
        mock_response = {
            "adaptations": ["test"],
            "paper_domain": "new_domain",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        # Verify input state values are unchanged (may be mutated by call_agent_with_metrics for metrics)
        assert base_state.get("paper_text") == original_state.get("paper_text")
        assert base_state.get("paper_domain") == original_state.get("paper_domain")
        assert base_state.get("paper_id") == original_state.get("paper_id")
        # prompt_adaptations may exist in base_state from initialization, but should not be changed by function
        assert base_state.get("prompt_adaptations") == original_prompt_adaptations
        assert base_state.get("workflow_phase") == original_workflow_phase
        
        # Verify result is separate dict with new values
        assert result is not base_state
        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == ["test"]

    def test_adapt_prompts_includes_paper_domain_in_user_content(self, base_state):
        """adapt_prompts_node should include paper_domain in user_content."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_domain"] = "plasmonics"
        mock_response = {
            "adaptations": [],
            "paper_domain": "plasmonics",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            adapt_prompts_node(base_state)

        call_kwargs = mock_llm.call_args.kwargs
        user_content = call_kwargs["user_content"]
        assert "Domain: plasmonics" in user_content

    def test_adapt_prompts_includes_paper_excerpt_in_user_content(self, base_state):
        """adapt_prompts_node should include paper excerpt in user_content."""
        from src.agents.planning import adapt_prompts_node

        base_state["paper_text"] = "This is a test paper about gold nanorods."
        mock_response = {
            "adaptations": [],
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            adapt_prompts_node(base_state)

        call_kwargs = mock_llm.call_args.kwargs
        user_content = call_kwargs["user_content"]
        assert "Paper excerpt:" in user_content
        assert "gold nanorods" in user_content

    def test_adapt_prompts_calls_build_agent_prompt(self, base_state):
        """adapt_prompts_node should call build_agent_prompt for system prompt."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": [],
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm, patch(
            "src.agents.planning.build_agent_prompt"
        ) as mock_build_prompt:
            mock_build_prompt.return_value = "System prompt"
            adapt_prompts_node(base_state)

        # Verify build_agent_prompt was called
        assert mock_build_prompt.called
        assert mock_build_prompt.call_args[0][0] == "prompt_adaptor"
        # State may be mutated by decorator, so check key fields
        state_passed = mock_build_prompt.call_args[0][1]
        assert state_passed.get("paper_id") == base_state.get("paper_id")
        assert state_passed.get("paper_text") == base_state.get("paper_text")
        
        # Verify system_prompt was passed to LLM
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["system_prompt"] == "System prompt"

    def test_adapt_prompts_handles_various_exception_types(self, base_state):
        """adapt_prompts_node should handle various exception types."""
        from src.agents.planning import adapt_prompts_node

        exception_types = [
            ValueError("Value error"),
            KeyError("Key error"),
            RuntimeError("Runtime error"),
            ConnectionError("Connection error"),
        ]

        for exc in exception_types:
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                side_effect=exc,
            ):
                result = adapt_prompts_node(base_state)
                
                assert result["workflow_phase"] == "adapting_prompts"
                assert result["prompt_adaptations"] == []

    def test_adapt_prompts_handles_awaiting_user_input_in_state(self, base_state):
        """adapt_prompts_node should return empty dict if already awaiting user input."""
        from src.agents.planning import adapt_prompts_node

        base_state["awaiting_user_input"] = True
        
        # Should return early without calling LLM
        with patch(
            "src.agents.planning.call_agent_with_metrics"
        ) as mock_llm:
            result = adapt_prompts_node(base_state)
            
            # Decorator should return empty dict
            assert result == {}
            assert not mock_llm.called

    def test_adapt_prompts_handles_context_check_returning_metrics(self, base_state):
        """adapt_prompts_node should continue when context check returns metrics only."""
        from src.agents.planning import adapt_prompts_node

        metrics_only = {
            "metrics": {"tokens_used": 1000},
        }
        
        mock_response = {
            "adaptations": ["test"],
            "paper_domain": "test",
        }

        with patch(
            "src.agents.base.check_context_or_escalate", return_value=metrics_only
        ), patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            result = adapt_prompts_node(base_state)

        # Should continue and call LLM
        assert mock_llm.called
        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == ["test"]

    def test_adapt_prompts_handles_large_adaptations_list(self, base_state):
        """adapt_prompts_node should handle large adaptations list."""
        from src.agents.planning import adapt_prompts_node

        large_adaptations = [f"adaptation_{i}" for i in range(100)]
        mock_response = {
            "adaptations": large_adaptations,
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert len(result["prompt_adaptations"]) == 100
        assert result["prompt_adaptations"] == large_adaptations

    def test_adapt_prompts_handles_complex_adaptation_objects(self, base_state):
        """adapt_prompts_node should handle complex adaptation objects."""
        from src.agents.planning import adapt_prompts_node

        complex_adaptations = [
            {"agent": "planner", "adaptation": "Focus on materials", "priority": 1},
            {"agent": "designer", "adaptation": "Check boundaries", "priority": 2},
            {"nested": {"key": "value"}},
        ]
        mock_response = {
            "adaptations": complex_adaptations,
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ):
            result = adapt_prompts_node(base_state)

        assert result["workflow_phase"] == "adapting_prompts"
        assert result["prompt_adaptations"] == complex_adaptations
        assert result["prompt_adaptations"][0]["priority"] == 1
        assert result["prompt_adaptations"][2]["nested"]["key"] == "value"

    def test_adapt_prompts_passes_state_to_llm(self, base_state):
        """adapt_prompts_node should pass state to call_agent_with_metrics."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": [],
            "paper_domain": "test",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock_llm:
            adapt_prompts_node(base_state)

        call_kwargs = mock_llm.call_args.kwargs
        assert "state" in call_kwargs
        # State may be mutated by call_agent_with_metrics (adds metrics), so check key fields
        state_passed = call_kwargs["state"]
        assert state_passed.get("paper_id") == base_state.get("paper_id")
        assert state_passed.get("paper_text") == base_state.get("paper_text")

