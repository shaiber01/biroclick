"""Tests for simulation_designer_node."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.design import simulation_designer_node


@pytest.fixture(name="base_state")
def design_base_state(design_state):
    return design_state


class TestSimulationDesignerNode:
    """Tests for simulation_designer_node."""

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_success(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test successful design generation with all integrations."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_get_spec.return_value = "complex_simulation"
        mock_build_content.return_value = "User Content"
        
        expected_design = {
            "design_description": "FDTD simulation setup...",
            "simulation_type": "FDTD",
            "parameters": {"mesh_size": "2nm"},
            "new_assumptions": [{"id": "A1", "description": "Use PML"}]
        }
        mock_llm.return_value = expected_design
        
        result = simulation_designer_node(base_state)
        
        # Verify State Updates
        assert result["workflow_phase"] == "design"
        assert result["design_description"] == expected_design
        
        # Verify Assumption Merging
        assert "assumptions" in result
        assert len(result["assumptions"]["global_assumptions"]) == 2
        assert result["assumptions"]["global_assumptions"][0]["id"] == "existing_1"
        assert result["assumptions"]["global_assumptions"][1]["id"] == "A1"
        
        # Strict assertion: ensure no error flags
        assert "ask_user_trigger" not in result
        assert "awaiting_user_input" not in result

        # Verify LLM Call Construction
        mock_check.assert_called_once_with(base_state, "design")
        mock_prompt.assert_called_once_with("simulation_designer", base_state)
        mock_build_content.assert_called_once_with(base_state)
        mock_get_spec.assert_called_once_with(base_state, "stage_1_sim", "complexity_class", "unknown")
        
        # Verify arguments passed to LLM
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["agent_name"] == "simulation_designer"
        assert "System Prompt" in call_kwargs["system_prompt"]
        assert "Complexity class" in call_kwargs["system_prompt"]
        assert "complex_simulation" in call_kwargs["system_prompt"] # Injected complexity
        assert call_kwargs["user_content"] == "User Content"
        assert call_kwargs["state"] == base_state
        mock_llm.assert_called_once()

    def test_designer_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = simulation_designer_node(base_state)
        
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design" # Should still set phase

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_handles_llm_failure(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = simulation_designer_node(base_state)
        
        # Should escalate to user
        assert result["ask_user_trigger"] == "llm_error"
        assert result["awaiting_user_input"] is True
        # Should contain error info
        assert "pending_user_questions" in result
        assert any("API Error" in q for q in result["pending_user_questions"])
        mock_llm.assert_called_once()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_injects_feedback(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test feedback injection into prompt."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        mock_llm.return_value = {}
        base_state["reviewer_feedback"] = "Fix mesh size"
        
        simulation_designer_node(base_state)
        
        # Verify prompt contains feedback
        call_kwargs = mock_llm.call_args[1]
        assert "Base Prompt" in call_kwargs["system_prompt"]
        assert "REVISION FEEDBACK: Fix mesh size" in call_kwargs["system_prompt"]
        mock_prompt.assert_called_once_with("simulation_designer", base_state)

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_missing_assumptions_in_output(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that missing 'new_assumptions' in LLM output is handled gracefully."""
        mock_check.return_value = None
        mock_llm.return_value = {"design_description": "Just description"}
        
        result = simulation_designer_node(base_state)
        
        assert "assumptions" not in result # Should not modify assumptions if none returned
        assert result["design_description"] == {"design_description": "Just description"}
        mock_llm.assert_called_once()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_preserves_existing_assumptions(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that existing assumptions are preserved when new ones are added."""
        mock_check.return_value = None
        mock_llm.return_value = {
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        
        result = simulation_designer_node(base_state)
        
        assert "assumptions" in result
        global_assumptions = result["assumptions"]["global_assumptions"]
        assert len(global_assumptions) == 2
        # Check presence of existing
        assert any(a["id"] == "existing_1" for a in global_assumptions)
        # Check presence of new
        assert any(a["id"] == "new" for a in global_assumptions)
        mock_llm.assert_called_once()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_designer_handles_none_assumptions(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when state['assumptions'] is explicitly None (bug repro)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "design_description": "Desc",
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        base_state["assumptions"] = None # Simulate corrupted state or init
        
        # This expects the code to handle None safely.
        # If it raises AttributeError, the test will fail, revealing the bug.
        try:
            result = simulation_designer_node(base_state)
        except AttributeError:
            pytest.fail("simulation_designer_node crashed because state['assumptions'] was None")
            
        assert "assumptions" in result
        global_assumptions = result["assumptions"]["global_assumptions"]
        assert len(global_assumptions) == 1
        assert global_assumptions[0]["id"] == "new"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_respects_context_escalation(
        self,
        mock_build_content,
        mock_get_spec,
        mock_llm,
        mock_check,
        mock_prompt,
        base_state,
    ):
        """If context check asks for user input, downstream calls must be skipped."""
        escalation = {
            "workflow_phase": "design",
            "awaiting_user_input": True,
            "ask_user_trigger": "context_invalid",
        }
        mock_check.return_value = escalation

        result = simulation_designer_node(base_state)

        assert result == escalation
        mock_prompt.assert_not_called()
        mock_build_content.assert_not_called()
        mock_get_spec.assert_not_called()
        mock_llm.assert_not_called()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_merges_context_updates_before_llm_call(
        self,
        mock_build_content,
        mock_get_spec,
        mock_llm,
        mock_check,
        mock_prompt,
        base_state,
    ):
        """Non-blocking context updates should be merged into the downstream state."""
        mock_check.return_value = {"context_refresh": True}
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}

        simulation_designer_node(base_state)

        merged_state = mock_llm.call_args[1]["state"]
        assert merged_state["context_refresh"] is True
        mock_prompt.assert_called_once_with("simulation_designer", merged_state)
        mock_llm.assert_called_once()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_does_not_mutate_input_state_when_adding_assumptions(
        self,
        mock_build_content,
        mock_get_spec,
        mock_llm,
        mock_check,
        mock_prompt,
        base_state,
    ):
        """State merging must not mutate the incoming state object."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "design_description": {"details": "something"},
            "new_assumptions": [{"id": "new", "description": "from llm"}],
        }

        original_assumptions = list(base_state["assumptions"]["global_assumptions"])

        result = simulation_designer_node(base_state)

        assert result["assumptions"]["global_assumptions"][-1]["id"] == "new"
        assert base_state["assumptions"]["global_assumptions"] == original_assumptions

# ═══════════════════════════════════════════════════════════════════════
# design_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════
