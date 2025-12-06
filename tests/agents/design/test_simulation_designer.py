"""Tests for simulation_designer_node."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.design import simulation_designer_node


@pytest.fixture(name="base_state")
def design_base_state(design_state):
    return design_state


class TestSimulationDesignerNode:
    """Tests for simulation_designer_node."""

    @patch("src.agents.design.log_agent_call")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_success(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, mock_log, base_state):
        """Test successful design generation with all integrations."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_get_spec.return_value = "complex_simulation"
        mock_build_content.return_value = "User Content"
        mock_log.return_value = lambda state, result: None  # Return a no-op function
        
        expected_design = {
            "design_description": "FDTD simulation setup...",
            "simulation_type": "FDTD",
            "parameters": {"mesh_size": "2nm"},
            "new_assumptions": [{"id": "A1", "description": "Use PML"}]
        }
        mock_llm.return_value = expected_design
        
        result = simulation_designer_node(base_state)
        
        # Verify exact structure of result - no extra keys, no missing keys
        expected_keys = {"workflow_phase", "design_description", "assumptions"}
        assert set(result.keys()) == expected_keys, f"Unexpected keys in result: {set(result.keys()) - expected_keys}"
        
        # Verify State Updates
        assert result["workflow_phase"] == "design"
        assert result["design_description"] == expected_design
        
        # Verify Assumption Merging - exact structure
        assert "assumptions" in result
        assert isinstance(result["assumptions"], dict)
        assert "global_assumptions" in result["assumptions"]
        assert isinstance(result["assumptions"]["global_assumptions"], list)
        assert len(result["assumptions"]["global_assumptions"]) == 2
        
        # Verify exact order and content of assumptions
        assert result["assumptions"]["global_assumptions"][0]["id"] == "existing_1"
        assert result["assumptions"]["global_assumptions"][0]["description"] == "Existing assumption"
        assert result["assumptions"]["global_assumptions"][1]["id"] == "A1"
        assert result["assumptions"]["global_assumptions"][1]["description"] == "Use PML"
        
        # Strict assertion: ensure no error flags
        assert "ask_user_trigger" not in result
        assert "pending_user_questions" not in result

        # Verify LLM Call Construction - exact calls
        mock_check.assert_called_once_with(base_state, "design")
        mock_prompt.assert_called_once_with("simulation_designer", base_state)
        mock_build_content.assert_called_once_with(base_state)
        mock_get_spec.assert_called_once_with(base_state, "stage_1_sim", "complexity_class", "unknown")
        
        # Verify arguments passed to LLM - exact values
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["agent_name"] == "simulation_designer"
        assert call_kwargs["system_prompt"] == "System Prompt\n\nComplexity class for this stage: complex_simulation"
        assert call_kwargs["user_content"] == "User Content"
        assert call_kwargs["state"] == base_state
        mock_llm.assert_called_once()
        
        # Verify metrics logging was called
        assert mock_log.called

    def test_designer_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = simulation_designer_node(base_state)
        
        # Verify exact error response structure (awaiting_user_input removed)
        expected_keys = {"workflow_phase", "ask_user_trigger", "pending_user_questions"}
        assert set(result.keys()) == expected_keys, f"Unexpected keys: {set(result.keys()) - expected_keys}"
        
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result.get("ask_user_trigger") is not None
        assert result["workflow_phase"] == "design"  # Should still set phase
        
        # Verify exact error message content
        assert "pending_user_questions" in result
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        assert "ERROR: No stage selected" in result["pending_user_questions"][0]
        assert "workflow error" in result["pending_user_questions"][0].lower()
    
    def test_designer_empty_string_stage_id(self, base_state):
        """Test error when current_stage_id is empty string (falsy but not None)."""
        base_state["current_stage_id"] = ""
        result = simulation_designer_node(base_state)
        
        # Empty string should be treated as missing
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result.get("ask_user_trigger") is not None
        assert result["workflow_phase"] == "design"
    
    def test_designer_missing_stage_id_key(self, base_state):
        """Test error when current_stage_id key doesn't exist."""
        del base_state["current_stage_id"]
        result = simulation_designer_node(base_state)
        
        # Missing key should be treated as missing
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result.get("ask_user_trigger") is not None
        assert result["workflow_phase"] == "design"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_llm_failure(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.side_effect = Exception("API Error")
        
        result = simulation_designer_node(base_state)
        
        # Verify exact error response structure (awaiting_user_input removed)
        assert "ask_user_trigger" in result
        assert "pending_user_questions" in result
        assert "workflow_phase" in result
        
        # Should escalate to user
        assert result["ask_user_trigger"] == "llm_error"
        assert result.get("ask_user_trigger") is not None
        assert result["workflow_phase"] == "design"
        
        # Should contain error info - verify exact content
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) > 0
        assert any("API Error" in str(q) for q in result["pending_user_questions"])
        
        # Verify LLM was called before error
        mock_llm.assert_called_once()
        
        # Verify all setup calls happened
        mock_check.assert_called_once()
        mock_prompt.assert_called_once()
        mock_get_spec.assert_called_once()
        mock_build_content.assert_called_once()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_injects_feedback(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test feedback injection into prompt."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        base_state["reviewer_feedback"] = "Fix mesh size"
        
        simulation_designer_node(base_state)
        
        # Verify prompt contains feedback - exact format
        call_kwargs = mock_llm.call_args[1]
        expected_prompt = "Base Prompt\n\nComplexity class for this stage: standard\n\nREVISION FEEDBACK: Fix mesh size"
        assert call_kwargs["system_prompt"] == expected_prompt
        
        mock_prompt.assert_called_once_with("simulation_designer", base_state)
    
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_no_feedback_when_empty(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that empty feedback string is not injected."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        base_state["reviewer_feedback"] = ""
        
        simulation_designer_node(base_state)
        
        # Verify prompt does NOT contain feedback section
        call_kwargs = mock_llm.call_args[1]
        expected_prompt = "Base Prompt\n\nComplexity class for this stage: standard"
        assert call_kwargs["system_prompt"] == expected_prompt
        assert "REVISION FEEDBACK" not in call_kwargs["system_prompt"]
    
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_no_feedback_when_missing_key(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that missing reviewer_feedback key is handled."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        if "reviewer_feedback" in base_state:
            del base_state["reviewer_feedback"]
        
        simulation_designer_node(base_state)
        
        # Verify prompt does NOT contain feedback section
        call_kwargs = mock_llm.call_args[1]
        expected_prompt = "Base Prompt\n\nComplexity class for this stage: standard"
        assert call_kwargs["system_prompt"] == expected_prompt
        assert "REVISION FEEDBACK" not in call_kwargs["system_prompt"]

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_missing_assumptions_in_output(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that missing 'new_assumptions' in LLM output is handled gracefully."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {"design_description": "Just description"}
        
        result = simulation_designer_node(base_state)
        
        # Verify exact result structure
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design"
        assert "design_description" in result
        assert result["design_description"] == {"design_description": "Just description"}
        
        # Should not modify assumptions if none returned
        assert "assumptions" not in result
        
        # Verify original assumptions are unchanged
        assert base_state["assumptions"]["global_assumptions"] == [{"id": "existing_1", "description": "Existing assumption"}]
        
        mock_llm.assert_called_once()
    
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_empty_assumptions_list(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that empty 'new_assumptions' list is handled correctly."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "design_description": "Desc",
            "new_assumptions": []  # Empty list
        }
        
        result = simulation_designer_node(base_state)
        
        # Empty list should be treated as no assumptions
        assert "assumptions" not in result
        assert result["design_description"]["design_description"] == "Desc"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_preserves_existing_assumptions(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that existing assumptions are preserved when new ones are added."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        
        result = simulation_designer_node(base_state)
        
        # Verify exact structure
        assert "assumptions" in result
        assert isinstance(result["assumptions"], dict)
        assert "global_assumptions" in result["assumptions"]
        
        global_assumptions = result["assumptions"]["global_assumptions"]
        assert isinstance(global_assumptions, list)
        assert len(global_assumptions) == 2
        
        # Verify exact order: existing first, then new
        assert global_assumptions[0]["id"] == "existing_1"
        assert global_assumptions[0]["description"] == "Existing assumption"
        assert global_assumptions[1]["id"] == "new"
        assert global_assumptions[1]["description"] == "new"
        
        # Verify original state unchanged
        assert base_state["assumptions"]["global_assumptions"] == [{"id": "existing_1", "description": "Existing assumption"}]
        
        mock_llm.assert_called_once()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_none_assumptions(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when state['assumptions'] is explicitly None (bug repro)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "design_description": "Desc",
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        base_state["assumptions"] = None  # Simulate corrupted state or init
        
        # This expects the code to handle None safely.
        # If it raises AttributeError, the test will fail, revealing the bug.
        try:
            result = simulation_designer_node(base_state)
        except (AttributeError, TypeError) as e:
            pytest.fail(f"simulation_designer_node crashed because state['assumptions'] was None: {e}")
            
        # Verify exact structure
        assert "assumptions" in result
        assert isinstance(result["assumptions"], dict)
        assert "global_assumptions" in result["assumptions"]
        
        global_assumptions = result["assumptions"]["global_assumptions"]
        assert isinstance(global_assumptions, list)
        assert len(global_assumptions) == 1
        assert global_assumptions[0]["id"] == "new"
        assert global_assumptions[0]["description"] == "new"
    
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_empty_assumptions_dict(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when state['assumptions'] is empty dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "design_description": "Desc",
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        base_state["assumptions"] = {}
        
        result = simulation_designer_node(base_state)
        
        assert "assumptions" in result
        assert isinstance(result["assumptions"], dict)
        assert "global_assumptions" in result["assumptions"]
        assert len(result["assumptions"]["global_assumptions"]) == 1
        assert result["assumptions"]["global_assumptions"][0]["id"] == "new"
    
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_missing_global_assumptions_key(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when assumptions dict exists but global_assumptions key is missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "design_description": "Desc",
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        base_state["assumptions"] = {"other_key": "value"}  # Missing global_assumptions
        
        result = simulation_designer_node(base_state)
        
        assert "assumptions" in result
        assert "global_assumptions" in result["assumptions"]
        assert len(result["assumptions"]["global_assumptions"]) == 1
        assert result["assumptions"]["global_assumptions"][0]["id"] == "new"
        # Verify other keys are preserved
        assert result["assumptions"]["other_key"] == "value"

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
            "ask_user_trigger": "context_overflow",
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

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_get_stage_design_spec_exception(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when get_stage_design_spec raises an exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.side_effect = KeyError("Stage not found")
        mock_build_content.return_value = "User"
        
        # Should propagate exception - this reveals a bug if not handled
        with pytest.raises(KeyError):
            simulation_designer_node(base_state)
        
        # Verify it got to get_stage_design_spec
        mock_get_spec.assert_called_once()
        # Should not have called LLM
        mock_llm.assert_not_called()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_build_user_content_exception(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when build_user_content_for_designer raises an exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.side_effect = ValueError("Invalid state")
        
        # Should propagate exception - this reveals a bug if not handled
        with pytest.raises(ValueError):
            simulation_designer_node(base_state)
        
        # Verify it got to build_user_content
        mock_build_content.assert_called_once()
        # Should not have called LLM
        mock_llm.assert_not_called()

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_build_agent_prompt_exception(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when build_agent_prompt raises an exception."""
        mock_check.return_value = None
        mock_prompt.side_effect = RuntimeError("Prompt build failed")
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        
        # Should propagate exception - this reveals a bug if not handled
        with pytest.raises(RuntimeError):
            simulation_designer_node(base_state)
        
        # Verify it got to build_agent_prompt
        mock_prompt.assert_called_once()
        # Should not have called LLM
        mock_llm.assert_not_called()

    @patch("src.agents.design.log_agent_call")
    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_calls_log_agent_call(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, mock_log, base_state):
        """Test that log_agent_call is invoked with correct parameters."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {"design_description": "Desc"}
        log_func = MagicMock()
        mock_log.return_value = log_func
        
        result = simulation_designer_node(base_state)
        
        # Verify log_agent_call was called with correct agent name and phase
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0]
        assert call_args[0] == "SimulationDesignerAgent"
        assert call_args[1] == "design"
        assert isinstance(call_args[2], type(mock_log.call_args[0][2]))  # datetime object
        
        # Verify the returned function was called with state and result
        log_func.assert_called_once()
        log_call_args = log_func.call_args[0]
        assert log_call_args[0] == base_state
        assert log_call_args[1] == result

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_empty_llm_output(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when LLM returns empty dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        
        result = simulation_designer_node(base_state)
        
        # Should still return valid structure
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design"
        assert "design_description" in result
        assert result["design_description"] == {}
        assert "assumptions" not in result

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_non_dict_llm_output(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when LLM returns non-dict (string or None)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = "Just a string"
        
        result = simulation_designer_node(base_state)
        
        # Should still return valid structure
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design"
        assert "design_description" in result
        assert result["design_description"] == "Just a string"
        # Should not crash trying to get new_assumptions from string
        assert "assumptions" not in result

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_handles_multiple_new_assumptions(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test merging multiple new assumptions."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "design_description": "Desc",
            "new_assumptions": [
                {"id": "A1", "description": "First"},
                {"id": "A2", "description": "Second"},
                {"id": "A3", "description": "Third"}
            ]
        }
        
        result = simulation_designer_node(base_state)
        
        assert "assumptions" in result
        global_assumptions = result["assumptions"]["global_assumptions"]
        assert len(global_assumptions) == 4  # 1 existing + 3 new
        
        # Verify order: existing first, then new ones in order
        assert global_assumptions[0]["id"] == "existing_1"
        assert global_assumptions[1]["id"] == "A1"
        assert global_assumptions[2]["id"] == "A2"
        assert global_assumptions[3]["id"] == "A3"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_preserves_all_assumption_dict_keys(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that all keys in assumptions dict are preserved, not just global_assumptions."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "new_assumptions": [{"id": "new", "description": "new"}]
        }
        
        # Add extra keys to assumptions dict
        base_state["assumptions"]["stage_specific"] = {"stage_1": ["assumption1"]}
        base_state["assumptions"]["metadata"] = {"version": "1.0"}
        
        result = simulation_designer_node(base_state)
        
        # Verify all keys are preserved
        assert "global_assumptions" in result["assumptions"]
        assert "stage_specific" in result["assumptions"]
        assert "metadata" in result["assumptions"]
        assert result["assumptions"]["stage_specific"] == {"stage_1": ["assumption1"]}
        assert result["assumptions"]["metadata"] == {"version": "1.0"}

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_complexity_class_injection_format(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test exact format of complexity class injection in prompt."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        mock_get_spec.return_value = "high_complexity"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        
        simulation_designer_node(base_state)
        
        call_kwargs = mock_llm.call_args[1]
        expected_prompt = "Base Prompt\n\nComplexity class for this stage: high_complexity"
        assert call_kwargs["system_prompt"] == expected_prompt

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_complexity_class_default_value(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that default complexity class 'unknown' is used when get_stage_design_spec returns it."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        mock_get_spec.return_value = "unknown"  # Default value
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        
        simulation_designer_node(base_state)
        
        # Verify get_stage_design_spec was called with default "unknown"
        mock_get_spec.assert_called_once_with(base_state, "stage_1_sim", "complexity_class", "unknown")
        
        call_kwargs = mock_llm.call_args[1]
        assert "Complexity class for this stage: unknown" in call_kwargs["system_prompt"]

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_state_passed_to_llm_is_merged_state(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that state passed to LLM includes context updates."""
        context_updates = {"context_refresh": True, "metrics": {"tokens": 100}}
        mock_check.return_value = context_updates
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {}
        
        simulation_designer_node(base_state)
        
        # Verify merged state includes context updates
        merged_state = mock_llm.call_args[1]["state"]
        assert merged_state["context_refresh"] is True
        assert merged_state["metrics"]["tokens"] == 100
        # Verify original state keys are still present
        assert merged_state["current_stage_id"] == "stage_1_sim"
        assert merged_state["paper_id"] == "test_paper"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_new_assumptions_not_list_handled_gracefully(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test that non-list new_assumptions is handled gracefully (logs warning, doesn't crash)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "new_assumptions": "not a list"  # Wrong type
        }
        
        # Should handle gracefully without crashing
        result = simulation_designer_node(base_state)
        
        # Should not add invalid assumptions - only existing ones should remain
        # Verify assumptions are not modified (or if they are, only valid ones)
        if "assumptions" in result:
            # If assumptions exist, verify they only contain valid assumption objects
            global_assumptions = result["assumptions"]["global_assumptions"]
            # Should only have the existing assumption, not the invalid string
            assert len(global_assumptions) == 1
            assert global_assumptions[0]["id"] == "existing_1"
            # Verify no string characters were added
            assert all(isinstance(a, dict) for a in global_assumptions)
        else:
            # Or assumptions might not be in result at all if none were valid
            pass

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.check_context_or_escalate")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_stage_design_spec")
    @patch("src.agents.design.build_user_content_for_designer")
    def test_designer_new_assumptions_items_missing_id(self, mock_build_content, mock_get_spec, mock_llm, mock_check, mock_prompt, base_state):
        """Test behavior when new_assumptions items are missing required fields."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_get_spec.return_value = "standard"
        mock_build_content.return_value = "User"
        mock_llm.return_value = {
            "new_assumptions": [
                {"description": "missing id"},  # Missing id field
                {"id": "has_id", "description": "complete"}
            ]
        }
        
        result = simulation_designer_node(base_state)
        
        # Should handle gracefully - add what it can
        assert "assumptions" in result
        global_assumptions = result["assumptions"]["global_assumptions"]
        # Should have 1 existing + 2 new (even if one is malformed)
        assert len(global_assumptions) >= 2
        # Verify the complete one is there
        assert any(a.get("id") == "has_id" for a in global_assumptions)

# ═══════════════════════════════════════════════════════════════════════
# design_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════
