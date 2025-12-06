"""Tests for physics_sanity_node."""

from unittest.mock import patch

import pytest

from src.agents.execution import physics_sanity_node


@pytest.fixture(name="base_state")
def execution_base_state(execution_state):
    return execution_state


class TestPhysicsSanityNode:
    """Tests for physics_sanity_node."""

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_pass(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics pass."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "pass",
            "summary": "Physics OK."
        }
        
        result = physics_sanity_node(base_state)
        
        # Verify all required fields are present and correct
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Physics OK."
        assert "backtrack_suggestion" not in result
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        assert "design_feedback" not in result
        
        # Verify prompt was built correctly
        mock_prompt.assert_called_once()
        prompt_args, prompt_kwargs = mock_prompt.call_args
        assert prompt_args[0] == "physics_sanity"
        # State may have additional fields from decorator, so check ALL critical fields
        # that physics_sanity_node reads: stage_outputs, current_stage_id, design_description, runtime_config
        prompt_state = prompt_args[1]
        assert prompt_state["current_stage_id"] == base_state["current_stage_id"]
        assert prompt_state["design_description"] == base_state["design_description"]
        assert prompt_state["stage_outputs"] == base_state["stage_outputs"]
        assert prompt_state["runtime_config"] == base_state["runtime_config"]
        # Verify decorator didn't corrupt other critical fields
        assert prompt_state.get("paper_id") == base_state.get("paper_id")
        
        # Verify LLM was called with correct parameters
        mock_llm.assert_called_once()
        args, kwargs = mock_llm.call_args
        assert kwargs["agent_name"] == "physics_sanity"
        assert kwargs["system_prompt"] == "Prompt"
        assert "PHYSICS SANITY CHECK FOR STAGE" in kwargs["user_content"]
        assert "stage_1_sim" in kwargs["user_content"]
        assert "## Stage Outputs" in kwargs["user_content"]
        assert "## Design Spec" in kwargs["user_content"]
        assert "p1" in kwargs["user_content"]  # Design parameters should be in context
        # State may have additional fields, so check ALL critical fields
        # that physics_sanity_node reads: stage_outputs, current_stage_id, design_description, runtime_config
        llm_state = kwargs["state"]
        assert llm_state["current_stage_id"] == base_state["current_stage_id"]
        assert llm_state["design_description"] == base_state["design_description"]
        assert llm_state["stage_outputs"] == base_state["stage_outputs"]
        assert llm_state["runtime_config"] == base_state["runtime_config"]
        # Verify decorator didn't corrupt other critical fields
        assert llm_state.get("paper_id") == base_state.get("paper_id")

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_fail_increments_physics_count(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics fail increments physics_failure_count."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Unphysical energy."
        }
        
        # Verify initial state
        assert base_state.get("physics_failure_count", 0) == 0
        
        result = physics_sanity_node(base_state)
        
        # Verify all fields are correct
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 1
        assert result["physics_feedback"] == "Unphysical energy."
        assert "design_revision_count" not in result
        assert "design_feedback" not in result
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_design_flaw_increments_design_count(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test design_flaw increments design_revision_count."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "design_flaw",
            "summary": "Impossible geometry."
        }
        
        # Verify initial state
        assert base_state.get("design_revision_count", 0) == 0
        
        result = physics_sanity_node(base_state)
        
        # Verify all fields are correct
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1
        assert result["design_feedback"] == "Impossible geometry."
        assert result["physics_feedback"] == "Impossible geometry."
        assert "physics_failure_count" not in result
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_backtrack_suggestion(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test capturing backtrack suggestion."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Backtrack needed.",
            "backtrack_suggestion": {"suggest_backtrack": True, "reason": "Bad materials"}
        }
        
        result = physics_sanity_node(base_state)
        
        # Verify all fields are correct
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Backtrack needed."
        assert result["physics_failure_count"] == 1
        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["backtrack_suggestion"]["reason"] == "Bad materials"
        assert "design_revision_count" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_llm_exception(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics handling LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = physics_sanity_node(base_state)
        
        # Verify exception handling - should auto-approve with "pass"
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert "API Error" in result["physics_feedback"]
        assert "physics_sanity" in result["physics_feedback"].lower() or "auto" in result["physics_feedback"].lower()
        assert "backtrack_suggestion" not in result
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_no_design(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics sanity with missing design."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["design_description"] = None
        
        result = physics_sanity_node(base_state)
        
        # Verify result is correct
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "OK"
        
        # Verify design section is missing when design is None
        args, kwargs = mock_llm.call_args
        assert "## Design Spec" not in kwargs["user_content"]
        assert "## Stage Outputs" in kwargs["user_content"]  # Stage outputs should still be present

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_includes_string_design_spec(self, mock_llm, mock_check, mock_prompt, base_state):
        """Design specs provided as strings must be passed through verbatim."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "warning", "summary": "Manual inspection suggested."}
        base_state["design_description"] = "Design v1 (text-only)"
        
        result = physics_sanity_node(base_state)
        
        # Verify warning verdict is handled correctly
        assert result["physics_verdict"] == "warning"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Manual inspection suggested."
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        assert "backtrack_suggestion" not in result
        
        # Verify string design spec is included verbatim (not as JSON)
        args, kwargs = mock_llm.call_args
        assert "\n## Design Spec\nDesign v1 (text-only)" in kwargs["user_content"]
        assert "\n## Design Spec\n```json" not in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_failure_count_respects_runtime_max(self, mock_llm, mock_check, mock_prompt, base_state):
        """Physics failure counter should not exceed configured max."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Still broken."}
        base_state["physics_failure_count"] = 2
        base_state["runtime_config"]["max_physics_failures"] = 2
        
        result = physics_sanity_node(base_state)
        
        # Verify counter is clamped at max (should not increment from 2 to 3)
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 2  # Should remain at max, not increment
        assert result["physics_feedback"] == "Still broken."
        assert "backtrack_suggestion" not in result
        assert "design_revision_count" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_design_revision_count_respects_runtime_max(self, mock_llm, mock_check, mock_prompt, base_state):
        """Design revision counter should clamp at runtime limit."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "design_flaw", "summary": "Needs redesign."}
        base_state["design_revision_count"] = 1
        base_state["runtime_config"]["max_design_revisions"] = 1
        
        result = physics_sanity_node(base_state)
        
        # Verify counter is clamped at max (should not increment from 1 to 2)
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1  # Should remain at max, not increment
        assert result["design_feedback"] == "Needs redesign."
        assert result["physics_feedback"] == "Needs redesign."
        assert "physics_failure_count" not in result
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_stage_outputs_none_crash(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of None stage_outputs."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = None
        
        # Should not crash - code uses state.get("stage_outputs") or {}
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "OK"
        
        # Verify empty dict was used for stage_outputs in user_content
        args, kwargs = mock_llm.call_args
        assert "## Stage Outputs" in kwargs["user_content"]
        assert "{}" in kwargs["user_content"] or "```json" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_llm_missing_verdict(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM response missing verdict.
        
        Missing verdict is handled gracefully (not as an exception):
        - Defaults to 'pass' with a warning
        - Does NOT treat this as 'LLM unavailable' error
        - Consistent with design_reviewer and code_reviewer patterns
        """
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"summary": "Oops no verdict"}
        
        result = physics_sanity_node(base_state)
        
        # Should default to pass gracefully
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        # Should NOT increment counters (pass verdict)
        assert "physics_failure_count" not in result
        # Feedback should indicate missing verdict (graceful handling)
        assert "Missing verdict" in result["physics_feedback"] or "Oops no verdict" in result["physics_feedback"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_backtrack_malformed(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of malformed backtrack suggestion (not a dict)."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        # backtrack_suggestion should be a dict, but LLM might return string or list
        mock_llm.return_value = {
            "verdict": "fail", 
            "summary": "Backtrack", 
            "backtrack_suggestion": "Yes, please backtrack."
        }
        
        # Code uses isinstance(backtrack, dict) check, so non-dict should be ignored
        result = physics_sanity_node(base_state)
        
        # Should not crash - malformed backtrack_suggestion should be ignored
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 1
        # backtrack_suggestion should not be in result since it's not a dict
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_warning_verdict(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test warning verdict is handled correctly."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "warning", "summary": "Minor concerns but proceed."}
        
        result = physics_sanity_node(base_state)
        
        # Warning should not increment any counters
        assert result["physics_verdict"] == "warning"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Minor concerns but proceed."
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_empty_stage_outputs(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of empty stage_outputs dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = {}
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        
        # Verify empty dict is serialized in user_content
        args, kwargs = mock_llm.call_args
        assert "## Stage Outputs" in kwargs["user_content"]
        assert "{}" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_empty_design_dict(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of empty design dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["design_description"] = {}
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        
        # Empty dict should be treated as falsy, so design spec should not appear
        args, kwargs = mock_llm.call_args
        assert "## Design Spec" not in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_empty_design_string(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of empty design string."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["design_description"] = ""
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        
        # Empty string should be treated as falsy
        args, kwargs = mock_llm.call_args
        assert "## Design Spec" not in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_counter_initialization(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test counter initialization when counter doesn't exist in state."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "First failure."}
        
        # Remove counter from state to test initialization
        if "physics_failure_count" in base_state:
            del base_state["physics_failure_count"]
        
        result = physics_sanity_node(base_state)
        
        # Counter should be initialized to 1
        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_design_revision_counter_initialization(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test design_revision_count initialization when counter doesn't exist."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "design_flaw", "summary": "First design flaw."}
        
        # Remove counter from state to test initialization
        if "design_revision_count" in base_state:
            del base_state["design_revision_count"]
        
        result = physics_sanity_node(base_state)
        
        # Counter should be initialized to 1
        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 1

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_missing_runtime_config(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when runtime_config is missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Failure."}
        
        # Remove runtime_config to test default behavior
        if "runtime_config" in base_state:
            del base_state["runtime_config"]
        
        result = physics_sanity_node(base_state)
        
        # Should use default max from MAX_PHYSICS_FAILURES constant
        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_empty_runtime_config(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when runtime_config is empty dict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Failure."}
        
        base_state["runtime_config"] = {}
        
        result = physics_sanity_node(base_state)
        
        # Should use default max from MAX_PHYSICS_FAILURES constant
        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_missing_summary(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when LLM response is missing summary."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass"}
        
        result = physics_sanity_node(base_state)
        
        # Should use default "No feedback provided."
        assert result["physics_verdict"] == "pass"
        assert result["physics_feedback"] == "No feedback provided."

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_backtrack_false_not_included(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that backtrack_suggestion with suggest_backtrack=False is not included."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Failure.",
            "backtrack_suggestion": {"suggest_backtrack": False, "reason": "No need"}
        }
        
        result = physics_sanity_node(base_state)
        
        # Should not include backtrack_suggestion when suggest_backtrack is False
        assert result["physics_verdict"] == "fail"
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_backtrack_missing_key(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when backtrack_suggestion dict is missing suggest_backtrack key."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Failure.",
            "backtrack_suggestion": {"reason": "Some reason"}  # Missing suggest_backtrack
        }
        
        result = physics_sanity_node(base_state)
        
        # Should not include backtrack_suggestion when suggest_backtrack key is missing
        assert result["physics_verdict"] == "fail"
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_backtrack_none(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when backtrack_suggestion is None."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "fail",
            "summary": "Failure.",
            "backtrack_suggestion": None
        }
        
        result = physics_sanity_node(base_state)
        
        # Should not crash and should not include backtrack_suggestion
        assert result["physics_verdict"] == "fail"
        assert "backtrack_suggestion" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.base.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_context_escalation(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that context escalation is handled correctly."""
        escalation_result = {
            "ask_user_trigger": "context_overflow",
            "ask_user_trigger": "missing_context",
            "pending_user_questions": ["Need more context"]
        }
        mock_check.return_value = escalation_result
        mock_prompt.return_value = "Prompt"
        
        # When context check returns escalation with ask_user_trigger set,
        # decorator should return immediately without calling the function
        result = physics_sanity_node(base_state)
        
        # Should return escalation result without calling LLM
        assert result == escalation_result
        assert result.get("ask_user_trigger") is not None
        mock_llm.assert_not_called()
        mock_prompt.assert_not_called()

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_context_state_update(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that context check state updates are merged."""
        state_update = {"some_context_field": "updated_value"}
        mock_check.return_value = state_update
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        # When context check returns state update (not escalation), it should be merged
        result = physics_sanity_node(base_state)
        
        # Should proceed normally after merging state
        assert result["physics_verdict"] == "pass"
        mock_llm.assert_called_once()

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_stage_id_missing(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when current_stage_id is missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        if "current_stage_id" in base_state:
            del base_state["current_stage_id"]
        
        result = physics_sanity_node(base_state)
        
        # Should use "unknown" as default
        assert result["physics_verdict"] == "pass"
        args, kwargs = mock_llm.call_args
        assert "unknown" in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_design_dict_formatting(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that dict design_description is formatted as JSON."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        design_dict = {"parameters": {"p1": 10, "p2": 20}, "geometry": "sphere"}
        base_state["design_description"] = design_dict
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        
        # Verify dict is formatted as JSON in user_content
        args, kwargs = mock_llm.call_args
        assert "## Design Spec" in kwargs["user_content"]
        assert "```json" in kwargs["user_content"]
        assert '"p1": 10' in kwargs["user_content"] or '"p1":10' in kwargs["user_content"]

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_multiple_failures_increment(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that multiple failures increment counter correctly."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Failure."}
        
        base_state["physics_failure_count"] = 0
        
        # First failure
        result1 = physics_sanity_node(base_state)
        assert result1["physics_failure_count"] == 1
        
        # Update state for second call
        base_state["physics_failure_count"] = 1
        result2 = physics_sanity_node(base_state)
        assert result2["physics_failure_count"] == 2

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_backtrack_default_false(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test that backtrack_suggestion defaults to False when missing."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "fail", "summary": "Failure."}
        # Missing backtrack_suggestion - code should add default
        
        result = physics_sanity_node(base_state)
        
        # Should not include backtrack_suggestion since default is False
        assert result["physics_verdict"] == "fail"
        assert "backtrack_suggestion" not in result
