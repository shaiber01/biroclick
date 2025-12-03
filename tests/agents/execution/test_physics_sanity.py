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
        
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Physics OK."
        assert "backtrack_suggestion" not in result
        
        # Verify context passed
        args, kwargs = mock_llm.call_args
        assert "p1" in kwargs["user_content"] # Design parameters should be in context

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
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 1
        assert "design_revision_count" not in result
        assert result["physics_feedback"] == "Unphysical energy."
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
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1
        assert "physics_failure_count" not in result
        # Should return design_feedback for design flaws
        assert result["design_feedback"] == "Impossible geometry." 
        assert result["physics_feedback"] == "Impossible geometry."

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
        
        assert result["workflow_phase"] == "physics_validation"
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["physics_feedback"] == "Backtrack needed."

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_llm_exception(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test physics handling LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        assert "API Error" in result["physics_feedback"]

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
        
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"
        # Verify design section is missing or handled
        args, kwargs = mock_llm.call_args
        # Should not have "## Design Spec" if design is None/empty?
        # Let's check the code: if design: ...
        assert "## Design Spec" not in kwargs["user_content"]

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
        
        assert result["physics_verdict"] == "warning"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_feedback"] == "Manual inspection suggested."
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
        
        assert result["physics_verdict"] == "fail"
        assert result["workflow_phase"] == "physics_validation"
        assert result["physics_failure_count"] == 2
        assert result["physics_feedback"] == "Still broken."
        assert "backtrack_suggestion" not in result

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
        
        assert result["physics_verdict"] == "design_flaw"
        assert result["workflow_phase"] == "physics_validation"
        assert result["design_revision_count"] == 1
        assert result["design_feedback"] == "Needs redesign."
        assert result["physics_feedback"] == "Needs redesign."
        assert "physics_failure_count" not in result

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_stage_outputs_none_crash(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of None stage_outputs."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
        
        base_state["stage_outputs"] = None
        
        try:
            result = physics_sanity_node(base_state)
        except AttributeError:
             pytest.fail("physics_sanity_node crashed due to None stage_outputs")

        assert result["physics_verdict"] == "pass"

    @patch("src.agents.execution.build_agent_prompt")
    @patch("src.agents.execution.check_context_or_escalate")
    @patch("src.agents.execution.call_agent_with_metrics")
    def test_physics_llm_missing_verdict(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM response missing verdict."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"summary": "Oops no verdict"}
        
        try:
            result = physics_sanity_node(base_state)
            assert result["physics_verdict"] in ["pass", "fail", "warning", "design_flaw"]
        except KeyError:
             pytest.fail("physics_sanity_node crashed due to missing 'verdict' key")

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
        
        try:
            result = physics_sanity_node(base_state)
            # Should not crash and ideally not return the malformed suggestion if it expects a dict
        except AttributeError:
             pytest.fail("physics_sanity_node crashed due to malformed backtrack_suggestion")

        # Check if it handled it gracefully (e.g. ignored it or wrapped it)
        # The code expects .get() so likely crashed.
