"""
Tests for Design Agents (SimulationDesignerAgent, DesignReviewerAgent).
"""

import pytest
from unittest.mock import patch, MagicMock, ANY
from src.agents.design import simulation_designer_node, design_reviewer_node

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Base state for design tests."""
    return {
        "paper_id": "test_paper",
        "current_stage_id": "stage_1_sim",
        "plan": {
            "stages": [
                {
                    "stage_id": "stage_1_sim",
                    "type": "simulation",
                    "name": "Simulation Stage",
                    "targets": ["Fig1"],
                    "complexity_class": "standard"
                }
            ]
        },
        "design_revision_count": 0,
        "runtime_config": {
            "max_design_revisions": 3
        },
        "assumptions": {
            "global_assumptions": [
                {"id": "existing_1", "description": "Existing assumption"}
            ]
        },
        "validated_materials": {},
        "paper_text": "Full paper text...",
        "paper_domain": "nanophotonics"
    }

# ═══════════════════════════════════════════════════════════════════════
# simulation_designer_node Tests
# ═══════════════════════════════════════════════════════════════════════

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
        assert len(result["assumptions"]["global_assumptions"]) == 2
        assert result["assumptions"]["global_assumptions"][0]["id"] == "existing_1"
        assert result["assumptions"]["global_assumptions"][1]["id"] == "A1"
        
        # Verify LLM Call Construction
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


# ═══════════════════════════════════════════════════════════════════════
# design_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDesignReviewerNode:
    """Tests for design_reviewer_node."""

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer approving design."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        base_state["design_description"] = {"some": "design"}
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        assert result["design_revision_count"] == 0
        assert result["reviewer_issues"] == []
        assert result["workflow_phase"] == "design_review"

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    @patch("src.agents.design.get_plan_stage")
    def test_reviewer_constructs_prompt_correctly(self, mock_get_plan_stage, mock_llm, mock_prompt, base_state):
        """Test that user content includes design, stage spec, and feedback."""
        mock_prompt.return_value = "System Prompt"
        mock_llm.return_value = {"verdict": "approve"}
        
        # Setup state
        base_state["design_description"] = {"method": "FDTD"}
        base_state["reviewer_feedback"] = "Previous feedback"
        mock_get_plan_stage.return_value = {"stage_id": "stage_1", "name": "Test Stage"}
        
        design_reviewer_node(base_state)
        
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["user_content"]
        
        # Verify Design inclusion
        assert "DESIGN TO REVIEW" in user_content
        assert '"method": "FDTD"' in user_content
        
        # Verify Stage Spec inclusion
        assert "PLAN STAGE SPEC" in user_content
        assert '"name": "Test Stage"' in user_content
        
        # Verify Feedback inclusion
        assert "REVISION FEEDBACK" in user_content
        assert "Previous feedback" in user_content

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, base_state):
        """Test reviewer requesting revision."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Add more details",
            "issues": ["Missing parameters"]
        }
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1
        assert "Add more details" in result["reviewer_feedback"]
        assert result["reviewer_issues"] == ["Missing parameters"]

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, base_state):
        """Test reviewer hitting max revisions."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision", "feedback": "more"}
        
        base_state["design_revision_count"] = 3 # Already at max
        
        result = design_reviewer_node(base_state)
        
        # Should check what happens when max is reached.
        # The current logic calls increment_counter_with_max.
        # If max is reached, it should NOT increment, and retain current count.
        assert result["design_revision_count"] == 3
        assert result["last_design_review_verdict"] == "needs_revision"
        # Crucially, verify what happens to feedback. It should still update feedback.
        assert "more" in result["reviewer_feedback"]

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer defaults to auto-approve on LLM failure."""
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = design_reviewer_node(base_state)
        
        assert result["last_design_review_verdict"] == "approve"
        # Should verify we log or mark this somehow? The current impl just approves.
        # Ideally it should maybe add a note about auto-approval?
        # The code uses create_llm_error_auto_approve which sets verdict to approve.

    @patch("src.agents.design.build_agent_prompt")
    @patch("src.agents.design.call_agent_with_metrics")
    def test_reviewer_missing_verdict(self, mock_llm, mock_prompt, base_state):
        """Test handling when LLM returns JSON without 'verdict'."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"feedback": "Some feedback but no verdict"}
        
        # The code should handle missing verdict gracefully, e.g., by auto-approving or escalating.
        # It should NOT raise KeyError.
        
        try:
            result = design_reviewer_node(base_state)
            # If it returns result, check if it handled it safely
            assert "last_design_review_verdict" in result
        except KeyError:
            pytest.fail("design_reviewer_node raised KeyError on missing 'verdict' field")

