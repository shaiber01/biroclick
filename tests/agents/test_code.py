"""Unit tests for src/agents/code.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.code import (
    code_generator_node,
    code_reviewer_node,
)


class TestCodeGeneratorNode:
    """Tests for code_generator_node function."""

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generates_code_on_success(
        self, mock_user, mock_prompt, mock_context, mock_call, validated_code_generator_response
    ):
        """Should generate code from LLM output (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        
        # Use the validated mock response
        # Ensure it meets the length requirement for valid code (>50 chars)
        mock_response = validated_code_generator_response.copy()
        if len(mock_response.get("code", "")) < 50:
            mock_response["code"] = mock_response.get("code", "") + " # " + ("x" * 50)
        mock_call.return_value = mock_response
        
        # Design must be >50 chars and not contain stub markers
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",  # Avoids validated_materials check
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation.",
        }
        
        result = code_generator_node(state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "code" in result
        # Basic check that we got code back
        assert len(result["code"]) > 0

    @patch("src.agents.code.check_context_or_escalate")
    def test_errors_on_stub_design(self, mock_context):
        """Should error when design_description is a stub."""
        mock_context.return_value = None
        
        state = {
            "current_stage_id": "stage1",
            "design_description": "STUB design TODO replace",  # Contains stub marker
        }
        
        result = code_generator_node(state)
        
        # Returns supervisor_verdict to continue and tries again
        assert result.get("supervisor_verdict") == "ok_continue"

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_handles_llm_error(self, mock_user, mock_prompt, mock_context, mock_call):
        """Should handle LLM call failure gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_user.return_value = "content"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation.",
        }
        
        result = code_generator_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "llm_error"

    @patch("src.agents.code.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "design_description": {}}
        
        result = code_generator_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.code.check_context_or_escalate")
    def test_fails_on_missing_stage_id(self, mock_context):
        """Should fail when current_stage_id is missing."""
        mock_context.return_value = None
        
        state = {
            "design_description": "Valid design"
        }
        
        result = code_generator_node(state)
        
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True

    @patch("src.agents.code.check_context_or_escalate")
    def test_fails_on_missing_materials(self, mock_context):
        """Should fail when validated_materials is empty for Stage 1+."""
        mock_context.return_value = None
        
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "SIMULATION",
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation. It is definitely longer than 50 characters to pass the validation check.",
            "validated_materials": [] # Empty
        }
        
        result = code_generator_node(state)
        
        assert "run_error" in result
        assert "validated_materials is empty" in result["run_error"]
        assert result["workflow_phase"] == "code_generation"

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_extracts_simulation_code_fallback(
        self, mock_user, mock_prompt, mock_context, mock_call, validated_code_generator_response
    ):
        """Should extract code from simulation_code field if code is empty."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        
        mock_response = validated_code_generator_response.copy()
        mock_response["code"] = "" # Empty primary field
        mock_response["simulation_code"] = "import meep as mp\n# Simulation code fallback\nsim = mp.Simulation()"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation. It is definitely longer than 50 characters to pass the validation check.",
        }
        
        result = code_generator_node(state)
        
        assert "import meep" in result["code"]
        assert "# Simulation code fallback" in result["code"]

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_handles_empty_generated_code(
        self, mock_user, mock_prompt, mock_context, mock_call, validated_code_generator_response
    ):
        """Should reject empty generated code."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        
        mock_response = validated_code_generator_response.copy()
        mock_response["code"] = ""
        mock_response.pop("simulation_code", None) # Ensure no fallback
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": "This is a detailed simulation design specification with geometry, sources, and monitors for the FDTD simulation. It is definitely longer than 50 characters to pass the validation check.",
        }
        
        result = code_generator_node(state)
        
        # The implementation tries JSON dump as last resort fallback
        assert "Failed generation" in result.get("explanation", "") or "code" in result["code"]
        
        # Let's test specifically the stub/empty rejection logic by forcing a stub
        mock_response["code"] = "TODO: Implement simulation"
        mock_call.return_value = mock_response
        
        result = code_generator_node(state)
        assert "ERROR: Generated code is empty or contains stub markers" in result["reviewer_feedback"]
        assert result["code_revision_count"] == 1

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_returns_generated_code_on_fallback(self, mock_user, mock_prompt, mock_context, mock_call):
        """Should return code even if fallback to JSON dump is used."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_user.return_value = "user content"
        
        # Return dict without code field
        mock_call.return_value = {"something_else": "value", "key": "data"}
        
        state = {
            "current_stage_id": "stage1",
            "current_stage_type": "MATERIAL_VALIDATION",
            "design_description": "Valid design description > 50 chars long............................."
        }
        
        result = code_generator_node(state)
        
        # Should use JSON dump
        assert "something_else" in result["code"]
        # But since it's just a JSON dump of random dict, it might be flagged as stub/empty if validation is strict
        # The current validation logic checks length < 50.
        # {"something_else": "value", "key": "data"} is < 50 chars.
        # So let's make it longer.
        mock_call.return_value = {"long_key": "x" * 60}
        result = code_generator_node(state)
        assert result["workflow_phase"] == "code_generation"
        assert "long_key" in result["code"]


class TestCodeReviewerNode:
    """Tests for code_reviewer_node function."""

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_approves_valid_code(self, mock_prompt, mock_context, mock_call, validated_code_reviewer_response):
        """Should approve valid code (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        
        mock_response = validated_code_reviewer_response.copy()
        mock_response["verdict"] = "approve"
        mock_response["issues"] = []
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "generated_code": "import meep as mp\nsim = mp.Simulation()",
        }
        
        result = code_reviewer_node(state)
        
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_rejects_with_feedback(self, mock_prompt, mock_context, mock_call, validated_code_reviewer_response):
        """Should reject code and provide feedback (using validated mock)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_code_reviewer_response.copy()
        mock_response["verdict"] = "needs_revision"
        mock_response["issues"] = [{"severity": "major", "description": "Missing output saving"}]
        mock_response["feedback"] = "Add np.savetxt to save results"
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "code": "import meep as mp",  # Uses 'code' not 'generated_code'
            "code_revision_count": 0,
        }
        
        result = code_reviewer_node(state)
        
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1
        assert "reviewer_feedback" in result  # Uses 'reviewer_feedback' not 'code_reviewer_feedback'

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_auto_approves_on_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should auto-approve when LLM call fails."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "generated_code": "import meep",
        }
        
        result = code_reviewer_node(state)
        
        assert result["last_code_review_verdict"] == "approve"

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow.
        
        Note: Patches base.py because code_reviewer_node uses @with_context_check decorator.
        """
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "generated_code": "code"}
        
        result = code_reviewer_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.build_agent_prompt")
    def test_respects_max_revisions(self, mock_prompt, mock_context, mock_call, validated_code_reviewer_response):
        """Should not exceed max revisions."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        
        mock_response = validated_code_reviewer_response.copy()
        mock_response["verdict"] = "needs_revision"
        mock_response["issues"] = [{"severity": "minor", "description": "issue"}]
        mock_call.return_value = mock_response
        
        state = {
            "current_stage_id": "stage1",
            "generated_code": "code",
            "code_revision_count": 5,
            "runtime_config": {"max_code_revisions": 5},
        }
        
        result = code_reviewer_node(state)
        
        # Should not increment past max
        assert result["code_revision_count"] == 5
