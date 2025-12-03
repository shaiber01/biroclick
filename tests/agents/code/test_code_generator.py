"""Tests for code_generator_node."""

from unittest.mock import ANY, patch

import pytest

from schemas.state import MAX_DESIGN_REVISIONS

from src.agents.code import code_generator_node
from tests.agents.shared_objects import LONG_FALLBACK_JSON, LONG_FALLBACK_PAYLOAD


@pytest.fixture(name="base_state")
def code_base_state(code_state):
    return code_state


class TestCodeGeneratorNode:
    """Tests for code_generator_node."""

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_success(self, mock_user_content, mock_llm, mock_check, mock_prompt, base_state):
        """Test successful code generation."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_user_content.return_value = "User Prompt"
        mock_llm.return_value = {
            "code": "import meep as mp\nprint('success')\n# This code must be longer than 50 characters to pass validation checks in code.py",
            "expected_outputs": ["output.csv"]
        }
        
        result = code_generator_node(base_state)
        
        # Assertions
        assert result["workflow_phase"] == "code_generation"
        assert "import meep" in result["code"]
        assert "output.csv" in result["expected_outputs"]
        
        # Verify dependencies called correctly
        mock_check.assert_called_once_with(base_state, "generate_code")
        mock_prompt.assert_called_once_with("code_generator", base_state)
        mock_user_content.assert_called_once_with(base_state)
        mock_llm.assert_called_once_with(
            agent_name="code_generator",
            system_prompt="System Prompt",
            user_content="User Prompt",
            state=base_state
        )

    def test_generator_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True
        assert "No stage selected" in result["pending_user_questions"][0]

    @pytest.mark.parametrize("design_desc", [
        "TODO: Add design",
        "Placeholder for design",
        "STUB: design here",
        "",
        "Too short",
        "A" * 49  # Just under 50 chars
    ])
    def test_generator_invalid_design(self, base_state, design_desc):
        """Test error when design is a stub or too short."""
        base_state["design_description"] = design_desc
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "design_revision_count" in result
        # Should increment design revision
        assert result["design_revision_count"] == 1
        assert "Design description is missing or contains stub" in result["reviewer_feedback"]
        assert result["supervisor_verdict"] == "ok_continue"
        assert "code" not in result

    def test_generator_design_revision_max_cap(self, base_state):
        """Test that design revision count doesn't exceed max."""
        base_state["design_description"] = "TODO: stub"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        
        result = code_generator_node(base_state)
        
        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS
        assert "Design description is missing or contains stub" in result["reviewer_feedback"]

    def test_generator_missing_materials_stage1(self, base_state):
        """Test error when validated_materials is missing for Stage 1+."""
        base_state["validated_materials"] = [] # Empty
        # current_stage_type is SINGLE_STRUCTURE by default in fixture (Stage 1+)
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "validated_materials is empty" in result["run_error"]
        assert result["code_revision_count"] == 1
        assert "expected_outputs" not in result

    def test_generator_skip_materials_validation_stage0(self, base_state):
        """Test that materials validation is skipped for MATERIAL_VALIDATION stage."""
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["validated_materials"] = [] # Empty, but should be allowed
        
        # We need to mock dependencies to get past this check to success path
        with patch("src.agents.code.build_agent_prompt") as mock_prompt, \
             patch("src.agents.code.check_context_or_escalate") as mock_check, \
             patch("src.agents.code.call_agent_with_metrics") as mock_llm, \
             patch("src.agents.code.build_user_content_for_code_generator") as mock_uc:
             
            mock_check.return_value = None
            mock_llm.return_value = {
                "code": "import meep\n# valid code length > 50 chars................................", 
                "expected_outputs": []
            }
            
            result = code_generator_node(base_state)
            
            assert "run_error" not in result
            assert "code" in result
            assert result["expected_outputs"] == []
            mock_llm.assert_called_once()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_includes_feedback(self, mock_uc, mock_llm, mock_check, mock_prompt, base_state):
        """Test that reviewer feedback is appended to the prompt."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        base_state["reviewer_feedback"] = "Fix the geometry."
        mock_llm.return_value = {"code": "code"*20, "expected_outputs": ["field_map.h5"]} # Valid length
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert result["expected_outputs"] == ["field_map.h5"]
        
        # Verify feedback in prompt
        call_args = mock_llm.call_args[1]
        assert "REVISION FEEDBACK: Fix the geometry." in call_args["system_prompt"]

    @pytest.mark.parametrize(
        "llm_output, expected_code, expected_outputs",
        [
            (
                {"code": "valid code" * 10, "expected_outputs": ["Ez.csv"]},
                "valid code" * 10,
                ["Ez.csv"],
            ),
            (
                {"simulation_code": "sim code" * 10},
                "sim code" * 10,
                [],
            ),
            (
                dict(LONG_FALLBACK_PAYLOAD),
                LONG_FALLBACK_JSON,
                [],
            ),
        ],
    )
    def test_generator_code_extraction(self, llm_output, expected_code, expected_outputs, base_state):
        """Test extraction of code from various LLM output formats."""
        with patch("src.agents.code.build_agent_prompt"), \
             patch("src.agents.code.check_context_or_escalate", return_value=None), \
             patch("src.agents.code.call_agent_with_metrics") as mock_llm, \
             patch("src.agents.code.build_user_content_for_code_generator"):
            
            mock_llm.return_value = llm_output
            
            result = code_generator_node(base_state)
            
            assert result["workflow_phase"] == "code_generation"
            assert result["code"] == expected_code
            assert result["expected_outputs"] == expected_outputs

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_stub_code_output(self, mock_uc, mock_llm, mock_check, mock_prompt, base_state):
        """Test error when LLM returns stub code."""
        mock_check.return_value = None
        mock_llm.return_value = {"code": "# TODO: Implement simulation"}
        
        result = code_generator_node(base_state)
        
        # Should detect stub and request revision
        assert result["workflow_phase"] == "code_generation"
        assert result["code_revision_count"] == 1
        assert "Generated code is empty or contains stub" in result["reviewer_feedback"]
        assert result["code"] == "# TODO: Implement simulation"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_generator_handles_llm_failure(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM exception."""
        mock_check.return_value = None
        mock_llm.side_effect = Exception("API Error")
        
        result = code_generator_node(base_state)
        
        # Should escalate to user via create_llm_error_escalation
        assert result["ask_user_trigger"] == "llm_error"
        assert result["awaiting_user_input"] is True
        # Expect human-readable name "Code Generator"
        assert "Code Generator" in result["pending_user_questions"][0]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_stub_code_respects_max_revisions(self, mock_uc, mock_llm, mock_check, mock_prompt, base_state):
        """Test that code revision count respects max limit even when stub code is generated."""
        mock_check.return_value = None
        mock_llm.return_value = {"code": "# TODO: Implement simulation"}
        
        # Set current revisions to max
        max_revs = base_state["runtime_config"]["max_code_revisions"]
        base_state["code_revision_count"] = max_revs
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        # Should not exceed max
        assert result["code_revision_count"] == max_revs
        assert "Generated code is empty or contains stub" in result["reviewer_feedback"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    def test_generator_context_escalation_short_circuits(self, mock_check, mock_llm, mock_user_content, mock_prompt, base_state):
        """Ensure context escalation halts execution before making LLM calls."""
        escalation = {
            "workflow_phase": "code_generation",
            "ask_user_trigger": "context_overflow",
            "pending_user_questions": ["Context overflow, need guidance"],
            "awaiting_user_input": True,
        }
        mock_check.return_value = escalation
        
        result = code_generator_node(base_state)
        
        assert result == escalation
        mock_llm.assert_not_called()
        mock_prompt.assert_not_called()
        mock_user_content.assert_not_called()

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    def test_generator_context_state_updates_passed_to_llm(self, mock_check, mock_llm, mock_user_content, mock_prompt, base_state):
        """Ensure non-blocking context updates are merged into downstream state."""
        mock_check.return_value = {"context_trimmed": True}
        mock_prompt.return_value = "System Prompt"
        mock_user_content.return_value = "User Content"
        mock_llm.return_value = {"code": "print('ok')" * 20}
        
        code_generator_node(base_state)
        
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["state"]["context_trimmed"] is True
        assert call_kwargs["state"]["paper_id"] == base_state["paper_id"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    def test_generator_expected_outputs_default_empty(self, mock_check, mock_llm, mock_user_content, mock_prompt, base_state):
        """Ensure expected_outputs defaults to [] when LLM omits it."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_user_content.return_value = "User Content"
        mock_llm.return_value = {"code": "print('ok')" * 20}
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert result["expected_outputs"] == []
        mock_llm.assert_called_once_with(
            agent_name="code_generator",
            system_prompt="System Prompt",
            user_content="User Content",
            state=ANY,
        )

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.build_user_content_for_code_generator")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.check_context_or_escalate")
    def test_generator_short_code_without_stub_markers(self, mock_check, mock_llm, mock_user_content, mock_prompt, base_state):
        """Short code lacking stub markers should still trigger revision path."""
        mock_check.return_value = None
        mock_prompt.return_value = "System Prompt"
        mock_user_content.return_value = "User Content"
        mock_llm.return_value = {"code": "print('short run')"}
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert result["code_revision_count"] == 1
        assert "Generated code is empty or contains stub" in result["reviewer_feedback"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_llm_returns_string(self, mock_uc, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling when LLM returns a string instead of a dict."""
        mock_check.return_value = None
        # LLM returns a string directly (e.g. raw code)
        mock_llm.return_value = "import meep as mp\n" * 5
        
        # This should NOT raise an AttributeError when calling .get()
        # It should treat the string as the code itself or handle gracefully
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "import meep" in result["code"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_validated_materials_none(self, mock_uc, mock_llm, mock_check, mock_prompt, base_state):
        """Test error when validated_materials is explicitly None (not just empty list)."""
        mock_check.return_value = None
        base_state["validated_materials"] = None
        # current_stage_type is SINGLE_STRUCTURE
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "validated_materials is empty" in result["run_error"]
        assert result["code_revision_count"] == 1

# ═══════════════════════════════════════════════════════════════════════
# code_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════
