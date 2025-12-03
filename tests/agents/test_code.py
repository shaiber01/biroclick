"""
Tests for Code Agents (CodeGeneratorAgent, CodeReviewerAgent).
"""

import pytest
import json
from unittest.mock import patch, MagicMock, ANY
from src.agents.code import code_generator_node, code_reviewer_node
from schemas.state import MAX_CODE_REVISIONS, MAX_DESIGN_REVISIONS

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Base state for code tests."""
    return {
        "paper_id": "test_paper",
        "current_stage_id": "stage_1_sim",
        "current_stage_type": "SINGLE_STRUCTURE",
        "design_description": "Simulate a gold nanorod with length 100nm and diameter 40nm using FDTD. This description is long enough to pass validation checks (>50 chars).",
        "plan": {
            "stages": [
                {"stage_id": "stage_1_sim", "targets": ["Fig1"]}
            ]
        },
        "validated_materials": [{"material_id": "gold", "path": "/materials/gold.csv"}],
        "code": "import meep as mp\n# Valid simulation code structure\n# ... more lines ...\n# ... more lines ...",
        "code_revision_count": 0,
        "design_revision_count": 0,
        "runtime_config": {
            "max_code_revisions": 3,
            "max_design_revisions": 3
        }
    }

# ═══════════════════════════════════════════════════════════════════════
# code_generator_node Tests
# ═══════════════════════════════════════════════════════════════════════

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

    def test_generator_design_revision_max_cap(self, base_state):
        """Test that design revision count doesn't exceed max."""
        base_state["design_description"] = "TODO: stub"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        
        result = code_generator_node(base_state)
        
        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS

    def test_generator_missing_materials_stage1(self, base_state):
        """Test error when validated_materials is missing for Stage 1+."""
        base_state["validated_materials"] = [] # Empty
        # current_stage_type is SINGLE_STRUCTURE by default in fixture (Stage 1+)
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "validated_materials is empty" in result["run_error"]
        assert result["code_revision_count"] == 1

    def test_generator_skip_materials_validation_stage0(self, base_state):
        """Test that materials validation is skipped for MATERIAL_VALIDATION stage."""
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["validated_materials"] = [] # Empty, but should be allowed
        
        # We need to mock dependencies to get past this check to success path
        with patch("src.agents.code.build_agent_prompt") as mock_prompt,              patch("src.agents.code.check_context_or_escalate") as mock_check,              patch("src.agents.code.call_agent_with_metrics") as mock_llm,              patch("src.agents.code.build_user_content_for_code_generator") as mock_uc:
             
            mock_check.return_value = None
            mock_llm.return_value = {
                "code": "import meep\n# valid code length > 50 chars................................", 
                "expected_outputs": []
            }
            
            result = code_generator_node(base_state)
            
            assert "run_error" not in result
            assert "code" in result

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    @patch("src.agents.code.build_user_content_for_code_generator")
    def test_generator_includes_feedback(self, mock_uc, mock_llm, mock_check, mock_prompt, base_state):
        """Test that reviewer feedback is appended to the prompt."""
        mock_check.return_value = None
        mock_prompt.return_value = "Base Prompt"
        base_state["reviewer_feedback"] = "Fix the geometry."
        mock_llm.return_value = {"code": "code"*20} # Valid length
        
        code_generator_node(base_state)
        
        # Verify feedback in prompt
        call_args = mock_llm.call_args[1]
        assert "REVISION FEEDBACK: Fix the geometry." in call_args["system_prompt"]

    @pytest.mark.parametrize("llm_output, expected_code", [
        ({"code": "valid code"*10}, "valid code"*10),
        ({"simulation_code": "sim code"*10}, "sim code"*10),
        # Fallback to JSON dump if no code key
        ({"something": "else"}, '{\n  "something": "else"\n}') 
    ])
    def test_generator_code_extraction(self, llm_output, expected_code, base_state):
        """Test extraction of code from various LLM output formats."""
        with patch("src.agents.code.build_agent_prompt"),              patch("src.agents.code.check_context_or_escalate", return_value=None),              patch("src.agents.code.call_agent_with_metrics") as mock_llm,              patch("src.agents.code.build_user_content_for_code_generator"):
            
            mock_llm.return_value = llm_output
            # Ensure expected code is long enough to pass validation or we expect failure
            if len(expected_code) < 50:
                 pass

            result = code_generator_node(base_state)
            
            assert result["code"] == expected_code

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
        assert result["code_revision_count"] <= max_revs

# ═══════════════════════════════════════════════════════════════════════
# code_reviewer_node Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCodeReviewerNode:
    """Tests for code_reviewer_node."""

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer approving code."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "approve",
            "issues": []
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "approve"
        assert result["code_revision_count"] == 0
        assert result["reviewer_issues"] == []
        assert "reviewer_feedback" not in result # Should not set feedback on approval

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, base_state):
        """Test reviewer requesting revision."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix boundary conditions",
            "issues": ["Boundary issue"]
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["workflow_phase"] == "code_review"
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1
        assert result["reviewer_feedback"] == "Fix boundary conditions"
        assert result["reviewer_issues"] == ["Boundary issue"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_uses_summary_fallback(self, mock_llm, mock_prompt, base_state):
        """Test reviewer uses summary as feedback if feedback is missing."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "summary": "Summary of issues",
            # No feedback key
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["reviewer_feedback"] == "Summary of issues"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, base_state):
        """Test reviewer hitting max revisions."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision"}
        
        base_state["code_revision_count"] = MAX_CODE_REVISIONS
        
        result = code_reviewer_node(base_state)
        
        # Should not increment past max
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result["last_code_review_verdict"] == "needs_revision"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer defaults to auto-approve on LLM failure (non-critical)."""
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = code_reviewer_node(base_state)
        
        # Auto-approve logic for reviewers
        assert result["last_code_review_verdict"] == "approve" 
        # Should usually log error but continue

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_dict_design(self, mock_llm, mock_prompt, base_state):
        """Test that user content is constructed correctly with dict design."""
        base_state["design_description"] = {"key": "value"}
        base_state["reviewer_feedback"] = "Previous feedback"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        assert "CODE TO REVIEW" in user_content
        assert "DESIGN SPEC" in user_content
        assert '"key": "value"' in user_content # JSON dump of dict
        assert "REVISION FEEDBACK" in user_content
        assert "Previous feedback" in user_content

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_context_construction_str_design(self, mock_llm, mock_prompt, base_state):
        """Test that user content is constructed correctly with string design."""
        base_state["design_description"] = "String design"
        mock_llm.return_value = {"verdict": "approve"}
        
        code_reviewer_node(base_state)
        
        call_args = mock_llm.call_args[1]
        user_content = call_args["user_content"]
        
        assert "DESIGN SPEC" in user_content
        assert "String design" in user_content
