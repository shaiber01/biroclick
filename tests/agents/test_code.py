"""
Tests for Code Agents (CodeGeneratorAgent, CodeReviewerAgent).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.code import code_generator_node, code_reviewer_node

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
        "code": "import meep as mp...",
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
    def test_generator_success(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test successful code generation."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "code": "import meep as mp\nprint('success')\n# This code must be longer than 50 characters to pass validation checks in code.py",
            "expected_outputs": ["output.csv"]
        }
        
        result = code_generator_node(base_state)
        
        assert result["workflow_phase"] == "code_generation"
        assert "import meep" in result["code"]
        assert "output.csv" in result["expected_outputs"]

    def test_generator_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = code_generator_node(base_state)
        
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True

    def test_generator_stub_design(self, base_state):
        """Test error when design is a stub."""
        base_state["design_description"] = "TODO: Add design"
        result = code_generator_node(base_state)
        
        assert "design_revision_count" in result
        assert "Design description is missing or contains stub" in result["reviewer_feedback"]

    def test_generator_missing_materials_stage1(self, base_state):
        """Test error when validated_materials is missing for Stage 1+."""
        base_state["validated_materials"] = [] # Empty
        result = code_generator_node(base_state)
        
        assert "validated_materials is empty" in result["run_error"]
        assert result["code_revision_count"] == 1

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_generator_stub_code_output(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test error when LLM returns stub code."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"code": "# TODO: Implement simulation"}
        
        result = code_generator_node(base_state)
        
        # Should detect stub and request revision
        assert result["code_revision_count"] == 1
        assert "Generated code is empty or contains stub" in result["reviewer_feedback"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.check_context_or_escalate")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_generator_handles_llm_failure(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test handling of LLM exception."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = code_generator_node(base_state)
        
        # Should escalate to user
        assert result["ask_user_trigger"] == "llm_error"
        assert result["awaiting_user_input"] is True


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
        mock_llm.return_value = {"verdict": "approve"}
        
        result = code_reviewer_node(base_state)
        
        assert result["last_code_review_verdict"] == "approve"
        assert result["code_revision_count"] == 0

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_needs_revision(self, mock_llm, mock_prompt, base_state):
        """Test reviewer requesting revision."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {
            "verdict": "needs_revision",
            "feedback": "Fix boundary conditions"
        }
        
        result = code_reviewer_node(base_state)
        
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1
        assert "Fix boundary conditions" in result["reviewer_feedback"]

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_max_revisions(self, mock_llm, mock_prompt, base_state):
        """Test reviewer hitting max revisions."""
        mock_prompt.return_value = "Prompt"
        mock_llm.return_value = {"verdict": "needs_revision"}
        
        base_state["code_revision_count"] = 3 # Already at max
        
        result = code_reviewer_node(base_state)
        
        # Should not increment past max
        assert result["code_revision_count"] == 3
        assert result["last_code_review_verdict"] == "needs_revision"

    @patch("src.agents.code.build_agent_prompt")
    @patch("src.agents.code.call_agent_with_metrics")
    def test_reviewer_llm_failure_auto_approve(self, mock_llm, mock_prompt, base_state):
        """Test reviewer defaults to auto-approve on LLM failure (non-critical)."""
        mock_prompt.return_value = "Prompt"
        mock_llm.side_effect = Exception("API Error")
        
        result = code_reviewer_node(base_state)
        
        # Auto-approve logic for reviewers
        assert result["last_code_review_verdict"] == "approve" # Or whatever default is configured
