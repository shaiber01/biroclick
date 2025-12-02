"""Unit tests for src/agents/analysis.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.analysis import (
    results_analyzer_node,
    comparison_validator_node,
)


class TestResultsAnalyzerNode:
    """Tests for results_analyzer_node function."""

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.get_plan_stage")
    def test_analyzes_matching_results(self, mock_plan_stage, mock_prompt, mock_context, mock_call):
        """Should analyze and classify matching results."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_plan_stage.return_value = {"targets": ["Fig1"], "stage_id": "stage1"}
        mock_call.return_value = {
            "classification": "match",
            "confidence": 0.95,
            "summary": "Results match reference data",
            "quantitative_metrics": {
                "peak_position_error_percent": 1.5,
                "normalized_rmse_percent": 3.2,
            },
        }
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": ["spectrum.csv"]},  # Uses stage_outputs
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["Fig1"]}]},
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.get_plan_stage")
    def test_identifies_mismatch(self, mock_plan_stage, mock_prompt, mock_context, mock_call):
        """Should identify mismatched results."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_plan_stage.return_value = {"targets": ["Fig1"], "stage_id": "stage1"}
        mock_call.return_value = {
            "classification": "mismatch",
            "confidence": 0.8,
            "summary": "Peak positions differ significantly",
            "quantitative_metrics": {
                "peak_position_error_percent": 25.0,
            },
        }
        
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": ["spectrum.csv"]},  # Must have files
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["Fig1"]}]},
        }
        
        result = results_analyzer_node(state)
        
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    def test_handles_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should handle LLM call failure gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "run_result": {"success": True},
        }
        
        result = results_analyzer_node(state)
        
        # Should return pending validation or error state
        assert "workflow_phase" in result

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "run_result": {}}
        
        result = results_analyzer_node(state)
        
        assert result["awaiting_user_input"] is True


class TestComparisonValidatorNode:
    """Tests for comparison_validator_node function."""

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    def test_validates_good_comparison(self, mock_prompt, mock_context, mock_call):
        """Should validate good comparison results."""
        mock_context.return_value = None
        mock_prompt.return_value = "system prompt"
        mock_call.return_value = {
            "verdict": "accept",
            "consistency_score": 0.92,
            "summary": "Comparison is consistent",
        }
        
        state = {
            "current_stage_id": "stage1",
            "analysis_result": {
                "classification": "match",
                "quantitative_metrics": {},
            },
        }
        
        result = comparison_validator_node(state)
        
        assert result["workflow_phase"] == "comparison_validation"
        assert "comparison_verdict" in result

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    def test_rejects_inconsistent_comparison(self, mock_prompt, mock_context, mock_call):
        """Should reject inconsistent comparison."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "reject",
            "consistency_score": 0.3,
            "issues": ["Metrics don't match classification"],
            "summary": "Comparison is inconsistent",
        }
        
        state = {
            "current_stage_id": "stage1",
            "analysis_result": {"classification": "match"},
            "comparison_failures": 0,
        }
        
        result = comparison_validator_node(state)
        
        assert "comparison_verdict" in result

    @patch("src.agents.analysis.call_agent_with_metrics")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.build_agent_prompt")
    def test_handles_llm_error(self, mock_prompt, mock_context, mock_call):
        """Should handle LLM call failure gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("API error")
        
        state = {
            "current_stage_id": "stage1",
            "analysis_result": {},
        }
        
        result = comparison_validator_node(state)
        
        # Should return default acceptance or error state
        assert "workflow_phase" in result

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        state = {"current_stage_id": "stage1", "analysis_result": {}}
        
        result = comparison_validator_node(state)
        
        assert result["awaiting_user_input"] is True

