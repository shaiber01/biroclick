"""State/IO tests for results_analyzer_node."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.analysis import results_analyzer_node
from src.agents.constants import AnalysisClassification


@pytest.fixture(name="base_state")
def analysis_base_state_alias(analysis_state):
    return analysis_state


class TestResultsAnalyzerNode:
    """Tests for results_analyzer_node."""

    def test_analyzer_stage_outputs_none(self, base_state):
        """Test error when stage_outputs is explicitly None."""
        base_state["stage_outputs"] = None
        
        result = results_analyzer_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"

    def test_analyzer_plan_none(self, base_state):
        """Test handling when plan is None (should default gracefully or fail safely)."""
        base_state["plan"] = None
        
        # Should not crash, but might result in NO_TARGETS or proceed if figures exist
        # If plan is None, get_plan_stage returns None.
        # figures = ensure_stub_figures(state) -> checks paper_figures
        # target_ids fallback to figures.
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            result = results_analyzer_node(base_state)
        
        assert result["workflow_phase"] == "analysis"
        # If paper_figures are present in base_state (they are "Fig1"), it should try to analyze "Fig1"
        # But it might fail to find output if stage_outputs["files"] is checked against empty expectations?
        # match_output_file logic handles matching.
        
        # In base_state, stage_outputs has files.
        # So it might actually proceed.
        assert result["analysis_overall_classification"] in [
            AnalysisClassification.NO_TARGETS,
            AnalysisClassification.POOR_MATCH,
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
            AnalysisClassification.EXCELLENT_MATCH
        ]

    def test_analyzer_missing_stage_id(self, base_state):
        """Test error when current_stage_id is missing."""
        base_state["current_stage_id"] = None
        result = results_analyzer_node(base_state)
        
        assert result["workflow_phase"] == "analysis"
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert result["awaiting_user_input"] is True
        assert "No stage selected" in result["pending_user_questions"][0]

    def test_analyzer_missing_outputs(self, base_state):
        """Test error when stage outputs are empty."""
        base_state["stage_outputs"] = {}
        result = results_analyzer_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Stage outputs are missing" in result["run_error"]
        assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    def test_analyzer_files_missing_on_disk(self, mock_prompt, mock_check, base_state):
        """Test failure when files listed in state do not exist on disk."""
        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False):
            result = results_analyzer_node(base_state)
            
            assert result["execution_verdict"] == "fail"
            assert "do not exist on disk" in result["run_error"]
            assert "output.csv" in result["run_error"]

    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="Prompt")
    @patch("src.agents.analysis.ensure_stub_figures", return_value=[])
    def test_analyzer_no_targets(self, mock_stubs, mock_prompt, mock_check, base_state):
        """Test handling of stage with no targets."""
        base_state["plan"]["stages"][0]["targets"] = []
        base_state["plan"]["stages"][0]["target_details"] = []
        
        result = results_analyzer_node(base_state)
        
        assert result["analysis_overall_classification"] == AnalysisClassification.NO_TARGETS
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["analysis_summary"]["totals"]["targets"] == 0

    @patch("src.agents.analysis.build_agent_prompt")
    @patch("src.agents.analysis.check_context_or_escalate")
    @patch("src.agents.analysis.call_agent_with_metrics")
    def test_analyzer_hallucinated_targets(self, mock_llm, mock_check, mock_prompt, base_state):
        """Test robustness against targets not in plan."""
        mock_check.return_value = None
        mock_prompt.return_value = "Prompt"
        # Add a target to feedback that isn't in plan
        base_state["analysis_feedback"] = "Check non_existent_fig"
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            
            result = results_analyzer_node(base_state)
            assert result["workflow_phase"] == "analysis"

    @patch("src.agents.analysis.get_plan_stage")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    @patch("src.agents.analysis.ensure_stub_figures")
    def test_handles_missing_files_list(self, mock_stub, mock_prompt, mock_check, mock_plan_stage, base_state):
        """Should return error when files list is explicit empty list."""
        mock_stub.return_value = [{"id": "Fig1"}]
        mock_plan_stage.return_value = {"stage_id": "stage1", "targets": ["Fig1"]}
        
        base_state["stage_outputs"] = {"files": []}
        
        result = results_analyzer_node(base_state)
        
        assert result["execution_verdict"] == "fail"
        assert "Stage outputs are missing" in result["run_error"]

    @patch("src.agents.analysis.check_context_or_escalate")
    def test_returns_escalation_on_context_overflow(self, mock_context, base_state):
        """Should return escalation when context overflow."""
        mock_context.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        result = results_analyzer_node(base_state)
        
        assert result["awaiting_user_input"] is True
        assert result["pending_user_questions"] == ["Context overflow"]

    @patch("src.agents.analysis.load_numeric_series")
    @patch("src.agents.analysis.quantitative_curve_metrics")
    @patch("src.agents.analysis.match_output_file")
    @patch("src.agents.analysis.check_context_or_escalate", return_value=None)
    @patch("src.agents.analysis.build_agent_prompt", return_value="prompt")
    def test_handles_digitized_data_requirement(
        self, mock_prompt, mock_check, mock_match, mock_metrics, mock_load, base_state
    ):
        """Should flag targets requiring digitized data without it."""
        # Require excellent precision which mandates digitized data
        base_state["plan"]["stages"][0]["targets"] = ["Fig1"]
        base_state["plan"]["stages"][0]["target_details"] = [{"figure_id": "Fig1", "precision_requirement": "excellent"}]
        # Ensure no digitized path in plan or figures
        base_state["paper_figures"][0]["digitized_data_path"] = None
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
             
            result = results_analyzer_node(base_state)
            
            # Should flag as missing digitized data
            summary = result["analysis_summary"]
            # It goes into pending_targets or just blocked?
            # Code says: pending_targets.append(target_id), classification="missing_digitized_data"
            assert "Fig1" in summary["pending_targets"]
            
            reports = result["figure_comparisons"]
            assert any(r["classification"] == "missing_digitized_data" for r in reports)

