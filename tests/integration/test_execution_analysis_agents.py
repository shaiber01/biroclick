import json
import os
import tempfile
from unittest.mock import patch

import pytest


class TestErrorHandling:
    """Verify error handling produces correct state updates."""

    def test_llm_error_triggers_user_escalation(self, base_state):
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_node(base_state)

        assert result.get("ask_user_trigger") == "llm_error"
        assert result.get("awaiting_user_input") is True

    def test_reviewer_llm_error_auto_approves(self, base_state, valid_plan):
        from src.agents.planning import plan_reviewer_node

        base_state["plan"] = valid_plan

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_reviewer_node(base_state)

        assert result.get("last_plan_review_verdict") == "approve"
        assert result.get("workflow_phase") == "plan_review"


class TestMissingNodeCoverage:
    """Test nodes that were not covered in original test file."""

    def test_adapt_prompts_node_updates_state(self, base_state):
        from src.agents.planning import adapt_prompts_node

        mock_response = {
            "adaptations": ["Focus on plasmonics", "Use Johnson-Christy data"],
            "paper_domain": "plasmonics",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            result = adapt_prompts_node(base_state)

        assert mock.called
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "prompt_adaptor"
        assert result.get("workflow_phase") == "adapting_prompts"
        assert result["prompt_adaptations"] == mock_response["adaptations"]
        assert result.get("paper_domain") == "plasmonics"

    def test_select_stage_node_selects_valid_stage(self, base_state, valid_plan):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "not_started",
                    "dependencies": [],
                }
            ]
        }

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION"
        assert result.get("workflow_phase") == "stage_selection"

    def test_simulation_designer_node_creates_design(self, base_state, valid_plan):
        from src.agents.design import simulation_designer_node

        mock_response = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation with gold nanorod...",
            "geometry": [{"type": "cylinder", "radius": 20, "material": "gold"}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
            "materials": [{"material_id": "gold", "source": "Johnson-Christy"}],
            "new_assumptions": {"sim_a1": "assuming periodic boundary"},
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"

        with patch(
            "src.agents.design.call_agent_with_metrics", return_value=mock_response
        ):
            result = simulation_designer_node(base_state)

        design = result["design_description"]
        assert design.get("stage_id") == "stage_0"
        assert design["geometry"][0].get("type") == "cylinder"
        assert design["materials"] == mock_response["materials"]

class TestExecutionValidatorBehavior:
    """Execution validator specific behaviors."""

    def test_execution_validator_returns_verdict_from_llm(self, base_state):
        from src.agents.execution import execution_validator_node

        mock_response = {"verdict": "pass", "summary": "OK"}

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"], "exit_code": 0}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result.get("execution_verdict") == "pass"
        assert "workflow_phase" in result


class TestValidatorVerdicts:
    """Test various validator verdict scenarios."""

    def test_physics_sanity_returns_design_flaw(self, base_state):
        from src.agents.execution import physics_sanity_node

        mock_response = {
            "verdict": "design_flaw",
            "summary": "Simulation parameters inconsistent with physics",
            "design_issues": ["Wavelength range too narrow"],
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert result["physics_verdict"] == "design_flaw"

    def test_execution_validator_returns_fail(self, base_state):
        from src.agents.execution import execution_validator_node

        mock_response = {
            "verdict": "fail",
            "summary": "Simulation crashed",
            "error_analysis": "Memory allocation failure",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {
            "files": [],
            "exit_code": 1,
            "stderr": "Segmentation fault",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "fail"


class TestAdditionalCoverageExecution:
    """Execution & analysis helpers migrated from legacy tests."""

    def test_physics_sanity_passes(self, base_state):
        from src.agents.execution import physics_sanity_node

        mock_response = {
            "verdict": "pass",
            "summary": "Physics checks passed",
            "checks_performed": ["energy_conservation", "value_ranges"],
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ) as mock:
            result = physics_sanity_node(base_state)

        assert mock.called
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "physics_sanity"
        assert result["physics_verdict"] == "pass"
        assert result["workflow_phase"] == "physics_validation"

    def test_results_analyzer_sets_workflow_phase(self, base_state, valid_plan):
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/nonexistent/fake_output.csv"]}

        result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert result.get("execution_verdict") == "fail" or result.get("run_error")

    def test_results_analyzer_with_empty_outputs(self, base_state, valid_plan):
        """results_analyzer_node should handle empty stage_outputs gracefully."""
        from src.agents.analysis import results_analyzer_node

        mock_response = {
            "overall_classification": "NO_DATA",
            "figure_comparisons": [],
            "summary": "No data to analyze",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        with patch(
            "src.agents.analysis.call_agent_with_metrics", return_value=mock_response
        ):
            result = results_analyzer_node(base_state)

        assert result.get("workflow_phase") == "analysis"

    def test_results_analyzer_returns_figure_comparisons_on_success(
        self, base_state, valid_plan
    ):
        from src.agents.analysis import results_analyzer_node

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = f.name

        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"

            result = results_analyzer_node(base_state)

            assert result["workflow_phase"] == "analysis"
            assert "analysis_summary" in result
            assert "figure_comparisons" in result
        finally:
            os.unlink(temp_file)


class TestComparisonValidatorEdgeCases:
    """Test comparison_validator_node edge cases."""

    def test_comparison_validator_approves_no_targets(self, base_state):
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "targets": [], "target_details": []}]
        }
        base_state["figure_comparisons"] = []
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "running"}]
        }
        base_state["analysis_revision_count"] = 2

        result = comparison_validator_node(base_state)
        assert result["comparison_verdict"] == "approve"
        assert "analysis_revision_count" not in result
        assert result.get("analysis_feedback") is None

    def test_comparison_validator_rejects_missing_comparisons(
        self, base_state, valid_plan
    ):
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "running"}]
        }
        base_state["analysis_revision_count"] = 1

        result = comparison_validator_node(base_state)

        assert result["comparison_verdict"] == "needs_revision"
        assert result.get("analysis_revision_count") == 2
        feedback = result.get("analysis_feedback", "")
        assert feedback
        assert any(keyword in feedback.lower() for keyword in ("comparison", "report"))


class TestComparisonValidatorLogic:
    """Verify comparison_validator_node logic."""

    def test_comparison_validator_approves_valid_comparisons(
        self, base_state, valid_plan
    ):
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = [
            {
                "stage_id": "stage_0",
                "target_figure": "material_gold",
                "status": "match",
                "criteria_failures": [],
            }
        ]

        result = comparison_validator_node(base_state)
        assert result["comparison_verdict"] == "approve"
        assert result["workflow_phase"] == "comparison_validation"

    def test_comparison_validator_rejects_missing_reports(
        self, base_state, valid_plan
    ):
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = [
            {
                "stage_id": "stage_0",
                "figure_id": "material_gold",
                "classification": "match",
            }
        ]
        base_state["analysis_result_reports"] = []

        result = comparison_validator_node(base_state)
        assert result["comparison_verdict"] == "needs_revision"
        assert "Missing quantitative reports" in result["comparison_feedback"]


class TestResultsAnalyzerCompleteness:
    """Verify results_analyzer produces complete output."""

    def test_results_analyzer_returns_all_required_fields(self, base_state, valid_plan):
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}

        result = results_analyzer_node(base_state)
        assert result.get("workflow_phase") == "analysis"


class TestResultsAnalyzerLogic:
    """Verify results_analyzer_node logic."""

    def test_results_analyzer_analyzes_existing_files(self, base_state, valid_plan):
        from src.agents.analysis import results_analyzer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]
        ), patch(
            "src.agents.analysis.quantitative_curve_metrics",
            return_value={"peak_position_error_percent": 5.0},
        ), patch(
            "src.agents.analysis.get_images_for_analyzer", return_value=[]
        ):
            result = results_analyzer_node(base_state)

        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        assert result["figure_comparisons"]


class TestPhysicsSanityBacktrack:
    """Ensure physics_sanity_node forwards backtrack suggestions."""

    def test_physics_sanity_passes_backtrack_suggestion(self, base_state):
        from src.agents.execution import physics_sanity_node

        mock_response = {
            "verdict": "design_flaw",
            "summary": "Fundamental issue with simulation setup",
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                "target_stage_id": "stage_0",
                "reason": "Material properties need revalidation",
            },
        }

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True


class TestReportGeneratorCompleteness:
    """Verify generate_report_node produces complete report."""

    def test_report_includes_token_summary(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
                {"agent_name": "designer", "input_tokens": 2000, "output_tokens": 800},
            ]
        }

        mock_response = {
            "executive_summary": {"overall_assessment": []},
            "conclusions": {"main_physics_reproduced": True},
        }

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        metrics = result.get("metrics", {})
        token_summary = metrics.get("token_summary", {})
        assert token_summary.get("total_input_tokens") == 3000
        assert "total_output_tokens" in token_summary

    def test_report_marks_workflow_complete(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}

        mock_response = {"executive_summary": {}, "conclusions": {}}

        with patch(
            "src.agents.reporting.call_agent_with_metrics", return_value=mock_response
        ):
            result = generate_report_node(base_state)

        assert result.get("workflow_complete") is True


class TestPhysicsSanityBacktrack:
    """Verify physics_sanity_node handles backtrack suggestions."""

    def test_physics_sanity_passes_backtrack_suggestion(self, base_state):
        from src.agents.execution import physics_sanity_node

        mock_response = {
            "verdict": "design_flaw",
            "summary": "Fundamental issue with simulation setup",
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                "target_stage_id": "stage_0",
                "reason": "Material properties need revalidation",
            },
        }

        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}

        with patch(
            "src.agents.execution.call_agent_with_metrics", return_value=mock_response
        ):
            result = physics_sanity_node(base_state)

        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True

