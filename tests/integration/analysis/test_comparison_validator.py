"""Integration tests for comparison_validator_node."""


class TestComparisonValidatorEdgeCases:
    """Edge cases covering missing targets and comparisons."""

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

