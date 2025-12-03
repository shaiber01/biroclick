from unittest.mock import patch

import pytest


class TestHandleBacktrackNode:
    """Integration checks for handle_backtrack_node."""

    def test_backtrack_marks_target_as_needs_rerun(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                }
            ]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
            "reason": "Need to revalidate materials",
        }

        result = handle_backtrack_node(base_state)
        progress = result.get("progress", {})
        stages = progress.get("stages", [])
        target_stage = next((s for s in stages if s["stage_id"] == "stage_0"), None)

        assert target_stage is not None
        assert target_stage["status"] == "needs_rerun"

    def test_backtrack_invalidates_dependent_stages(self, base_state):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION"},
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "dependencies": ["stage_0"],
                },
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "status": "completed_success",
                },
            ]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": ["stage_1"],
            "reason": "Material validation needs rerun",
        }

        result = handle_backtrack_node(base_state)
        stages = result.get("progress", {}).get("stages", [])
        stage_1 = next((s for s in stages if s["stage_id"] == "stage_1"), None)
        assert stage_1 is not None
        assert stage_1["status"] == "invalidated"

    def test_backtrack_increments_counter(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                }
            ]
        }
        base_state["backtrack_count"] = 0
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result["backtrack_count"] == 1

    def test_backtrack_clears_working_state(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        base_state["code"] = "print('old code')"
        base_state["design_description"] = {"old": "design"}
        base_state["stage_outputs"] = {"files": ["/old/file.csv"]}

        result = handle_backtrack_node(base_state)
        assert result.get("code") is None
        assert result.get("design_description") is None
        assert result.get("stage_outputs") == {}

    def test_backtrack_rejects_missing_decision(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["backtrack_decision"] = None

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") is not None
        assert result.get("awaiting_user_input") is True

    def test_backtrack_respects_max_limit(self, base_state, valid_plan):
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = 2
        base_state["runtime_config"] = {"max_backtracks": 2}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") == "backtrack_limit"


class TestReportGeneratorCompleteness:
    """Verify generate_report_node produces complete reports."""

    def test_generate_report_node_creates_report(self, base_state, valid_plan):
        from src.agents.reporting import generate_report_node

        mock_response = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
            "conclusions": {"main_physics_reproduced": True, "key_findings": ["Test finding"]},
            "paper_citation": {"title": "Test Paper", "authors": "Test Author"},
        }

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
            ]
        }

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)

        assert result.get("workflow_complete") is True
        assert result["workflow_phase"] == "reporting"
        assert "metrics" in result and "token_summary" in result["metrics"]

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

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)

        metrics = result.get("metrics", {})
        token_summary = metrics.get("token_summary", {})
        assert token_summary.get("total_input_tokens") == 3000
        assert token_summary.get("total_output_tokens") == 1300

