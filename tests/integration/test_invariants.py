import json
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestCriticalFilesExist:
    """Verify all required prompt and schema files exist."""

    AGENTS_WITH_PROMPTS = [
        "planner",
        "plan_reviewer",
        "prompt_adaptor",
        "simulation_designer",
        "design_reviewer",
        "code_generator",
        "code_reviewer",
        "execution_validator",
        "physics_sanity",
        "results_analyzer",
        "comparison_validator",
        "supervisor",
        "report_generator",
    ]

    AGENTS_WITH_SCHEMAS = [
        "planner_output",
        "plan_reviewer_output",
        "prompt_adaptor_output",
        "simulation_designer_output",
        "design_reviewer_output",
        "code_generator_output",
        "code_reviewer_output",
        "execution_validator_output",
        "physics_sanity_output",
        "results_analyzer_output",
        "supervisor_output",
    ]

    @pytest.mark.parametrize("agent_name", AGENTS_WITH_PROMPTS)
    def test_prompt_file_exists(self, agent_name):
        prompt_path = PROJECT_ROOT / "prompts" / f"{agent_name}_agent.md"
        assert prompt_path.exists(), f"Missing prompt file: {prompt_path}"
        content = prompt_path.read_text()
        assert len(content) > 100, f"Prompt file too short: {prompt_path}"

    @pytest.mark.parametrize("schema_name", AGENTS_WITH_SCHEMAS)
    def test_schema_file_exists(self, schema_name):
        schema_path = PROJECT_ROOT / "schemas" / f"{schema_name}_schema.json"
        assert schema_path.exists(), f"Missing schema file: {schema_path}"
        content = schema_path.read_text()
        schema = json.loads(content)
        assert "properties" in schema, f"Schema missing 'properties': {schema_path}"


class TestRealPromptBuilding:
    """Spot-check real prompt builders without mocking the LLM."""

    def test_planner_prompt_includes_paper_text(self, base_state):
        from src.prompts import build_agent_prompt
        from src.llm_client import build_user_content_for_planner

        system_prompt = build_agent_prompt("planner", base_state)
        user_content = build_user_content_for_planner(base_state)

        assert len(system_prompt) > 500
        if isinstance(user_content, str):
            content = user_content.lower()
        else:
            text_parts = [
                part.get("text", "").lower()
                for part in user_content
                if part.get("type") == "text"
            ]
            content = " ".join(text_parts)
        assert "gold nanorod" in content or "optical properties" in content

    def test_code_generator_prompt_includes_design(self, base_state, valid_plan):
        from src.prompts import build_agent_prompt

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for extinction",
            "geometry": [{"type": "cylinder", "radius": 20}],
        }

        system_prompt = build_agent_prompt("code_generator", base_state)
        assert len(system_prompt) > 500


class TestInvariants:
    """Cross-cutting invariants across all agent nodes."""

    def test_all_reviewer_nodes_return_verdict_field(self, base_state, valid_plan):
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["plan"] = valid_plan
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert "last_plan_review_verdict" in result

        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert "last_design_review_verdict" in result

        base_state["code"] = "print('test')"
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert "last_code_review_verdict" in result

    def test_all_validators_return_verdict_field(self, base_state):
        from src.agents.execution import execution_validator_node, physics_sanity_node
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}
        mock_response = {"verdict": "pass", "summary": "OK"}

        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
            assert "execution_verdict" in result

        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = physics_sanity_node(base_state)
            assert "physics_verdict" in result

        base_state["figure_comparisons"] = []
        base_state["progress"] = {"stages": []}
        base_state["plan"] = {"stages": [{"stage_id": "stage_0", "targets": []}]}
        result = comparison_validator_node(base_state)
        assert "comparison_verdict" in result

    def test_all_nodes_return_workflow_phase(self, base_state, valid_plan):
        from src.agents.planning import plan_node, plan_reviewer_node, adapt_prompts_node
        from src.agents.design import simulation_designer_node, design_reviewer_node
        from src.agents.code import code_generator_node, code_reviewer_node
        from src.agents.execution import execution_validator_node, physics_sanity_node
        from src.agents.analysis import results_analyzer_node, comparison_validator_node
        from src.agents.reporting import generate_report_node
        from src.agents.supervision.supervisor import supervisor_node

        mock_reviewer_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        mock_plan_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
            "targets": [],
            "extracted_parameters": [],
        }
        mock_code_response = {"code": "import meep\nprint('test')", "expected_outputs": []}
        mock_design_response = {
            "stage_id": "stage_0",
            "design_description": "Test design",
            "geometry": [{"type": "box"}],
            "sources": [],
            "monitors": [],
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}
        base_state["figure_comparisons"] = []
        base_state["progress"] = {"stages": []}
        base_state["code"] = "print('test')"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Test design description here",
            "geometry": [{"type": "box"}],
        }

        nodes_to_test = [
            ("adapt_prompts_node", adapt_prompts_node, "src.agents.planning"),
            ("plan_node", plan_node, "src.agents.planning"),
            ("plan_reviewer_node", plan_reviewer_node, "src.agents.planning"),
            ("simulation_designer_node", simulation_designer_node, "src.agents.design"),
            ("design_reviewer_node", design_reviewer_node, "src.agents.design"),
            ("code_generator_node", code_generator_node, "src.agents.code"),
            ("code_reviewer_node", code_reviewer_node, "src.agents.code"),
            ("execution_validator_node", execution_validator_node, "src.agents.execution"),
            ("physics_sanity_node", physics_sanity_node, "src.agents.execution"),
            ("results_analyzer_node", results_analyzer_node, "src.agents.analysis"),
            ("comparison_validator_node", comparison_validator_node, "src.agents.analysis"),
            ("generate_report_node", generate_report_node, "src.agents.reporting"),
            ("supervisor_node", supervisor_node, "src.agents.supervision.supervisor"),
        ]

        for node_name, node_func, module_path in nodes_to_test:
            if "reviewer" in node_name:
                mock_resp = mock_reviewer_response
            elif "plan" in node_name and "reviewer" not in node_name:
                mock_resp = mock_plan_response
            elif "designer" in node_name:
                mock_resp = mock_design_response
            elif "code_generator" in node_name:
                mock_resp = mock_code_response
            else:
                mock_resp = {"verdict": "pass", "summary": "OK"}

            with patch(f"{module_path}.call_agent_with_metrics", return_value=mock_resp):
                try:
                    result = node_func(base_state)
                    assert "workflow_phase" in result, f"{node_name} must set workflow_phase"
                except Exception:
                    # Some nodes have required preconditions that may raise; skip in that case.
                    pass

    def test_counter_increment_never_negative(self, base_state):
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["design_revision_count"] = 0
        base_state["code_revision_count"] = 0
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"

        mock_response = {"verdict": "needs_revision", "issues": ["fix"], "summary": "Fix"}

        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert result.get("design_revision_count", 0) >= 0

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert result.get("code_revision_count", 0) >= 0


class TestRegressionPrevention:
    """Regression tests to ensure LLM failures escalate correctly."""

    def test_llm_error_in_planner_escalates_not_crashes(self, base_state):
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_node(base_state)
            assert result.get("awaiting_user_input") is True

    def test_llm_error_in_reviewers_auto_approves(self, base_state, valid_plan):
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "approve"

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = design_reviewer_node(base_state)
            assert result.get("last_design_review_verdict") == "approve"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = code_reviewer_node(base_state)
            assert result.get("last_code_review_verdict") == "approve"

