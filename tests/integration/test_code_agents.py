from unittest.mock import patch

import pytest


class TestCodeRevisionCounters:
    """Verify code reviewer counters respect bounds and increments."""

    def test_code_revision_counter_bounded(self, base_state):
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": ["bug"],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        max_revisions = 5
        base_state["code_revision_count"] = max_revisions
        base_state["runtime_config"] = {"max_code_revisions": max_revisions}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert "code_revision_count" in result
        assert result["code_revision_count"] == max_revisions

    def test_code_revision_counter_increments_under_max(self, base_state):
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": ["bug"],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 2

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["code_revision_count"] == 3


class TestCodeReviewerCounterIncrements:
    """Verify code reviewer increments revision counters on rejection."""

    def test_code_reviewer_increments_counter_on_rejection(self, base_state):
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": ["bug"],
            "summary": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 0

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result.get("code_revision_count") == 1


class TestCodeReviewerFeedback:
    """Verify code reviewer feedback fields are preserved."""

    def test_code_reviewer_populates_feedback_on_rejection(self, base_state):
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "critical", "description": "Missing import statement"},
                {"severity": "major", "description": "Incorrect parameter value"},
            ],
            "summary": "Code has critical issues",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('incomplete code')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert "reviewer_feedback" in result
        feedback = result["reviewer_feedback"]
        assert len(feedback) > 20


class TestCodeReviewerOutputFields:
    """Verify code reviewer sets required fields on approval."""

    def test_code_reviewer_sets_all_fields_on_approve(self, base_state):
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('good code')"
        base_state["code_revision_count"] = 3

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result.get("last_code_review_verdict") == "approve"
        assert "workflow_phase" in result
        assert "code_revision_count" in result


class TestCodeGeneratorBehavior:
    """Verify code generator passes through all required data."""

    def test_code_generator_node_creates_code(self, base_state, valid_plan):
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nimport numpy as np\nprint('Simulation started')",
            "expected_outputs": ["output.csv", "spectrum.png"],
            "explanation": "Simple FDTD test simulation",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod extinction",
            "geometry": [{"type": "cylinder", "radius": 20, "height": 100}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 900]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
        }
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert "code" in result and len(result["code"]) > 10
        assert result["workflow_phase"] == "code_generation"
        assert result["expected_outputs"] == ["output.csv", "spectrum.png"]

    def test_code_generator_requires_validated_materials_for_stage1(
        self, base_state, valid_plan
    ):
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_1"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = {
            "stage_id": "stage_1",
            "design_description": "FDTD simulation",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = []

        result = code_generator_node(base_state)
        assert "code" not in result or result.get("run_error")


class TestCodeGeneratorExpectedOutputs:
    """Verify code_generator expected outputs handling."""

    def test_expected_outputs_passed_through(self, base_state, valid_plan):
        from src.agents.code import code_generator_node

        expected = ["spectrum.csv", "field_map.png", "resonance_data.json"]
        mock_response = {
            "code": "import meep as mp\nimport numpy as np\n\nprint('Running simulation')",
            "expected_outputs": expected,
            "explanation": "Test simulation",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod",
            "geometry": [{"type": "cylinder", "radius": 20}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result["expected_outputs"] == expected

    def test_empty_expected_outputs_defaults_to_empty_list(self, base_state, valid_plan):
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nimport numpy as np\nprint('Simulation')",
            "explanation": "Test",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Test design description here",
            "geometry": [{"type": "box"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result["expected_outputs"] == []


class TestCodeGeneratorContent:
    """Verify prompts and schema selection for code generator."""

    def test_code_generator_uses_correct_agent_name(self, base_state, valid_plan):
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nprint('test')",
            "expected_outputs": ["output.csv"],
            "explanation": "Test",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod",
            "geometry": [{"type": "cylinder", "radius": 20}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_generator_node(base_state)

        assert mock.called
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "code_generator"
        system_prompt = call_kwargs.get("system_prompt", "")
        assert len(system_prompt) > 100, "System prompt too short for code_generator"


class TestCodeEdgeCases:
    """Code-generator-specific edge cases."""

    def test_code_generator_with_stub_design(self, base_state, valid_plan):
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "TODO: Add design",
            "geometry": [],
        }

        result = code_generator_node(base_state)
        assert "code" not in result or result.get("run_error")


class TestCodeLLMContent:
    """Verify code reviewer receives the required design context."""

    def test_code_reviewer_receives_design_spec(self, base_state):
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('test')"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD for gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_reviewer_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        assert "DESIGN" in user_content.upper() or "design" in user_content.lower()

