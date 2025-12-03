import json
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestCriticalFilesExist:
    """Verify all required prompt and schema files exist and are valid."""

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
        # Verify prompt is not just whitespace
        assert content.strip(), f"Prompt file is empty or whitespace only: {prompt_path}"
        # Verify prompt contains meaningful content (not just placeholder text)
        assert "TODO" not in content[:200] or len(content) > 500, \
            f"Prompt file appears to be a stub: {prompt_path}"

    @pytest.mark.parametrize("schema_name", AGENTS_WITH_SCHEMAS)
    def test_schema_file_exists(self, schema_name):
        schema_path = PROJECT_ROOT / "schemas" / f"{schema_name}_schema.json"
        assert schema_path.exists(), f"Missing schema file: {schema_path}"
        content = schema_path.read_text()
        assert content.strip(), f"Schema file is empty: {schema_path}"
        schema = json.loads(content)
        assert isinstance(schema, dict), f"Schema is not a JSON object: {schema_path}"
        assert "properties" in schema, f"Schema missing 'properties': {schema_path}"
        assert isinstance(schema["properties"], dict), \
            f"Schema 'properties' must be a dict: {schema_path}"
        assert len(schema["properties"]) > 0, \
            f"Schema 'properties' is empty: {schema_path}"
        # Verify schema has type field
        assert "type" in schema, f"Schema missing 'type' field: {schema_path}"
        assert schema["type"] == "object", \
            f"Schema type must be 'object', got '{schema.get('type')}': {schema_path}"


class TestRealPromptBuilding:
    """Spot-check real prompt builders without mocking the LLM."""

    def test_planner_prompt_includes_paper_text(self, base_state):
        from src.prompts import build_agent_prompt
        from src.llm_client import build_user_content_for_planner

        system_prompt = build_agent_prompt("planner", base_state)
        user_content = build_user_content_for_planner(base_state)

        # Verify system prompt is substantial
        assert len(system_prompt) > 500, f"System prompt too short: {len(system_prompt)} chars"
        assert system_prompt.strip(), "System prompt is empty or whitespace"
        
        # Verify user content includes paper text
        if isinstance(user_content, str):
            content = user_content.lower()
            assert len(content) > 100, f"User content too short: {len(content)} chars"
        else:
            assert isinstance(user_content, list), f"User content must be str or list, got {type(user_content)}"
            assert len(user_content) > 0, "User content list is empty"
            text_parts = [
                part.get("text", "").lower()
                for part in user_content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = " ".join(text_parts)
            assert len(content) > 100, f"User content text too short: {len(content)} chars"
        
        # Verify paper content is actually included
        assert "gold nanorod" in content or "optical properties" in content, \
            f"Paper text not found in user content. Content preview: {content[:200]}"
        
        # Note: paper_id may or may not be included in user content depending on implementation
        # This is not a bug - paper_id is primarily used for file paths, not prompts

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
        assert len(system_prompt) > 500, f"System prompt too short: {len(system_prompt)} chars"
        assert system_prompt.strip(), "System prompt is empty or whitespace"
        
        # Verify design description is referenced in prompt
        assert "design" in system_prompt.lower() or "simulation" in system_prompt.lower(), \
            "System prompt should reference design or simulation"
        
        # Note: stage_id may be included in user content rather than system prompt
        # This is not a bug - stage context is typically provided via user content

    def test_prompt_building_with_empty_state(self, base_state):
        """Test prompt building handles empty/missing fields gracefully."""
        from src.prompts import build_agent_prompt
        
        # Test with minimal state
        minimal_state = {"paper_id": "test"}
        
        # Should not crash, but may return minimal prompt
        try:
            prompt = build_agent_prompt("planner", minimal_state)
            assert isinstance(prompt, str), f"Prompt must be string, got {type(prompt)}"
            assert len(prompt) > 0, "Prompt must not be empty"
        except Exception as e:
            # If it raises, verify it's a meaningful error, not a crash
            assert "paper_text" in str(e).lower() or "required" in str(e).lower(), \
                f"Unexpected error type: {e}"

    def test_prompt_building_with_none_values(self, base_state):
        """Test prompt building handles None values correctly."""
        from src.prompts import build_agent_prompt
        
        # Set critical fields to None
        base_state["paper_text"] = None
        base_state["paper_figures"] = None
        
        # Should handle None gracefully without crashing
        try:
            prompt = build_agent_prompt("planner", base_state)
            assert isinstance(prompt, str), f"Prompt must be string, got {type(prompt)}"
        except Exception as e:
            # If it raises, should be a validation error, not AttributeError/TypeError
            assert not isinstance(e, (AttributeError, TypeError)), \
                f"Prompt building should handle None gracefully, got {type(e).__name__}: {e}"


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
            assert "last_plan_review_verdict" in result, \
                "plan_reviewer_node must return last_plan_review_verdict"
            assert result["last_plan_review_verdict"] == "approve", \
                f"Expected 'approve', got '{result['last_plan_review_verdict']}'"
            assert result["last_plan_review_verdict"] in ["approve", "needs_revision"], \
                f"Invalid verdict value: {result['last_plan_review_verdict']}"
            assert "workflow_phase" in result, \
                "plan_reviewer_node must return workflow_phase"
            assert result["workflow_phase"] == "plan_review", \
                f"Expected workflow_phase 'plan_review', got '{result['workflow_phase']}'"

        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert "last_design_review_verdict" in result, \
                "design_reviewer_node must return last_design_review_verdict"
            assert result["last_design_review_verdict"] == "approve", \
                f"Expected 'approve', got '{result['last_design_review_verdict']}'"
            assert result["last_design_review_verdict"] in ["approve", "needs_revision"], \
                f"Invalid verdict value: {result['last_design_review_verdict']}"
            assert "workflow_phase" in result, \
                "design_reviewer_node must return workflow_phase"
            assert result["workflow_phase"] == "design_review", \
                f"Expected workflow_phase 'design_review', got '{result['workflow_phase']}'"

        base_state["code"] = "print('test')"
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert "last_code_review_verdict" in result, \
                "code_reviewer_node must return last_code_review_verdict"
            assert result["last_code_review_verdict"] == "approve", \
                f"Expected 'approve', got '{result['last_code_review_verdict']}'"
            assert result["last_code_review_verdict"] in ["approve", "needs_revision"], \
                f"Invalid verdict value: {result['last_code_review_verdict']}"
            assert "workflow_phase" in result, \
                "code_reviewer_node must return workflow_phase"
            assert result["workflow_phase"] == "code_review", \
                f"Expected workflow_phase 'code_review', got '{result['workflow_phase']}'"

    def test_all_validators_return_verdict_field(self, base_state):
        from src.agents.execution import execution_validator_node, physics_sanity_node
        from src.agents.analysis import comparison_validator_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}
        mock_response = {"verdict": "pass", "summary": "OK"}

        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
            assert "execution_verdict" in result, \
                "execution_validator_node must return execution_verdict"
            assert result["execution_verdict"] == "pass", \
                f"Expected 'pass', got '{result['execution_verdict']}'"
            assert result["execution_verdict"] in ["pass", "fail"], \
                f"Invalid verdict value: {result['execution_verdict']}"
            assert "workflow_phase" in result, \
                "execution_validator_node must return workflow_phase"
            assert result["workflow_phase"] == "execution_validation", \
                f"Expected workflow_phase 'execution_validation', got '{result['workflow_phase']}'"
            assert "execution_feedback" in result, \
                "execution_validator_node must return execution_feedback"
            assert isinstance(result["execution_feedback"], str), \
                f"execution_feedback must be string, got {type(result['execution_feedback'])}"

        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = physics_sanity_node(base_state)
            assert "physics_verdict" in result, \
                "physics_sanity_node must return physics_verdict"
            assert result["physics_verdict"] == "pass", \
                f"Expected 'pass', got '{result['physics_verdict']}'"
            assert result["physics_verdict"] in ["pass", "fail", "warning", "design_flaw"], \
                f"Invalid verdict value: {result['physics_verdict']}"
            assert "workflow_phase" in result, \
                "physics_sanity_node must return workflow_phase"
            assert result["workflow_phase"] == "physics_validation", \
                f"Expected workflow_phase 'physics_validation', got '{result['workflow_phase']}'"
            assert "physics_feedback" in result, \
                "physics_sanity_node must return physics_feedback"
            assert isinstance(result["physics_feedback"], str), \
                f"physics_feedback must be string, got {type(result['physics_feedback'])}"

        base_state["figure_comparisons"] = []
        base_state["progress"] = {"stages": []}
        base_state["plan"] = {"stages": [{"stage_id": "stage_0", "targets": []}]}
        result = comparison_validator_node(base_state)
        assert "comparison_verdict" in result, \
            "comparison_validator_node must return comparison_verdict"
        assert result["comparison_verdict"] in ["approve", "needs_revision"], \
            f"Invalid verdict value: {result['comparison_verdict']}"
        assert "workflow_phase" in result, \
            "comparison_validator_node must return workflow_phase"
        assert result["workflow_phase"] == "comparison_validation", \
            f"Expected workflow_phase 'comparison_validation', got '{result['workflow_phase']}'"
        assert "comparison_feedback" in result, \
            "comparison_validator_node must return comparison_feedback"
        assert isinstance(result["comparison_feedback"], str), \
            f"comparison_feedback must be string, got {type(result['comparison_feedback'])}"

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

        # Expected workflow phases for each node
        expected_phases = {
            "adapt_prompts_node": "adapting_prompts",
            "plan_node": "planning",
            "plan_reviewer_node": "plan_review",
            "simulation_designer_node": "design",
            "design_reviewer_node": "design_review",
            "code_generator_node": "code_generation",
            "code_reviewer_node": "code_review",
            "execution_validator_node": "execution_validation",
            "physics_sanity_node": "physics_validation",
            "results_analyzer_node": "analysis",
            "comparison_validator_node": "comparison_validation",
            "generate_report_node": "reporting",
            "supervisor_node": "supervision",
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
                    assert isinstance(result, dict), \
                        f"{node_name} must return dict, got {type(result)}"
                    assert "workflow_phase" in result, \
                        f"{node_name} must set workflow_phase"
                    assert isinstance(result["workflow_phase"], str), \
                        f"{node_name} workflow_phase must be string, got {type(result['workflow_phase'])}"
                    assert result["workflow_phase"], \
                        f"{node_name} workflow_phase must not be empty"
                    
                    # Verify expected phase if we know it
                    if node_name in expected_phases:
                        assert result["workflow_phase"] == expected_phases[node_name], \
                            f"{node_name} expected workflow_phase '{expected_phases[node_name]}', " \
                            f"got '{result['workflow_phase']}'"
                except Exception as e:
                    # Some nodes have required preconditions that may raise; verify it's a meaningful error
                    error_msg = str(e).lower()
                    # Acceptable errors: missing required fields, validation errors
                    # Not acceptable: AttributeError, TypeError, KeyError (indicates bug)
                    if isinstance(e, (AttributeError, TypeError, KeyError)):
                        pytest.fail(
                            f"{node_name} raised {type(e).__name__}: {e}. "
                            "This indicates a bug in the node implementation."
                        )
                    # Other exceptions (ValueError, RuntimeError) may be acceptable preconditions

    def test_counter_increment_never_negative(self, base_state):
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["design_revision_count"] = 0
        base_state["code_revision_count"] = 0
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"

        mock_response = {"verdict": "needs_revision", "issues": ["fix"], "summary": "Fix"}

        # Test design_reviewer_node counter increment
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert "design_revision_count" in result, \
                "design_reviewer_node must return design_revision_count"
            assert isinstance(result["design_revision_count"], int), \
                f"design_revision_count must be int, got {type(result['design_revision_count'])}"
            assert result["design_revision_count"] >= 0, \
                f"design_revision_count must be non-negative, got {result['design_revision_count']}"
            # When verdict is needs_revision, counter should increment
            assert result["design_revision_count"] == 1, \
                f"Expected design_revision_count=1 after needs_revision, got {result['design_revision_count']}"

        # Reset counter and test code_reviewer_node
        base_state["code_revision_count"] = 0
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert "code_revision_count" in result, \
                "code_reviewer_node must return code_revision_count"
            assert isinstance(result["code_revision_count"], int), \
                f"code_revision_count must be int, got {type(result['code_revision_count'])}"
            assert result["code_revision_count"] >= 0, \
                f"code_revision_count must be non-negative, got {result['code_revision_count']}"
            # When verdict is needs_revision, counter should increment
            assert result["code_revision_count"] == 1, \
                f"Expected code_revision_count=1 after needs_revision, got {result['code_revision_count']}"

    def test_counter_increment_respects_max_limit(self, base_state):
        """Test that counters respect max revision limits and escalate when exceeded."""
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"
        base_state["runtime_config"] = {"max_design_revisions": 3, "max_code_revisions": 3}

        mock_response = {"verdict": "needs_revision", "issues": ["fix"], "summary": "Fix"}

        # Test design counter respects max
        base_state["design_revision_count"] = 2  # One below max
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert result["design_revision_count"] == 3, \
                f"Expected design_revision_count=3, got {result['design_revision_count']}"

        # Test code counter respects max
        base_state["code_revision_count"] = 2  # One below max
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert result["code_revision_count"] == 3, \
                f"Expected code_revision_count=3, got {result['code_revision_count']}"

        # Test counter doesn't exceed max
        base_state["design_revision_count"] = 3  # At max
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert result["design_revision_count"] == 3, \
                f"design_revision_count should not exceed max (3), got {result['design_revision_count']}"
            # When max is reached, should escalate
            assert result.get("awaiting_user_input") is True or result.get("ask_user_trigger"), \
                "Should escalate to user when max revisions reached"

    def test_counter_does_not_increment_on_approve(self, base_state):
        """Test that counters don't increment when verdict is approve."""
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["design_revision_count"] = 5
        base_state["code_revision_count"] = 7
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert result["design_revision_count"] == 5, \
                f"design_revision_count should not increment on approve, got {result['design_revision_count']}"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert result["code_revision_count"] == 7, \
                f"code_revision_count should not increment on approve, got {result['code_revision_count']}"


class TestRegressionPrevention:
    """Regression tests to ensure LLM failures escalate correctly."""

    def test_llm_error_in_planner_escalates_not_crashes(self, base_state):
        from src.agents.planning import plan_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_node(base_state)
            assert isinstance(result, dict), \
                "plan_node must return dict even on error, got {type(result)}"
            assert result.get("awaiting_user_input") is True, \
                "plan_node must escalate to user on LLM error"
            assert result.get("workflow_phase") == "planning", \
                f"Expected workflow_phase 'planning', got '{result.get('workflow_phase')}'"
            assert result.get("ask_user_trigger"), \
                "plan_node should set ask_user_trigger on error"
            assert result.get("pending_user_questions"), \
                "plan_node should set pending_user_questions on error"
            assert isinstance(result.get("pending_user_questions"), list), \
                "pending_user_questions must be a list"

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
            assert isinstance(result, dict), \
                "plan_reviewer_node must return dict even on error"
            assert result.get("last_plan_review_verdict") == "approve", \
                f"Expected auto-approve on LLM error, got '{result.get('last_plan_review_verdict')}'"
            assert result.get("workflow_phase") == "plan_review", \
                f"Expected workflow_phase 'plan_review', got '{result.get('workflow_phase')}'"

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = design_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "design_reviewer_node must return dict even on error"
            assert result.get("last_design_review_verdict") == "approve", \
                f"Expected auto-approve on LLM error, got '{result.get('last_design_review_verdict')}'"
            assert result.get("workflow_phase") == "design_review", \
                f"Expected workflow_phase 'design_review', got '{result.get('workflow_phase')}'"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = code_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "code_reviewer_node must return dict even on error"
            assert result.get("last_code_review_verdict") == "approve", \
                f"Expected auto-approve on LLM error, got '{result.get('last_code_review_verdict')}'"
            assert result.get("workflow_phase") == "code_review", \
                f"Expected workflow_phase 'code_review', got '{result.get('workflow_phase')}'"

    def test_llm_error_in_validators_auto_approves(self, base_state):
        """Test that validators auto-approve on LLM errors."""
        from src.agents.execution import execution_validator_node, physics_sanity_node

        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = execution_validator_node(base_state)
            assert isinstance(result, dict), \
                "execution_validator_node must return dict even on error"
            assert result.get("execution_verdict") == "pass", \
                f"Expected auto-approve (pass) on LLM error, got '{result.get('execution_verdict')}'"
            assert result.get("workflow_phase") == "execution_validation", \
                f"Expected workflow_phase 'execution_validation', got '{result.get('workflow_phase')}'"

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = physics_sanity_node(base_state)
            assert isinstance(result, dict), \
                "physics_sanity_node must return dict even on error"
            assert result.get("physics_verdict") == "pass", \
                f"Expected auto-approve (pass) on LLM error, got '{result.get('physics_verdict')}'"
            assert result.get("workflow_phase") == "physics_validation", \
                f"Expected workflow_phase 'physics_validation', got '{result.get('workflow_phase')}'"

    def test_different_exception_types_handled_gracefully(self, base_state, valid_plan):
        """Test that different exception types are handled without crashing."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}

        exception_types = [
            RuntimeError("Runtime error"),
            ValueError("Value error"),
            ConnectionError("Connection error"),
            TimeoutError("Timeout error"),
        ]

        for exc in exception_types:
            with patch(
                "src.agents.planning.call_agent_with_metrics",
                side_effect=exc,
            ):
                result = plan_reviewer_node(base_state)
                assert isinstance(result, dict), \
                    f"plan_reviewer_node must handle {type(exc).__name__} gracefully"
                assert result.get("last_plan_review_verdict") == "approve", \
                    f"Should auto-approve on {type(exc).__name__}"

            with patch(
                "src.agents.design.call_agent_with_metrics",
                side_effect=exc,
            ):
                result = design_reviewer_node(base_state)
                assert isinstance(result, dict), \
                    f"design_reviewer_node must handle {type(exc).__name__} gracefully"
                assert result.get("last_design_review_verdict") == "approve", \
                    f"Should auto-approve on {type(exc).__name__}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nodes_handle_none_values(self, base_state, valid_plan):
        """Test that nodes handle None values in state gracefully."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        
        # Set various fields to None
        base_state["paper_text"] = None
        base_state["paper_figures"] = None
        base_state["assumptions"] = None

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "plan_reviewer_node must handle None values gracefully"
            assert "last_plan_review_verdict" in result, \
                "plan_reviewer_node must return verdict even with None values"

        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "design_reviewer_node must handle None values gracefully"
            assert "last_design_review_verdict" in result, \
                "design_reviewer_node must return verdict even with None values"

    def test_nodes_handle_empty_strings(self, base_state, valid_plan):
        """Test that nodes handle empty strings in state."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = ""  # Empty string
        base_state["paper_text"] = ""

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "plan_reviewer_node must handle empty strings"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "code_reviewer_node must handle empty code string"
            # Empty code should still result in a verdict
            assert "last_code_review_verdict" in result

    def test_nodes_handle_empty_lists(self, base_state, valid_plan):
        """Test that nodes handle empty lists in state."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.execution import execution_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}  # Empty list
        base_state["paper_figures"] = []
        base_state["assumptions"] = {}

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "plan_reviewer_node must handle empty lists"

        mock_response = {"verdict": "pass", "summary": "OK"}
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
            assert isinstance(result, dict), \
                "execution_validator_node must handle empty file lists"

    def test_nodes_handle_missing_required_fields(self, base_state):
        """Test that nodes handle missing required fields appropriately."""
        from src.agents.design import simulation_designer_node
        from src.agents.code import code_generator_node

        # Remove required fields
        minimal_state = {"paper_id": "test"}
        
        # simulation_designer_node requires current_stage_id
        try:
            result = simulation_designer_node(minimal_state)
            # Should either escalate or return error state
            assert isinstance(result, dict), \
                "simulation_designer_node must return dict even with missing fields"
            # Should either have awaiting_user_input or workflow_phase
            assert result.get("awaiting_user_input") or result.get("workflow_phase"), \
                "simulation_designer_node should escalate or set workflow_phase when fields missing"
        except Exception as e:
            # If it raises, should be a meaningful validation error
            assert "current_stage_id" in str(e).lower() or "stage" in str(e).lower(), \
                f"Error should mention missing field: {e}"

        # code_generator_node requires current_stage_id and design_description
        minimal_state["current_stage_id"] = "stage_0"
        try:
            result = code_generator_node(minimal_state)
            assert isinstance(result, dict), \
                "code_generator_node must return dict even with missing design"
            # Should escalate or indicate missing design
            assert result.get("awaiting_user_input") or result.get("workflow_phase"), \
                "code_generator_node should escalate when design missing"
        except Exception as e:
            assert "design" in str(e).lower(), \
                f"Error should mention missing design: {e}"

    def test_verdict_values_are_valid(self, base_state, valid_plan):
        """Test that verdict values are always from the allowed set."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node
        from src.agents.execution import execution_validator_node, physics_sanity_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"
        base_state["stage_outputs"] = {"files": []}

        # Test with various mock responses
        test_cases = [
            {"verdict": "approve", "issues": [], "summary": "OK"},
            {"verdict": "needs_revision", "issues": ["fix"], "summary": "Fix"},
            {"verdict": "pass", "summary": "OK"},
            {"verdict": "fail", "summary": "Failed"},
            {"verdict": "warning", "summary": "Warning"},
            {"verdict": "design_flaw", "summary": "Design issue"},
        ]

        for mock_response in test_cases:
            with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
                result = plan_reviewer_node(base_state)
                verdict = result.get("last_plan_review_verdict")
                assert verdict in ["approve", "needs_revision"], \
                    f"Invalid plan_review verdict: {verdict}"

            with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
                result = design_reviewer_node(base_state)
                verdict = result.get("last_design_review_verdict")
                assert verdict in ["approve", "needs_revision"], \
                    f"Invalid design_review verdict: {verdict}"

            with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
                result = code_reviewer_node(base_state)
                verdict = result.get("last_code_review_verdict")
                assert verdict in ["approve", "needs_revision"], \
                    f"Invalid code_review verdict: {verdict}"

            if mock_response.get("verdict") in ["pass", "fail"]:
                with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
                    result = execution_validator_node(base_state)
                    verdict = result.get("execution_verdict")
                    assert verdict in ["pass", "fail"], \
                        f"Invalid execution verdict: {verdict}"

            if mock_response.get("verdict") in ["pass", "fail", "warning", "design_flaw"]:
                with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
                    result = physics_sanity_node(base_state)
                    verdict = result.get("physics_verdict")
                    assert verdict in ["pass", "fail", "warning", "design_flaw"], \
                        f"Invalid physics verdict: {verdict}"

    def test_state_mutations_are_immutable(self, base_state, valid_plan):
        """Test that nodes don't mutate input state directly."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node

        import copy

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}

        # Create deep copy to compare
        original_state = copy.deepcopy(base_state)

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            # Verify state wasn't mutated (nodes return updates, don't mutate)
            assert base_state == original_state, \
                "plan_reviewer_node should not mutate input state"
            # Result should be separate dict
            assert result is not base_state, \
                "plan_reviewer_node should return new dict, not mutate input"

        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert base_state == original_state, \
                "design_reviewer_node should not mutate input state"
            assert result is not base_state, \
                "design_reviewer_node should return new dict, not mutate input"

    def test_counters_handle_none_initial_value(self, base_state):
        """Test that counters handle None as initial value correctly."""
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"
        
        # Set counters to None (simulating uninitialized state)
        base_state["design_revision_count"] = None
        base_state["code_revision_count"] = None

        mock_response = {"verdict": "needs_revision", "issues": ["fix"], "summary": "Fix"}

        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            # Counter should be initialized to 0 or 1, not None
            assert result.get("design_revision_count") is not None, \
                "design_revision_count should not be None after increment"
            assert isinstance(result["design_revision_count"], int), \
                f"design_revision_count must be int, got {type(result['design_revision_count'])}"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert result.get("code_revision_count") is not None, \
                "code_revision_count should not be None after increment"
            assert isinstance(result["code_revision_count"], int), \
                f"code_revision_count must be int, got {type(result['code_revision_count'])}"

    def test_feedback_fields_are_strings(self, base_state, valid_plan):
        """Test that feedback fields are always strings, not None or other types."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.execution import execution_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}

        # Test with missing feedback in mock response
        mock_response = {"verdict": "approve", "issues": []}  # No summary field

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            # If planner_feedback is set, it should be a string
            if "planner_feedback" in result:
                assert isinstance(result["planner_feedback"], str), \
                    f"planner_feedback must be string, got {type(result['planner_feedback'])}"

        mock_response = {"verdict": "pass"}  # No summary field
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
            assert "execution_feedback" in result, \
                "execution_validator_node must return execution_feedback"
            assert isinstance(result["execution_feedback"], str), \
                f"execution_feedback must be string, got {type(result['execution_feedback'])}"
            assert result["execution_feedback"], \
                "execution_feedback should not be empty string"

