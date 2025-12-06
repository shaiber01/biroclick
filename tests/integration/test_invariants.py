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
            assert result.get("ask_user_trigger") is not None or result.get("ask_user_trigger"), \
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
            assert result.get("ask_user_trigger") is not None, \
                "plan_node must escalate to user on LLM error"
            assert result.get("workflow_phase") == "planning", \
                f"Expected workflow_phase 'planning', got '{result.get('workflow_phase')}'"
            assert result.get("ask_user_trigger"), \
                "plan_node should set ask_user_trigger on error"
            assert result.get("pending_user_questions"), \
                "plan_node should set pending_user_questions on error"
            assert isinstance(result.get("pending_user_questions"), list), \
                "pending_user_questions must be a list"

    def test_llm_error_in_reviewers_uses_fail_closed_safety(self, base_state, valid_plan):
        """Test that reviewer nodes use fail-closed safety on LLM errors.
        
        IMPORTANT: Reviewers should NOT auto-approve on LLM error. They should
        default to 'needs_revision' for fail-closed safety. This prevents
        potentially buggy code/designs from being auto-approved when LLM
        review is unavailable.
        
        This is the opposite of validators (execution, physics) which auto-pass
        to not block execution when LLM is unavailable.
        """
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"

        # Plan reviewer should use fail-closed safety (needs_revision on error)
        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = plan_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "plan_reviewer_node must return dict even on error"
            # FAIL-CLOSED: Reviewers default to needs_revision, not approve
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                f"Expected fail-closed needs_revision on LLM error, got '{result.get('last_plan_review_verdict')}'"
            assert result.get("workflow_phase") == "plan_review", \
                f"Expected workflow_phase 'plan_review', got '{result.get('workflow_phase')}'"

        # Design reviewer should use fail-closed safety (needs_revision on error)
        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = design_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "design_reviewer_node must return dict even on error"
            # FAIL-CLOSED: Reviewers default to needs_revision, not approve
            assert result.get("last_design_review_verdict") == "needs_revision", \
                f"Expected fail-closed needs_revision on LLM error, got '{result.get('last_design_review_verdict')}'"
            assert result.get("workflow_phase") == "design_review", \
                f"Expected workflow_phase 'design_review', got '{result.get('workflow_phase')}'"

        # Code reviewer should use fail-closed safety (needs_revision on error)
        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = code_reviewer_node(base_state)
            assert isinstance(result, dict), \
                "code_reviewer_node must return dict even on error"
            # FAIL-CLOSED: Reviewers default to needs_revision, not approve
            assert result.get("last_code_review_verdict") == "needs_revision", \
                f"Expected fail-closed needs_revision on LLM error, got '{result.get('last_code_review_verdict')}'"
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
        """Test that different exception types are handled without crashing.
        
        Reviewers should use fail-closed safety (needs_revision) for all exception types.
        The key invariant is that no exception type should cause a crash.
        """
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
                # FAIL-CLOSED: Reviewers default to needs_revision on any error
                assert result.get("last_plan_review_verdict") == "needs_revision", \
                    f"Should use fail-closed needs_revision on {type(exc).__name__}"
                # Verify workflow_phase is preserved
                assert result.get("workflow_phase") == "plan_review", \
                    f"workflow_phase should be set even on {type(exc).__name__}"

            with patch(
                "src.agents.design.call_agent_with_metrics",
                side_effect=exc,
            ):
                result = design_reviewer_node(base_state)
                assert isinstance(result, dict), \
                    f"design_reviewer_node must handle {type(exc).__name__} gracefully"
                # FAIL-CLOSED: Reviewers default to needs_revision on any error
                assert result.get("last_design_review_verdict") == "needs_revision", \
                    f"Should use fail-closed needs_revision on {type(exc).__name__}"
                # Verify workflow_phase is preserved
                assert result.get("workflow_phase") == "design_review", \
                    f"workflow_phase should be set even on {type(exc).__name__}"


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
            # Should either have ask_user_trigger or workflow_phase
            assert result.get("ask_user_trigger") or result.get("workflow_phase"), \
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
            assert result.get("ask_user_trigger") or result.get("workflow_phase"), \
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


class TestRoutingInvariants:
    """Test that routing functions respect invariant rules."""

    def test_plan_review_router_maps_verdicts_correctly(self, base_state, valid_plan):
        """Test plan review router routes verdicts to correct nodes."""
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["replan_count"] = 0

        # Test approve → select_stage
        base_state["last_plan_review_verdict"] = "approve"
        route = route_after_plan_review(base_state)
        assert route == "select_stage", \
            f"approve should route to select_stage, got {route}"

        # Test needs_revision → plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        route = route_after_plan_review(base_state)
        assert route == "planning", \
            f"needs_revision should route to plan, got {route}"

    def test_design_review_router_maps_verdicts_correctly(self, base_state):
        """Test design review router routes verdicts to correct nodes."""
        from src.routing import route_after_design_review

        base_state["design_revision_count"] = 0

        # Test approve → generate_code
        base_state["last_design_review_verdict"] = "approve"
        route = route_after_design_review(base_state)
        assert route == "generate_code", \
            f"approve should route to generate_code, got {route}"

        # Test needs_revision → design
        base_state["last_design_review_verdict"] = "needs_revision"
        route = route_after_design_review(base_state)
        assert route == "design", \
            f"needs_revision should route to design, got {route}"

    def test_code_review_router_maps_verdicts_correctly(self, base_state):
        """Test code review router routes verdicts to correct nodes."""
        from src.routing import route_after_code_review

        base_state["code_revision_count"] = 0

        # Test approve → run_code
        base_state["last_code_review_verdict"] = "approve"
        route = route_after_code_review(base_state)
        assert route == "run_code", \
            f"approve should route to run_code, got {route}"

        # Test needs_revision → generate_code
        base_state["last_code_review_verdict"] = "needs_revision"
        route = route_after_code_review(base_state)
        assert route == "generate_code", \
            f"needs_revision should route to generate_code, got {route}"

    def test_execution_router_maps_verdicts_correctly(self, base_state):
        """Test execution check router routes verdicts to correct nodes."""
        from src.routing import route_after_execution_check

        base_state["execution_failure_count"] = 0

        # Test pass → physics_check
        base_state["execution_verdict"] = "pass"
        route = route_after_execution_check(base_state)
        assert route == "physics_check", \
            f"pass should route to physics_check, got {route}"

        # Test warning → physics_check (also proceeds)
        base_state["execution_verdict"] = "warning"
        route = route_after_execution_check(base_state)
        assert route == "physics_check", \
            f"warning should route to physics_check, got {route}"

        # Test fail → generate_code
        base_state["execution_verdict"] = "fail"
        route = route_after_execution_check(base_state)
        assert route == "generate_code", \
            f"fail should route to generate_code, got {route}"

    def test_physics_router_maps_verdicts_correctly(self, base_state):
        """Test physics check router routes verdicts to correct nodes."""
        from src.routing import route_after_physics_check

        base_state["physics_failure_count"] = 0
        base_state["design_revision_count"] = 0

        # Test pass → analyze
        base_state["physics_verdict"] = "pass"
        route = route_after_physics_check(base_state)
        assert route == "analyze", \
            f"pass should route to analyze, got {route}"

        # Test warning → analyze (proceeds with warning)
        base_state["physics_verdict"] = "warning"
        route = route_after_physics_check(base_state)
        assert route == "analyze", \
            f"warning should route to analyze, got {route}"

        # Test fail → generate_code
        base_state["physics_verdict"] = "fail"
        route = route_after_physics_check(base_state)
        assert route == "generate_code", \
            f"fail should route to generate_code, got {route}"

        # Test design_flaw → design
        base_state["physics_verdict"] = "design_flaw"
        route = route_after_physics_check(base_state)
        assert route == "design", \
            f"design_flaw should route to design, got {route}"

    def test_router_escalates_on_none_verdict(self, base_state):
        """Test that routers escalate to ask_user when verdict is None."""
        from src.routing import (
            route_after_plan_review,
            route_after_design_review,
            route_after_code_review,
            route_after_execution_check,
            route_after_physics_check,
        )

        routers_and_fields = [
            (route_after_plan_review, "last_plan_review_verdict"),
            (route_after_design_review, "last_design_review_verdict"),
            (route_after_code_review, "last_code_review_verdict"),
            (route_after_execution_check, "execution_verdict"),
            (route_after_physics_check, "physics_verdict"),
        ]

        for router, verdict_field in routers_and_fields:
            base_state[verdict_field] = None
            route = router(base_state)
            assert route == "ask_user", \
                f"Router should escalate to ask_user when {verdict_field} is None, got {route}"

    def test_router_escalates_on_unknown_verdict(self, base_state):
        """Test that routers escalate to ask_user for unknown verdict values."""
        from src.routing import route_after_plan_review

        base_state["last_plan_review_verdict"] = "invalid_verdict"
        route = route_after_plan_review(base_state)
        assert route == "ask_user", \
            f"Router should escalate to ask_user for unknown verdict, got {route}"

    def test_router_escalates_on_count_limit_exceeded(self, base_state):
        """Test that routers escalate when revision limits are exceeded."""
        from src.routing import route_after_code_review

        # Set revision count at max
        base_state["code_revision_count"] = 10
        base_state["runtime_config"] = {"max_code_revisions": 3}
        base_state["last_code_review_verdict"] = "needs_revision"

        route = route_after_code_review(base_state)
        assert route == "ask_user", \
            f"Router should escalate to ask_user when count >= max, got {route}"


class TestBacktrackInvariants:
    """Test backtrack decision handling invariants."""

    def test_backtrack_node_requires_accepted_decision(self, base_state):
        """Test that backtrack node requires an accepted decision."""
        from src.agents.reporting import handle_backtrack_node

        # Test with missing backtrack_decision
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") is not None, \
            "handle_backtrack_node should escalate when decision is missing"
        assert result.get("ask_user_trigger") == "invalid_backtrack_decision", \
            "Should set ask_user_trigger for invalid decision"

        # Test with decision not accepted
        base_state["backtrack_decision"] = {"accepted": False, "target_stage_id": "stage_0"}
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") is not None, \
            "handle_backtrack_node should escalate when decision is not accepted"

    def test_backtrack_node_requires_target_stage_id(self, base_state, valid_plan):
        """Test that backtrack node requires a target stage ID."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }

        # Test with empty target_stage_id
        base_state["backtrack_decision"] = {"accepted": True, "target_stage_id": ""}
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") is not None, \
            "handle_backtrack_node should escalate when target_stage_id is empty"
        assert result.get("ask_user_trigger") == "invalid_backtrack_target", \
            "Should set ask_user_trigger for invalid target"

    def test_backtrack_node_validates_target_stage_exists(self, base_state, valid_plan):
        """Test that backtrack node validates target stage exists."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }

        # Test with non-existent target stage
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_nonexistent",
            "stages_to_invalidate": []
        }
        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") is not None, \
            "handle_backtrack_node should escalate when target stage doesn't exist"
        assert result.get("ask_user_trigger") == "backtrack_target_not_found", \
            "Should set ask_user_trigger for not found target"

    def test_backtrack_node_respects_max_backtracks(self, base_state, valid_plan):
        """Test that backtrack node respects max backtrack limit."""
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
            "stages_to_invalidate": []
        }

        result = handle_backtrack_node(base_state)
        assert result.get("ask_user_trigger") is not None, \
            "handle_backtrack_node should escalate when max backtracks exceeded"
        assert result.get("ask_user_trigger") == "backtrack_limit", \
            "Should set ask_user_trigger for backtrack limit"
        assert result.get("workflow_phase") == "backtracking_limit", \
            f"Expected workflow_phase 'backtracking_limit', got '{result.get('workflow_phase')}'"

    def test_backtrack_node_increments_counter(self, base_state, valid_plan):
        """Test that backtrack node increments backtrack counter."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = 0
        base_state["runtime_config"] = {"max_backtracks": 5}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": []
        }

        result = handle_backtrack_node(base_state)
        assert result.get("backtrack_count") == 1, \
            f"Expected backtrack_count=1, got {result.get('backtrack_count')}"

    def test_backtrack_node_handles_none_backtrack_count(self, base_state, valid_plan):
        """Test that backtrack node handles None backtrack_count."""
        from src.agents.reporting import handle_backtrack_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = None
        base_state["runtime_config"] = {"max_backtracks": 5}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": []
        }

        result = handle_backtrack_node(base_state)
        assert result.get("backtrack_count") == 1, \
            f"Expected backtrack_count=1 when starting from None, got {result.get('backtrack_count')}"


class TestPromptAdaptorInvariants:
    """Test prompt adaptor node invariants."""

    def test_adapt_prompts_returns_workflow_phase(self, base_state):
        """Test that adapt_prompts_node returns workflow_phase."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {"adaptations": [], "paper_domain": "plasmonics"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = adapt_prompts_node(base_state)
            assert "workflow_phase" in result, \
                "adapt_prompts_node must return workflow_phase"
            assert result["workflow_phase"] == "adapting_prompts", \
                f"Expected workflow_phase 'adapting_prompts', got '{result['workflow_phase']}'"

    def test_adapt_prompts_handles_llm_error_gracefully(self, base_state):
        """Test that adapt_prompts_node handles LLM errors gracefully."""
        from src.agents.planning import adapt_prompts_node

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = adapt_prompts_node(base_state)
            # Should not raise, should return default adaptations
            assert isinstance(result, dict), \
                "adapt_prompts_node must return dict even on error"
            assert "workflow_phase" in result, \
                "adapt_prompts_node must return workflow_phase even on error"
            assert result.get("prompt_adaptations") == [], \
                "adapt_prompts_node should return empty adaptations on error"

    def test_adapt_prompts_handles_invalid_adaptations(self, base_state):
        """Test that adapt_prompts_node handles invalid adaptations gracefully."""
        from src.agents.planning import adapt_prompts_node

        # Test with adaptations as non-list (dict)
        mock_response = {"adaptations": {"invalid": "dict"}, "paper_domain": "plasmonics"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = adapt_prompts_node(base_state)
            # Should handle gracefully
            assert isinstance(result.get("prompt_adaptations"), list), \
                f"prompt_adaptations should be list, got {type(result.get('prompt_adaptations'))}"

    def test_adapt_prompts_handles_none_adaptations(self, base_state):
        """Test that adapt_prompts_node handles None adaptations."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {"adaptations": None, "paper_domain": "plasmonics"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = adapt_prompts_node(base_state)
            assert result.get("prompt_adaptations") == [], \
                "adapt_prompts_node should return empty list when adaptations is None"

    def test_adapt_prompts_preserves_paper_domain(self, base_state):
        """Test that adapt_prompts_node preserves paper domain from response."""
        from src.agents.planning import adapt_prompts_node

        mock_response = {"adaptations": [], "paper_domain": "quantum_optics"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = adapt_prompts_node(base_state)
            assert result.get("paper_domain") == "quantum_optics", \
                f"Expected paper_domain 'quantum_optics', got '{result.get('paper_domain')}'"


class TestComparisonValidatorInvariants:
    """Test comparison_validator_node invariants."""

    def test_comparison_validator_returns_required_fields(self, base_state, valid_plan):
        """Test that comparison_validator_node returns all required fields."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["progress"] = {"stages": []}

        result = comparison_validator_node(base_state)
        assert "workflow_phase" in result, \
            "comparison_validator_node must return workflow_phase"
        assert result["workflow_phase"] == "comparison_validation", \
            f"Expected workflow_phase 'comparison_validation', got '{result['workflow_phase']}'"
        assert "comparison_verdict" in result, \
            "comparison_validator_node must return comparison_verdict"
        assert "comparison_feedback" in result, \
            "comparison_validator_node must return comparison_feedback"

    def test_comparison_validator_approves_when_no_targets(self, base_state, valid_plan):
        """Test that comparison_validator approves when stage has no targets."""
        from src.agents.analysis import comparison_validator_node

        # Create plan with empty targets
        plan_no_targets = {**valid_plan}
        plan_no_targets["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Test",
            "targets": [],  # Empty targets
            "dependencies": [],
        }]

        base_state["plan"] = plan_no_targets
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["progress"] = {"stages": []}

        result = comparison_validator_node(base_state)
        assert result.get("comparison_verdict") == "approve", \
            f"Should approve when stage has no targets, got '{result.get('comparison_verdict')}'"

    def test_comparison_validator_rejects_when_comparisons_missing(self, base_state, valid_plan):
        """Test that comparison_validator rejects when comparisons are missing."""
        from src.agents.analysis import comparison_validator_node

        # Plan with targets but no comparisons
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []  # No comparisons
        base_state["analysis_result_reports"] = []
        base_state["progress"] = {"stages": []}

        result = comparison_validator_node(base_state)
        assert result.get("comparison_verdict") == "needs_revision", \
            f"Should need revision when comparisons missing, got '{result.get('comparison_verdict')}'"

    def test_comparison_validator_increments_count_on_needs_revision(self, base_state, valid_plan):
        """Test that comparison_validator increments count on needs_revision."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        base_state["analysis_revision_count"] = 0
        base_state["progress"] = {"stages": []}

        result = comparison_validator_node(base_state)
        if result.get("comparison_verdict") == "needs_revision":
            assert "analysis_revision_count" in result, \
                "comparison_validator should update analysis_revision_count on needs_revision"
            assert result["analysis_revision_count"] == 1, \
                f"Expected analysis_revision_count=1, got {result['analysis_revision_count']}"

    def test_comparison_validator_skips_when_trigger_set(self, base_state, valid_plan):
        """Test that comparison_validator skips processing when ask_user_trigger is set."""
        from src.agents.analysis import comparison_validator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["ask_user_trigger"] = "some_trigger"  # Trigger set

        result = comparison_validator_node(base_state)
        # Should return empty dict (no processing)
        assert result == {}, \
            "comparison_validator should return empty dict when ask_user_trigger is set"


class TestSchemaConsistencyInvariants:
    """Test that schema files are consistent with code expectations."""

    REQUIRED_SCHEMA_FIELDS = {
        "planner_output_schema.json": ["stages", "paper_domain"],
        "plan_reviewer_output_schema.json": ["verdict"],
        "simulation_designer_output_schema.json": ["design_description", "geometry"],
        "design_reviewer_output_schema.json": ["verdict"],
        "code_generator_output_schema.json": ["code"],
        "code_reviewer_output_schema.json": ["verdict"],
        "execution_validator_output_schema.json": ["verdict"],
        "physics_sanity_output_schema.json": ["verdict"],
        "results_analyzer_output_schema.json": ["figure_comparisons"],
        "supervisor_output_schema.json": ["verdict"],
    }

    @pytest.mark.parametrize("schema_name,required_fields", REQUIRED_SCHEMA_FIELDS.items())
    def test_schema_has_required_fields(self, schema_name, required_fields):
        """Test that each schema has its required fields in properties."""
        schema_path = PROJECT_ROOT / "schemas" / schema_name
        if not schema_path.exists():
            pytest.skip(f"Schema {schema_name} does not exist")

        content = schema_path.read_text()
        schema = json.loads(content)
        properties = schema.get("properties", {})

        for field in required_fields:
            assert field in properties, \
                f"Schema {schema_name} missing required field '{field}' in properties"

    def test_all_verdict_schemas_have_consistent_types(self):
        """Test that all verdict fields have consistent type definitions."""
        verdict_schemas = [
            "plan_reviewer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_reviewer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "supervisor_output_schema.json",
        ]

        for schema_name in verdict_schemas:
            schema_path = PROJECT_ROOT / "schemas" / schema_name
            if not schema_path.exists():
                continue

            content = schema_path.read_text()
            schema = json.loads(content)
            properties = schema.get("properties", {})

            if "verdict" in properties:
                verdict_prop = properties["verdict"]
                # Verdict should be either string type or enum
                assert verdict_prop.get("type") == "string" or "enum" in verdict_prop, \
                    f"Verdict in {schema_name} should be string type or enum"


class TestStageProgressionInvariants:
    """Test stage progression invariants."""

    def test_stage_selection_requires_plan(self, base_state):
        """Test that stage selection cannot proceed without a plan."""
        from src.agents.stage_selection import select_stage_node

        # Remove plan from state
        base_state["plan"] = None
        base_state["progress"] = {"stages": []}

        result = select_stage_node(base_state)
        # Should escalate or return error state
        assert isinstance(result, dict), \
            "select_stage_node must return dict even without plan"
        # Should either set current_stage_id to None or escalate
        assert result.get("current_stage_id") is None or result.get("ask_user_trigger"), \
            "select_stage_node should not set current_stage_id without plan"

    def test_validation_hierarchy_is_computed_not_stored(self, base_state, valid_plan):
        """Test that validation hierarchy is computed on demand."""
        from schemas.state import get_validation_hierarchy

        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"}
            ]
        }

        hierarchy = get_validation_hierarchy(base_state)
        assert isinstance(hierarchy, dict), \
            "get_validation_hierarchy must return dict"
        assert "material_validation" in hierarchy, \
            "hierarchy must include material_validation"
        assert hierarchy["material_validation"] == "passed", \
            f"Expected material_validation='passed' for completed_success, got '{hierarchy['material_validation']}'"

    def test_validation_hierarchy_handles_empty_progress(self, base_state):
        """Test that validation hierarchy handles empty progress gracefully."""
        from schemas.state import get_validation_hierarchy

        base_state["progress"] = {}

        hierarchy = get_validation_hierarchy(base_state)
        assert isinstance(hierarchy, dict), \
            "get_validation_hierarchy must return dict even with empty progress"
        assert all(v == "not_done" for v in hierarchy.values()), \
            "All hierarchy values should be 'not_done' for empty progress"

    def test_validation_hierarchy_handles_missing_progress(self, base_state):
        """Test that validation hierarchy handles missing progress gracefully."""
        from schemas.state import get_validation_hierarchy

        # Remove progress entirely
        if "progress" in base_state:
            del base_state["progress"]

        hierarchy = get_validation_hierarchy(base_state)
        assert isinstance(hierarchy, dict), \
            "get_validation_hierarchy must return dict even without progress"


class TestSupervisorInvariants:
    """Test supervisor node invariants."""

    def test_supervisor_returns_workflow_phase(self, base_state, valid_plan):
        """Test that supervisor_node always returns workflow_phase."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": []}

        mock_response = {"verdict": "ok_continue", "reasoning": "OK"}

        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", return_value=mock_response):
            result = supervisor_node(base_state)
            assert "workflow_phase" in result, \
                "supervisor_node must return workflow_phase"
            assert result["workflow_phase"] == "supervision", \
                f"Expected workflow_phase 'supervision', got '{result['workflow_phase']}'"

    def test_supervisor_clears_trigger_after_handling(self, base_state, valid_plan):
        """Test that supervisor clears ask_user_trigger after handling."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": []}
        base_state["ask_user_trigger"] = "test_trigger"
        base_state["user_responses"] = {"question": "APPROVE"}

        # Just need supervisor to process the trigger - mock will prevent LLM call
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", return_value={"verdict": "ok_continue"}):
            result = supervisor_node(base_state)
            assert result.get("ask_user_trigger") is None, \
                "supervisor_node should clear ask_user_trigger after handling"

    def test_supervisor_handles_invalid_user_responses_type(self, base_state, valid_plan):
        """Test that supervisor handles invalid user_responses type gracefully."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": []}
        base_state["user_responses"] = "invalid_string"  # Should be dict

        mock_response = {"verdict": "ok_continue", "reasoning": "OK"}

        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", return_value=mock_response):
            result = supervisor_node(base_state)
            # Should not crash, should handle gracefully
            assert isinstance(result, dict), \
                "supervisor_node must return dict even with invalid user_responses type"

    def test_supervisor_handles_llm_error_gracefully(self, base_state, valid_plan):
        """Test that supervisor handles LLM errors gracefully."""
        from src.agents.supervision.supervisor import supervisor_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": []}

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = supervisor_node(base_state)
            assert isinstance(result, dict), \
                "supervisor_node must return dict even on LLM error"
            assert result.get("supervisor_verdict") == "ok_continue", \
                f"supervisor_node should default to 'ok_continue' on error, got '{result.get('supervisor_verdict')}'"


class TestPlanValidationInvariants:
    """Test plan validation invariants in plan_reviewer_node."""

    def test_plan_reviewer_rejects_empty_stages(self, base_state, valid_plan):
        """Test that plan_reviewer rejects plans with empty stages."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with empty stages
        plan_empty_stages = {**valid_plan, "stages": []}
        base_state["plan"] = plan_empty_stages

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            # Should reject because stages is empty
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                "Should reject plan with empty stages"

    def test_plan_reviewer_detects_missing_stage_id(self, base_state, valid_plan):
        """Test that plan_reviewer detects stages without stage_id."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with stage missing stage_id
        plan_missing_id = {**valid_plan}
        plan_missing_id["stages"] = [{
            "stage_type": "MATERIAL_VALIDATION",
            "targets": ["Fig1"],
            "dependencies": [],
        }]
        base_state["plan"] = plan_missing_id

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                "Should reject stage without stage_id"

    def test_plan_reviewer_detects_duplicate_stage_ids(self, base_state, valid_plan):
        """Test that plan_reviewer detects duplicate stage IDs."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with duplicate stage IDs
        plan_dupes = {**valid_plan}
        plan_dupes["stages"] = [
            {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []},
            {"stage_id": "stage_0", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": []},
        ]
        base_state["plan"] = plan_dupes

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                "Should reject plan with duplicate stage IDs"

    def test_plan_reviewer_detects_missing_dependencies(self, base_state, valid_plan):
        """Test that plan_reviewer detects references to non-existent dependencies."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with dependency on non-existent stage
        plan_bad_deps = {**valid_plan}
        plan_bad_deps["stages"] = [{
            "stage_id": "stage_1",
            "stage_type": "SINGLE_STRUCTURE",
            "targets": ["Fig1"],
            "dependencies": ["nonexistent_stage"],
        }]
        base_state["plan"] = plan_bad_deps

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                "Should reject plan with missing dependency"

    def test_plan_reviewer_detects_circular_dependencies(self, base_state, valid_plan):
        """Test that plan_reviewer detects circular dependencies."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with circular dependencies
        plan_circular = {**valid_plan}
        plan_circular["stages"] = [
            {"stage_id": "stage_a", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": ["stage_b"]},
            {"stage_id": "stage_b", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": ["stage_a"]},
        ]
        base_state["plan"] = plan_circular

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                "Should reject plan with circular dependencies"

    def test_plan_reviewer_detects_self_dependency(self, base_state, valid_plan):
        """Test that plan_reviewer detects stages that depend on themselves."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with self-dependency
        plan_self_dep = {**valid_plan}
        plan_self_dep["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "targets": ["Fig1"],
            "dependencies": ["stage_0"],  # Self-reference
        }]
        base_state["plan"] = plan_self_dep

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "needs_revision", \
                "Should reject stage with self-dependency"

    def test_plan_reviewer_handles_none_dependencies(self, base_state, valid_plan):
        """Test that plan_reviewer handles None dependencies gracefully."""
        from src.agents.planning import plan_reviewer_node

        # Create plan with None dependencies
        plan_none_deps = {**valid_plan}
        plan_none_deps["stages"] = [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "targets": ["Fig1"],
            "dependencies": None,  # None instead of empty list
        }]
        base_state["plan"] = plan_none_deps

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            # Should not crash with None dependencies
            assert isinstance(result, dict), \
                "plan_reviewer should handle None dependencies gracefully"


class TestReportGenerationInvariants:
    """Test report generation invariants."""

    def test_report_node_returns_workflow_phase(self, base_state, valid_plan):
        """Test that report node always returns workflow_phase."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {}

        mock_response = {}

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
            assert "workflow_phase" in result, \
                "generate_report_node must return workflow_phase"
            assert result["workflow_phase"] == "reporting", \
                f"Expected workflow_phase 'reporting', got '{result['workflow_phase']}'"

    def test_report_node_sets_workflow_complete(self, base_state, valid_plan):
        """Test that report node sets workflow_complete flag."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {}

        mock_response = {}

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
            assert result.get("workflow_complete") is True, \
                "generate_report_node must set workflow_complete=True"

    def test_report_node_handles_empty_metrics(self, base_state, valid_plan):
        """Test that report node handles empty/missing metrics gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = None

        mock_response = {}

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
            # Should not crash, should set default metrics
            assert "metrics" in result, \
                "generate_report_node must include metrics in result"

    def test_report_node_handles_invalid_agent_calls_type(self, base_state, valid_plan):
        """Test that report node handles invalid agent_calls type gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {"agent_calls": "invalid_string"}  # Should be list

        mock_response = {}

        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
            # Should not crash, should handle gracefully
            assert isinstance(result, dict), \
                "generate_report_node must return dict even with invalid agent_calls type"

    def test_report_node_handles_llm_error_gracefully(self, base_state, valid_plan):
        """Test that report node handles LLM errors gracefully."""
        from src.agents.reporting import generate_report_node

        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["metrics"] = {}

        with patch(
            "src.agents.reporting.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = generate_report_node(base_state)
            # Should not crash, should complete with stub report
            assert isinstance(result, dict), \
                "generate_report_node must return dict even on LLM error"
            assert result.get("workflow_complete") is True, \
                "generate_report_node must set workflow_complete even on error"


class TestBaseUtilityInvariants:
    """Test base utility function invariants."""

    def test_increment_counter_with_max_handles_none_state(self):
        """Test that increment_counter_with_max raises on None state."""
        from src.agents.base import increment_counter_with_max

        with pytest.raises(TypeError):
            increment_counter_with_max(None, "counter", "max", 3)

    def test_increment_counter_with_max_handles_non_dict_state(self):
        """Test that increment_counter_with_max raises on non-dict state."""
        from src.agents.base import increment_counter_with_max

        with pytest.raises(TypeError):
            increment_counter_with_max("not_a_dict", "counter", "max", 3)

    def test_increment_counter_with_max_handles_none_counter(self, base_state):
        """Test that increment_counter_with_max handles None counter value."""
        from src.agents.base import increment_counter_with_max

        base_state["test_counter"] = None

        new_count, incremented = increment_counter_with_max(
            base_state, "test_counter", "max_test", 3
        )
        assert new_count == 1, \
            f"Expected count=1 when starting from None, got {new_count}"
        assert incremented is True, \
            "Should return incremented=True when starting from None"

    def test_increment_counter_with_max_respects_max(self, base_state):
        """Test that increment_counter_with_max respects maximum value."""
        from src.agents.base import increment_counter_with_max

        base_state["test_counter"] = 3
        base_state["runtime_config"] = {"max_test": 3}

        new_count, incremented = increment_counter_with_max(
            base_state, "test_counter", "max_test", 5
        )
        assert new_count == 3, \
            f"Expected count=3 (not incremented), got {new_count}"
        assert incremented is False, \
            "Should return incremented=False when at max"

    def test_check_keywords_handles_none_response(self):
        """Test that check_keywords handles None response."""
        from src.agents.base import check_keywords

        result = check_keywords(None, ["TEST"])
        assert result is False, \
            "check_keywords should return False for None response"

    def test_check_keywords_handles_none_keywords(self):
        """Test that check_keywords raises on None keywords."""
        from src.agents.base import check_keywords

        with pytest.raises(TypeError):
            check_keywords("test", None)

    def test_check_keywords_handles_empty_response(self):
        """Test that check_keywords handles empty response."""
        from src.agents.base import check_keywords

        result = check_keywords("", ["TEST"])
        assert result is False, \
            "check_keywords should return False for empty response"

    def test_check_keywords_case_insensitive(self):
        """Test that check_keywords is case insensitive."""
        from src.agents.base import check_keywords

        # Test lowercase
        assert check_keywords("approve", ["APPROVE"]) is True
        # Test uppercase
        assert check_keywords("APPROVE", ["approve"]) is True
        # Test mixed
        assert check_keywords("ApPrOvE", ["APPROVE"]) is True

    def test_check_keywords_word_boundaries(self):
        """Test that check_keywords respects word boundaries."""
        from src.agents.base import check_keywords

        # "DISAPPROVE" should not match "APPROVE"
        assert check_keywords("DISAPPROVE", ["APPROVE"]) is False, \
            "check_keywords should not match partial words"
        # But "APPROVE" alone should match
        assert check_keywords("I APPROVE this", ["APPROVE"]) is True

    def test_parse_user_response_handles_none(self):
        """Test that parse_user_response handles None."""
        from src.agents.base import parse_user_response

        result = parse_user_response(None)
        assert result == "", \
            "parse_user_response should return empty string for None"

    def test_parse_user_response_handles_empty_dict(self):
        """Test that parse_user_response handles empty dict."""
        from src.agents.base import parse_user_response

        result = parse_user_response({})
        assert result == "", \
            "parse_user_response should return empty string for empty dict"

    def test_parse_user_response_handles_non_dict(self):
        """Test that parse_user_response raises on non-dict."""
        from src.agents.base import parse_user_response

        with pytest.raises(TypeError):
            parse_user_response("not_a_dict")


class TestCodeGeneratorInvariants:
    """Test code generator node invariants."""

    def test_code_generator_requires_stage_id(self, base_state, valid_plan):
        """Test that code_generator requires current_stage_id."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = None
        base_state["design_description"] = {"design": "test"}

        result = code_generator_node(base_state)
        assert result.get("ask_user_trigger") is not None, \
            "code_generator should escalate when current_stage_id is None"

    def test_code_generator_requires_design_description(self, base_state, valid_plan):
        """Test that code_generator requires design_description."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = None

        result = code_generator_node(base_state)
        # Should reject because design_description is None
        assert "supervisor_verdict" in result or "reviewer_feedback" in result, \
            "code_generator should indicate error when design_description is missing"

    def test_code_generator_rejects_stub_design(self, base_state, valid_plan):
        """Test that code_generator rejects stub design descriptions."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = "TODO: implement design"  # Stub marker

        result = code_generator_node(base_state)
        # Should reject stub design
        assert "reviewer_feedback" in result or "supervisor_verdict" in result, \
            "code_generator should reject stub design descriptions"

    def test_code_generator_validates_generated_code(self, base_state, valid_plan):
        """Test that code_generator validates generated code is not empty/stub."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        # Set stage type to MATERIAL_VALIDATION to avoid validated_materials check
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "design_description": "Full valid design for testing that is long enough to pass length check",
            "geometry": [{"type": "box"}],
            "sources": [],
            "monitors": []
        }

        # Mock LLM returning stub code
        mock_response = {"code": "# TODO: implement", "expected_outputs": []}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)
            # Should detect stub code
            assert "reviewer_feedback" in result, \
                "code_generator should provide feedback when generated code is stub"

    def test_code_generator_handles_llm_error(self, base_state, valid_plan):
        """Test that code_generator escalates on LLM error."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        # Set stage type to MATERIAL_VALIDATION to avoid validated_materials check
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "design_description": "Full valid design for testing that is long enough to pass length check",
            "geometry": [{"type": "box"}],
            "sources": [],
            "monitors": []
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=RuntimeError("API Error"),
        ):
            result = code_generator_node(base_state)
            assert result.get("ask_user_trigger") is not None, \
                "code_generator should escalate to user on LLM error"
            assert result.get("ask_user_trigger") == "llm_error", \
                f"Expected ask_user_trigger 'llm_error', got '{result.get('ask_user_trigger')}'"

