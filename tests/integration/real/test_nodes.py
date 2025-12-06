"""Integration tests ensuring node functions execute with minimal state."""

from unittest.mock import patch
import pytest


class TestNodeFunctionsCallable:
    """
    Test that node functions can be called with minimal state.

    Only mock the LLM call itself - everything else runs for real.
    These tests verify:
    1. The node doesn't crash.
    2. The node returns a dict with expected state keys.
    3. The node returns correct data structures and values.
    4. Edge cases are handled properly.
    """

    def test_plan_node_returns_valid_plan(self, minimal_state, mock_llm_response):
        """plan_node must return a result containing a complete plan structure."""
        from src.agents.planning import plan_node

        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = plan_node(minimal_state)

        assert result is not None, "plan_node must return a result"
        assert isinstance(result, dict), "plan_node must return a dict"
        assert "plan" in result, "plan_node must return 'plan' key"
        assert "workflow_phase" in result, "plan_node must return 'workflow_phase'"
        assert result["workflow_phase"] == "planning", "workflow_phase must be 'planning'"
        
        plan = result["plan"]
        assert isinstance(plan, dict), "plan must be a dict"
        assert "stages" in plan, "Plan must have 'stages' key"
        assert "targets" in plan, "Plan must have 'targets' key"
        assert "paper_id" in plan, "Plan must have 'paper_id' key"
        assert "extracted_parameters" in plan, "Plan must have 'extracted_parameters' key"
        
        # Validate stages structure
        stages = plan["stages"]
        assert isinstance(stages, list), "stages must be a list"
        
        # Validate targets structure
        targets = plan["targets"]
        assert isinstance(targets, list), "targets must be a list"
        
        # Validate extracted_parameters structure
        extracted_parameters = plan["extracted_parameters"]
        assert isinstance(extracted_parameters, list), "extracted_parameters must be a list"
        
        # Check for planned_materials and assumptions
        assert "planned_materials" in result, "plan_node must return 'planned_materials'"
        assert isinstance(result["planned_materials"], list), "planned_materials must be a list"
        assert "assumptions" in result, "plan_node must return 'assumptions'"
        assert isinstance(result["assumptions"], dict), "assumptions must be a dict"
        
        # Validate paper_domain
        assert "paper_domain" in result, "plan_node must return 'paper_domain'"
        assert isinstance(result["paper_domain"], str), "paper_domain must be a string"
        
        # If progress was initialized, validate its structure
        if "progress" in result:
            progress = result["progress"]
            assert isinstance(progress, dict), "progress must be a dict"
            if "stages" in progress:
                assert isinstance(progress["stages"], list), "progress.stages must be a list"

    def test_supervisor_node_returns_verdict(self, minimal_state, mock_llm_response):
        """supervisor_node must return a supervisor_verdict with valid structure."""
        from src.agents.supervision.supervisor import supervisor_node

        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            mock_llm_response,
        ):
            result = supervisor_node(minimal_state)

        assert result is not None, "supervisor_node must return a result"
        assert isinstance(result, dict), "supervisor_node must return a dict"
        assert "supervisor_verdict" in result, "supervisor_node must return supervisor_verdict"
        assert "supervisor_feedback" in result, "supervisor_node must return supervisor_feedback"
        assert "workflow_phase" in result, "supervisor_node must return workflow_phase"
        assert result["workflow_phase"] == "supervision", "workflow_phase must be 'supervision'"
        
        # Validate verdict is a valid string
        verdict = result["supervisor_verdict"]
        assert isinstance(verdict, str), "supervisor_verdict must be a string"
        assert len(verdict) > 0, "supervisor_verdict must not be empty"
        # Valid verdicts include: ok_continue, backtrack_to_stage, needs_revision, etc.
        # We don't restrict to specific values as the supervisor may return various verdicts
        
        # Validate feedback is a string
        feedback = result["supervisor_feedback"]
        assert isinstance(feedback, str), "supervisor_feedback must be a string"
        
        # Check for archive_errors (should always be present, even if empty)
        assert "archive_errors" in result, "supervisor_node must return archive_errors"
        assert isinstance(result["archive_errors"], list), "archive_errors must be a list"

    def test_report_node_completes_workflow(self, minimal_state, mock_llm_response):
        """generate_report_node must mark workflow as complete with valid report structure."""
        from src.agents.reporting import generate_report_node

        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)

        assert result is not None, "generate_report_node must return a result"
        assert isinstance(result, dict), "generate_report_node must return a dict"
        assert "workflow_complete" in result, "generate_report_node must set workflow_complete"
        assert result["workflow_complete"] is True, "workflow_complete must be True"
        assert "executive_summary" in result, "generate_report_node must return executive_summary"
        assert "workflow_phase" in result, "generate_report_node must return workflow_phase"
        assert result["workflow_phase"] == "reporting", "workflow_phase must be 'reporting'"
        
        # Validate executive_summary structure
        executive_summary = result["executive_summary"]
        assert isinstance(executive_summary, dict), "executive_summary must be a dict"
        assert "overall_assessment" in executive_summary, "executive_summary must have overall_assessment"
        assert isinstance(executive_summary["overall_assessment"], list), "overall_assessment must be a list"
        
        # Validate metrics structure
        assert "metrics" in result, "generate_report_node must return metrics"
        metrics = result["metrics"]
        assert isinstance(metrics, dict), "metrics must be a dict"
        assert "token_summary" in metrics, "metrics must have token_summary"
        token_summary = metrics["token_summary"]
        assert isinstance(token_summary, dict), "token_summary must be a dict"
        assert "total_input_tokens" in token_summary, "token_summary must have total_input_tokens"
        assert "total_output_tokens" in token_summary, "token_summary must have total_output_tokens"
        assert "estimated_cost" in token_summary, "token_summary must have estimated_cost"
        assert isinstance(token_summary["total_input_tokens"], int), "total_input_tokens must be int"
        assert isinstance(token_summary["total_output_tokens"], int), "total_output_tokens must be int"
        assert isinstance(token_summary["estimated_cost"], (int, float)), "estimated_cost must be numeric"
        assert token_summary["total_input_tokens"] >= 0, "total_input_tokens must be non-negative"
        assert token_summary["total_output_tokens"] >= 0, "total_output_tokens must be non-negative"
        assert token_summary["estimated_cost"] >= 0, "estimated_cost must be non-negative"
        
        # Validate paper_citation structure
        assert "paper_citation" in result, "generate_report_node must return paper_citation"
        paper_citation = result["paper_citation"]
        assert isinstance(paper_citation, dict), "paper_citation must be a dict"

    def test_design_node_returns_design(self, minimal_state):
        """simulation_designer_node must return design_description with valid structure."""
        from src.agents.design import simulation_designer_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        design_response = {
            "design": {
                "stage_id": "stage_0",
                "geometry": [],
                "simulation_parameters": {},
            },
            "explanation": "Designed",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=design_response,
        ):
            result = simulation_designer_node(minimal_state)

        assert result is not None, "simulation_designer_node must return a result"
        assert isinstance(result, dict), "simulation_designer_node must return a dict"
        assert "design_description" in result, "simulation_designer_node must return design_description"
        assert "workflow_phase" in result, "simulation_designer_node must return workflow_phase"
        assert result["workflow_phase"] == "design", "workflow_phase must be 'design'"
        
        # Validate design_description structure
        design_description = result["design_description"]
        assert isinstance(design_description, dict), "design_description must be a dict"
        assert "design" in design_description, "design_description must have 'design' key"
        
        design = design_description["design"]
        assert isinstance(design, dict), "design must be a dict"
        assert "stage_id" in design, "design must have 'stage_id'"
        assert design["stage_id"] == "stage_0", "design.stage_id must match current_stage_id"
        assert "geometry" in design, "design must have 'geometry'"
        assert isinstance(design["geometry"], list), "geometry must be a list"
        assert "simulation_parameters" in design, "design must have 'simulation_parameters'"
        assert isinstance(design["simulation_parameters"], dict), "simulation_parameters must be a dict"

    def test_code_generator_returns_code(self, minimal_state):
        """code_generator_node must return generated code with valid structure."""
        from src.agents.code import code_generator_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        # design_description must have a 'design_description' key with the main description text
        minimal_state["design_description"] = {
            "design_description": "A valid FDTD simulation design for aluminum nanoantenna arrays",
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design",
        }
        # Provide code that's at least 50 characters to pass validation
        code_response = {
            "code": "import meep as mp\nimport numpy as np\n\n# Simulation setup\nsim = mp.Simulation()\nprint('Simulation initialized')",
            "explanation": "Generated",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)

        assert result is not None, "code_generator_node must return a result"
        assert isinstance(result, dict), "code_generator_node must return a dict"
        assert "code" in result, "code_generator_node must return code"
        assert "workflow_phase" in result, "code_generator_node must return workflow_phase"
        assert result["workflow_phase"] == "code_generation", "workflow_phase must be 'code_generation'"
        
        # Validate code is a non-empty string
        code = result["code"]
        assert isinstance(code, str), "code must be a string"
        assert len(code.strip()) > 0, "code must not be empty"
        assert len(code.strip()) >= 50, "code must be substantial (at least 50 chars) - component validates this"
        assert "import meep" in code, "code must contain expected content"
        
        # Check for expected_outputs if present
        if "expected_outputs" in result:
            assert isinstance(result["expected_outputs"], list), "expected_outputs must be a list"

    def test_execution_validator_node_returns_verdict(self, minimal_state, mock_llm_response):
        """execution_validator_node must return execution_verdict with valid structure."""
        from src.agents.execution import execution_validator_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"stdout": "run complete", "stderr": ""}

        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = execution_validator_node(minimal_state)

        assert result is not None, "execution_validator_node must return a result"
        assert isinstance(result, dict), "execution_validator_node must return a dict"
        assert "execution_verdict" in result, "execution_validator_node must return execution_verdict"
        assert "execution_feedback" in result, "execution_validator_node must return execution_feedback"
        assert "workflow_phase" in result, "execution_validator_node must return workflow_phase"
        assert result["workflow_phase"] == "execution_validation", "workflow_phase must be 'execution_validation'"
        
        # Validate verdict is one of the expected values
        verdict = result["execution_verdict"]
        assert isinstance(verdict, str), "execution_verdict must be a string"
        assert verdict in ["pass", "fail"], f"execution_verdict must be 'pass' or 'fail', got '{verdict}'"
        
        # Validate feedback is a string
        feedback = result["execution_feedback"]
        assert isinstance(feedback, str), "execution_feedback must be a string"
        assert len(feedback) > 0, "execution_feedback must not be empty"
        
        # If verdict is "fail", check for failure count
        if verdict == "fail":
            assert "execution_failure_count" in result, "execution_failure_count must be present when verdict is 'fail'"
            assert isinstance(result["execution_failure_count"], int), "execution_failure_count must be an int"
            assert result["execution_failure_count"] > 0, "execution_failure_count must be > 0 when verdict is 'fail'"
            assert "total_execution_failures" in result, "total_execution_failures must be present when verdict is 'fail'"
            assert isinstance(result["total_execution_failures"], int), "total_execution_failures must be an int"
            assert result["total_execution_failures"] > 0, "total_execution_failures must be > 0 when verdict is 'fail'"

    def test_physics_sanity_node_returns_verdict(self, minimal_state, mock_llm_response):
        """physics_sanity_node must return physics_verdict with valid structure."""
        from src.agents.execution import physics_sanity_node

        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"files": ["spectrum.csv"]}

        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = physics_sanity_node(minimal_state)

        assert result is not None, "physics_sanity_node must return a result"
        assert isinstance(result, dict), "physics_sanity_node must return a dict"
        assert "physics_verdict" in result, "physics_sanity_node must return physics_verdict"
        assert "physics_feedback" in result, "physics_sanity_node must return physics_feedback"
        assert "workflow_phase" in result, "physics_sanity_node must return workflow_phase"
        assert result["workflow_phase"] == "physics_validation", "workflow_phase must be 'physics_validation'"
        
        # Validate verdict is one of the expected values
        verdict = result["physics_verdict"]
        assert isinstance(verdict, str), "physics_verdict must be a string"
        assert verdict in ["pass", "fail", "warning", "design_flaw"], \
            f"physics_verdict must be one of ['pass', 'fail', 'warning', 'design_flaw'], got '{verdict}'"
        
        # Validate feedback is a string
        feedback = result["physics_feedback"]
        assert isinstance(feedback, str), "physics_feedback must be a string"
        assert len(feedback) > 0, "physics_feedback must not be empty"
        
        # If verdict is "fail", check for physics_failure_count
        if verdict == "fail":
            assert "physics_failure_count" in result, "physics_failure_count must be present when verdict is 'fail'"
            assert isinstance(result["physics_failure_count"], int), "physics_failure_count must be an int"
            assert result["physics_failure_count"] > 0, "physics_failure_count must be > 0 when verdict is 'fail'"
        
        # If verdict is "design_flaw", check for design_revision_count
        if verdict == "design_flaw":
            assert "design_revision_count" in result, "design_revision_count must be present when verdict is 'design_flaw'"
            assert isinstance(result["design_revision_count"], int), "design_revision_count must be an int"
            assert result["design_revision_count"] > 0, "design_revision_count must be > 0 when verdict is 'design_flaw'"
            assert "design_feedback" in result, "design_feedback must be present when verdict is 'design_flaw'"
            assert isinstance(result["design_feedback"], str), "design_feedback must be a string"

    def test_prompt_adaptor_node_returns_adaptations(self, minimal_state, mock_llm_response):
        """prompt_adaptor_node must return prompt_adaptations with valid structure."""
        from src.agents.planning import adapt_prompts_node

        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = adapt_prompts_node(minimal_state)

        assert result is not None, "adapt_prompts_node must return a result"
        assert isinstance(result, dict), "adapt_prompts_node must return a dict"
        assert "prompt_adaptations" in result, "adapt_prompts_node must return prompt_adaptations"
        assert "workflow_phase" in result, "adapt_prompts_node must return workflow_phase"
        assert result["workflow_phase"] == "adapting_prompts", "workflow_phase must be 'adapting_prompts'"
        
        # Validate prompt_adaptations is a list
        adaptations = result["prompt_adaptations"]
        assert isinstance(adaptations, list), "prompt_adaptations must be a list"
        
        # If paper_domain is returned, validate it
        if "paper_domain" in result:
            assert isinstance(result["paper_domain"], str), "paper_domain must be a string"


class TestNodeEdgeCases:
    """Test edge cases and error conditions to catch bugs."""
    
    def test_plan_node_handles_missing_paper_text(self, minimal_state, mock_llm_response):
        """plan_node must handle missing or too-short paper_text."""
        from src.agents.planning import plan_node
        
        minimal_state["paper_text"] = ""  # Empty paper text
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = plan_node(minimal_state)
        
        # Should escalate to user when paper_text is missing/too short
        assert result is not None, "plan_node must return a result even with missing paper_text"
        assert "ask_user_trigger" in result, "plan_node must set ask_user_trigger when paper_text is missing"
        assert result["ask_user_trigger"] == "missing_paper_text", "ask_user_trigger must be 'missing_paper_text'"
        assert "pending_user_questions" in result, "plan_node must set pending_user_questions"
        assert isinstance(result["pending_user_questions"], list), "pending_user_questions must be a list"
        assert len(result["pending_user_questions"]) > 0, "pending_user_questions must not be empty"
        assert result.get("ask_user_trigger") is not None, "ask_user_trigger must be set"
    
    def test_plan_node_handles_none_paper_text(self, minimal_state, mock_llm_response):
        """plan_node must handle None paper_text."""
        from src.agents.planning import plan_node
        
        minimal_state["paper_text"] = None
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = plan_node(minimal_state)
        
        assert result is not None, "plan_node must return a result even with None paper_text"
        assert "ask_user_trigger" in result, "plan_node must set ask_user_trigger when paper_text is None"
    
    def test_design_node_handles_missing_stage_id(self, minimal_state):
        """simulation_designer_node must handle missing current_stage_id."""
        from src.agents.design import simulation_designer_node
        
        # Remove current_stage_id
        if "current_stage_id" in minimal_state:
            del minimal_state["current_stage_id"]
        
        design_response = {
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Designed",
        }
        
        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=design_response,
        ):
            result = simulation_designer_node(minimal_state)
        
        assert result is not None, "simulation_designer_node must return a result even with missing stage_id"
        assert "ask_user_trigger" in result, "simulation_designer_node must set ask_user_trigger when stage_id is missing"
        assert result["ask_user_trigger"] == "missing_stage_id", "ask_user_trigger must be 'missing_stage_id'"
    
    def test_code_generator_handles_missing_stage_id(self, minimal_state):
        """code_generator_node must handle missing current_stage_id."""
        from src.agents.code import code_generator_node
        
        # Remove current_stage_id
        if "current_stage_id" in minimal_state:
            del minimal_state["current_stage_id"]
        
        minimal_state["design_description"] = {
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design",
        }
        
        code_response = {"code": "import meep as mp\nprint('hello')"}
        
        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)
        
        assert result is not None, "code_generator_node must return a result even with missing stage_id"
        assert "ask_user_trigger" in result, "code_generator_node must set ask_user_trigger when stage_id is missing"
        assert result["ask_user_trigger"] == "missing_stage_id", "ask_user_trigger must be 'missing_stage_id'"
    
    def test_code_generator_handles_missing_design_description(self, minimal_state):
        """code_generator_node must handle missing or stub design_description."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        minimal_state["design_description"] = None  # Missing design
        
        code_response = {"code": "import meep as mp\nprint('hello')"}
        
        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)
        
        assert result is not None, "code_generator_node must return a result even with missing design"
        # Should increment design_revision_count or handle error
        assert "design_revision_count" in result or "reviewer_feedback" in result, \
            "code_generator_node must handle missing design_description"
    
    def test_code_generator_handles_stub_design_description(self, minimal_state):
        """code_generator_node must handle stub design_description."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        minimal_state["design_description"] = "STUB: This is a placeholder design"
        
        code_response = {"code": "import meep as mp\nprint('hello')"}
        
        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)
        
        assert result is not None, "code_generator_node must return a result even with stub design"
        # Should handle stub design appropriately
        assert "design_revision_count" in result or "reviewer_feedback" in result, \
            "code_generator_node must handle stub design_description"
    
    def test_code_generator_handles_empty_code_response(self, minimal_state):
        """code_generator_node must handle empty code from LLM."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        minimal_state["design_description"] = {
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design",
        }
        
        code_response = {"code": ""}  # Empty code
        
        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)
        
        assert result is not None, "code_generator_node must return a result even with empty code"
        # Should handle empty code appropriately (increment revision count or error)
        assert "code_revision_count" in result or "reviewer_feedback" in result, \
            "code_generator_node must handle empty code"
    
    def test_code_generator_handles_stub_code_response(self, minimal_state):
        """code_generator_node must handle stub code from LLM."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        minimal_state["design_description"] = {
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design",
        }
        
        code_response = {"code": "STUB"}  # Stub code
        
        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)
        
        assert result is not None, "code_generator_node must return a result even with stub code"
        # Should handle stub code appropriately
        assert "code_revision_count" in result or "reviewer_feedback" in result, \
            "code_generator_node must handle stub code"
    
    def test_code_generator_handles_missing_validated_materials_for_stage1(self, minimal_state):
        """code_generator_node must handle missing validated_materials for Stage 1+."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_1"
        minimal_state["current_stage_type"] = "SINGLE_STRUCTURE"  # Not MATERIAL_VALIDATION
        # design_description must have a 'design_description' key with the main description text
        minimal_state["design_description"] = {
            "design_description": "A valid FDTD simulation design for aluminum nanoantenna arrays",
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design",
        }
        # Don't set validated_materials
        
        code_response = {"code": "import meep as mp\nprint('hello')"}
        
        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=code_response,
        ):
            result = code_generator_node(minimal_state)
        
        assert result is not None, "code_generator_node must return a result even with missing materials"
        # Should handle missing validated_materials for Stage 1+
        assert "run_error" in result or "code_revision_count" in result, \
            "code_generator_node must handle missing validated_materials for Stage 1+"
    
    def test_execution_validator_handles_missing_stage_outputs(self, minimal_state, mock_llm_response):
        """execution_validator_node must handle missing stage_outputs."""
        from src.agents.execution import execution_validator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        # Don't set stage_outputs
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = execution_validator_node(minimal_state)
        
        assert result is not None, "execution_validator_node must return a result even with missing stage_outputs"
        assert "execution_verdict" in result, "execution_validator_node must return execution_verdict"
        assert "execution_feedback" in result, "execution_validator_node must return execution_feedback"
    
    def test_execution_validator_handles_none_stage_outputs(self, minimal_state, mock_llm_response):
        """execution_validator_node must handle None stage_outputs."""
        from src.agents.execution import execution_validator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = None
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = execution_validator_node(minimal_state)
        
        assert result is not None, "execution_validator_node must return a result even with None stage_outputs"
        assert "execution_verdict" in result, "execution_validator_node must return execution_verdict"
    
    def test_physics_sanity_handles_missing_stage_outputs(self, minimal_state, mock_llm_response):
        """physics_sanity_node must handle missing stage_outputs."""
        from src.agents.execution import physics_sanity_node
        
        minimal_state["current_stage_id"] = "stage_0"
        # Don't set stage_outputs
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = physics_sanity_node(minimal_state)
        
        assert result is not None, "physics_sanity_node must return a result even with missing stage_outputs"
        assert "physics_verdict" in result, "physics_sanity_node must return physics_verdict"
        assert "physics_feedback" in result, "physics_sanity_node must return physics_feedback"
    
    def test_physics_sanity_handles_none_stage_outputs(self, minimal_state, mock_llm_response):
        """physics_sanity_node must handle None stage_outputs."""
        from src.agents.execution import physics_sanity_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = None
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = physics_sanity_node(minimal_state)
        
        assert result is not None, "physics_sanity_node must return a result even with None stage_outputs"
        assert "physics_verdict" in result, "physics_sanity_node must return physics_verdict"
    
    def test_supervisor_handles_none_user_responses(self, minimal_state, mock_llm_response):
        """supervisor_node must handle None user_responses."""
        from src.agents.supervision.supervisor import supervisor_node
        
        minimal_state["user_responses"] = None
        
        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            mock_llm_response,
        ):
            result = supervisor_node(minimal_state)
        
        assert result is not None, "supervisor_node must return a result even with None user_responses"
        assert "supervisor_verdict" in result, "supervisor_node must return supervisor_verdict"
        assert "archive_errors" in result, "supervisor_node must return archive_errors"
    
    def test_supervisor_handles_invalid_user_responses_type(self, minimal_state, mock_llm_response):
        """supervisor_node must handle invalid user_responses type."""
        from src.agents.supervision.supervisor import supervisor_node
        
        minimal_state["user_responses"] = "not a dict"  # Invalid type
        
        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            mock_llm_response,
        ):
            result = supervisor_node(minimal_state)
        
        assert result is not None, "supervisor_node must return a result even with invalid user_responses type"
        assert "supervisor_verdict" in result, "supervisor_node must return supervisor_verdict"
    
    def test_report_node_handles_missing_metrics(self, minimal_state, mock_llm_response):
        """generate_report_node must handle missing metrics."""
        from src.agents.reporting import generate_report_node
        
        if "metrics" in minimal_state:
            del minimal_state["metrics"]
        
        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)
        
        assert result is not None, "generate_report_node must return a result even with missing metrics"
        assert "metrics" in result, "generate_report_node must return metrics"
        assert "token_summary" in result["metrics"], "metrics must have token_summary"
        assert result["metrics"]["token_summary"]["total_input_tokens"] == 0, \
            "token_summary should default to 0 when metrics are missing"
    
    def test_report_node_handles_empty_progress(self, minimal_state, mock_llm_response):
        """generate_report_node must handle empty progress."""
        from src.agents.reporting import generate_report_node
        
        minimal_state["progress"] = {}
        
        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)
        
        assert result is not None, "generate_report_node must return a result even with empty progress"
        assert "workflow_complete" in result, "generate_report_node must set workflow_complete"
        assert result["workflow_complete"] is True, "workflow_complete must be True"
    
    def test_adapt_prompts_handles_none_paper_text(self, minimal_state, mock_llm_response):
        """adapt_prompts_node must handle None paper_text."""
        from src.agents.planning import adapt_prompts_node
        
        minimal_state["paper_text"] = None
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = adapt_prompts_node(minimal_state)
        
        assert result is not None, "adapt_prompts_node must return a result even with None paper_text"
        assert "prompt_adaptations" in result, "adapt_prompts_node must return prompt_adaptations"
        assert isinstance(result["prompt_adaptations"], list), "prompt_adaptations must be a list"
    
    def test_adapt_prompts_handles_empty_paper_text(self, minimal_state, mock_llm_response):
        """adapt_prompts_node must handle empty paper_text."""
        from src.agents.planning import adapt_prompts_node
        
        minimal_state["paper_text"] = ""
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = adapt_prompts_node(minimal_state)
        
        assert result is not None, "adapt_prompts_node must return a result even with empty paper_text"
        assert "prompt_adaptations" in result, "adapt_prompts_node must return prompt_adaptations"
        assert isinstance(result["prompt_adaptations"], list), "prompt_adaptations must be a list"
    
    def test_execution_validator_handles_fail_verdict(self, minimal_state):
        """execution_validator_node must handle 'fail' verdict correctly."""
        from src.agents.execution import execution_validator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"stdout": "error occurred", "stderr": "Error: failed"}
        
        def mock_fail_response(*args, **kwargs):
            return {"verdict": "fail", "summary": "Execution failed"}
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_fail_response):
            result = execution_validator_node(minimal_state)
        
        assert result is not None, "execution_validator_node must return a result"
        assert result["execution_verdict"] == "fail", "execution_verdict must be 'fail'"
        assert "execution_failure_count" in result, "execution_failure_count must be present when verdict is 'fail'"
        assert result["execution_failure_count"] > 0, "execution_failure_count must be incremented"
        assert "total_execution_failures" in result, "total_execution_failures must be present when verdict is 'fail'"
        assert result["total_execution_failures"] > 0, "total_execution_failures must be incremented"
    
    def test_physics_sanity_handles_design_flaw_verdict(self, minimal_state):
        """physics_sanity_node must handle 'design_flaw' verdict correctly."""
        from src.agents.execution import physics_sanity_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"files": ["spectrum.csv"]}
        
        def mock_design_flaw_response(*args, **kwargs):
            return {"verdict": "design_flaw", "summary": "Design flaw detected"}
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_design_flaw_response):
            result = physics_sanity_node(minimal_state)
        
        assert result is not None, "physics_sanity_node must return a result"
        assert result["physics_verdict"] == "design_flaw", "physics_verdict must be 'design_flaw'"
        assert "design_revision_count" in result, "design_revision_count must be present when verdict is 'design_flaw'"
        assert result["design_revision_count"] > 0, "design_revision_count must be incremented"
        assert "design_feedback" in result, "design_feedback must be present when verdict is 'design_flaw'"
        assert isinstance(result["design_feedback"], str), "design_feedback must be a string"
        assert len(result["design_feedback"]) > 0, "design_feedback must not be empty"
    
    def test_physics_sanity_handles_fail_verdict(self, minimal_state):
        """physics_sanity_node must handle 'fail' verdict correctly."""
        from src.agents.execution import physics_sanity_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"files": ["spectrum.csv"]}
        
        def mock_fail_response(*args, **kwargs):
            return {"verdict": "fail", "summary": "Physics check failed"}
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_fail_response):
            result = physics_sanity_node(minimal_state)
        
        assert result is not None, "physics_sanity_node must return a result"
        assert result["physics_verdict"] == "fail", "physics_verdict must be 'fail'"
        assert "physics_failure_count" in result, "physics_failure_count must be present when verdict is 'fail'"
        assert result["physics_failure_count"] > 0, "physics_failure_count must be incremented"
    
    def test_plan_node_validates_plan_structure(self, minimal_state):
        """plan_node must validate plan structure and handle initialization errors."""
        from src.agents.planning import plan_node
        
        def mock_invalid_plan_response(*args, **kwargs):
            # Return plan with invalid structure that would cause initialization to fail
            return {
                "paper_id": "test",
                "title": "Test",
                "stages": [{"invalid": "structure"}],  # Missing required fields
                "targets": [],
                "extracted_parameters": [],
            }
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_invalid_plan_response):
            result = plan_node(minimal_state)
        
        assert result is not None, "plan_node must return a result even with invalid plan structure"
        # Should either initialize progress successfully or handle the error
        # If initialization fails, should set last_plan_review_verdict to needs_revision
        if "last_plan_review_verdict" in result:
            assert result["last_plan_review_verdict"] == "needs_revision", \
                "plan_node must mark plan for revision if initialization fails"
    
    def test_supervisor_handles_archive_errors(self, minimal_state, mock_llm_response):
        """supervisor_node must handle archive_errors correctly."""
        from src.agents.supervision.supervisor import supervisor_node
        
        # Set archive_errors to test retry logic
        minimal_state["archive_errors"] = [
            {"stage_id": "stage_0", "error": "Test error", "timestamp": "2023-01-01T00:00:00Z"}
        ]
        
        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            mock_llm_response,
        ):
            result = supervisor_node(minimal_state)
        
        assert result is not None, "supervisor_node must return a result"
        assert "archive_errors" in result, "supervisor_node must return archive_errors"
        assert isinstance(result["archive_errors"], list), "archive_errors must be a list"
    
    def test_report_node_computes_metrics_correctly(self, minimal_state, mock_llm_response):
        """generate_report_node must compute token metrics correctly."""
        from src.agents.reporting import generate_report_node
        
        # Set up metrics with known values
        minimal_state["metrics"] = {
            "agent_calls": [
                {"input_tokens": 100, "output_tokens": 50},
                {"input_tokens": 200, "output_tokens": 100},
            ]
        }
        
        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)
        
        assert result is not None, "generate_report_node must return a result"
        assert "metrics" in result, "generate_report_node must return metrics"
        assert "token_summary" in result["metrics"], "metrics must have token_summary"
        
        token_summary = result["metrics"]["token_summary"]
        assert token_summary["total_input_tokens"] == 300, "total_input_tokens must sum correctly (100+200)"
        assert token_summary["total_output_tokens"] == 150, "total_output_tokens must sum correctly (50+100)"
        # Estimated cost = (300 * 3.0 + 150 * 15.0) / 1_000_000 = (900 + 2250) / 1_000_000 = 0.00315
        expected_cost = (300 * 3.0 + 150 * 15.0) / 1_000_000
        assert abs(token_summary["estimated_cost"] - expected_cost) < 0.0001, \
            f"estimated_cost must be calculated correctly (expected {expected_cost}, got {token_summary['estimated_cost']})"
    
    def test_report_node_handles_none_metrics(self, minimal_state, mock_llm_response):
        """generate_report_node must handle None metrics."""
        from src.agents.reporting import generate_report_node
        
        minimal_state["metrics"] = None
        
        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)
        
        assert result is not None, "generate_report_node must return a result even with None metrics"
        assert "metrics" in result, "generate_report_node must return metrics"
        assert "token_summary" in result["metrics"], "metrics must have token_summary"
        assert result["metrics"]["token_summary"]["total_input_tokens"] == 0, \
            "token_summary should default to 0 when metrics is None"
    
    def test_adapt_prompts_handles_non_list_adaptations(self, minimal_state):
        """adapt_prompts_node must handle non-list adaptations from LLM."""
        from src.agents.planning import adapt_prompts_node
        
        def mock_non_list_response(*args, **kwargs):
            return {"adaptations": "not a list"}  # Invalid type
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_non_list_response):
            result = adapt_prompts_node(minimal_state)
        
        assert result is not None, "adapt_prompts_node must return a result even with invalid adaptations type"
        assert "prompt_adaptations" in result, "adapt_prompts_node must return prompt_adaptations"
        assert isinstance(result["prompt_adaptations"], list), "prompt_adaptations must be a list (converted from invalid type)"
        assert len(result["prompt_adaptations"]) == 0, "prompt_adaptations must be empty list when LLM returns invalid type"
    
    def test_adapt_prompts_handles_none_adaptations(self, minimal_state):
        """adapt_prompts_node must handle None adaptations from LLM."""
        from src.agents.planning import adapt_prompts_node
        
        def mock_none_response(*args, **kwargs):
            return {"adaptations": None}  # None value
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_none_response):
            result = adapt_prompts_node(minimal_state)
        
        assert result is not None, "adapt_prompts_node must return a result even with None adaptations"
        assert "prompt_adaptations" in result, "adapt_prompts_node must return prompt_adaptations"
        assert isinstance(result["prompt_adaptations"], list), "prompt_adaptations must be a list (converted from None)"
        assert len(result["prompt_adaptations"]) == 0, "prompt_adaptations must be empty list when LLM returns None"


