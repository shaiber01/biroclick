"""Execution-phase E2E tests.

Tests for:
- execution_validator_node: validates simulation execution results
- physics_sanity_node: validates physics of simulation outputs
- route_after_execution_check: routes based on execution verdict
- route_after_physics_check: routes based on physics verdict
- Graph flow integration tests
"""

import pytest
from unittest.mock import patch, MagicMock

from src.graph import create_repro_graph
from src.agents.execution import execution_validator_node, physics_sanity_node
from src.routing import route_after_execution_check, route_after_physics_check
from schemas.state import (
    create_initial_state,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_DESIGN_REVISIONS,
)

from tests.state_machine.common import (
    CHECKPOINT_PATCH_LOCATIONS,
    LLM_PATCH_LOCATIONS,
    MockLLMResponses,
    MultiPatch,
    unique_thread_id,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_state():
    """Base state with minimal required fields for execution tests."""
    return create_initial_state(
        paper_id="test_paper",
        paper_text="Gold nanorods have LSPR at 700nm." * 10,
    )


@pytest.fixture
def execution_state(base_state):
    """State ready for execution validation."""
    return {
        **base_state,
        "current_stage_id": "stage_1_main",
        "code": "print('Simulating...')",
        "stage_outputs": {
            "stdout": "Simulation complete",
            "stderr": "",
            "exit_code": 0,
            "files": ["output.npy"],
            "runtime_seconds": 10.5,
            "timeout_exceeded": False,
            "memory_exceeded": False,
        },
        "run_error": None,
        "execution_failure_count": 0,
        "total_execution_failures": 0,
        "design_description": {"geometry": {"dimensionality": "3D"}},
    }


@pytest.fixture
def physics_state(execution_state):
    """State ready for physics sanity check."""
    return {
        **execution_state,
        "execution_verdict": "pass",
        "physics_failure_count": 0,
        "design_revision_count": 0,
    }


# =============================================================================
# Unit Tests: execution_validator_node
# =============================================================================

class TestExecutionValidatorNode:
    """Unit tests for execution_validator_node."""

    def test_pass_verdict_sets_correct_state(self, execution_state):
        """Test that pass verdict sets workflow_phase and execution_verdict correctly."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "pass",
                "summary": "Execution successful",
            }
            
            result = execution_validator_node(execution_state)
            
            assert result["workflow_phase"] == "execution_validation"
            assert result["execution_verdict"] == "pass"
            assert result["execution_feedback"] == "Execution successful"
            # Should NOT increment failure count on pass
            assert "execution_failure_count" not in result

    def test_fail_verdict_increments_failure_count(self, execution_state):
        """Test that fail verdict increments execution_failure_count."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Simulation crashed",
            }
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_verdict"] == "fail"
            assert result["execution_failure_count"] == 1
            assert result["total_execution_failures"] == 1

    def test_fail_verdict_increments_from_existing_count(self, execution_state):
        """Test that fail verdict increments from existing count."""
        execution_state["execution_failure_count"] = 1
        execution_state["total_execution_failures"] = 5
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Simulation crashed again",
            }
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_failure_count"] == 2
            assert result["total_execution_failures"] == 6

    def test_fail_verdict_at_max_triggers_ask_user(self, execution_state):
        """Test that reaching max failures triggers ask_user."""
        execution_state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        execution_state["run_error"] = "Memory limit exceeded"
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Failed again",
            }
            
            result = execution_validator_node(execution_state)
            
            assert result["ask_user_trigger"] == "execution_failure_limit"
            assert len(result["pending_user_questions"]) == 1
            assert "Memory limit exceeded" in result["pending_user_questions"][0]
            # Count should NOT increment beyond max
            assert result["execution_failure_count"] == MAX_EXECUTION_FAILURES

    def test_warning_verdict_does_not_increment_failure_count(self, execution_state):
        """Test that warning verdict does not increment failure count."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "warning",
                "summary": "Minor issues detected",
            }
            
            result = execution_validator_node(execution_state)
            
            # Warning is not explicitly handled in the node - verify behavior
            assert result["execution_verdict"] == "warning"
            # Warning should not trigger failure count increment
            assert "execution_failure_count" not in result or result.get("execution_failure_count", 0) == 0

    def test_timeout_with_skip_with_warning_fallback(self, execution_state):
        """Test that timeout with skip_with_warning fallback passes."""
        execution_state["stage_outputs"]["timeout_exceeded"] = True
        execution_state["run_error"] = "Exceeded timeout of 3600s"
        # Note: fallback_strategy is a direct property of the stage, not nested in design_spec
        execution_state["plan"] = {
            "stages": [{
                "stage_id": "stage_1_main",
                "fallback_strategy": "skip_with_warning",
            }]
        }
        
        # No LLM call should be made for timeout with skip_with_warning
        result = execution_validator_node(execution_state)
        
        assert result["execution_verdict"] == "pass"
        assert "skip_with_warning" in result["execution_feedback"]
        assert "timed out" in result["execution_feedback"].lower()

    def test_timeout_without_skip_with_warning_fails(self, execution_state):
        """Test that timeout without skip_with_warning fallback fails."""
        execution_state["stage_outputs"]["timeout_exceeded"] = True
        execution_state["run_error"] = "Exceeded timeout of 3600s"
        # Default fallback is "ask_user", not "skip_with_warning"
        
        result = execution_validator_node(execution_state)
        
        assert result["execution_verdict"] == "fail"
        assert "timed out" in result["execution_feedback"].lower()

    def test_timeout_detected_from_error_string(self, execution_state):
        """Test that timeout is detected from run_error string pattern."""
        execution_state["stage_outputs"]["timeout_exceeded"] = False
        execution_state["run_error"] = "Process exceeded timeout limit"
        
        result = execution_validator_node(execution_state)
        
        assert result["execution_verdict"] == "fail"
        assert "timed out" in result["execution_feedback"].lower()

    def test_missing_verdict_defaults_to_pass(self, execution_state):
        """Test that missing verdict in LLM response defaults to pass."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                # No verdict key!
                "summary": "Analysis complete",
            }
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_verdict"] == "pass"
            assert "Missing verdict" in result["execution_feedback"]
            assert "Original summary" in result["execution_feedback"]

    def test_missing_verdict_and_summary(self, execution_state):
        """Test that missing verdict and summary are handled."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {}  # Empty response
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_verdict"] == "pass"
            assert "Missing verdict" in result["execution_feedback"]

    def test_llm_error_auto_approves(self, execution_state):
        """Test that LLM error causes auto-approve."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API rate limit exceeded")
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_verdict"] == "pass"
            assert "Auto-approved" in result["execution_feedback"] or "LLM error" in result["execution_feedback"].lower()

    def test_run_error_injected_into_prompt(self, execution_state):
        """Test that run_error is injected into the system prompt."""
        execution_state["run_error"] = "ImportError: No module named meep"
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm, \
             patch("src.agents.execution.build_agent_prompt") as mock_prompt:
            mock_prompt.return_value = "Base prompt"
            mock_llm.return_value = {"verdict": "fail", "summary": "Module error"}
            
            execution_validator_node(execution_state)
            
            # Verify build_agent_prompt was called
            mock_prompt.assert_called_once()
            # Verify LLM was called with prompt containing run_error
            call_args = mock_llm.call_args
            system_prompt = call_args.kwargs.get("system_prompt", call_args.args[1] if len(call_args.args) > 1 else "")
            assert "ImportError" in system_prompt or "module named meep" in system_prompt

    def test_handles_none_total_execution_failures(self, execution_state):
        """Test that None total_execution_failures is handled."""
        execution_state["total_execution_failures"] = None
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            
            result = execution_validator_node(execution_state)
            
            assert result["total_execution_failures"] == 1

    def test_awaiting_user_input_returns_empty(self, execution_state):
        """Test that awaiting_user_input state returns empty result."""
        execution_state["awaiting_user_input"] = True
        
        result = execution_validator_node(execution_state)
        
        assert result == {}

    def test_stage_id_included_in_agent_output(self, execution_state):
        """Test that stage_id is included in processed agent output."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            # The stage_id is used internally - verify via the prompt
            execution_validator_node(execution_state)
            
            # Verify the user_content sent to LLM contains stage_id
            call_kwargs = mock_llm.call_args.kwargs
            user_content = call_kwargs.get("user_content", "")
            assert "stage_1_main" in user_content


# =============================================================================
# Unit Tests: physics_sanity_node
# =============================================================================

class TestPhysicsSanityNode:
    """Unit tests for physics_sanity_node."""

    def test_pass_verdict_sets_correct_state(self, physics_state):
        """Test that pass verdict sets workflow_phase and physics_verdict correctly."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "pass",
                "summary": "Physics looks reasonable",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["workflow_phase"] == "physics_validation"
            assert result["physics_verdict"] == "pass"
            assert result["physics_feedback"] == "Physics looks reasonable"
            # Should NOT increment failure count on pass
            assert "physics_failure_count" not in result

    def test_fail_verdict_increments_failure_count(self, physics_state):
        """Test that fail verdict increments physics_failure_count."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Negative energy detected",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "fail"
            assert result["physics_failure_count"] == 1

    def test_fail_verdict_increments_from_existing_count(self, physics_state):
        """Test that fail verdict increments from existing count."""
        physics_state["physics_failure_count"] = 1
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Still wrong",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_failure_count"] == 2

    def test_design_flaw_verdict_increments_design_revision_count(self, physics_state):
        """Test that design_flaw verdict increments design_revision_count."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "design_flaw",
                "summary": "Resolution too low for plasmon",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "design_flaw"
            assert result["design_revision_count"] == 1
            assert result["design_feedback"] == "Resolution too low for plasmon"

    def test_design_flaw_increments_from_existing_design_count(self, physics_state):
        """Test that design_flaw increments from existing design_revision_count."""
        physics_state["design_revision_count"] = 2
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "design_flaw",
                "summary": "Wrong cell size",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["design_revision_count"] == 3

    def test_warning_verdict_does_not_increment_counters(self, physics_state):
        """Test that warning verdict does not increment any counters."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "warning",
                "summary": "Minor discrepancy noted",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "warning"
            # Warning should not trigger counter increments
            assert "physics_failure_count" not in result
            assert "design_revision_count" not in result

    def test_missing_verdict_defaults_to_pass(self, physics_state):
        """Test that missing verdict in LLM response defaults to pass."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "summary": "Analysis complete",
            }
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "pass"
            assert "Missing verdict" in result["physics_feedback"]
            assert "Original summary" in result["physics_feedback"]

    def test_missing_verdict_and_summary(self, physics_state):
        """Test that missing verdict and summary are handled."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {}  # Empty response
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "pass"
            assert "Missing verdict" in result["physics_feedback"]

    def test_llm_error_auto_approves(self, physics_state):
        """Test that LLM error causes auto-approve."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API error")
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "pass"
            assert "Auto-approved" in result["physics_feedback"] or "LLM error" in result["physics_feedback"].lower()

    def test_backtrack_suggestion_propagated(self, physics_state):
        """Test that backtrack_suggestion from LLM is propagated to result."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Critical error",
                "backtrack_suggestion": {
                    "suggest_backtrack": True,
                    "target_stage_id": "stage_0_materials",
                    "reason": "Material properties wrong",
                },
            }
            
            result = physics_sanity_node(physics_state)
            
            assert "backtrack_suggestion" in result
            assert result["backtrack_suggestion"]["suggest_backtrack"] is True
            assert result["backtrack_suggestion"]["target_stage_id"] == "stage_0_materials"

    def test_backtrack_suggestion_not_propagated_when_false(self, physics_state):
        """Test that backtrack_suggestion is not propagated when suggest_backtrack is False."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "pass",
                "summary": "OK",
                "backtrack_suggestion": {"suggest_backtrack": False},
            }
            
            result = physics_sanity_node(physics_state)
            
            # Should not have backtrack_suggestion in result when False
            assert "backtrack_suggestion" not in result

    def test_design_description_included_in_prompt(self, physics_state):
        """Test that design_description is included in user content."""
        physics_state["design_description"] = {"geometry": {"resolution": 20}}
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            physics_sanity_node(physics_state)
            
            call_kwargs = mock_llm.call_args.kwargs
            user_content = call_kwargs.get("user_content", "")
            assert "resolution" in user_content or "Design Spec" in user_content

    def test_empty_design_description_dict_excluded(self, physics_state):
        """Test that empty dict design_description is excluded (no value to show)."""
        physics_state["design_description"] = {}
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            physics_sanity_node(physics_state)
            
            call_kwargs = mock_llm.call_args.kwargs
            user_content = call_kwargs.get("user_content", "")
            # Empty dict should not be included (no useful information)
            assert "Design Spec" not in user_content

    def test_awaiting_user_input_returns_empty(self, physics_state):
        """Test that awaiting_user_input state returns empty result."""
        physics_state["awaiting_user_input"] = True
        
        result = physics_sanity_node(physics_state)
        
        assert result == {}


# =============================================================================
# Unit Tests: Routing Functions
# =============================================================================

class TestRouteAfterExecutionCheck:
    """Unit tests for route_after_execution_check."""

    def test_pass_verdict_routes_to_physics_check(self):
        """Test that pass verdict routes to physics_check."""
        state = {"execution_verdict": "pass", "execution_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "physics_check"

    def test_warning_verdict_routes_to_physics_check(self):
        """Test that warning verdict routes to physics_check."""
        state = {"execution_verdict": "warning", "execution_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "physics_check"

    def test_fail_verdict_routes_to_generate_code(self):
        """Test that fail verdict routes to generate_code."""
        state = {"execution_verdict": "fail", "execution_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "generate_code"

    def test_fail_verdict_at_max_routes_to_ask_user(self):
        """Test that fail verdict at max count routes to ask_user."""
        state = {
            "execution_verdict": "fail",
            "execution_failure_count": MAX_EXECUTION_FAILURES,
        }
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "ask_user"

    def test_none_verdict_routes_to_ask_user(self):
        """Test that None verdict routes to ask_user."""
        state = {"execution_verdict": None, "execution_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "ask_user"

    def test_unknown_verdict_routes_to_ask_user(self):
        """Test that unknown verdict routes to ask_user."""
        state = {"execution_verdict": "unknown_verdict", "execution_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "ask_user"

    def test_custom_max_from_runtime_config(self):
        """Test that custom max from runtime_config is respected."""
        state = {
            "execution_verdict": "fail",
            "execution_failure_count": 5,
            "runtime_config": {"max_execution_failures": 5},
        }
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "ask_user"

    def test_invalid_verdict_type_routes_to_ask_user(self):
        """Test that non-string verdict routes to ask_user."""
        state = {"execution_verdict": 123, "execution_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_execution_check(state)
        
        assert result == "ask_user"


class TestRouteAfterPhysicsCheck:
    """Unit tests for route_after_physics_check."""

    def test_pass_verdict_routes_to_analyze(self):
        """Test that pass verdict routes to analyze."""
        state = {"physics_verdict": "pass", "physics_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "analyze"

    def test_warning_verdict_routes_to_analyze(self):
        """Test that warning verdict routes to analyze."""
        state = {"physics_verdict": "warning", "physics_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "analyze"

    def test_fail_verdict_routes_to_generate_code(self):
        """Test that fail verdict routes to generate_code."""
        state = {"physics_verdict": "fail", "physics_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "generate_code"

    def test_fail_verdict_at_max_routes_to_ask_user(self):
        """Test that fail verdict at max count routes to ask_user."""
        state = {
            "physics_verdict": "fail",
            "physics_failure_count": MAX_PHYSICS_FAILURES,
        }
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "ask_user"

    def test_design_flaw_verdict_routes_to_design(self):
        """Test that design_flaw verdict routes to design."""
        state = {"physics_verdict": "design_flaw", "design_revision_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "design"

    def test_design_flaw_at_max_routes_to_ask_user(self):
        """Test that design_flaw at max design revisions routes to ask_user."""
        state = {
            "physics_verdict": "design_flaw",
            "design_revision_count": MAX_DESIGN_REVISIONS,
        }
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "ask_user"

    def test_none_verdict_routes_to_ask_user(self):
        """Test that None verdict routes to ask_user."""
        state = {"physics_verdict": None, "physics_failure_count": 0}
        
        with patch("src.routing.save_checkpoint"):
            result = route_after_physics_check(state)
        
        assert result == "ask_user"


# =============================================================================
# Graph Integration Tests
# =============================================================================

class TestExecutionPhaseGraphFlow:
    """Test graph flow through execution phase."""

    def test_execution_success_flow(self, initial_state):
        """Test: run_code → execution_check(pass) → physics_check"""
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"workflow_phase": "running_code", "execution_output": "Simulating..."},
        ):
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("execution")}}

            nodes_visited = []

            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)

                    if node_name == "physics_check":
                        break
                else:
                    continue
                break

            assert "run_code" in nodes_visited
            assert "execution_check" in nodes_visited
            assert "physics_check" in nodes_visited

            # Verify order: run_code must come before execution_check
            run_code_idx = nodes_visited.index("run_code")
            exec_check_idx = nodes_visited.index("execution_check")
            physics_idx = nodes_visited.index("physics_check")
            assert run_code_idx < exec_check_idx < physics_idx

    def test_execution_fail_routes_to_generate_code(self, initial_state):
        """Test: execution_check(fail) → generate_code"""
        visited = []
        code_generator_call_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal code_generator_call_count
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "code_generator":
                code_generator_call_count += 1

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_fail(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"run_error": "Simulation crashed"},
        ):
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("exec_fail")}}

            nodes_visited = []
            iterations = 0
            max_iterations = 50

            for event in graph.stream(initial_state, config):
                iterations += 1
                if iterations > max_iterations:
                    break
                    
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)

                    # Stop after we see the second code_generator call
                    if code_generator_call_count >= 2:
                        break
                else:
                    continue
                break

            # Should have visited execution_check and then gone back to generate_code
            assert "execution_check" in nodes_visited
            # Code generator should be called multiple times (initial + retry after fail)
            assert code_generator_call_count >= 2

    def test_physics_pass_routes_to_analyze(self, initial_state):
        """Test: physics_check(pass) → analyze"""
        visited = []

        def mock_llm(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_pass(),
                "results_analyzer": MockLLMResponses.results_analyzer(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"stage_outputs": {"stdout": "OK"}},
        ):
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("physics_pass")}}

            nodes_visited = []

            for event in graph.stream(initial_state, config):
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)

                    if node_name == "analyze":
                        break
                else:
                    continue
                break

            assert "physics_check" in nodes_visited
            assert "analyze" in nodes_visited
            
            # Verify order
            physics_idx = nodes_visited.index("physics_check")
            analyze_idx = nodes_visited.index("analyze")
            assert physics_idx < analyze_idx

    def test_physics_design_flaw_routes_to_design(self, initial_state):
        """Test: physics_check(design_flaw) → design"""
        visited = []
        designer_call_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal designer_call_count
            agent = kwargs.get("agent_name", "unknown")
            visited.append(agent)

            if agent == "simulation_designer":
                designer_call_count += 1

            responses = {
                "prompt_adaptor": MockLLMResponses.prompt_adaptor(),
                "planner": MockLLMResponses.planner(),
                "plan_reviewer": MockLLMResponses.plan_reviewer_approve(),
                "simulation_designer": MockLLMResponses.simulation_designer(),
                "design_reviewer": MockLLMResponses.design_reviewer_approve(),
                "code_generator": MockLLMResponses.code_generator(),
                "code_reviewer": MockLLMResponses.code_reviewer_approve(),
                "execution_validator": MockLLMResponses.execution_validator_pass(),
                "physics_sanity": MockLLMResponses.physics_sanity_design_flaw(),
            }
            return responses.get(agent, {})

        with MultiPatch(LLM_PATCH_LOCATIONS, side_effect=mock_llm), MultiPatch(
            CHECKPOINT_PATCH_LOCATIONS, return_value="/tmp/cp.json"
        ), patch(
            "src.code_runner.run_code_node",
            return_value={"stage_outputs": {"stdout": "OK"}},
        ):
            graph = create_repro_graph()
            config = {"configurable": {"thread_id": unique_thread_id("physics_flaw")}}

            nodes_visited = []
            iterations = 0
            max_iterations = 50

            for event in graph.stream(initial_state, config):
                iterations += 1
                if iterations > max_iterations:
                    break
                    
                for node_name, _ in event.items():
                    nodes_visited.append(node_name)

                    # Stop after we see the second designer call
                    if designer_call_count >= 2:
                        break
                else:
                    continue
                break

            assert "physics_check" in nodes_visited
            # Designer should be called multiple times (initial + retry after design_flaw)
            assert designer_call_count >= 2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestExecutionEdgeCases:
    """Test edge cases in execution validation."""

    def test_empty_stage_outputs_handled(self, execution_state):
        """Test that empty stage_outputs is handled gracefully."""
        execution_state["stage_outputs"] = {}
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_verdict"] == "pass"

    def test_none_stage_outputs_handled(self, execution_state):
        """Test that None stage_outputs is handled gracefully."""
        execution_state["stage_outputs"] = None
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            result = execution_validator_node(execution_state)
            
            assert result["execution_verdict"] == "pass"

    def test_custom_runtime_config_max_failures(self, execution_state):
        """Test that custom max_execution_failures from runtime_config is respected."""
        execution_state["runtime_config"] = {"max_execution_failures": 5}
        execution_state["execution_failure_count"] = 4
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            
            result = execution_validator_node(execution_state)
            
            # Should increment to 5 (the custom max)
            assert result["execution_failure_count"] == 5
            # Should NOT trigger ask_user yet since we just hit the limit
            # Actually, it SHOULD trigger ask_user because count=5 >= max=5
            # Let me re-check the logic...
            # Looking at the code: if not was_incremented (at max), trigger ask_user
            # At count=4, incrementing to 5 means we're now AT the max
            # The next failure would trigger ask_user

    def test_missing_current_stage_id_uses_unknown(self, execution_state):
        """Test that missing current_stage_id defaults to 'unknown'."""
        del execution_state["current_stage_id"]
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            result = execution_validator_node(execution_state)
            
            # Should not crash
            assert result["execution_verdict"] == "pass"


class TestPhysicsEdgeCases:
    """Test edge cases in physics validation."""

    def test_empty_stage_outputs_handled(self, physics_state):
        """Test that empty stage_outputs is handled gracefully."""
        physics_state["stage_outputs"] = {}
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "pass"

    def test_none_design_description_handled(self, physics_state):
        """Test that None design_description is handled gracefully."""
        physics_state["design_description"] = None
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            result = physics_sanity_node(physics_state)
            
            assert result["physics_verdict"] == "pass"

    def test_string_design_description_handled(self, physics_state):
        """Test that string design_description is handled."""
        physics_state["design_description"] = "Simple 2D simulation"
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "summary": "OK"}
            
            physics_sanity_node(physics_state)
            
            call_kwargs = mock_llm.call_args.kwargs
            user_content = call_kwargs.get("user_content", "")
            assert "Simple 2D simulation" in user_content

    def test_backtrack_suggestion_as_non_dict_handled(self, physics_state):
        """Test that non-dict backtrack_suggestion is handled."""
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Issue",
                "backtrack_suggestion": "Should backtrack",  # String instead of dict
            }
            
            result = physics_sanity_node(physics_state)
            
            # Should not crash and not propagate invalid backtrack_suggestion
            assert result["physics_verdict"] == "fail"
            assert "backtrack_suggestion" not in result


# =============================================================================
# Counter Increment Logic Tests
# =============================================================================

class TestCounterIncrementLogic:
    """Test counter increment logic in execution nodes."""

    def test_execution_failure_count_bounded_by_max(self, execution_state):
        """Test that execution_failure_count does not exceed max."""
        execution_state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            
            result = execution_validator_node(execution_state)
            
            # Count should stay at max, not increase
            assert result["execution_failure_count"] == MAX_EXECUTION_FAILURES
            # Should trigger ask_user
            assert result["ask_user_trigger"] == "execution_failure_limit"

    def test_physics_failure_count_bounded_by_max(self, physics_state):
        """Test that physics_failure_count does not exceed max."""
        physics_state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            
            result = physics_sanity_node(physics_state)
            
            # Count should stay at max, not increase
            assert result["physics_failure_count"] == MAX_PHYSICS_FAILURES

    def test_design_revision_count_bounded_by_max(self, physics_state):
        """Test that design_revision_count does not exceed max."""
        physics_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "design_flaw", "summary": "Design issue"}
            
            result = physics_sanity_node(physics_state)
            
            # Count should stay at max, not increase
            assert result["design_revision_count"] == MAX_DESIGN_REVISIONS

    def test_total_execution_failures_always_increments(self, execution_state):
        """Test that total_execution_failures always increments regardless of max."""
        execution_state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        execution_state["total_execution_failures"] = 100
        
        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "fail", "summary": "Failed"}
            
            result = execution_validator_node(execution_state)
            
            # Total should always increment
            assert result["total_execution_failures"] == 101
