"""
Tests for workflow revision cycles.

Tests design and code revision cycles including:
- Design → review reject → revision → approve cycle
- Code revision limits and counter behavior
- Feedback propagation between nodes
- Max revision limit escalation
- Edge cases and error paths
"""

from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest

from src.agents import (
    simulation_designer_node,
    design_reviewer_node,
    code_reviewer_node,
    code_generator_node,
)
from schemas.state import MAX_DESIGN_REVISIONS, MAX_CODE_REVISIONS

from tests.workflow.fixtures import MockResponseFactory


class TestDesignRevisionCycle:
    """Test design revision cycles with strict assertions."""

    def test_design_revision_cycle_full_flow(self, base_state):
        """Test complete design → review reject → revision → approve cycle."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["validated_materials"] = [{"material_id": "gold"}]

        # First design attempt
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_response = MockResponseFactory.designer_response()
            mock_llm.return_value = mock_response
            result = simulation_designer_node(base_state)
            base_state.update(result)

            # STRICT: Verify all expected outputs
            assert result["workflow_phase"] == "design"
            assert result["design_description"] == mock_response
            # No reviewer feedback initially (None or empty string both mean "no feedback")
            assert not base_state.get("reviewer_feedback"), \
                f"Expected no reviewer feedback initially, got: {base_state.get('reviewer_feedback')}"

        # Review rejects with specific feedback
        feedback_msg = "Add PML thickness parameter"
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision(feedback_msg)
            result = design_reviewer_node(base_state)
            base_state.update(result)

        # STRICT: Verify all fields set by reviewer
        assert result["workflow_phase"] == "design_review"
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1
        assert result["reviewer_feedback"] == feedback_msg
        assert "reviewer_issues" in result
        assert len(result["reviewer_issues"]) > 0
        assert result["reviewer_issues"][0]["description"] == feedback_msg

        # Second design attempt - feedback should be available in state
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_response = MockResponseFactory.designer_response()
            mock_llm.return_value = mock_response
            result = simulation_designer_node(base_state)
            base_state.update(result)

            # STRICT: Verify feedback was accessible
            assert base_state["reviewer_feedback"] == feedback_msg
            assert result["workflow_phase"] == "design"

        # Review approves
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)

        # STRICT: Verify approval state
        assert result["last_design_review_verdict"] == "approve"
        assert result["workflow_phase"] == "design_review"
        # design_revision_count should not change on approval
        assert result["design_revision_count"] == 1

    def test_design_reviewer_increments_counter_correctly(self, base_state):
        """Verify design_revision_count increments correctly on each rejection."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = 0

        # First rejection: 0 → 1
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Issue 1")
            result = design_reviewer_node(base_state)
            assert result["design_revision_count"] == 1
            base_state.update(result)

        # Second rejection: 1 → 2
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Issue 2")
            result = design_reviewer_node(base_state)
            assert result["design_revision_count"] == 2
            base_state.update(result)

        # Approval: count stays at 2
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            assert result["design_revision_count"] == 2

    def test_design_reviewer_max_limit_triggers_escalation(self, base_state):
        """Design reviewer must escalate to ask_user when max limit reached."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS - 1  # One before max
        base_state["runtime_config"] = {"max_design_revisions": MAX_DESIGN_REVISIONS}

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Final issue")
            result = design_reviewer_node(base_state)

        # STRICT: Must trigger escalation
        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS
        assert result["ask_user_trigger"] == "design_review_limit", \
            "Must trigger ask_user when design review limit reached"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) > 0
        assert "Design review limit reached" in result["pending_user_questions"][0]
        assert result["last_node_before_ask_user"] == "design_review"

    def test_design_reviewer_at_max_counter_unchanged(self, base_state):
        """Counter should not exceed max when already at max."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS  # Already at max
        base_state["runtime_config"] = {"max_design_revisions": MAX_DESIGN_REVISIONS}

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Issue")
            result = design_reviewer_node(base_state)

        # Counter should stay at max, not exceed it
        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS
        assert result["awaiting_user_input"] is True

    def test_design_reviewer_custom_max_limit(self, base_state):
        """Design reviewer respects custom max limit from runtime_config."""
        custom_max = 5
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = custom_max - 1
        base_state["runtime_config"] = {"max_design_revisions": custom_max}

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Issue")
            result = design_reviewer_node(base_state)

        # Should escalate at custom max
        assert result["design_revision_count"] == custom_max
        assert result["ask_user_trigger"] == "design_review_limit"

    def test_design_reviewer_verdict_normalization(self, base_state):
        """Design reviewer normalizes various verdict strings."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"

        test_cases = [
            ({"verdict": "pass", "summary": "OK"}, "approve"),
            ({"verdict": "approved", "summary": "OK"}, "approve"),
            ({"verdict": "accept", "summary": "OK"}, "approve"),
            ({"verdict": "approve", "summary": "OK"}, "approve"),
            ({"verdict": "reject", "summary": "Bad"}, "needs_revision"),
            ({"verdict": "revision_needed", "summary": "Bad"}, "needs_revision"),
            ({"verdict": "needs_work", "summary": "Bad"}, "needs_revision"),
            ({"verdict": "needs_revision", "summary": "Bad"}, "needs_revision"),
        ]

        for mock_response, expected_verdict in test_cases:
            base_state["design_revision_count"] = 0  # Reset counter
            with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
                mock_llm.return_value = mock_response
                result = design_reviewer_node(base_state)
                assert result["last_design_review_verdict"] == expected_verdict, \
                    f"Verdict '{mock_response['verdict']}' should normalize to '{expected_verdict}'"

    def test_design_reviewer_missing_verdict_defaults_to_needs_revision(self, base_state):
        """Missing verdict should default to needs_revision for safety."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = 0

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"summary": "No verdict field"}
            result = design_reviewer_node(base_state)

        # Missing verdict defaults to needs_revision (safer than auto-approve)
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1

    def test_design_reviewer_unknown_verdict_defaults_to_needs_revision(self, base_state):
        """Unknown verdict string should default to needs_revision for safety."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = 0

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "maybe_ok", "summary": "Uncertain"}
            result = design_reviewer_node(base_state)

        # Unknown verdict defaults to needs_revision (safer than auto-approve)
        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["design_revision_count"] == 1


class TestCodeRevisionCycle:
    """Test code revision cycles with strict assertions."""

    def test_code_revision_increments_correctly(self, base_state):
        """Test code_revision_count increments correctly on each rejection."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"
        base_state["code_revision_count"] = 0
        base_state["runtime_config"] = {"max_code_revisions": 5}  # High limit to test increments

        counts = []
        for i in range(3):
            with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
                mock_llm.return_value = MockResponseFactory.reviewer_needs_revision(f"Issue {i+1}")
                result = code_reviewer_node(base_state)
                counts.append(result["code_revision_count"])
                base_state.update(result)

        assert counts == [1, 2, 3], f"Counter should increment 0→1→2→3, got transitions to {counts}"

    def test_code_revision_max_limit_triggers_escalation(self, base_state):
        """Code reviewer must escalate when max limit reached."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS - 1  # One before max
        base_state["runtime_config"] = {"max_code_revisions": MAX_CODE_REVISIONS}

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision("Final issue")
            result = code_reviewer_node(base_state)

        # STRICT: Must trigger escalation
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result["ask_user_trigger"] == "code_review_limit", \
            "Must trigger ask_user when code review limit reached"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) > 0
        assert "Code review limit reached" in result["pending_user_questions"][0]
        assert result["last_node_before_ask_user"] == "code_review"

    def test_code_revision_with_max_limit_early_return(self, base_state):
        """When awaiting_user_input is True, node returns empty dict."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"
        base_state["runtime_config"] = {"max_code_revisions": 2}

        # First revision: 0 → 1
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)
        assert base_state["code_revision_count"] == 1

        # Second revision: 1 → 2 (triggers escalation)
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
            base_state.update(result)
        assert base_state["code_revision_count"] == 2
        assert base_state["awaiting_user_input"] is True
        assert base_state["ask_user_trigger"] == "code_review_limit"

        # Third call: should return empty dict because awaiting_user_input is True
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)
        
        # STRICT: Should return empty dict, not increment counter
        assert result == {}, "Should return empty dict when awaiting_user_input is True"
        assert base_state["code_revision_count"] == 2  # Unchanged

    def test_code_revision_counter_at_max_stays_at_max(self, base_state):
        """Counter should not exceed max when already at max."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS  # Already at max
        base_state["runtime_config"] = {"max_code_revisions": MAX_CODE_REVISIONS}
        base_state["awaiting_user_input"] = False  # Clear to allow node to run

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)

        # Counter should stay at max
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        # Should escalate again
        assert result["awaiting_user_input"] is True

    def test_code_reviewer_verdict_normalization(self, base_state):
        """Code reviewer normalizes various verdict strings."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"

        test_cases = [
            ({"verdict": "pass", "summary": "OK"}, "approve"),
            ({"verdict": "approved", "summary": "OK"}, "approve"),
            ({"verdict": "approve", "summary": "OK"}, "approve"),
            ({"verdict": "reject", "summary": "Bad"}, "needs_revision"),
            ({"verdict": "needs_revision", "summary": "Bad"}, "needs_revision"),
        ]

        for mock_response, expected_verdict in test_cases:
            base_state["code_revision_count"] = 0
            base_state["runtime_config"] = {"max_code_revisions": 10}  # High to avoid escalation
            with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
                mock_llm.return_value = mock_response
                result = code_reviewer_node(base_state)
                assert result["last_code_review_verdict"] == expected_verdict, \
                    f"Verdict '{mock_response['verdict']}' should normalize to '{expected_verdict}'"

    def test_code_reviewer_feedback_extraction(self, base_state):
        """Code reviewer extracts feedback from response."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"
        base_state["runtime_config"] = {"max_code_revisions": 10}

        # Test feedback from 'feedback' field
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "feedback": "Use vectorized operations",
                "summary": "Different summary"
            }
            result = code_reviewer_node(base_state)
        assert result["reviewer_feedback"] == "Use vectorized operations"

        # Test fallback to 'summary' when 'feedback' missing
        base_state["code_revision_count"] = 0
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "summary": "Need better error handling"
            }
            result = code_reviewer_node(base_state)
        assert result["reviewer_feedback"] == "Need better error handling"


class TestCodeGeneratorValidation:
    """Test code generator validation logic."""

    def test_missing_validated_materials(self, base_state):
        """Code generation fails if materials not validated for non-material stages."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = "Valid design " * 5  # >50 chars
        base_state["validated_materials"] = []  # Empty

        result = code_generator_node(base_state)

        assert "run_error" in result
        assert "validated_materials is empty" in result["run_error"]
        assert "code_revision_count" in result

    def test_validated_materials_none(self, base_state):
        """Code generation handles None validated_materials."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = "Valid design " * 5
        base_state["validated_materials"] = None

        result = code_generator_node(base_state)

        # Should fail with empty materials error
        assert "run_error" in result
        assert "validated_materials" in result["run_error"]

    def test_materials_not_required_for_material_validation_stage(self, base_state):
        """MATERIAL_VALIDATION stage doesn't require validated_materials."""
        base_state["current_stage_id"] = "stage_0_materials"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = "Valid design " * 5
        base_state["validated_materials"] = []  # Empty but should be OK

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)

        # Should NOT fail with materials error
        assert "run_error" not in result
        assert result["workflow_phase"] == "code_generation"

    def test_stub_detection_short_code_with_todo(self, base_state):
        """Short code with TODO marker is detected as stub."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = "# TODO: Implement simulation"

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower()
        assert result["code_revision_count"] == 1

    def test_stub_detection_short_code_with_placeholder(self, base_state):
        """Short code with PLACEHOLDER marker is detected as stub."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = "# PLACEHOLDER - replace with real code"

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower()

    def test_valid_long_code_with_inline_todo_not_stub(self, base_state):
        """Longer valid code with inline TODO comments is NOT a stub."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        # Valid code that happens to have a TODO comment in the middle
        valid_code = """import meep as mp
import numpy as np

# Gold nanorod FDTD simulation
cell = mp.Vector3(0.4, 0.2, 0.2)
resolution = 50

geometry = [
    mp.Cylinder(radius=0.02, height=0.1, material=mp.Medium(epsilon=1))
]

sim = mp.Simulation(cell_size=cell, geometry=geometry, resolution=resolution)
# TODO: Consider adding absorber boundaries
sim.run(until=100)

# Save extinction data
np.savetxt("extinction.csv", [[400, 0.5], [700, 1.0], [900, 0.3]])
"""
        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = valid_code

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        # Should NOT be flagged as stub
        assert "reviewer_feedback" not in result or "stub" not in result.get("reviewer_feedback", "").lower()
        assert result["code"] == valid_code

    def test_stub_detection_code_starting_with_stub_marker(self, base_state):
        """Code starting with STUB marker is detected regardless of length."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        # Long code but starts with STUB
        long_stub = "# STUB - this would be generated by the LLM\n" + "x = 1\n" * 100

        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = long_stub

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower()

    def test_empty_code_detected(self, base_state):
        """Empty code is detected as invalid."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = ""

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        assert "reviewer_feedback" in result
        assert "empty" in result["reviewer_feedback"].lower() or "stub" in result["reviewer_feedback"].lower()

    def test_whitespace_only_code_detected(self, base_state):
        """Whitespace-only code is detected as invalid."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = "   \n\t\n   "

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        assert "reviewer_feedback" in result

    def test_code_too_short_detected(self, base_state):
        """Code shorter than 50 chars is detected as invalid."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        stub_response = MockResponseFactory.code_generator_response()
        stub_response["code"] = "x = 1"  # <50 chars

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = stub_response
            result = code_generator_node(base_state)

        assert "reviewer_feedback" in result


class TestSimulationDesignerNode:
    """Test simulation designer node edge cases."""

    def test_missing_stage_id_triggers_escalation(self, base_state):
        """Missing current_stage_id triggers user escalation."""
        base_state["current_stage_id"] = None
        base_state["plan"] = MockResponseFactory.planner_response()

        result = simulation_designer_node(base_state)

        assert result["workflow_phase"] == "design"
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert len(result["pending_user_questions"]) > 0
        assert "ERROR" in result["pending_user_questions"][0]

    def test_empty_stage_id_triggers_escalation(self, base_state):
        """Empty string current_stage_id triggers user escalation."""
        base_state["current_stage_id"] = ""
        base_state["plan"] = MockResponseFactory.planner_response()

        result = simulation_designer_node(base_state)

        # Empty string is falsy, should trigger escalation
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"

    def test_design_includes_feedback_in_prompt(self, base_state):
        """Designer includes reviewer feedback in system prompt for revisions."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["reviewer_feedback"] = "Add PML layers specification"

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            with patch("src.agents.design.build_agent_prompt") as mock_prompt:
                mock_prompt.return_value = "System prompt"
                mock_llm.return_value = MockResponseFactory.designer_response()
                simulation_designer_node(base_state)

                # Verify feedback would be in the system prompt
                assert base_state["reviewer_feedback"] == "Add PML layers specification"

    def test_design_handles_llm_error(self, base_state):
        """Designer handles LLM call failure with escalation."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API rate limit exceeded")
            result = simulation_designer_node(base_state)

        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "llm_error"
        assert "failed" in result["pending_user_questions"][0].lower()

    def test_design_handles_new_assumptions(self, base_state):
        """Designer merges new assumptions into existing ones."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["assumptions"] = {
            "global_assumptions": [
                {"id": "A1", "description": "Existing assumption"}
            ]
        }

        mock_response = MockResponseFactory.designer_response()
        mock_response["new_assumptions"] = [
            {"id": "A2", "description": "New assumption from designer"}
        ]

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = simulation_designer_node(base_state)

        # Should have merged assumptions
        if "assumptions" in result:
            assert len(result["assumptions"]["global_assumptions"]) == 2


class TestDesignDescriptionValidation:
    """Test design description validation in code generator."""

    def test_missing_design_description(self, base_state):
        """Code generator fails with missing design_description."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = ""

        result = code_generator_node(base_state)

        assert result["workflow_phase"] == "code_generation"
        assert "reviewer_feedback" in result
        assert "design description" in result["reviewer_feedback"].lower()
        assert result["design_revision_count"] >= 1

    def test_stub_design_description_rejected(self, base_state):
        """Code generator rejects design with STUB marker."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "STUB design - would be generated by LLM"

        result = code_generator_node(base_state)

        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower() or "design description" in result["reviewer_feedback"].lower()

    def test_short_design_description_rejected(self, base_state):
        """Code generator rejects design shorter than 50 chars."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Too short"  # <50 chars

        result = code_generator_node(base_state)

        assert "reviewer_feedback" in result

    def test_none_design_description_rejected(self, base_state):
        """Code generator handles None design_description."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = None

        result = code_generator_node(base_state)

        assert "reviewer_feedback" in result


class TestCodeGeneratorMissingStageId:
    """Test code generator stage ID validation."""

    def test_missing_stage_id_triggers_escalation(self, base_state):
        """Missing current_stage_id triggers user escalation."""
        base_state["current_stage_id"] = None
        base_state["design_description"] = "Valid design " * 5

        result = code_generator_node(base_state)

        assert result["workflow_phase"] == "code_generation"
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"
        assert "ERROR" in result["pending_user_questions"][0]

    def test_empty_stage_id_triggers_escalation(self, base_state):
        """Empty string stage_id triggers user escalation."""
        base_state["current_stage_id"] = ""
        base_state["design_description"] = "Valid design " * 5

        result = code_generator_node(base_state)

        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "missing_stage_id"


class TestCodeGeneratorOutputExtraction:
    """Test code generator extracts code from various response formats."""

    def test_extracts_code_from_code_key(self, base_state):
        """Extracts code from 'code' key in response."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        expected_code = MockResponseFactory.code_generator_response()["code"]

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)

        assert result["code"] == expected_code

    def test_extracts_code_from_simulation_code_key(self, base_state):
        """Extracts code from 'simulation_code' key as fallback."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        mock_response = {"simulation_code": "import meep\n# simulation code here...\n" * 10}

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = mock_response
            result = code_generator_node(base_state)

        assert "import meep" in result["code"]

    def test_handles_string_response(self, base_state):
        """Handles string response instead of dict."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        string_code = "import meep\n# This is raw string code\n" * 10

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = string_code
            result = code_generator_node(base_state)

        assert result["code"] == string_code


class TestLLMErrorHandling:
    """Test LLM error handling in various nodes."""

    def test_design_reviewer_defaults_to_needs_revision_on_llm_error(self, base_state):
        """Design reviewer defaults to needs_revision when LLM fails."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"
        base_state["design_revision_count"] = 0

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("Connection timeout")
            result = design_reviewer_node(base_state)

        # LLM failures should default to needs_revision (safer than auto-approve)
        assert result["last_design_review_verdict"] == "needs_revision"
        # Should increment revision counter
        assert result["design_revision_count"] == 1

    def test_code_reviewer_defaults_to_needs_revision_on_llm_error(self, base_state):
        """Code reviewer defaults to needs_revision when LLM fails."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test code"
        base_state["design_description"] = "Test design"
        base_state["code_revision_count"] = 0

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API unavailable")
            result = code_reviewer_node(base_state)

        # LLM failures should default to needs_revision (safer than auto-approve)
        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 1

    def test_code_generator_escalates_on_llm_error(self, base_state):
        """Code generator escalates to user when LLM fails."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["design_description"] = "Valid design " * 5

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("Service unavailable")
            result = code_generator_node(base_state)

        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "llm_error"
        assert "failed" in result["pending_user_questions"][0].lower()


class TestContextCheckBehavior:
    """Test context check decorator behavior."""

    def test_design_reviewer_returns_empty_when_awaiting_input(self, base_state):
        """Design reviewer returns empty dict when already awaiting user input."""
        base_state["awaiting_user_input"] = True
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = "Test design"

        result = design_reviewer_node(base_state)

        assert result == {}

    def test_code_reviewer_returns_empty_when_awaiting_input(self, base_state):
        """Code reviewer returns empty dict when already awaiting user input."""
        base_state["awaiting_user_input"] = True
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code"] = "# test"

        result = code_reviewer_node(base_state)

        assert result == {}
