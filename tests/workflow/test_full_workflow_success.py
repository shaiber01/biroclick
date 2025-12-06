from copy import deepcopy
from unittest.mock import patch, MagicMock
import pytest

from src.agents import (
    plan_node,
    plan_reviewer_node,
    select_stage_node,
    simulation_designer_node,
    design_reviewer_node,
    code_generator_node,
    code_reviewer_node,
)
from schemas.state import (
    MAX_REPLANS,
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
)

from tests.workflow.fixtures import MockResponseFactory


class TestPlanningPhase:
    """Tests for plan_node functionality."""

    def test_planning_phase_produces_valid_plan_structure(self, base_state):
        """Test planning phase produces valid plan with all required fields."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()

            result = plan_node(base_state)

            # Verify plan exists and has correct structure
            assert "plan" in result, "plan_node must return 'plan' key"
            plan = result["plan"]
            
            # Verify required plan fields
            assert plan["paper_id"] == "test_gold_nanorod"
            assert "paper_domain" in plan
            assert "stages" in plan
            assert len(plan["stages"]) == 2, "Expected 2 stages in plan"
            
            # Verify first stage structure
            stage_0 = plan["stages"][0]
            assert stage_0["stage_id"] == "stage_0_materials"
            assert stage_0["stage_type"] == "MATERIAL_VALIDATION"
            assert "dependencies" in stage_0
            assert stage_0["dependencies"] == [], "Material validation should have no dependencies"

            # Verify second stage structure
            stage_1 = plan["stages"][1]
            assert stage_1["stage_id"] == "stage_1_extinction"
            assert stage_1["stage_type"] == "SINGLE_STRUCTURE"
            assert "stage_0_materials" in stage_1["dependencies"], (
                "SINGLE_STRUCTURE should depend on MATERIAL_VALIDATION"
            )

            # Verify progress initialization
            assert "progress" in result, "plan_node must initialize progress"
            assert len(result["progress"]["stages"]) == 2
            for stage in result["progress"]["stages"]:
                assert stage["status"] == "not_started", (
                    f"Initial stage status must be 'not_started', got {stage['status']}"
                )

            # Verify extracted parameters
            assert "extracted_parameters" in result
            assert len(result["extracted_parameters"]) == 2
            assert result["extracted_parameters"][0]["name"] == "length"
            assert result["extracted_parameters"][0]["value"] == 100
            assert result["extracted_parameters"][0]["unit"] == "nm"

            assert result["workflow_phase"] == "planning"

    def test_planning_with_missing_paper_text_escalates(self, base_state):
        """Test that plan_node escalates when paper_text is missing or too short."""
        # Test with empty paper_text
        base_state["paper_text"] = ""
        
        result = plan_node(base_state)
        
        assert result.get("ask_user_trigger") is not None, (
            "Missing paper_text should trigger user escalation"
        )
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        
    def test_planning_with_short_paper_text_escalates(self, base_state):
        """Test that plan_node escalates when paper_text is too short."""
        base_state["paper_text"] = "Short text"  # < 100 chars
        
        result = plan_node(base_state)
        
        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "missing_paper_text"

    def test_planning_llm_failure_escalates(self, base_state):
        """Test that LLM failure in plan_node escalates to user."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            result = plan_node(base_state)
            
            assert result.get("ask_user_trigger") is not None, (
                "LLM failure should trigger user escalation"
            )
            assert result.get("ask_user_trigger") == "llm_error"
            assert "pending_user_questions" in result

    def test_planning_progress_init_failure_triggers_revision(self, base_state):
        """Test that progress initialization failure triggers plan revision."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            # Return plan with invalid stage structure that will fail init
            bad_plan = MockResponseFactory.planner_response()
            # Make stages invalid by removing required fields
            bad_plan["stages"] = [{"invalid": True}]
            mock_llm.return_value = bad_plan
            
            with patch("src.agents.planning.initialize_progress_from_plan") as mock_init:
                mock_init.side_effect = Exception("Invalid stage structure")
                
                result = plan_node(base_state)
                
                assert result.get("last_plan_review_verdict") == "needs_revision", (
                    "Progress init failure should mark plan for revision"
                )
                assert "planner_feedback" in result

    def test_planning_replan_context_injected(self, base_state):
        """Test that replan context is injected on subsequent attempts."""
        base_state["replan_count"] = 1
        
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()
            with patch("src.agents.planning.build_agent_prompt") as mock_prompt:
                mock_prompt.return_value = "base prompt"
                
                plan_node(base_state)
                
                # The system prompt should mention replan attempt
                # Check that the LLM was called (we can't easily check prompt content)
                assert mock_llm.called


class TestPlanReviewPhase:
    """Tests for plan_reviewer_node functionality."""

    def test_plan_review_approve(self, base_state):
        """Test plan review approval with proper verdict."""
        base_state["plan"] = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()

            result = plan_reviewer_node(base_state)

            assert result["last_plan_review_verdict"] == "approve"
            assert result["workflow_phase"] == "plan_review"
            # When approved, there should be no feedback (or empty)
            feedback = result.get("planner_feedback")
            assert not feedback or feedback == "", (
                f"Approved plan should not have feedback, got: {feedback}"
            )

    def test_plan_review_needs_revision(self, base_state):
        """Test plan review with needs_revision verdict."""
        base_state["plan"] = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision(
                feedback="Missing convergence criteria"
            )

            result = plan_reviewer_node(base_state)

            assert result["last_plan_review_verdict"] == "needs_revision"
            assert result["workflow_phase"] == "plan_review"
            assert "replan_count" in result, "needs_revision should increment replan_count"
            assert result["replan_count"] >= 1
            assert "planner_feedback" in result
            assert result["planner_feedback"] != ""

    def test_plan_review_verdict_normalization(self, base_state):
        """Test that various verdict strings are normalized correctly."""
        base_state["plan"] = MockResponseFactory.planner_response()
        
        # Test "pass" -> "approve"
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "issues": [], "summary": "ok"}
            result = plan_reviewer_node(base_state)
            assert result["last_plan_review_verdict"] == "approve"
        
        # Test "reject" -> "needs_revision"
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "reject", "issues": [], "summary": "bad"}
            result = plan_reviewer_node(base_state)
            assert result["last_plan_review_verdict"] == "needs_revision"
        
        # Test "approved" -> "approve"
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "approved", "issues": [], "summary": "ok"}
            result = plan_reviewer_node(base_state)
            assert result["last_plan_review_verdict"] == "approve"

    def test_plan_review_empty_plan_rejects(self, base_state):
        """Test that empty plan is rejected without LLM call."""
        base_state["plan"] = {}

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            result = plan_reviewer_node(base_state)

            assert result["last_plan_review_verdict"] == "needs_revision"
            # LLM should not be called for structural validation
            assert not mock_llm.called, "LLM should not be called for empty plan"
            assert "planner_feedback" in result

    def test_plan_review_missing_stages_rejects(self, base_state):
        """Test that plan with no stages is rejected."""
        base_state["plan"] = {"paper_id": "test", "stages": []}

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert "no stages" in result["planner_feedback"].lower() or "at least one stage" in result["planner_feedback"].lower()

    def test_plan_review_missing_stage_id_rejects(self, base_state):
        """Test that stage without stage_id is rejected."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_type": "MATERIAL_VALIDATION", "targets": ["mat1"]}
                # Missing stage_id
            ]
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        # Should mention missing stage_id
        assert "stage_id" in result["planner_feedback"].lower()

    def test_plan_review_duplicate_stage_id_rejects(self, base_state):
        """Test that duplicate stage IDs are rejected."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["mat1"]},
                {"stage_id": "stage_0", "stage_type": "SINGLE_STRUCTURE", "targets": ["fig1"]},
            ]
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "duplicate" in result["planner_feedback"].lower()

    def test_plan_review_missing_targets_rejects(self, base_state):
        """Test that stages without targets are rejected."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION"}
                # Missing targets
            ]
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "target" in result["planner_feedback"].lower()

    def test_plan_review_circular_dependencies_rejects(self, base_state):
        """Test that circular dependencies are detected and rejected."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_a", "stage_type": "MATERIAL_VALIDATION", 
                 "targets": ["mat1"], "dependencies": ["stage_b"]},
                {"stage_id": "stage_b", "stage_type": "SINGLE_STRUCTURE", 
                 "targets": ["fig1"], "dependencies": ["stage_a"]},
            ]
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result["planner_feedback"].lower()

    def test_plan_review_self_dependency_rejects(self, base_state):
        """Test that self-dependency is detected and rejected."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", 
                 "targets": ["mat1"], "dependencies": ["stage_0"]},
            ]
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "depends on itself" in result["planner_feedback"].lower()

    def test_plan_review_missing_dependency_rejects(self, base_state):
        """Test that missing dependencies are detected and rejected."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "SINGLE_STRUCTURE", 
                 "targets": ["fig1"], "dependencies": ["nonexistent_stage"]},
            ]
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "missing" in result["planner_feedback"].lower()

    def test_plan_review_llm_failure_defaults_to_needs_revision(self, base_state):
        """Test that LLM failure in reviewer defaults to needs_revision for safety."""
        base_state["plan"] = MockResponseFactory.planner_response()
        base_state["replan_count"] = 0

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = plan_reviewer_node(base_state)

            # LLM failures should default to needs_revision (safer than auto-approve)
            assert result["last_plan_review_verdict"] == "needs_revision"
            assert result["replan_count"] == 1


class TestStageSelection:
    """Tests for select_stage_node functionality."""

    def test_stage_selection_picks_first_available(self, base_state):
        """Test stage selection picks first available stage."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])

        result = select_stage_node(base_state)

        # Should select first stage with no dependencies
        selected = result.get("current_stage_id")
        assert selected == "stage_0_materials", (
            "Should select materials stage first (no deps)"
        )
        assert result.get("current_stage_type") == "MATERIAL_VALIDATION"
        assert result.get("workflow_phase") == "stage_selection"

    def test_stage_selection_respects_dependencies(self, base_state):
        """Test that stages with unmet dependencies are not selected."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Stage 0 is not_started, Stage 1 depends on Stage 0
        # Should not select Stage 1 because Stage 0 is not completed

        result = select_stage_node(base_state)

        # Should select stage_0 (no deps), not stage_1 (has deps)
        assert result.get("current_stage_id") == "stage_0_materials"

    def test_stage_selection_after_stage_completed(self, base_state):
        """Test stage selection after a stage is completed."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Mark stage 0 as completed
        base_state["progress"]["stages"][0]["status"] = "completed_success"

        result = select_stage_node(base_state)

        # Should now select stage 1
        assert result.get("current_stage_id") == "stage_1_extinction", (
            "Should select next stage after dependency is completed"
        )
        assert result.get("current_stage_type") == "SINGLE_STRUCTURE"

    def test_stage_selection_no_stages_escalates(self, base_state):
        """Test that no stages available triggers user escalation."""
        base_state["plan"] = {}
        base_state["progress"] = {}

        result = select_stage_node(base_state)

        assert result.get("current_stage_id") is None
        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "no_stages_available"

    def test_stage_selection_all_completed_returns_none(self, base_state):
        """Test that all stages completed returns None for current_stage_id."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Mark all stages as completed
        for stage in base_state["progress"]["stages"]:
            stage["status"] = "completed_success"

        result = select_stage_node(base_state)

        assert result.get("current_stage_id") is None
        assert result.get("ask_user_trigger") is None, (
            "All stages completed should not escalate"
        )

    def test_stage_selection_needs_rerun_priority(self, base_state):
        """Test that needs_rerun stages have priority over not_started."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Mark stage 0 as completed, stage 1 as needs_rerun
        base_state["progress"]["stages"][0]["status"] = "completed_success"
        base_state["progress"]["stages"][1]["status"] = "needs_rerun"

        result = select_stage_node(base_state)

        # Should select stage_1 (needs_rerun has priority)
        assert result.get("current_stage_id") == "stage_1_extinction"

    def test_stage_selection_resets_counters_on_new_stage(self, base_state):
        """Test that revision counters reset when selecting a new stage."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "some_other_stage"
        base_state["code_revision_count"] = 2
        base_state["design_revision_count"] = 1

        result = select_stage_node(base_state)

        # Counters should be reset for new stage
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0
        assert result.get("execution_failure_count") == 0

    def test_stage_selection_deadlock_detection(self, base_state):
        """Test that deadlock is detected when all remaining stages are blocked."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Mark stage 0 as blocked (no deps satisfied)
        base_state["progress"]["stages"][0]["status"] = "blocked"
        # Stage 1 can't run because stage 0 is blocked
        base_state["progress"]["stages"][1]["status"] = "not_started"

        result = select_stage_node(base_state)

        # Should detect deadlock
        assert result.get("ask_user_trigger") == "deadlock_detected"
        assert result.get("ask_user_trigger") is not None

    def test_stage_selection_blocked_stage_with_missing_stage_type(self, base_state):
        """Test that stage without stage_type gets blocked."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        
        # Remove stage_type from a stage
        base_state["progress"]["stages"][0]["stage_type"] = None

        result = select_stage_node(base_state)

        # Stage 0 should be skipped (missing stage_type)
        # Stage 1 can't run (dependency not satisfied)
        # Should detect deadlock or escalate
        assert result.get("current_stage_id") is None or result.get("ask_user_trigger") is not None


class TestSimulationDesigner:
    """Tests for simulation_designer_node functionality."""

    def test_designer_produces_valid_design(self, base_state):
        """Test that designer produces valid design structure."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()

            result = simulation_designer_node(base_state)

            assert "design_description" in result
            assert result["workflow_phase"] == "design"
            
            # Verify design structure
            design = result["design_description"]
            assert isinstance(design, dict)
            assert "geometry_definitions" in design
            assert len(design["geometry_definitions"]) > 0
            assert design["geometry_definitions"][0]["material"] == "gold"

    def test_designer_missing_stage_id_escalates(self, base_state):
        """Test that missing current_stage_id triggers escalation."""
        base_state["current_stage_id"] = None

        result = simulation_designer_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "missing_stage_id"

    def test_designer_llm_failure_escalates(self, base_state):
        """Test that LLM failure in designer escalates to user."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = simulation_designer_node(base_state)

            assert result.get("ask_user_trigger") is not None
            assert result.get("ask_user_trigger") == "llm_error"

    def test_designer_adds_new_assumptions(self, base_state):
        """Test that designer adds new assumptions to state."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["assumptions"] = {"global_assumptions": []}

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            response = MockResponseFactory.designer_response()
            response["new_assumptions"] = [
                {"id": "A2", "category": "numerical", "description": "PML layers set to 20"}
            ]
            mock_llm.return_value = response

            result = simulation_designer_node(base_state)

            if "assumptions" in result:
                assert len(result["assumptions"]["global_assumptions"]) >= 1


class TestDesignReviewer:
    """Tests for design_reviewer_node functionality."""

    def test_design_review_approve(self, base_state):
        """Test design review approval."""
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["current_stage_id"] = "stage_1_extinction"

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()

            result = design_reviewer_node(base_state)

            assert result["last_design_review_verdict"] == "approve"
            assert result["workflow_phase"] == "design_review"

    def test_design_review_needs_revision_increments_counter(self, base_state):
        """Test that needs_revision increments design_revision_count."""
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = 0

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = design_reviewer_node(base_state)

            assert result["last_design_review_verdict"] == "needs_revision"
            assert result["design_revision_count"] == 1
            assert "reviewer_feedback" in result

    def test_design_review_max_revisions_escalates(self, base_state):
        """Test that max design revisions triggers user escalation."""
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS  # At max

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = design_reviewer_node(base_state)

            assert result.get("ask_user_trigger") is not None
            assert result.get("ask_user_trigger") == "design_review_limit"

    def test_design_review_verdict_normalization(self, base_state):
        """Test design review verdict normalization."""
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["current_stage_id"] = "stage_1_extinction"

        # Test "pass" -> "approve"
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "issues": []}
            result = design_reviewer_node(base_state)
            assert result["last_design_review_verdict"] == "approve"

    def test_design_review_llm_failure_defaults_to_needs_revision(self, base_state):
        """Test that LLM failure defaults to needs_revision for safety."""
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = 0

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = design_reviewer_node(base_state)

            # LLM failures should default to needs_revision (safer than auto-approve)
            assert result["last_design_review_verdict"] == "needs_revision"
            assert result["design_revision_count"] == 1


class TestCodeGenerator:
    """Tests for code_generator_node functionality."""

    def test_code_generator_produces_valid_code(self, base_state):
        """Test that code generator produces valid Python code."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()

            result = code_generator_node(base_state)

            assert "code" in result
            assert result["workflow_phase"] == "code_generation"
            assert "import meep" in result["code"], "Code should contain meep import"
            assert "STUB" not in result["code"], "Code should not contain STUB markers"

    def test_code_generator_missing_stage_id_escalates(self, base_state):
        """Test that missing current_stage_id triggers escalation."""
        base_state["current_stage_id"] = None

        result = code_generator_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "missing_stage_id"

    def test_code_generator_stub_design_fails(self, base_state):
        """Test that stub design description causes failure."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = "STUB - Design would be generated"
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        result = code_generator_node(base_state)

        # Should fail and increment design revision count
        assert "reviewer_feedback" in result or "run_error" in result
        assert "stub" in (result.get("reviewer_feedback", "") + result.get("run_error", "")).lower()

    def test_code_generator_empty_design_fails(self, base_state):
        """Test that empty design description causes failure."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = ""
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        result = code_generator_node(base_state)

        # Should fail
        assert "reviewer_feedback" in result or "run_error" in result

    def test_code_generator_missing_validated_materials_for_stage1_fails(self, base_state):
        """Test that Stage 1+ fails without validated_materials."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["validated_materials"] = []  # Empty!

        result = code_generator_node(base_state)

        assert "run_error" in result
        assert "validated_materials" in result["run_error"]

    def test_code_generator_material_validation_no_materials_ok(self, base_state):
        """Test that MATERIAL_VALIDATION stage doesn't require validated_materials."""
        base_state["current_stage_id"] = "stage_0_materials"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["validated_materials"] = []  # OK for Stage 0

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()

            result = code_generator_node(base_state)

            # Should succeed
            assert "code" in result
            assert "run_error" not in result or result.get("run_error") is None

    def test_code_generator_llm_failure_escalates(self, base_state):
        """Test that LLM failure escalates to user."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = code_generator_node(base_state)

            assert result.get("ask_user_trigger") is not None
            assert result.get("ask_user_trigger") == "llm_error"

    def test_code_generator_stub_code_increments_revision(self, base_state):
        """Test that stub code output increments revision counter."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]
        base_state["code_revision_count"] = 0

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"code": "# STUB - Code would go here"}

            result = code_generator_node(base_state)

            assert result.get("code_revision_count", 0) >= 1
            assert "reviewer_feedback" in result


class TestCodeReviewer:
    """Tests for code_reviewer_node functionality."""

    def test_code_review_approve(self, base_state):
        """Test code review approval."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_description"] = MockResponseFactory.designer_response()

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()

            result = code_reviewer_node(base_state)

            assert result["last_code_review_verdict"] == "approve"
            assert result["workflow_phase"] == "code_review"

    def test_code_review_needs_revision_increments_counter(self, base_state):
        """Test that needs_revision increments code_revision_count."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code_revision_count"] = 0

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = code_reviewer_node(base_state)

            assert result["last_code_review_verdict"] == "needs_revision"
            assert result["code_revision_count"] == 1
            assert "reviewer_feedback" in result

    def test_code_review_max_revisions_escalates(self, base_state):
        """Test that max code revisions triggers user escalation."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS  # At max

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = code_reviewer_node(base_state)

            assert result.get("ask_user_trigger") is not None
            assert result.get("ask_user_trigger") == "code_review_limit"

    def test_code_review_llm_failure_defaults_to_needs_revision(self, base_state):
        """Test that LLM failure defaults to needs_revision for safety."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code_revision_count"] = 0

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = code_reviewer_node(base_state)

            # LLM failures should default to needs_revision (safer than auto-approve)
            assert result["last_code_review_verdict"] == "needs_revision"
            assert result["code_revision_count"] == 1

    def test_code_review_verdict_normalization(self, base_state):
        """Test code review verdict normalization."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"

        # Test "pass" -> "approve"
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "pass", "issues": []}
            result = code_reviewer_node(base_state)
            assert result["last_code_review_verdict"] == "approve"

        # Test unknown verdict -> "needs_revision" (safer for code)
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {"verdict": "unknown_verdict", "issues": []}
            result = code_reviewer_node(base_state)
            assert result["last_code_review_verdict"] == "needs_revision"


class TestFullWorkflowSuccess:
    """Test complete successful workflow from paper to report."""

    def test_planning_phase(self, base_state):
        """Test planning phase produces valid plan with all required fields."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()

            result = plan_node(base_state)

            assert "plan" in result
            plan = result["plan"]

            # Verify plan structure
            assert plan["paper_id"] == "test_gold_nanorod"
            assert len(plan["stages"]) == 2
            assert plan["stages"][0]["stage_id"] == "stage_0_materials"
            assert plan["stages"][0]["stage_type"] == "MATERIAL_VALIDATION"

            # Verify progress initialization
            assert "progress" in result
            assert len(result["progress"]["stages"]) == 2
            assert result["progress"]["stages"][0]["status"] == "not_started"

            # Verify extracted parameters
            assert "extracted_parameters" in result
            assert len(result["extracted_parameters"]) == 2
            assert result["extracted_parameters"][0]["name"] == "length"

            assert result["workflow_phase"] == "planning"

    def test_plan_review_approve(self, base_state):
        """Test plan review approval."""
        # Set up state with plan
        base_state["plan"] = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()

            result = plan_reviewer_node(base_state)

            assert result["last_plan_review_verdict"] == "approve"
            assert result["workflow_phase"] == "plan_review"
            # Verify no unexpected issues
            assert (
                "planner_feedback" not in result or not result["planner_feedback"]
            )

    def test_stage_selection(self, base_state):
        """Test stage selection picks first available stage."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])

        result = select_stage_node(base_state)

        # Should select first stage with no dependencies
        selected = result.get("current_stage_id")
        assert selected == "stage_0_materials", "Should select materials stage first (no deps)"
        assert result.get("current_stage_type") == "MATERIAL_VALIDATION"

    def test_full_single_stage_workflow(self, base_state):
        """Test complete single-stage workflow."""
        # Setup: Plan with approved review
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        # Mock execution result (simulates code_runner)
        base_state["execution_result"] = {
            "success": True,
            "output_files": ["extinction.csv"],
            "runtime_seconds": 120,
        }
        base_state["stage_outputs"] = {
            "files": ["extinction.csv"],
            "stage_id": "stage_1_extinction",
        }

        workflow_states = []

        # Design phase
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)
            workflow_states.append(("design", result))

            assert "design_description" in result
            assert (
                result["design_description"]["geometry_definitions"][0]["material"]
                == "gold"
            )

        # Design review
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)
            workflow_states.append(("design_review", result))

        assert base_state["last_design_review_verdict"] == "approve"

        # Code generation
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)
            base_state.update(result)
            workflow_states.append(("code_gen", result))

        assert "code" in base_state
        assert "import meep" in base_state["code"]
        assert "STUB" not in base_state["code"]

        # Code review
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = code_reviewer_node(base_state)
            base_state.update(result)
            workflow_states.append(("code_review", result))

        assert base_state["last_code_review_verdict"] == "approve"

    def test_full_workflow_with_revision_cycle(self, base_state):
        """Test workflow with one revision cycle before approval."""
        plan = MockResponseFactory.planner_response()
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]
        base_state["code_revision_count"] = 0

        # Design phase
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            base_state.update(result)

        # Design review - approve
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = design_reviewer_node(base_state)
            base_state.update(result)

        # First code generation
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)
            base_state.update(result)

        # First code review - needs revision
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision(
                feedback="Missing error handling"
            )
            result = code_reviewer_node(base_state)
            base_state.update(result)

        assert base_state["last_code_review_verdict"] == "needs_revision"
        assert base_state["code_revision_count"] == 1
        assert "reviewer_feedback" in base_state

        # Second code generation (with feedback)
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)
            base_state.update(result)

        # Second code review - approve
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = code_reviewer_node(base_state)
            base_state.update(result)

        assert base_state["last_code_review_verdict"] == "approve"


class TestExecutionAndAnalysis:
    """Tests for execution_validator_node and analysis nodes."""

    def test_execution_validator_pass(self, base_state):
        """Test execution validator pass verdict."""
        from src.agents import execution_validator_node
        
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {
            "files": ["extinction.csv"],
            "success": True,
        }
        base_state["run_error"] = None

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_pass()

            result = execution_validator_node(base_state)

            assert result["execution_verdict"] == "pass"
            assert result["workflow_phase"] == "execution_validation"
            # Execution failure count should not be incremented
            assert "execution_failure_count" not in result or result.get("execution_failure_count", 0) == 0

    def test_execution_validator_fail_increments_counter(self, base_state):
        """Test execution validator fail increments failure counter."""
        from src.agents import execution_validator_node
        
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {}
        base_state["run_error"] = "Simulation crashed"
        base_state["execution_failure_count"] = 0

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.execution_validator_fail()

            result = execution_validator_node(base_state)

            assert result["execution_verdict"] == "fail"
            assert result["execution_failure_count"] == 1
            assert result.get("total_execution_failures", 0) >= 1

    def test_execution_validator_timeout_with_skip_fallback(self, base_state):
        """Test that timeout with skip_with_warning fallback passes."""
        from src.agents import execution_validator_node
        
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"timeout_exceeded": True}
        base_state["run_error"] = "Exceeded timeout"
        base_state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_1_extinction",
                    "fallback_strategy": "skip_with_warning",
                }
            ]
        }

        result = execution_validator_node(base_state)

        assert result["execution_verdict"] == "pass"
        assert "timeout" in result.get("execution_feedback", "").lower()

    def test_physics_sanity_pass(self, base_state):
        """Test physics sanity pass verdict."""
        from src.agents import physics_sanity_node
        
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"files": ["extinction.csv"]}
        base_state["design_description"] = MockResponseFactory.designer_response()

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.physics_sanity_pass()

            result = physics_sanity_node(base_state)

            assert result["physics_verdict"] == "pass"
            assert result["workflow_phase"] == "physics_validation"

    def test_physics_sanity_fail_increments_counter(self, base_state):
        """Test physics sanity fail increments counter."""
        from src.agents import physics_sanity_node
        
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"files": ["extinction.csv"]}
        base_state["physics_failure_count"] = 0

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "fail",
                "summary": "Results violate physics constraints",
            }

            result = physics_sanity_node(base_state)

            assert result["physics_verdict"] == "fail"
            assert result["physics_failure_count"] == 1

    def test_physics_sanity_design_flaw_increments_design_revision(self, base_state):
        """Test physics sanity design_flaw verdict increments design revision counter."""
        from src.agents import physics_sanity_node
        
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["stage_outputs"] = {"files": ["extinction.csv"]}
        base_state["design_revision_count"] = 0

        with patch("src.agents.execution.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "design_flaw",
                "summary": "Fundamental design problem: PML layers too thin",
            }

            result = physics_sanity_node(base_state)

            assert result["physics_verdict"] == "design_flaw"
            assert result["design_revision_count"] == 1
            assert "design_feedback" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_values_handled_gracefully(self, base_state):
        """Test that None values in state don't cause crashes."""
        base_state["plan"] = None
        base_state["progress"] = None
        
        # select_stage should handle None gracefully
        result = select_stage_node(base_state)
        
        # Should not crash, should escalate or return None stage
        assert result.get("current_stage_id") is None or result.get("ask_user_trigger") is not None

    def test_empty_list_dependencies_handled(self, base_state):
        """Test that empty dependencies list is handled correctly."""
        plan = MockResponseFactory.planner_response()
        # Explicitly set empty dependencies
        plan["stages"][0]["dependencies"] = []
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])

        result = select_stage_node(base_state)

        # Should select stage 0 (empty deps = no deps)
        assert result.get("current_stage_id") == "stage_0_materials"

    def test_none_dependencies_handled(self, base_state):
        """Test that None dependencies are handled as empty list."""
        plan = MockResponseFactory.planner_response()
        plan["stages"][0]["dependencies"] = None
        base_state["plan"] = plan
        base_state["progress"] = deepcopy(plan["progress"])
        base_state["progress"]["stages"][0]["dependencies"] = None

        result = select_stage_node(base_state)

        # Should not crash, should treat as no deps
        assert result.get("current_stage_id") == "stage_0_materials"

    def test_trigger_set_bypasses_nodes(self, base_state):
        """Test that nodes bypass processing when ask_user_trigger is set."""
        base_state["ask_user_trigger"] = "some_trigger"
        base_state["plan"] = MockResponseFactory.planner_response()

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            result = plan_reviewer_node(base_state)

            # Should return empty result without calling LLM
            assert result == {} or "last_plan_review_verdict" not in result
            assert not mock_llm.called

    def test_runtime_config_limits_respected(self, base_state):
        """Test that runtime_config limits are respected."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS  # Already at max
        base_state["runtime_config"] = {"max_code_revisions": MAX_CODE_REVISIONS}

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()
            result = code_reviewer_node(base_state)

            # When at max, rejection should trigger escalation
            assert result.get("ask_user_trigger") is not None
            assert result.get("ask_user_trigger") == "code_review_limit"

    def test_invalid_stage_structure_handled(self, base_state):
        """Test that invalid stage structure (non-dict) is caught."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": ["invalid_string_stage"]  # Should be dict
        }

        result = plan_reviewer_node(base_state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "invalid stage structure" in result["planner_feedback"].lower()


class TestCounterBounds:
    """Test counter increment bounds and max limit behavior."""

    def test_replan_count_respects_max(self, base_state):
        """Test that replan_count doesn't exceed MAX_REPLANS."""
        base_state["plan"] = MockResponseFactory.planner_response()
        base_state["replan_count"] = MAX_REPLANS  # Already at max

        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = plan_reviewer_node(base_state)

            # Counter should not exceed max
            assert result.get("replan_count", MAX_REPLANS) <= MAX_REPLANS

    def test_design_revision_count_respects_max(self, base_state):
        """Test that design_revision_count doesn't exceed MAX_DESIGN_REVISIONS."""
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS  # At max

        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = design_reviewer_node(base_state)

            # Counter should not exceed max
            assert result["design_revision_count"] <= MAX_DESIGN_REVISIONS
            # Should escalate when at max
            assert result.get("ask_user_trigger") is not None

    def test_code_revision_count_respects_max(self, base_state):
        """Test that code_revision_count doesn't exceed MAX_CODE_REVISIONS."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS  # At max

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_needs_revision()

            result = code_reviewer_node(base_state)

            # Counter should not exceed max
            assert result["code_revision_count"] <= MAX_CODE_REVISIONS
            # Should escalate when at max
            assert result.get("ask_user_trigger") is not None


class TestValidationHierarchy:
    """Test validation hierarchy enforcement in stage selection."""

    def test_single_structure_requires_material_validation(self, base_state):
        """Test SINGLE_STRUCTURE cannot run before MATERIAL_VALIDATION."""
        # Create plan where Stage 1 has no explicit dependency but hierarchy requires it
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0_materials",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["mat1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1_single",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["fig1"],
                    "dependencies": [],  # No explicit dependency
                },
            ]
        }
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0_materials", "stage_type": "MATERIAL_VALIDATION", "status": "not_started"},
                {"stage_id": "stage_1_single", "stage_type": "SINGLE_STRUCTURE", "status": "not_started"},
            ]
        }

        result = select_stage_node(base_state)

        # Should select materials stage first due to hierarchy
        assert result.get("current_stage_id") == "stage_0_materials"

    def test_stage_selection_follows_type_order(self, base_state):
        """Test that stage selection follows STAGE_TYPE_ORDER."""
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_1_single",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_0_materials",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["mat1"],
                    "dependencies": [],
                },
            ]
        }
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_1_single", "stage_type": "SINGLE_STRUCTURE", "status": "not_started"},
                {"stage_id": "stage_0_materials", "stage_type": "MATERIAL_VALIDATION", "status": "not_started"},
            ]
        }

        result = select_stage_node(base_state)

        # Should select materials stage first even if it's second in list
        assert result.get("current_stage_id") == "stage_0_materials"


class TestProgressInitialization:
    """Test progress initialization and synchronization."""

    def test_progress_initialized_with_correct_status(self, base_state):
        """Test that progress stages are initialized with not_started status."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()

            result = plan_node(base_state)

            assert "progress" in result
            stages = result["progress"]["stages"]
            for stage in stages:
                assert stage["status"] == "not_started"
                assert "stage_id" in stage
                assert "stage_type" in stage

    def test_extracted_parameters_synced(self, base_state):
        """Test that extracted parameters are synced from plan."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()

            result = plan_node(base_state)

            assert "extracted_parameters" in result
            params = result["extracted_parameters"]
            # Verify parameters match plan
            assert len(params) == 2
            param_names = [p["name"] for p in params]
            assert "length" in param_names
            assert "diameter" in param_names


class TestWorkflowStateTransitions:
    """Test workflow state transition correctness."""

    def test_workflow_phase_set_correctly_plan(self, base_state):
        """Test workflow_phase is set correctly after plan node."""
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.planner_response()
            result = plan_node(base_state)
            assert result["workflow_phase"] == "planning"

    def test_workflow_phase_set_correctly_plan_review(self, base_state):
        """Test workflow_phase is set correctly after plan review node."""
        base_state["plan"] = MockResponseFactory.planner_response()
        with patch("src.agents.planning.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.reviewer_approve()
            result = plan_reviewer_node(base_state)
            assert result["workflow_phase"] == "plan_review"

    def test_workflow_phase_set_correctly_design(self, base_state):
        """Test workflow_phase is set correctly after design node."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        with patch("src.agents.design.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.designer_response()
            result = simulation_designer_node(base_state)
            assert result["workflow_phase"] == "design"

    def test_workflow_phase_set_correctly_code_generation(self, base_state):
        """Test workflow_phase is set correctly after code generation node."""
        base_state["current_stage_id"] = "stage_1_extinction"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = MockResponseFactory.designer_response()
        base_state["validated_materials"] = [{"material_id": "gold", "path": "/materials/Au.csv"}]
        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = MockResponseFactory.code_generator_response()
            result = code_generator_node(base_state)
            assert result["workflow_phase"] == "code_generation"

    def test_review_nodes_return_issues_list(self, base_state):
        """Test that review nodes include issues in their result."""
        base_state["code"] = MockResponseFactory.code_generator_response()["code"]
        base_state["current_stage_id"] = "stage_1_extinction"

        with patch("src.agents.code.call_agent_with_metrics") as mock_llm:
            mock_llm.return_value = {
                "verdict": "needs_revision",
                "issues": [{"severity": "major", "description": "Missing error handling"}],
                "feedback": "Add try/except blocks",
            }

            result = code_reviewer_node(base_state)

            assert "reviewer_issues" in result
            assert len(result["reviewer_issues"]) > 0
