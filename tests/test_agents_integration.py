"""
Integration Tests for Agent Nodes (Bug-Finding Tests).

These tests are designed to FIND BUGS, not just pass.

They verify:
1. LLM is called with correct parameters (agent_name, schema, prompt)
2. Node outputs have correct structure and values
3. State mutations are correct
4. Edge cases are handled
5. Business logic is enforced
6. All required files (prompts, schemas) exist
7. Routing functions return valid values
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from copy import deepcopy

from schemas.state import create_initial_state, ReproState

# Project root for file existence tests
PROJECT_ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    """Create a base state with realistic paper content."""
    paper_text = """
    We study the optical properties of gold nanorods with length 100nm and 
    diameter 40nm using FDTD simulations. The localized surface plasmon 
    resonance (LSPR) is observed at approximately 650nm wavelength.
    
    Materials: Gold optical constants from Johnson & Christy (1972).
    Surrounding medium: Water (n=1.33).
    
    Figure 1 shows the extinction spectrum with the longitudinal mode peak.
    Figure 2 shows near-field enhancement maps at resonance.
    """ * 5  # Make it long enough
    
    state = create_initial_state(
        paper_id="test_integration",
        paper_text=paper_text,
        paper_domain="plasmonics"
    )
    state["paper_figures"] = [
        {"figure_id": "Fig1", "description": "Extinction spectrum"},
        {"figure_id": "Fig2", "description": "Near-field map"},
    ]
    return state


@pytest.fixture
def valid_plan():
    """A valid plan structure for testing."""
    return {
        "paper_id": "test_integration",
        "title": "Gold Nanorod Optical Properties",
        "stages": [{
            "stage_id": "stage_0",
            "stage_type": "MATERIAL_VALIDATION",
            "description": "Validate gold optical constants",
            "targets": ["material_gold"],
            "dependencies": [],
        }],
        "targets": [{"figure_id": "Fig1", "description": "Test"}],
        "extracted_parameters": [
            {"name": "length", "value": 100, "unit": "nm", "source": "text"},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# Test: LLM is Called Correctly
# ═══════════════════════════════════════════════════════════════════════

class TestLLMCalledCorrectly:
    """Verify LLM is called with correct parameters."""
    
    def test_plan_node_calls_llm_with_correct_agent_name(self, base_state):
        """plan_node must call LLM with agent_name='planner'."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
            "targets": [],
            "extracted_parameters": [],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response) as mock:
            plan_node(base_state)
            
            # Verify LLM was called
            assert mock.called, "LLM should be called"
            
            # Verify agent_name
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("agent_name") == "planner", \
                f"Expected agent_name='planner', got '{call_kwargs.get('agent_name')}'"
    
    def test_supervisor_node_calls_llm_with_correct_agent_name(self, base_state):
        """supervisor_node must call LLM with agent_name='supervisor'."""
        from src.agents.supervision.supervisor import supervisor_node
        
        mock_response = {"verdict": "ok_continue", "feedback": "OK"}
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", return_value=mock_response) as mock:
            base_state["current_stage_id"] = "stage_0"
            supervisor_node(base_state)
            
            # LLM MUST be called - fail if not
            assert mock.called, "supervisor_node should call LLM"
            
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("agent_name") == "supervisor", \
                f"Expected agent_name='supervisor', got '{call_kwargs.get('agent_name')}'"
            
            # Verify system_prompt was provided and is substantial
            system_prompt = call_kwargs.get("system_prompt", "")
            assert len(system_prompt) > 100, \
                f"System prompt too short ({len(system_prompt)} chars)"
    
    def test_reviewer_calls_llm_with_system_prompt(self, base_state, valid_plan):
        """Reviewer nodes must call LLM with a system_prompt."""
        from src.agents.planning import plan_reviewer_node
        
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        
        base_state["plan"] = valid_plan
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response) as mock:
            plan_reviewer_node(base_state)
            
            # LLM MUST be called - fail if not
            assert mock.called, "plan_reviewer_node should call LLM"
            
            call_kwargs = mock.call_args.kwargs
            system_prompt = call_kwargs.get("system_prompt", "")
            assert len(system_prompt) > 100, \
                f"System prompt too short ({len(system_prompt)} chars) - prompt file may be missing"
            
            # Verify agent_name is correct
            assert call_kwargs.get("agent_name") == "plan_reviewer", \
                f"Expected agent_name='plan_reviewer', got '{call_kwargs.get('agent_name')}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Output Structure is Correct
# ═══════════════════════════════════════════════════════════════════════

class TestOutputStructure:
    """Verify node outputs have correct structure."""
    
    def test_plan_node_output_has_required_fields(self, base_state):
        """plan_node output must have workflow_phase and plan."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "paper_id": "test",
            "title": "Test Plan",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
            "targets": [{"figure_id": "Fig1"}],
            "extracted_parameters": [],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)
        
        # Must have workflow_phase
        assert "workflow_phase" in result, "Missing workflow_phase"
        assert result["workflow_phase"] == "planning", f"Wrong phase: {result['workflow_phase']}"
        
        # Must have plan (not error)
        assert "plan" in result, f"Missing plan. Got keys: {result.keys()}"
        
        # Plan must have stages
        assert "stages" in result["plan"], "Plan missing stages"
        assert len(result["plan"]["stages"]) > 0, "Plan has no stages"
    
    def test_reviewer_output_has_verdict(self, base_state, valid_plan):
        """All reviewer nodes must return a verdict."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node
        
        # Test plan_reviewer
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        base_state["plan"] = valid_plan
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
        
        assert "last_plan_review_verdict" in result, "Missing verdict field"
        assert result["last_plan_review_verdict"] in ["approve", "needs_revision"], \
            f"Invalid verdict: {result['last_plan_review_verdict']}"
    
    def test_execution_validator_output_has_verdict(self, base_state):
        """execution_validator_node must return execution_verdict."""
        from src.agents.execution import execution_validator_node
        
        mock_response = {"verdict": "pass", "summary": "OK"}
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
        
        assert "execution_verdict" in result, "Missing execution_verdict"
        assert result["execution_verdict"] in ["pass", "fail"], \
            f"Invalid verdict: {result['execution_verdict']}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Counter Bounds Are Enforced
# ═══════════════════════════════════════════════════════════════════════

class TestCounterBounds:
    """Verify revision counters respect bounds."""
    
    def test_design_revision_counter_bounded(self, base_state):
        """design_revision_count should not exceed max_design_revisions."""
        from src.agents.design import design_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "issues": ["test"], "summary": "Fix"}
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        
        # Set counter at max
        max_revisions = 3
        base_state["design_revision_count"] = max_revisions
        base_state["runtime_config"] = {"max_design_revisions": max_revisions}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
        
        # Counter should NOT exceed max
        assert result.get("design_revision_count", 0) <= max_revisions, \
            f"Counter {result.get('design_revision_count')} exceeded max {max_revisions}"
    
    def test_code_revision_counter_bounded(self, base_state):
        """code_revision_count should not exceed max_code_revisions."""
        from src.agents.code import code_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "issues": ["bug"], "summary": "Fix"}
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        
        max_revisions = 5
        base_state["code_revision_count"] = max_revisions
        base_state["runtime_config"] = {"max_code_revisions": max_revisions}
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
        
        assert result.get("code_revision_count", 0) <= max_revisions, \
            f"Counter {result.get('code_revision_count')} exceeded max {max_revisions}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Error Handling Produces Correct State
# ═══════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Verify error handling produces correct state updates."""
    
    def test_llm_error_triggers_user_escalation(self, base_state):
        """LLM errors in critical nodes should escalate to user."""
        from src.agents.planning import plan_node
        
        with patch("src.agents.planning.call_agent_with_metrics", side_effect=RuntimeError("API Error")):
            result = plan_node(base_state)
        
        # Must indicate error state
        assert result.get("ask_user_trigger") == "llm_error", \
            f"Expected ask_user_trigger='llm_error', got '{result.get('ask_user_trigger')}'"
        assert result.get("awaiting_user_input") is True, \
            "Should be awaiting user input on LLM error"
    
    def test_reviewer_llm_error_auto_approves(self, base_state, valid_plan):
        """Reviewer nodes should auto-approve on LLM error (not block workflow)."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["plan"] = valid_plan
        
        with patch("src.agents.planning.call_agent_with_metrics", side_effect=RuntimeError("API Error")):
            result = plan_reviewer_node(base_state)
        
        # Should auto-approve to not block workflow
        assert result.get("last_plan_review_verdict") == "approve", \
            f"Expected auto-approve on LLM error, got '{result.get('last_plan_review_verdict')}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Missing Nodes Coverage
# ═══════════════════════════════════════════════════════════════════════

class TestMissingNodeCoverage:
    """Test nodes that were not covered in original test file."""
    
    def test_adapt_prompts_node_updates_state(self, base_state):
        """adapt_prompts_node should update adapted_prompts in state."""
        from src.agents.planning import adapt_prompts_node
        
        mock_response = {
            "adapted_system_prompt": "You are an expert physicist...",
            "domain_specific_instructions": ["Focus on plasmonics"],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = adapt_prompts_node(base_state)
        
        # Should update workflow_phase
        assert result.get("workflow_phase") == "adapting_prompts", \
            f"Expected 'adapting_prompts', got '{result.get('workflow_phase')}'"
    
    def test_select_stage_node_selects_valid_stage(self, base_state, valid_plan):
        """select_stage_node should select a valid stage from plan."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "status": "not_started",
                "dependencies": [],
            }]
        }
        
        result = select_stage_node(base_state)
        
        # Should select a stage
        assert "current_stage_id" in result, "Should select a stage"
        selected = result["current_stage_id"]
        
        # Selected stage should be from plan
        plan_stage_ids = [s["stage_id"] for s in valid_plan["stages"]]
        assert selected in plan_stage_ids or selected is None, \
            f"Selected stage '{selected}' not in plan stages {plan_stage_ids}"
    
    def test_simulation_designer_node_creates_design(self, base_state, valid_plan):
        """simulation_designer_node should create a design with all required fields."""
        from src.agents.design import simulation_designer_node
        
        mock_response = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation with gold nanorod...",
            "geometry": [{"type": "cylinder", "radius": 20, "material": "gold"}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
            "materials": [{"material_id": "gold", "source": "Johnson-Christy"}],
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = simulation_designer_node(base_state)
        
        # Should create design_description
        assert "design_description" in result, f"Should create design_description. Got: {result.keys()}"
        assert result["workflow_phase"] == "design"
        
        # Verify ALL design fields are present
        design = result["design_description"]
        assert design.get("stage_id") == "stage_0", "Design should have correct stage_id"
        assert "design_description" in design, "Design should have description text"
        assert "geometry" in design, "Design should have geometry"
        assert "sources" in design, "Design should have sources"
        assert "monitors" in design, "Design should have monitors"
        
        # Verify geometry is not empty
        assert len(design.get("geometry", [])) > 0, "Design should have at least one geometry object"
        
        # Verify design_revision_count is initialized
        assert result.get("design_revision_count", 0) >= 0
    
    def test_code_generator_node_creates_code(self, base_state, valid_plan):
        """code_generator_node should generate code with all required fields."""
        from src.agents.code import code_generator_node
        
        mock_response = {
            "code": "import meep as mp\nimport numpy as np\nprint('Simulation started')",
            "expected_outputs": ["output.csv", "spectrum.png"],
            "explanation": "Simple FDTD test simulation",
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        # Code generator requires a VALID design_description
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod extinction",
            "geometry": [{"type": "cylinder", "radius": 20, "height": 100}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 900]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
        }
        
        base_state["validated_materials"] = [{"material_id": "gold", "path": "/materials/Au.csv"}]
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)
        
        # Verify required output fields
        assert "code" in result, f"Should generate code. Got: {result.keys()}"
        assert len(result["code"]) > 10, "Code too short"
        assert result["workflow_phase"] == "code_generation"
        
        # BUG: expected_outputs from LLM response should be passed through
        # so execution_validator can verify the right files were created
        assert "expected_outputs" in result, \
            "BUG: expected_outputs from LLM not passed through to state"
        assert result["expected_outputs"] == ["output.csv", "spectrum.png"], \
            f"BUG: expected_outputs mismatch: {result.get('expected_outputs')}"
    
    def test_code_generator_requires_validated_materials_for_stage1(self, base_state, valid_plan):
        """code_generator_node should fail for Stage 1+ without validated_materials."""
        from src.agents.code import code_generator_node
        
        # Set up for a SINGLE_STRUCTURE stage (not MATERIAL_VALIDATION)
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_1"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"  # Requires validated materials
        base_state["design_description"] = {
            "stage_id": "stage_1",
            "design_description": "FDTD simulation",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = []  # Empty! Should fail
        
        result = code_generator_node(base_state)
        
        # Should NOT generate code - should have error
        assert "code" not in result or result.get("run_error"), \
            "Should fail when validated_materials is empty for Stage 1+"
    
    def test_comparison_validator_node_validates(self, base_state, valid_plan):
        """comparison_validator_node should validate comparisons."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "running"}]}
        base_state["figure_comparisons"] = []
        
        # This node doesn't use LLM, it uses local validation
        result = comparison_validator_node(base_state)
        
        assert "comparison_verdict" in result, "Should have comparison_verdict"
        assert result["comparison_verdict"] in ["approve", "needs_revision"]
    
    def test_generate_report_node_creates_report(self, base_state, valid_plan):
        """generate_report_node should create a final report."""
        from src.agents.reporting import generate_report_node
        
        mock_response = {
            "executive_summary": {"overall_assessment": []},
            "conclusions": {"main_physics_reproduced": True, "key_findings": []},
        }
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "completed_success"}]}
        
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
        
        # Should mark workflow complete
        assert result.get("workflow_complete") is True, "Should mark workflow_complete"
        assert result["workflow_phase"] == "reporting"


# ═══════════════════════════════════════════════════════════════════════
# Test: Business Logic Verification
# ═══════════════════════════════════════════════════════════════════════

class TestBusinessLogic:
    """Test that business logic is correctly implemented."""
    
    def test_plan_reviewer_rejects_empty_stages(self, base_state):
        """plan_reviewer_node should reject plans with no stages."""
        from src.agents.planning import plan_reviewer_node
        
        # Plan with no stages
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Empty Plan",
            "stages": [],  # No stages!
            "targets": [],
        }
        
        # Don't need to mock LLM - internal validation should catch this
        result = plan_reviewer_node(base_state)
        
        assert result["last_plan_review_verdict"] == "needs_revision", \
            "Should reject empty plan"
    
    def test_plan_reviewer_rejects_stages_without_targets(self, base_state):
        """plan_reviewer_node should reject stages without targets."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Bad Plan",
            "stages": [{
                "stage_id": "s1",
                "stage_type": "SINGLE_STRUCTURE",
                "targets": [],  # No targets!
                "dependencies": [],
            }],
            "targets": [],
        }
        
        result = plan_reviewer_node(base_state)
        
        assert result["last_plan_review_verdict"] == "needs_revision", \
            "Should reject stage without targets"
    
    def test_execution_validator_returns_verdict_from_llm(self, base_state):
        """execution_validator_node should return the LLM's verdict."""
        from src.agents.execution import execution_validator_node
        
        mock_response = {"verdict": "pass", "summary": "OK"}
        
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {
            "files": ["/tmp/output.csv"],
            "exit_code": 0,
        }
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
        
        # Verdict should match what LLM returned
        assert result.get("execution_verdict") == "pass", \
            f"Expected 'pass', got '{result.get('execution_verdict')}'"
        
        # Should also set workflow_phase
        assert "workflow_phase" in result, "Should set workflow_phase"


# ═══════════════════════════════════════════════════════════════════════
# Test: State Isolation
# ═══════════════════════════════════════════════════════════════════════

class TestStateIsolation:
    """Verify nodes don't accidentally mutate input state."""
    
    def test_plan_node_doesnt_mutate_input(self, base_state):
        """plan_node should not mutate the input state."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
            "targets": [],
            "extracted_parameters": [],
        }
        
        original_state = deepcopy(base_state)
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)
        
        # Input state should not be modified (except for known side effects)
        # Check key fields that should NOT change
        assert base_state.get("paper_id") == original_state.get("paper_id")
        assert base_state.get("paper_text") == original_state.get("paper_text")


# ═══════════════════════════════════════════════════════════════════════
# Test: Additional Coverage (from original test file)
# ═══════════════════════════════════════════════════════════════════════

class TestAdditionalCoverage:
    """Additional tests migrated from original test_agents_integration.py."""
        
    def test_plan_node_handles_missing_paper_text(self):
        """plan_node should handle missing paper text gracefully."""
        from src.agents.planning import plan_node
        
        state = create_initial_state(
            paper_id="test",
            paper_text="",  # Empty paper text
        )
        
        result = plan_node(state)
        
        # Should trigger ask_user for missing paper
        assert result.get("ask_user_trigger") == "missing_paper_text", \
            f"Expected 'missing_paper_text', got '{result.get('ask_user_trigger')}'"
        assert result.get("awaiting_user_input") is True
    
    def test_physics_sanity_passes(self, base_state):
        """physics_sanity_node should pass valid physics."""
        from src.agents.execution import physics_sanity_node
        
        mock_response = {
            "verdict": "pass",
            "summary": "Physics checks passed",
            "checks_performed": ["energy_conservation", "value_ranges"],
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "pass"
    
    def test_supervisor_handles_material_checkpoint(self, base_state):
        """supervisor_node should handle material checkpoint approval."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = [{"name": "gold", "path": "/materials/Au.csv"}]
        base_state["pending_user_questions"] = ["Approve materials?"]
        
        result = supervisor_node(base_state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("validated_materials") is not None
    
    def test_results_analyzer_sets_workflow_phase(self, base_state, valid_plan):
        """results_analyzer_node should set workflow_phase to analysis."""
        from src.agents.analysis import results_analyzer_node
        
        mock_response = {
            "overall_classification": "PARTIAL_MATCH",
            "figure_comparisons": [],
            "summary": "Analysis complete",
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/output.csv"]}
        
        with patch("src.agents.analysis.call_agent_with_metrics", return_value=mock_response):
            result = results_analyzer_node(base_state)
        
        assert result["workflow_phase"] == "analysis"
    
    def test_supervisor_defaults_on_llm_error(self, base_state):
        """supervisor_node should default to ok_continue on LLM error."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["current_stage_id"] = "stage_0"
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", 
                   side_effect=RuntimeError("LLM API error")):
            result = supervisor_node(base_state)
        
        # Should default to ok_continue (not block workflow)
        assert result["supervisor_verdict"] == "ok_continue"


# ═══════════════════════════════════════════════════════════════════════
# Test: Critical Files Exist
# ═══════════════════════════════════════════════════════════════════════

class TestCriticalFilesExist:
    """Verify all required prompt and schema files exist."""
    
    # All agents that should have prompts
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
    
    # All agents that should have schemas (use _output_schema.json naming)
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
        """Each agent must have a prompt file."""
        prompt_path = PROJECT_ROOT / "prompts" / f"{agent_name}_agent.md"
        assert prompt_path.exists(), f"Missing prompt file: {prompt_path}"
        
        # Verify it's not empty
        content = prompt_path.read_text()
        assert len(content) > 100, f"Prompt file too short: {prompt_path} ({len(content)} chars)"
    
    @pytest.mark.parametrize("schema_name", AGENTS_WITH_SCHEMAS)
    def test_schema_file_exists(self, schema_name):
        """Each agent must have a schema file."""
        schema_path = PROJECT_ROOT / "schemas" / f"{schema_name}_schema.json"
        assert schema_path.exists(), f"Missing schema file: {schema_path}"
        
        # Verify it's valid JSON
        content = schema_path.read_text()
        try:
            schema = json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {schema_path}: {e}")
        
        # Schema must have required and properties
        assert "properties" in schema, f"Schema missing 'properties': {schema_path}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Routing Functions Return Valid Values
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingFunctions:
    """Test that routing functions return expected values."""
    
    def test_route_after_plan_review_returns_valid_values(self, base_state, valid_plan):
        """route_after_plan_review must return valid routing targets."""
        from src.routing import route_after_plan_review
        
        base_state["plan"] = valid_plan
        
        # Test approve path
        base_state["last_plan_review_verdict"] = "approve"
        result = route_after_plan_review(base_state)
        assert result in ["select_stage", "plan", "ask_user"], \
            f"Invalid routing target: {result}"
        
        # Test needs_revision path
        base_state["last_plan_review_verdict"] = "needs_revision"
        result = route_after_plan_review(base_state)
        assert result in ["select_stage", "plan", "ask_user"], \
            f"Invalid routing target: {result}"
    
    def test_route_after_design_review_returns_valid_values(self, base_state):
        """route_after_design_review must return valid routing targets."""
        from src.routing import route_after_design_review
        
        base_state["current_stage_id"] = "stage_0"
        
        # Test approve path
        base_state["last_design_review_verdict"] = "approve"
        result = route_after_design_review(base_state)
        assert result in ["generate_code", "design", "ask_user"], \
            f"Invalid routing target: {result}"
        
        # Test needs_revision path
        base_state["last_design_review_verdict"] = "needs_revision"
        result = route_after_design_review(base_state)
        assert result in ["generate_code", "design", "ask_user"], \
            f"Invalid routing target: {result}"
    
    def test_route_after_code_review_returns_valid_values(self, base_state):
        """route_after_code_review must return valid routing targets."""
        from src.routing import route_after_code_review
        
        base_state["current_stage_id"] = "stage_0"
        
        # Test approve path
        base_state["last_code_review_verdict"] = "approve"
        result = route_after_code_review(base_state)
        assert result in ["run_code", "generate_code", "ask_user"], \
            f"Invalid routing target: {result}"
    
    def test_route_after_execution_check_returns_valid_values(self, base_state):
        """route_after_execution_check must return valid routing targets."""
        from src.routing import route_after_execution_check
        
        base_state["current_stage_id"] = "stage_0"
        
        # Test pass path
        base_state["execution_verdict"] = "pass"
        result = route_after_execution_check(base_state)
        assert result in ["physics_check", "generate_code", "ask_user"], \
            f"Invalid routing target: {result}"
        
        # Test fail path
        base_state["execution_verdict"] = "fail"
        result = route_after_execution_check(base_state)
        assert result in ["physics_check", "generate_code", "ask_user"], \
            f"Invalid routing target: {result}"
    
    def test_route_after_physics_check_returns_valid_values(self, base_state):
        """route_after_physics_check must return valid routing targets."""
        from src.routing import route_after_physics_check
        
        base_state["current_stage_id"] = "stage_0"
        
        # Test pass path
        base_state["physics_verdict"] = "pass"
        result = route_after_physics_check(base_state)
        valid_targets = ["analyze", "design", "generate_code", "ask_user"]
        assert result in valid_targets, f"Invalid routing target: {result}"
        
        # Test design_flaw path
        base_state["physics_verdict"] = "design_flaw"
        result = route_after_physics_check(base_state)
        assert result in valid_targets, f"Invalid routing target: {result}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Reviewer Counter Increments
# ═══════════════════════════════════════════════════════════════════════

class TestReviewerCounterIncrements:
    """Verify reviewer nodes increment counters on rejection."""
    
    def test_design_reviewer_increments_counter_on_rejection(self, base_state):
        """design_reviewer_node should increment counter on needs_revision."""
        from src.agents.design import design_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "issues": ["test"], "summary": "Fix"}
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 0
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
        
        assert result.get("design_revision_count") == 1, \
            f"Counter should increment to 1, got {result.get('design_revision_count')}"
    
    def test_code_reviewer_increments_counter_on_rejection(self, base_state):
        """code_reviewer_node should increment counter on needs_revision."""
        from src.agents.code import code_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "issues": ["bug"], "summary": "Fix"}
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 0
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
        
        assert result.get("code_revision_count") == 1, \
            f"Counter should increment to 1, got {result.get('code_revision_count')}"
    
    def test_plan_reviewer_increments_replan_counter_on_rejection(self, base_state):
        """plan_reviewer_node should increment replan_count on needs_revision."""
        from src.agents.planning import plan_reviewer_node
        
        # Use a plan with invalid structure to trigger rejection
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Bad Plan",
            "stages": [],  # Empty stages triggers rejection
            "targets": [],
        }
        base_state["replan_count"] = 0
        
        result = plan_reviewer_node(base_state)
        
        # MUST reject the empty plan
        assert result.get("last_plan_review_verdict") == "needs_revision", \
            f"Should reject empty plan, got '{result.get('last_plan_review_verdict')}'"
        
        # MUST increment replan counter
        assert result.get("replan_count") == 1, \
            f"Should increment replan_count to 1, got {result.get('replan_count')}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Validator Verdicts
# ═══════════════════════════════════════════════════════════════════════

class TestValidatorVerdicts:
    """Test various validator verdict scenarios."""
    
    def test_physics_sanity_returns_design_flaw(self, base_state):
        """physics_sanity_node should return design_flaw verdict when appropriate."""
        from src.agents.execution import physics_sanity_node
        
        mock_response = {
            "verdict": "design_flaw",
            "summary": "Simulation parameters inconsistent with physics",
            "design_issues": ["Wavelength range too narrow"],
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = physics_sanity_node(base_state)
        
        assert result["physics_verdict"] == "design_flaw", \
            f"Expected 'design_flaw', got '{result['physics_verdict']}'"
    
    def test_execution_validator_returns_fail(self, base_state):
        """execution_validator_node should return fail verdict when appropriate."""
        from src.agents.execution import execution_validator_node
        
        mock_response = {
            "verdict": "fail",
            "summary": "Simulation crashed",
            "error_analysis": "Memory allocation failure",
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {
            "files": [],
            "exit_code": 1,
            "stderr": "Segmentation fault",
        }
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
        
        assert result["execution_verdict"] == "fail", \
            f"Expected 'fail', got '{result['execution_verdict']}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Schema Validation in LLM Calls  
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaInLLMCalls:
    """Verify correct agent_name is passed to LLM (determines schema)."""
    
    def test_planner_uses_correct_agent_name(self, base_state):
        """plan_node must call LLM with agent_name='planner'."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
            "targets": [],
            "extracted_parameters": [],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response) as mock:
            plan_node(base_state)
            
            call_kwargs = mock.call_args.kwargs
            agent_name = call_kwargs.get("agent_name", "")
            assert agent_name == "planner", \
                f"Expected agent_name='planner', got '{agent_name}'"
    
    def test_code_generator_uses_correct_agent_name(self, base_state, valid_plan):
        """code_generator_node must call LLM with agent_name='code_generator'."""
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
            "design_description": "FDTD simulation for gold nanorod",  # Not a stub
            "geometry": [{"type": "cylinder", "radius": 20}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_generator_node(base_state)
            
            # LLM MUST be called - fail if not
            assert mock.called, "code_generator_node should call LLM"
            
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("agent_name") == "code_generator", \
                f"Expected agent_name='code_generator', got '{call_kwargs.get('agent_name')}'"
            
            # Verify system_prompt is substantial
            system_prompt = call_kwargs.get("system_prompt", "")
            assert len(system_prompt) > 100, \
                f"System prompt too short ({len(system_prompt)} chars)"


# ═══════════════════════════════════════════════════════════════════════
# Test: Edge Cases and Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases that might reveal bugs."""
    
    def test_plan_node_with_very_short_paper(self):
        """plan_node should reject papers that are too short."""
        from src.agents.planning import plan_node
        
        state = create_initial_state(
            paper_id="test",
            paper_text="Short paper.",  # Too short to be useful
        )
        
        result = plan_node(state)
        
        # Should detect paper is too short
        assert result.get("ask_user_trigger") == "missing_paper_text" or \
               result.get("awaiting_user_input") is True, \
            "Should reject very short paper"
    
    def test_select_stage_with_no_plan(self, base_state):
        """select_stage_node should handle missing plan gracefully."""
        from src.agents.stage_selection import select_stage_node
        
        # No plan, empty progress (not None - that causes a crash)
        base_state["plan"] = {}
        base_state["progress"] = {}
        
        result = select_stage_node(base_state)
        
        # Should indicate error or select nothing
        assert result.get("current_stage_id") is None or \
               result.get("ask_user_trigger") is not None, \
            "Should handle missing plan gracefully"
    
    def test_select_stage_handles_none_progress(self, base_state):
        """select_stage_node should handle None progress gracefully, not crash."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = {}
        base_state["progress"] = None  # This should be handled gracefully
        
        # BUG: Currently crashes with AttributeError instead of handling gracefully
        # Should return error state, not crash
        result = select_stage_node(base_state)
        
        # Should not crash and should indicate error
        assert result.get("current_stage_id") is None or \
               result.get("ask_user_trigger") is not None, \
            "BUG: Should handle None progress gracefully"
    
    def test_code_generator_with_stub_design(self, base_state, valid_plan):
        """code_generator_node should reject stub/placeholder designs."""
        from src.agents.code import code_generator_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "TODO: Add design",  # Stub!
            "geometry": [],
        }
        
        result = code_generator_node(base_state)
        
        # Should detect stub and not generate code
        assert "code" not in result or result.get("run_error"), \
            "Should reject stub design"
    
    def test_supervisor_with_unknown_trigger(self, base_state):
        """supervisor_node should handle unknown ask_user_trigger."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["ask_user_trigger"] = "unknown_trigger_xyz"
        base_state["user_responses"] = {"Q1": "yes"}
        base_state["pending_user_questions"] = ["Unknown question"]
        
        result = supervisor_node(base_state)
        
        # Should not crash, should return some verdict
        assert "supervisor_verdict" in result, \
            "Should handle unknown trigger gracefully"
    
    def test_results_analyzer_with_empty_outputs(self, base_state, valid_plan):
        """results_analyzer_node should handle empty stage_outputs."""
        from src.agents.analysis import results_analyzer_node
        
        mock_response = {
            "overall_classification": "NO_DATA",
            "figure_comparisons": [],
            "summary": "No data to analyze",
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}  # Empty!
        
        with patch("src.agents.analysis.call_agent_with_metrics", return_value=mock_response):
            result = results_analyzer_node(base_state)
        
        # Should not crash, should set workflow_phase
        assert result.get("workflow_phase") == "analysis"


# ═══════════════════════════════════════════════════════════════════════
# Test: Field Mapping Correctness
# ═══════════════════════════════════════════════════════════════════════

class TestFieldMapping:
    """Verify LLM output fields are correctly mapped to state."""
    
    def test_planner_maps_stages_correctly(self, base_state):
        """plan_node should correctly map stages from LLM output."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "paper_id": "test_paper",
            "title": "Test Title",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "FDTD_DIRECT", "targets": ["Fig2"], "dependencies": ["stage_0"]},
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
            "extracted_parameters": [{"name": "wavelength", "value": 500}],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)
        
        # Verify stages are mapped correctly
        plan = result.get("plan", {})
        assert len(plan.get("stages", [])) == 2, "Should have 2 stages"
        assert plan["stages"][0]["stage_id"] == "stage_0"
        assert plan["stages"][1]["dependencies"] == ["stage_0"]
    
    def test_designer_maps_geometry_correctly(self, base_state, valid_plan):
        """simulation_designer_node should map geometry fields correctly."""
        from src.agents.design import simulation_designer_node
        
        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Test design",
            "geometry": [
                {"type": "cylinder", "radius": 20, "height": 100, "material": "gold"},
                {"type": "box", "size": [500, 500, 200], "material": "water"},
            ],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = simulation_designer_node(base_state)
        
        design = result.get("design_description", {})
        assert len(design.get("geometry", [])) == 2, "Should have 2 geometry objects"
        assert design["geometry"][0]["type"] == "cylinder"


# ═══════════════════════════════════════════════════════════════════════
# Test: Progress Initialization
# ═══════════════════════════════════════════════════════════════════════

class TestProgressInitialization:
    """Verify progress is correctly initialized from plan."""
    
    def test_plan_node_initializes_progress(self, base_state):
        """plan_node should initialize progress with stage statuses."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "FDTD_DIRECT", "targets": ["Fig1"], "dependencies": ["stage_0"]},
            ],
            "targets": [{"figure_id": "Fig1"}],
            "extracted_parameters": [],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_node(base_state)
        
        # Progress MUST be initialized
        assert "progress" in result, "Should initialize progress"
        progress = result["progress"]
        
        # Progress must have stages
        assert "stages" in progress, "Progress should have stages"
        assert len(progress["stages"]) == 2, \
            f"Progress should have 2 stages, got {len(progress.get('stages', []))}"
        
        # Each stage should have status
        for stage in progress["stages"]:
            assert "status" in stage, f"Stage {stage.get('stage_id')} missing status"
            assert stage["status"] == "not_started", \
                f"Initial status should be 'not_started', got '{stage['status']}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Feedback Fields
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackFields:
    """Verify feedback fields are populated correctly."""
    
    def test_design_reviewer_populates_feedback_on_rejection(self, base_state):
        """design_reviewer_node should populate reviewer_feedback on rejection."""
        from src.agents.design import design_reviewer_node
        
        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "major", "description": "Missing wavelength range"},
                {"severity": "minor", "description": "Consider adding symmetry"},
            ],
            "summary": "Design needs several improvements",
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0", "geometry": []}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
        
        # Should have feedback
        assert "reviewer_feedback" in result, "Should include reviewer_feedback"
        feedback = result["reviewer_feedback"]
        assert len(feedback) > 20, f"Feedback too short: '{feedback}'"
        
        # Should preserve issues for debugging
        assert "design_review_issues" in result or "issues" in str(result), \
            "Should preserve review issues"
    
    def test_code_reviewer_populates_feedback_on_rejection(self, base_state):
        """code_reviewer_node should populate reviewer_feedback on rejection."""
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
        
        # Should have feedback
        assert "reviewer_feedback" in result, "Should include reviewer_feedback"
        feedback = result["reviewer_feedback"]
        assert len(feedback) > 20, f"Feedback too short: '{feedback}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: All Reviewer Output Fields
# ═══════════════════════════════════════════════════════════════════════

class TestReviewerOutputFields:
    """Verify reviewers set all required output fields."""
    
    def test_design_reviewer_sets_all_fields_on_approve(self, base_state):
        """design_reviewer_node should set all fields on approve."""
        from src.agents.design import design_reviewer_node
        
        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Design looks good",
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0", "geometry": [{"type": "box"}]}
        base_state["design_revision_count"] = 2  # Already had some revisions
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
        
        # Verify required output fields
        assert result.get("last_design_review_verdict") == "approve"
        assert "workflow_phase" in result
        
        # BUG: design_revision_count should be included in result even on approve
        # so downstream nodes know how many revisions occurred
        assert "design_revision_count" in result, \
            "BUG: design_revision_count not included on approve - should preserve counter"
    
    def test_code_reviewer_sets_all_fields_on_approve(self, base_state):
        """code_reviewer_node should set all fields on approve."""
        from src.agents.code import code_reviewer_node
        
        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good",
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('good code')"
        base_state["code_revision_count"] = 3  # Already had some revisions
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
        
        # Verify required output fields
        assert result.get("last_code_review_verdict") == "approve"
        assert "workflow_phase" in result
        
        # BUG: code_revision_count should be included in result even on approve
        # so downstream nodes know how many revisions occurred
        assert "code_revision_count" in result, \
            "BUG: code_revision_count not included on approve - should preserve counter"


# ═══════════════════════════════════════════════════════════════════════
# Test: Real Prompt Building (No LLM Mock)
# ═══════════════════════════════════════════════════════════════════════

class TestRealPromptBuilding:
    """Test that prompts are built correctly using real code."""
    
    def test_planner_prompt_includes_paper_text(self, base_state):
        """Planner prompt should include paper text."""
        from src.prompts import build_agent_prompt
        from src.llm_client import build_user_content_for_planner
        
        system_prompt = build_agent_prompt("planner", base_state)
        user_content = build_user_content_for_planner(base_state)
        
        # System prompt should be substantial
        assert len(system_prompt) > 500, \
            f"System prompt too short ({len(system_prompt)} chars)"
        
        # User content should include paper text
        if isinstance(user_content, str):
            assert "gold nanorod" in user_content.lower() or \
                   "optical properties" in user_content.lower(), \
                "User content should include paper text"
        elif isinstance(user_content, list):
            # Multi-modal content
            text_parts = [p.get("text", "") for p in user_content if p.get("type") == "text"]
            combined = " ".join(text_parts).lower()
            assert "gold nanorod" in combined or "optical properties" in combined, \
                "User content should include paper text"
    
    def test_code_generator_prompt_includes_design(self, base_state, valid_plan):
        """Code generator prompt should include design details."""
        from src.prompts import build_agent_prompt
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for extinction",
            "geometry": [{"type": "cylinder", "radius": 20}],
        }
        
        system_prompt = build_agent_prompt("code_generator", base_state)
        
        # Should be substantial
        assert len(system_prompt) > 500, \
            f"System prompt too short ({len(system_prompt)} chars)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

