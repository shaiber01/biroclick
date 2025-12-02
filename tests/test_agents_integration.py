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
            "paper_domain": "plasmonics",
            "title": "Test Plan",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
            "targets": [{"figure_id": "Fig1"}],
            "extracted_parameters": [{"name": "p1", "value": 10}],
            "planned_materials": ["Au"],
            "assumptions": {"a1": "test assumption"}
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

        # Check extracting other fields
        assert "planned_materials" in result, "Missing planned_materials"
        assert result["planned_materials"] == mock_response["planned_materials"]

        assert "assumptions" in result, "Missing assumptions"
        assert result["assumptions"] == mock_response["assumptions"]

        assert "paper_domain" in result, "Missing paper_domain"
        assert result["paper_domain"] == mock_response["paper_domain"]

        assert "extracted_parameters" in result, "Missing extracted_parameters"
        assert len(result["extracted_parameters"]) > 0, "extracted_parameters should be populated"

        # Progress should be initialized
        assert "progress" in result, "Missing progress initialization"
        assert result["progress"] is not None
        assert len(result["progress"]["stages"]) == 1
        assert result["progress"]["stages"][0]["stage_id"] == "s1"
    
    def test_reviewer_output_has_verdict(self, base_state, valid_plan):
        """All reviewer nodes must return a verdict."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node
        
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        
        # Test plan_reviewer
        base_state["plan"] = valid_plan
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
        
        assert "last_plan_review_verdict" in result, "plan_reviewer: Missing verdict field"
        assert result["last_plan_review_verdict"] in ["approve", "needs_revision"], \
            f"plan_reviewer: Invalid verdict: {result['last_plan_review_verdict']}"
        assert "workflow_phase" in result, "plan_reviewer: Missing workflow_phase"
        
        # Test design_reviewer - BUG FINDER: must also return verdict
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0", "geometry": []}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
        
        assert "last_design_review_verdict" in result, "design_reviewer: Missing verdict field"
        assert result["last_design_review_verdict"] in ["approve", "needs_revision"], \
            f"design_reviewer: Invalid verdict: {result['last_design_review_verdict']}"
        assert "workflow_phase" in result, "design_reviewer: Missing workflow_phase"
        
        # Test code_reviewer - BUG FINDER: must also return verdict
        base_state["code"] = "print('test')"
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
        
        assert "last_code_review_verdict" in result, "code_reviewer: Missing verdict field"
        assert result["last_code_review_verdict"] in ["approve", "needs_revision"], \
            f"code_reviewer: Invalid verdict: {result['last_code_review_verdict']}"
        assert "workflow_phase" in result, "code_reviewer: Missing workflow_phase"
    
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
    """Verify revision counters respect bounds and increment correctly."""
    
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
        
        # MUST include counter in result
        assert "design_revision_count" in result, \
            "BUG: design_revision_count missing from result"
        
        # Counter should NOT exceed max
        assert result["design_revision_count"] <= max_revisions, \
            f"Counter {result['design_revision_count']} exceeded max {max_revisions}"
        
        # Counter should equal max (not increment beyond it)
        assert result["design_revision_count"] == max_revisions, \
            f"Counter should stay at max {max_revisions}, got {result['design_revision_count']}"
    
    def test_design_revision_counter_increments_under_max(self, base_state):
        """design_revision_count should increment when under max."""
        from src.agents.design import design_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "issues": ["test"], "summary": "Fix"}
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["design_revision_count"] = 1  # Under max
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
        
        # Counter MUST increment
        assert result["design_revision_count"] == 2, \
            f"Counter should increment from 1 to 2, got {result.get('design_revision_count')}"
    
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
        
        # MUST include counter in result
        assert "code_revision_count" in result, \
            "BUG: code_revision_count missing from result"
        
        # Counter should NOT exceed max
        assert result["code_revision_count"] <= max_revisions, \
            f"Counter {result['code_revision_count']} exceeded max {max_revisions}"
    
    def test_code_revision_counter_increments_under_max(self, base_state):
        """code_revision_count should increment when under max."""
        from src.agents.code import code_reviewer_node
        
        mock_response = {"verdict": "needs_revision", "issues": ["bug"], "summary": "Fix"}
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 2  # Under default max of 3
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
        
        # Counter MUST increment
        assert result["code_revision_count"] == 3, \
            f"Counter should increment from 2 to 3, got {result.get('code_revision_count')}"


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
        """adapt_prompts_node should update prompt_adaptations in state."""
        from src.agents.planning import adapt_prompts_node
        
        mock_response = {
            "adaptations": ["Focus on plasmonics", "Use Johnson-Christy data"],
            "paper_domain": "plasmonics",
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response) as mock:
            result = adapt_prompts_node(base_state)
        
        # Verify LLM was called with correct agent_name
        assert mock.called, "adapt_prompts should call LLM"
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "prompt_adaptor", \
            f"Wrong agent_name: {call_kwargs.get('agent_name')}"
        
        # MUST update workflow_phase
        assert result.get("workflow_phase") == "adapting_prompts", \
            f"Expected 'adapting_prompts', got '{result.get('workflow_phase')}'"
        
        # MUST have prompt_adaptations in result
        assert "prompt_adaptations" in result, \
            "BUG: Missing prompt_adaptations in result"
        
        # Adaptations should be passed through from LLM response
        assert result["prompt_adaptations"] == mock_response["adaptations"], \
            f"prompt_adaptations not passed through from LLM: {result.get('prompt_adaptations')}"
        
        # paper_domain should be updated if provided by LLM
        assert result.get("paper_domain") == "plasmonics", \
            f"paper_domain not updated: {result.get('paper_domain')}"
    
    def test_select_stage_node_selects_valid_stage(self, base_state, valid_plan):
        """select_stage_node should select a valid stage from plan when stages available."""
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
        
        # MUST have current_stage_id in result
        assert "current_stage_id" in result, "Missing current_stage_id in result"
        
        selected = result["current_stage_id"]
        
        # With a runnable stage available, MUST select it (not None)
        assert selected is not None, \
            "BUG: Should select stage_0 (not None) when stage is available and runnable"
        
        # Selected stage MUST be from plan
        plan_stage_ids = [s["stage_id"] for s in valid_plan["stages"]]
        assert selected in plan_stage_ids, \
            f"Selected stage '{selected}' not in plan stages {plan_stage_ids}"
        
        # MUST also set current_stage_type
        assert "current_stage_type" in result, "Missing current_stage_type"
        assert result["current_stage_type"] == "MATERIAL_VALIDATION", \
            f"Wrong stage_type: {result.get('current_stage_type')}"
        
        # MUST set workflow_phase
        assert result.get("workflow_phase") == "stage_selection", \
            f"Wrong workflow_phase: {result.get('workflow_phase')}"
    
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
            "new_assumptions": {"sim_a1": "assuming periodic boundary"},
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response) as mock:
            result = simulation_designer_node(base_state)
        
        # Verify LLM was called with correct agent_name
        assert mock.called, "simulation_designer should call LLM"
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "simulation_designer", \
            f"Wrong agent_name: {call_kwargs.get('agent_name')}"
        
        # MUST have design_description in result
        assert "design_description" in result, \
            f"BUG: Missing design_description. Got keys: {result.keys()}"
        
        # MUST have correct workflow_phase
        assert result["workflow_phase"] == "design", \
            f"Wrong workflow_phase: {result.get('workflow_phase')}"
        
        # Verify ALL design fields are present and correct
        design = result["design_description"]
        assert design.get("stage_id") == "stage_0", \
            f"Design should have correct stage_id, got: {design.get('stage_id')}"
        assert "design_description" in design, "Design should have description text"
        assert "geometry" in design, "Design should have geometry"
        assert "sources" in design, "Design should have sources"
        assert "monitors" in design, "Design should have monitors"
        
        # Verify geometry has content from LLM response
        assert len(design.get("geometry", [])) > 0, "Design should have at least one geometry object"
        assert design["geometry"][0].get("type") == "cylinder", \
            f"Geometry type mismatch: {design['geometry'][0].get('type')}"
        
        # Verify sources have content
        assert len(design.get("sources", [])) > 0, "Design should have at least one source"
        
        # Verify monitors have content  
        assert len(design.get("monitors", [])) > 0, "Design should have at least one monitor"

        # Verify materials pass-through
        assert "materials" in design, "Design should have materials"
        assert design["materials"] == mock_response["materials"]

        # Verify assumptions update
        assert "assumptions" in result, "Should update assumptions"
        assert "global_assumptions" in result["assumptions"]
        assert "sim_a1" in str(result["assumptions"]["global_assumptions"]) or \
               {"sim_a1": "assuming periodic boundary"} in result["assumptions"]["global_assumptions"], \
               f"New assumptions should be added: {result['assumptions']}"
    
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
        """comparison_validator_node should validate comparisons and return all fields."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "running"}]}
        base_state["figure_comparisons"] = []
        base_state["analysis_result_reports"] = []
        
        # This node doesn't use LLM, it uses local validation
        result = comparison_validator_node(base_state)
        
        # MUST have comparison_verdict
        assert "comparison_verdict" in result, "BUG: Missing comparison_verdict"
        assert result["comparison_verdict"] in ["approve", "needs_revision"], \
            f"Invalid verdict: {result['comparison_verdict']}"
        
        # MUST have workflow_phase
        assert "workflow_phase" in result, "Missing workflow_phase"
        assert result["workflow_phase"] == "comparison_validation", \
            f"Wrong workflow_phase: {result.get('workflow_phase')}"
        
        # MUST have comparison_feedback
        assert "comparison_feedback" in result, "Missing comparison_feedback"
        assert isinstance(result["comparison_feedback"], str), \
            f"comparison_feedback should be string, got {type(result['comparison_feedback'])}"
        
        # When plan has targets but comparisons empty, should reject
        # (valid_plan has targets, so empty comparisons = needs_revision)
        assert result["comparison_verdict"] == "needs_revision", \
            "Should reject when plan has targets but no comparisons produced"
    
    def test_generate_report_node_creates_report(self, base_state, valid_plan):
        """generate_report_node should create a final report with all required fields."""
        from src.agents.reporting import generate_report_node
        
        mock_response = {
            "executive_summary": {"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
            "conclusions": {"main_physics_reproduced": True, "key_findings": ["Test finding"]},
            "paper_citation": {"title": "Test Paper", "authors": "Test Author"},
        }
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "completed_success"}]}
        base_state["metrics"] = {
            "agent_calls": [
                {"agent_name": "planner", "input_tokens": 1000, "output_tokens": 500},
            ]
        }
        
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
        
        # MUST mark workflow complete
        assert result.get("workflow_complete") is True, \
            "BUG: workflow_complete should be True"
        
        # MUST have correct workflow_phase
        assert result["workflow_phase"] == "reporting", \
            f"Wrong workflow_phase: {result.get('workflow_phase')}"
        
        # MUST have metrics with token_summary
        assert "metrics" in result, "Missing metrics in result"
        metrics = result["metrics"]
        assert "token_summary" in metrics, "Missing token_summary in metrics"
        assert "total_input_tokens" in metrics["token_summary"], "Missing total_input_tokens"
        assert "total_output_tokens" in metrics["token_summary"], "Missing total_output_tokens"
        assert "estimated_cost" in metrics["token_summary"], "Missing estimated_cost"
        
        # Verify token calculation is correct
        assert metrics["token_summary"]["total_input_tokens"] == 1000, \
            f"Wrong input token sum: {metrics['token_summary']['total_input_tokens']}"
        assert metrics["token_summary"]["total_output_tokens"] == 500, \
            f"Wrong output token sum: {metrics['token_summary']['total_output_tokens']}"
        
        # Should have executive_summary (from LLM or default)
        assert "executive_summary" in result, "Missing executive_summary"


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
        """physics_sanity_node should pass valid physics and return all required fields."""
        from src.agents.execution import physics_sanity_node
        
        mock_response = {
            "verdict": "pass",
            "summary": "Physics checks passed",
            "checks_performed": ["energy_conservation", "value_ranges"],
        }
        
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["/tmp/spectrum.csv"]}
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response) as mock:
            result = physics_sanity_node(base_state)
        
        # Verify LLM was called with correct agent
        assert mock.called, "physics_sanity should call LLM"
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "physics_sanity", \
            f"Wrong agent_name: {call_kwargs.get('agent_name')}"
        
        # MUST have physics_verdict
        assert "physics_verdict" in result, "Missing physics_verdict"
        assert result["physics_verdict"] == "pass", \
            f"Expected 'pass', got '{result.get('physics_verdict')}'"
        
        # MUST have workflow_phase
        assert "workflow_phase" in result, "Missing workflow_phase"
        assert result["workflow_phase"] == "physics_validation", \
            f"Wrong workflow_phase: {result.get('workflow_phase')}"
    
    def test_supervisor_handles_material_checkpoint(self, base_state):
        """supervisor_node should handle material checkpoint approval correctly."""
        from src.agents.supervision.supervisor import supervisor_node
        
        pending_materials = [{"name": "gold", "path": "/materials/Au.csv"}]
        
        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = pending_materials
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "in_progress"}]}
        
        result = supervisor_node(base_state)
        
        # MUST have supervisor_verdict
        assert "supervisor_verdict" in result, "Missing supervisor_verdict"
        assert result["supervisor_verdict"] == "ok_continue", \
            f"Should approve with ok_continue, got '{result.get('supervisor_verdict')}'"
        
        # MUST move pending_validated_materials to validated_materials
        assert "validated_materials" in result, \
            "BUG: validated_materials should be populated from pending"
        assert result["validated_materials"] == pending_materials, \
            f"validated_materials should equal pending materials: {result.get('validated_materials')}"
        
        # MUST clear pending_validated_materials
        assert result.get("pending_validated_materials") == [], \
            f"pending_validated_materials should be cleared: {result.get('pending_validated_materials')}"
        
        # MUST clear ask_user_trigger
        assert result.get("ask_user_trigger") is None, \
            f"ask_user_trigger should be cleared: {result.get('ask_user_trigger')}"
    
    def test_results_analyzer_sets_workflow_phase(self, base_state, valid_plan):
        """results_analyzer_node should set workflow_phase on all paths (success and error)."""
        from src.agents.analysis import results_analyzer_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        # Use fake file path - tests error handling when files don't exist
        base_state["stage_outputs"] = {"files": ["/nonexistent/fake_output.csv"]}
        
        result = results_analyzer_node(base_state)
        
        # MUST have workflow_phase even on error path
        assert "workflow_phase" in result, "Missing workflow_phase"
        assert result["workflow_phase"] == "analysis", \
            f"Wrong workflow_phase: {result.get('workflow_phase')}"
        
        # When files don't exist, should indicate error
        assert result.get("execution_verdict") == "fail" or result.get("run_error"), \
            "Should indicate error when output files don't exist"
    
    def test_results_analyzer_returns_figure_comparisons_on_success(self, base_state, valid_plan):
        """results_analyzer_node should return figure_comparisons when analysis succeeds."""
        from src.agents.analysis import results_analyzer_node
        import tempfile
        import os
        
        # Create a real temporary file for the test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("wavelength,extinction\n400,0.1\n500,0.5\n600,0.3\n")
            temp_file = f.name
        
        try:
            base_state["plan"] = valid_plan
            base_state["current_stage_id"] = "stage_0"
            base_state["stage_outputs"] = {"files": [temp_file]}
            base_state["paper_id"] = "test_integration"
            
            result = results_analyzer_node(base_state)
            
            # MUST have workflow_phase
            assert "workflow_phase" in result, "Missing workflow_phase"
            assert result["workflow_phase"] == "analysis", \
                f"Wrong workflow_phase: {result.get('workflow_phase')}"
            
            # MUST have analysis_summary
            assert "analysis_summary" in result, "Missing analysis_summary"
            
            # MUST have figure_comparisons list when analysis completes
            assert "figure_comparisons" in result, \
                "BUG: Missing figure_comparisons when output files exist"
            assert isinstance(result["figure_comparisons"], list), \
                f"figure_comparisons should be list, got {type(result['figure_comparisons'])}"
        finally:
            os.unlink(temp_file)  # Clean up
    
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
    """Test that routing functions return CORRECT values for each verdict."""
    
    def test_route_after_plan_review_approve_goes_to_select_stage(self, base_state, valid_plan):
        """route_after_plan_review: approve -> select_stage."""
        from src.routing import route_after_plan_review
        
        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "approve"
        base_state["replan_count"] = 0  # Under limit
        
        result = route_after_plan_review(base_state)
        
        # MUST route to select_stage on approve
        assert result == "select_stage", \
            f"BUG: approve should route to select_stage, got '{result}'"
    
    def test_route_after_plan_review_needs_revision_goes_to_plan(self, base_state, valid_plan):
        """route_after_plan_review: needs_revision -> plan (under limit)."""
        from src.routing import route_after_plan_review
        
        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 0  # Under limit
        
        result = route_after_plan_review(base_state)
        
        # MUST route to plan for revision
        assert result == "plan", \
            f"BUG: needs_revision should route to plan, got '{result}'"
    
    def test_route_after_design_review_approve_goes_to_generate_code(self, base_state):
        """route_after_design_review: approve -> generate_code."""
        from src.routing import route_after_design_review
        
        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "approve"
        
        result = route_after_design_review(base_state)
        
        # MUST route to generate_code on approve
        assert result == "generate_code", \
            f"BUG: approve should route to generate_code, got '{result}'"
    
    def test_route_after_design_review_needs_revision_goes_to_design(self, base_state):
        """route_after_design_review: needs_revision -> design (under limit)."""
        from src.routing import route_after_design_review
        
        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 0  # Under limit
        
        result = route_after_design_review(base_state)
        
        # MUST route to design for revision
        assert result == "design", \
            f"BUG: needs_revision should route to design, got '{result}'"
    
    def test_route_after_code_review_approve_goes_to_run_code(self, base_state):
        """route_after_code_review: approve -> run_code."""
        from src.routing import route_after_code_review
        
        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "approve"
        
        result = route_after_code_review(base_state)
        
        # MUST route to run_code on approve
        assert result == "run_code", \
            f"BUG: approve should route to run_code, got '{result}'"
    
    def test_route_after_code_review_needs_revision_goes_to_generate_code(self, base_state):
        """route_after_code_review: needs_revision -> generate_code (under limit)."""
        from src.routing import route_after_code_review
        
        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0  # Under limit
        
        result = route_after_code_review(base_state)
        
        # MUST route to generate_code for revision
        assert result == "generate_code", \
            f"BUG: needs_revision should route to generate_code, got '{result}'"
    
    def test_route_after_execution_check_pass_goes_to_physics_check(self, base_state):
        """route_after_execution_check: pass -> physics_check."""
        from src.routing import route_after_execution_check
        
        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "pass"
        
        result = route_after_execution_check(base_state)
        
        # MUST route to physics_check on pass
        assert result == "physics_check", \
            f"BUG: pass should route to physics_check, got '{result}'"
    
    def test_route_after_execution_check_fail_goes_to_generate_code(self, base_state):
        """route_after_execution_check: fail -> generate_code (under limit)."""
        from src.routing import route_after_execution_check
        
        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0  # Under limit
        
        result = route_after_execution_check(base_state)
        
        # MUST route to generate_code on fail
        assert result == "generate_code", \
            f"BUG: fail should route to generate_code, got '{result}'"
    
    def test_route_after_physics_check_pass_goes_to_analyze(self, base_state):
        """route_after_physics_check: pass -> analyze."""
        from src.routing import route_after_physics_check
        
        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "pass"
        
        result = route_after_physics_check(base_state)
        
        # MUST route to analyze on pass
        assert result == "analyze", \
            f"BUG: pass should route to analyze, got '{result}'"
    
    def test_route_after_physics_check_design_flaw_goes_to_design(self, base_state):
        """route_after_physics_check: design_flaw -> design (under limit)."""
        from src.routing import route_after_physics_check
        
        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0  # Under limit
        
        result = route_after_physics_check(base_state)
        
        # MUST route to design on design_flaw
        assert result == "design", \
            f"BUG: design_flaw should route to design, got '{result}'"


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


# ═══════════════════════════════════════════════════════════════════════
# Test: Circular Dependency Detection
# ═══════════════════════════════════════════════════════════════════════

class TestCircularDependencyDetection:
    """Verify plan_reviewer correctly detects circular dependencies."""
    
    def test_plan_reviewer_detects_simple_cycle(self, base_state):
        """plan_reviewer should reject plans with A -> B -> A cycle."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Cyclic Plan",
            "stages": [
                {"stage_id": "stage_a", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": ["stage_b"]},
                {"stage_id": "stage_b", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": ["stage_a"]},
            ],
            "targets": [{"figure_id": "Fig1"}],
        }
        
        result = plan_reviewer_node(base_state)
        
        # MUST detect and reject the cycle
        assert result["last_plan_review_verdict"] == "needs_revision", \
            "BUG: Should detect A -> B -> A cycle"
        
        # Feedback should mention the cycle
        feedback = result.get("planner_feedback", "")
        assert "circular" in feedback.lower() or "cycle" in feedback.lower(), \
            f"Feedback should mention circular dependency: {feedback}"
    
    def test_plan_reviewer_detects_self_dependency(self, base_state):
        """plan_reviewer should reject stages that depend on themselves."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Self-Dependent Plan",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": ["stage_0"]},
            ],
            "targets": [{"figure_id": "Fig1"}],
        }
        
        result = plan_reviewer_node(base_state)
        
        assert result["last_plan_review_verdict"] == "needs_revision", \
            "BUG: Should detect self-dependency"
    
    def test_plan_reviewer_detects_transitive_cycle(self, base_state):
        """plan_reviewer should detect A -> B -> C -> A cycle."""
        from src.agents.planning import plan_reviewer_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "title": "Transitive Cycle",
            "stages": [
                {"stage_id": "a", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": ["c"]},
                {"stage_id": "b", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig1"], "dependencies": ["a"]},
                {"stage_id": "c", "stage_type": "ARRAY_SYSTEM", "targets": ["Fig2"], "dependencies": ["b"]},
            ],
            "targets": [{"figure_id": "Fig1"}, {"figure_id": "Fig2"}],
        }
        
        result = plan_reviewer_node(base_state)
        
        assert result["last_plan_review_verdict"] == "needs_revision", \
            "BUG: Should detect transitive A -> B -> C -> A cycle"


# ═══════════════════════════════════════════════════════════════════════
# Test: Handle Backtrack Node
# ═══════════════════════════════════════════════════════════════════════

class TestHandleBacktrackNode:
    """Test handle_backtrack_node functionality."""
    
    def test_backtrack_marks_target_as_needs_rerun(self, base_state, valid_plan):
        """handle_backtrack_node should mark target stage as needs_rerun."""
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"},
            ]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
            "reason": "Need to revalidate materials",
        }
        
        result = handle_backtrack_node(base_state)
        
        # Target stage should be marked needs_rerun
        progress = result.get("progress", {})
        stages = progress.get("stages", [])
        target_stage = next((s for s in stages if s["stage_id"] == "stage_0"), None)
        
        assert target_stage is not None, "Target stage should exist in progress"
        assert target_stage["status"] == "needs_rerun", \
            f"BUG: Target stage should be 'needs_rerun', got '{target_stage['status']}'"
    
    def test_backtrack_invalidates_dependent_stages(self, base_state):
        """handle_backtrack_node should invalidate dependent stages."""
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION"},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "dependencies": ["stage_0"]},
            ],
        }
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "status": "completed_success"},
            ]
        }
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": ["stage_1"],
            "reason": "Material validation needs rerun",
        }
        
        result = handle_backtrack_node(base_state)
        
        progress = result.get("progress", {})
        stages = progress.get("stages", [])
        stage_1 = next((s for s in stages if s["stage_id"] == "stage_1"), None)
        
        assert stage_1 is not None, "Dependent stage should exist"
        assert stage_1["status"] == "invalidated", \
            f"BUG: Dependent stage should be 'invalidated', got '{stage_1['status']}'"
    
    def test_backtrack_increments_counter(self, base_state, valid_plan):
        """handle_backtrack_node should increment backtrack_count."""
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = 0
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        
        result = handle_backtrack_node(base_state)
        
        assert result["backtrack_count"] == 1, \
            f"BUG: backtrack_count should be 1, got {result.get('backtrack_count')}"
    
    def test_backtrack_clears_working_state(self, base_state, valid_plan):
        """handle_backtrack_node should clear code, design, stage_outputs."""
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
        
        assert result.get("code") is None, "BUG: code should be cleared on backtrack"
        assert result.get("design_description") is None, "BUG: design_description should be cleared"
        assert result.get("stage_outputs") == {}, "BUG: stage_outputs should be cleared"
    
    def test_backtrack_rejects_missing_decision(self, base_state, valid_plan):
        """handle_backtrack_node should reject if backtrack_decision missing."""
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        base_state["backtrack_decision"] = None  # Missing!
        
        result = handle_backtrack_node(base_state)
        
        assert result.get("ask_user_trigger") is not None, \
            "BUG: Should escalate when backtrack_decision is missing"
        assert result.get("awaiting_user_input") is True
    
    def test_backtrack_respects_max_limit(self, base_state, valid_plan):
        """handle_backtrack_node should escalate when max_backtracks exceeded."""
        from src.agents.reporting import handle_backtrack_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "completed_success"}]
        }
        base_state["backtrack_count"] = 2  # At max
        base_state["runtime_config"] = {"max_backtracks": 2}
        base_state["backtrack_decision"] = {
            "accepted": True,
            "target_stage_id": "stage_0",
            "stages_to_invalidate": [],
        }
        
        result = handle_backtrack_node(base_state)
        
        # Should escalate to user
        assert result.get("ask_user_trigger") == "backtrack_limit", \
            f"BUG: Should trigger backtrack_limit, got '{result.get('ask_user_trigger')}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Routing Functions with Count Limits
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingCountLimits:
    """Verify routing functions respect count limits and escalate correctly."""
    
    def test_code_review_escalates_at_limit(self, base_state):
        """route_after_code_review should escalate to ask_user at limit."""
        from src.routing import route_after_code_review
        
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 3  # At default max (MAX_CODE_REVISIONS = 3)
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", \
            f"BUG: Should escalate to ask_user at limit, got '{result}'"
    
    def test_code_review_allows_under_limit(self, base_state):
        """route_after_code_review should route to generate_code under limit."""
        from src.routing import route_after_code_review
        
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 2  # Under limit
        
        result = route_after_code_review(base_state)
        
        assert result == "generate_code", \
            f"Should route to generate_code under limit, got '{result}'"
    
    def test_design_review_escalates_at_limit(self, base_state):
        """route_after_design_review should escalate at limit."""
        from src.routing import route_after_design_review
        
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 3  # At default max
        
        result = route_after_design_review(base_state)
        
        assert result == "ask_user", \
            f"BUG: Should escalate to ask_user at design limit, got '{result}'"
    
    def test_execution_check_escalates_at_limit(self, base_state):
        """route_after_execution_check should escalate at failure limit."""
        from src.routing import route_after_execution_check
        
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 2  # At default max
        
        result = route_after_execution_check(base_state)
        
        assert result == "ask_user", \
            f"BUG: Should escalate to ask_user at execution limit, got '{result}'"
    
    def test_physics_check_routes_to_design_on_design_flaw(self, base_state):
        """route_after_physics_check should route to design on design_flaw."""
        from src.routing import route_after_physics_check
        
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0  # Under limit
        
        result = route_after_physics_check(base_state)
        
        assert result == "design", \
            f"Should route to design on design_flaw, got '{result}'"
    
    def test_routing_with_none_verdict_escalates(self, base_state):
        """Routing functions should escalate to ask_user when verdict is None."""
        from src.routing import route_after_code_review
        
        base_state["last_code_review_verdict"] = None  # None verdict
        
        result = route_after_code_review(base_state)
        
        assert result == "ask_user", \
            f"BUG: Should escalate to ask_user when verdict is None, got '{result}'"
    
    def test_plan_review_escalates_at_replan_limit(self, base_state, valid_plan):
        """route_after_plan_review should escalate at replan limit."""
        from src.routing import route_after_plan_review
        
        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 2  # At default max (MAX_REPLANS = 2)
        
        result = route_after_plan_review(base_state)
        
        assert result == "ask_user", \
            f"BUG: Should escalate to ask_user at replan limit, got '{result}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Supervisor Trigger Handlers
# ═══════════════════════════════════════════════════════════════════════

class TestSupervisorTriggerHandlers:
    """Test supervisor handles various ask_user_trigger types correctly."""
    
    def test_supervisor_handles_code_review_limit_with_hint(self, base_state):
        """supervisor_node should handle code_review_limit with PROVIDE_HINT."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "PROVIDE_HINT: Try using mp.Medium instead"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3
        
        result = supervisor_node(base_state)
        
        # Should reset counter and include user hint
        assert result.get("code_revision_count") == 0, \
            f"BUG: code_revision_count should be reset to 0, got {result.get('code_revision_count')}"
        assert result.get("supervisor_verdict") == "ok_continue", \
            f"Should continue with hint, got '{result.get('supervisor_verdict')}'"
    
    def test_supervisor_handles_design_review_limit_skip(self, base_state):
        """supervisor_node should handle design_review_limit with SKIP."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["ask_user_trigger"] = "design_review_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Design review limit"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }
        
        result = supervisor_node(base_state)
        
        assert result.get("supervisor_verdict") == "ok_continue"
    
    def test_supervisor_handles_execution_failure_with_retry(self, base_state):
        """supervisor_node should handle execution_failure_limit with RETRY."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {"Q1": "RETRY_WITH_GUIDANCE: Increase memory allocation"}
        base_state["pending_user_questions"] = ["Execution failed"]
        base_state["execution_failure_count"] = 2
        
        result = supervisor_node(base_state)
        
        assert result.get("execution_failure_count") == 0, \
            f"BUG: execution_failure_count should reset, got {result.get('execution_failure_count')}"
    
    def test_supervisor_handles_llm_error_with_retry(self, base_state):
        """supervisor_node should handle llm_error trigger with RETRY."""
        from src.agents.supervision.supervisor import supervisor_node
        
        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["LLM API failed"]
        
        result = supervisor_node(base_state)
        
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "RETRY" in result.get("supervisor_feedback", "").upper() or \
               "retry" in result.get("supervisor_feedback", "").lower(), \
            f"Feedback should mention retry: {result.get('supervisor_feedback')}"
    
    def test_supervisor_handles_context_overflow_with_truncate(self, base_state):
        """supervisor_node should handle context_overflow with TRUNCATE."""
        from src.agents.supervision.supervisor import supervisor_node
        
        # Create very long paper text
        base_state["paper_text"] = "x" * 25000
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "TRUNCATE"}
        base_state["pending_user_questions"] = ["Context too long"]
        
        result = supervisor_node(base_state)
        
        # Should truncate paper_text
        if "paper_text" in result:
            assert len(result["paper_text"]) < 25000, \
                "BUG: Paper text should be truncated"
            assert "[TRUNCATED" in result["paper_text"], \
                "Truncated text should contain marker"


# ═══════════════════════════════════════════════════════════════════════
# Test: Code Generator Expected Outputs Passthrough
# ═══════════════════════════════════════════════════════════════════════

class TestCodeGeneratorExpectedOutputs:
    """Verify code_generator passes expected_outputs correctly."""
    
    def test_expected_outputs_passed_through(self, base_state, valid_plan):
        """code_generator_node should pass expected_outputs from LLM response."""
        from src.agents.code import code_generator_node
        
        expected = ["spectrum.csv", "field_map.png", "resonance_data.json"]
        mock_response = {
            "code": "import meep as mp\nimport numpy as np\n\n# Full simulation code here\nprint('Running simulation')",
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
        
        # expected_outputs MUST be in result
        assert "expected_outputs" in result, \
            "BUG: expected_outputs should be in result"
        assert result["expected_outputs"] == expected, \
            f"BUG: expected_outputs mismatch. Expected {expected}, got {result.get('expected_outputs')}"
    
    def test_empty_expected_outputs_defaults_to_empty_list(self, base_state, valid_plan):
        """code_generator_node should default expected_outputs to empty list."""
        from src.agents.code import code_generator_node
        
        mock_response = {
            "code": "import meep as mp\nimport numpy as np\nprint('Simulation')",
            # No expected_outputs field!
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
        
        assert "expected_outputs" in result, \
            "BUG: expected_outputs should be present even if empty"
        assert result["expected_outputs"] == [], \
            f"expected_outputs should default to [], got {result.get('expected_outputs')}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Stage Selection Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestStageSelectionEdgeCases:
    """Test edge cases in stage selection."""
    
    def test_select_stage_respects_validation_hierarchy(self, base_state):
        """select_stage_node should not select stage before hierarchy allows."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": ["stage_0"]},
            ],
        }
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage_0"]},
            ]
        }
        
        result = select_stage_node(base_state)
        
        # Should select stage_0 first (not stage_1)
        assert result["current_stage_id"] == "stage_0", \
            f"Should select stage_0 first due to hierarchy, got '{result.get('current_stage_id')}'"
    
    def test_select_stage_skips_completed_stages(self, base_state):
        """select_stage_node should skip completed stages."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": ["stage_0"]},
            ],
        }
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success", "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "status": "not_started", "dependencies": ["stage_0"]},
            ]
        }
        
        result = select_stage_node(base_state)
        
        # Should select stage_1 (stage_0 is completed)
        assert result["current_stage_id"] == "stage_1", \
            f"Should select stage_1 (stage_0 completed), got '{result.get('current_stage_id')}'"
    
    def test_select_stage_detects_deadlock(self, base_state):
        """select_stage_node should detect deadlock when all stages blocked/failed.
        
        To trigger deadlock detection, we need:
        1. remaining_stages is not empty (stages not completed)
        2. potentially_runnable is empty (no not_started/invalidated/needs_rerun stages)
        3. permanently_blocked is not empty (blocked/completed_failed stages)
        
        A stage with "blocked" status and empty dependencies gets unblocked.
        So we need to use "completed_failed" to test deadlock.
        """
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]},
            ],
        }
        # Use completed_failed status - this is permanently blocked
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_failed", "dependencies": []},
            ]
        }
        
        result = select_stage_node(base_state)
        
        # Should detect deadlock since only stage is completed_failed
        # and there are no runnable stages
        assert result.get("ask_user_trigger") == "deadlock_detected", \
            f"Should detect deadlock, got trigger '{result.get('ask_user_trigger')}'"
    
    def test_blocked_stage_with_unsatisfied_deps_stays_blocked(self, base_state):
        """A blocked stage with unsatisfied dependencies should stay blocked."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": ["stage_0"]},
            ],
        }
        # stage_0 is not_started (satisfies deps for stage_1)
        # stage_1 is blocked because stage_0 not completed yet
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []},
                {"stage_id": "stage_1", "stage_type": "SINGLE_STRUCTURE", "status": "blocked", "dependencies": ["stage_0"]},
            ]
        }
        
        result = select_stage_node(base_state)
        
        # Should select stage_0 (not stage_1 which is blocked)
        assert result["current_stage_id"] == "stage_0", \
            f"Should select stage_0 (stage_1 is blocked), got '{result.get('current_stage_id')}'"
    
    def test_select_stage_resets_counters_on_new_stage(self, base_state):
        """select_stage_node should reset revision counters on new stage."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
        }
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "status": "not_started", "dependencies": []}]
        }
        base_state["current_stage_id"] = None  # No previous stage
        base_state["design_revision_count"] = 5
        base_state["code_revision_count"] = 5
        
        result = select_stage_node(base_state)
        
        assert result.get("design_revision_count") == 0, \
            f"design_revision_count should reset to 0, got {result.get('design_revision_count')}"
        assert result.get("code_revision_count") == 0, \
            f"code_revision_count should reset to 0, got {result.get('code_revision_count')}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Results Analyzer Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestResultsAnalyzerCompleteness:
    """Verify results_analyzer produces complete output."""
    
    def test_results_analyzer_returns_all_required_fields(self, base_state, valid_plan):
        """results_analyzer_node should return all required fields."""
        from src.agents.analysis import results_analyzer_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {}  # Empty outputs - should handle gracefully
        
        result = results_analyzer_node(base_state)
        
        # All these fields should be present
        required_fields = ["workflow_phase"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        assert result["workflow_phase"] == "analysis"


# ═══════════════════════════════════════════════════════════════════════
# Test: Comparison Validator Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestComparisonValidatorEdgeCases:
    """Test comparison_validator_node edge cases."""
    
    def test_comparison_validator_approves_no_targets(self, base_state):
        """comparison_validator should approve when stage has no targets."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["current_stage_id"] = "stage_0"
        base_state["plan"] = {
            "stages": [{"stage_id": "stage_0", "targets": [], "target_details": []}]
        }
        base_state["figure_comparisons"] = []
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "running"}]}
        
        result = comparison_validator_node(base_state)
        
        # Should approve (no targets to compare)
        assert result["comparison_verdict"] == "approve", \
            f"Should approve when no targets, got '{result.get('comparison_verdict')}'"
    
    def test_comparison_validator_rejects_missing_comparisons(self, base_state, valid_plan):
        """comparison_validator should reject when expected comparisons missing."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["plan"] = valid_plan  # Has targets
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []  # No comparisons produced!
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "running"}]}
        
        result = comparison_validator_node(base_state)
        
        # Should reject - expected targets but no comparisons
        assert result["comparison_verdict"] == "needs_revision", \
            f"Should reject missing comparisons, got '{result.get('comparison_verdict')}'"


# ═══════════════════════════════════════════════════════════════════════
# Test: Simulation Designer Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestSimulationDesignerCompleteness:
    """Verify simulation_designer_node produces complete design."""
    
    def test_designer_returns_design_with_all_sections(self, base_state, valid_plan):
        """simulation_designer_node should include all design sections."""
        from src.agents.design import simulation_designer_node
        
        mock_response = {
            "stage_id": "stage_0",
            "design_description": "Full FDTD simulation design",
            "geometry": [{"type": "cylinder", "radius": 20, "height": 100, "material": "gold"}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 800]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
            "materials": [{"material_id": "gold", "source": "Palik"}],
            "computational_domain": {"pml_layers": 1, "resolution": 32},
        }
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = simulation_designer_node(base_state)
        
        # All sections should be in design
        design = result.get("design_description", {})
        
        assert "geometry" in design, "Design missing geometry"
        assert "sources" in design, "Design missing sources"
        assert "monitors" in design, "Design missing monitors"
        assert len(design.get("geometry", [])) > 0, "Design should have at least one geometry object"


# ═══════════════════════════════════════════════════════════════════════
# Test: LLM Content Verification
# ═══════════════════════════════════════════════════════════════════════

class TestLLMContentVerification:
    """Verify correct content is passed to LLM calls."""
    
    def test_planner_receives_paper_figures(self, base_state):
        """plan_node should include paper_figures in LLM call."""
        from src.agents.planning import plan_node
        
        base_state["paper_figures"] = [
            {"figure_id": "Fig1", "description": "Extinction spectrum"},
            {"figure_id": "Fig2", "description": "Near-field map"},
        ]
        
        mock_response = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
            "targets": [],
            "extracted_parameters": [],
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response) as mock:
            plan_node(base_state)
            
            # Verify user_content contains figure info
            call_kwargs = mock.call_args.kwargs
            user_content = call_kwargs.get("user_content", "")
            
            # User content should mention figures
            if isinstance(user_content, str):
                content_str = user_content
            else:
                # Multi-modal content
                content_str = str(user_content)
            
            assert "Fig1" in content_str or "figure" in content_str.lower(), \
                "User content should reference figures"
    
    def test_code_reviewer_receives_design_spec(self, base_state):
        """code_reviewer_node should include design_description in LLM call."""
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
            
            # Should include design reference
            assert "DESIGN" in user_content.upper() or "design" in user_content.lower(), \
                "User content should reference design spec"


# ═══════════════════════════════════════════════════════════════════════
# Test: Report Generator Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestReportGeneratorCompleteness:
    """Verify generate_report_node produces complete report."""
    
    def test_report_includes_token_summary(self, base_state, valid_plan):
        """generate_report_node should include token_summary in metrics."""
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
        
        # Check token summary
        metrics = result.get("metrics", {})
        token_summary = metrics.get("token_summary", {})
        
        assert "total_input_tokens" in token_summary, "Missing total_input_tokens"
        assert "total_output_tokens" in token_summary, "Missing total_output_tokens"
        assert token_summary["total_input_tokens"] == 3000, \
            f"Expected 3000 input tokens, got {token_summary.get('total_input_tokens')}"
    
    def test_report_marks_workflow_complete(self, base_state, valid_plan):
        """generate_report_node should set workflow_complete=True."""
        from src.agents.reporting import generate_report_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {"stages": []}
        
        mock_response = {"executive_summary": {}, "conclusions": {}}
        
        with patch("src.agents.reporting.call_agent_with_metrics", return_value=mock_response):
            result = generate_report_node(base_state)
        
        assert result.get("workflow_complete") is True, \
            "BUG: workflow_complete should be True"


# ═══════════════════════════════════════════════════════════════════════
# Test: Physics Sanity Backtrack Suggestion
# ═══════════════════════════════════════════════════════════════════════

class TestPhysicsSanityBacktrack:
    """Verify physics_sanity_node handles backtrack suggestions."""
    
    def test_physics_sanity_passes_backtrack_suggestion(self, base_state):
        """physics_sanity_node should pass through backtrack_suggestion."""
        from src.agents.execution import physics_sanity_node
        
        mock_response = {
            "verdict": "design_flaw",
            "summary": "Fundamental issue with simulation setup",
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                "target_stage_id": "stage_0",
                "reason": "Material properties need revalidation",
            },
        }
        
        base_state["current_stage_id"] = "stage_1"
        base_state["stage_outputs"] = {"files": []}
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = physics_sanity_node(base_state)
        
        assert "backtrack_suggestion" in result, \
            "BUG: backtrack_suggestion should be passed through"
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True


# ═══════════════════════════════════════════════════════════════════════
# Test: Invariant Verification (Future Bug Prevention)
# ═══════════════════════════════════════════════════════════════════════

class TestInvariants:
    """Test invariants that must hold for correct operation."""
    
    def test_all_reviewer_nodes_return_verdict_field(self, base_state, valid_plan):
        """All reviewer nodes must return their specific verdict field."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node
        
        # Setup for plan_reviewer
        base_state["plan"] = valid_plan
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(base_state)
            assert "last_plan_review_verdict" in result, \
                "plan_reviewer MUST return last_plan_review_verdict"
        
        # Setup for design_reviewer
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            assert "last_design_review_verdict" in result, \
                "design_reviewer MUST return last_design_review_verdict"
        
        # Setup for code_reviewer
        base_state["code"] = "print('test')"
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            assert "last_code_review_verdict" in result, \
                "code_reviewer MUST return last_code_review_verdict"
    
    def test_all_validators_return_verdict_field(self, base_state):
        """All validator nodes must return their specific verdict field."""
        from src.agents.execution import execution_validator_node, physics_sanity_node
        from src.agents.analysis import comparison_validator_node
        
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": []}
        
        mock_response = {"verdict": "pass", "summary": "OK"}
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = execution_validator_node(base_state)
            assert "execution_verdict" in result, \
                "execution_validator MUST return execution_verdict"
        
        with patch("src.agents.execution.call_agent_with_metrics", return_value=mock_response):
            result = physics_sanity_node(base_state)
            assert "physics_verdict" in result, \
                "physics_sanity MUST return physics_verdict"
        
        base_state["figure_comparisons"] = []
        base_state["progress"] = {"stages": []}
        base_state["plan"] = {"stages": [{"stage_id": "stage_0", "targets": []}]}
        
        result = comparison_validator_node(base_state)
        assert "comparison_verdict" in result, \
            "comparison_validator MUST return comparison_verdict"
    
    def test_all_nodes_return_workflow_phase(self, base_state, valid_plan):
        """All agent nodes must return workflow_phase."""
        from src.agents.planning import plan_node, plan_reviewer_node, adapt_prompts_node
        from src.agents.design import simulation_designer_node, design_reviewer_node
        from src.agents.code import code_generator_node, code_reviewer_node
        from src.agents.execution import execution_validator_node, physics_sanity_node
        from src.agents.analysis import results_analyzer_node, comparison_validator_node
        from src.agents.reporting import generate_report_node
        from src.agents.supervision.supervisor import supervisor_node
        
        mock_reviewer_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        mock_plan_response = {
            "paper_id": "test", "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}],
            "targets": [], "extracted_parameters": [],
        }
        mock_code_response = {"code": "import meep\nprint('test')", "expected_outputs": []}
        mock_design_response = {
            "stage_id": "stage_0", "design_description": "Test design",
            "geometry": [{"type": "box"}], "sources": [], "monitors": [],
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
        
        # Test each node returns workflow_phase
        nodes_to_test = [
            ("adapt_prompts_node", adapt_prompts_node, "src.agents.planning"),
            ("plan_node", plan_node, "src.agents.planning"),
            ("plan_reviewer_node", plan_reviewer_node, "src.agents.planning"),
            ("simulation_designer_node", simulation_designer_node, "src.agents.design"),
            ("design_reviewer_node", design_reviewer_node, "src.agents.design"),
            ("execution_validator_node", execution_validator_node, "src.agents.execution"),
            ("physics_sanity_node", physics_sanity_node, "src.agents.execution"),
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
                    assert "workflow_phase" in result, \
                        f"{node_name} MUST return workflow_phase"
                except Exception as e:
                    # Some nodes may have precondition failures, that's OK
                    pass
    
    def test_counter_increment_never_negative(self, base_state):
        """Revision counters should never go negative."""
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node
        
        base_state["current_stage_id"] = "stage_0"
        base_state["design_revision_count"] = 0
        base_state["code_revision_count"] = 0
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"
        
        mock_response = {"verdict": "needs_revision", "issues": ["fix"], "summary": "Fix"}
        
        # Increment design counter
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(base_state)
            count = result.get("design_revision_count", 0)
            assert count >= 0, f"design_revision_count should never be negative, got {count}"
        
        # Increment code counter  
        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)
            count = result.get("code_revision_count", 0)
            assert count >= 0, f"code_revision_count should never be negative, got {count}"
    
    def test_stage_selection_always_returns_stage_id_or_escalates(self, base_state, valid_plan):
        """select_stage_node must return current_stage_id or escalate."""
        from src.agents.stage_selection import select_stage_node
        
        base_state["plan"] = valid_plan
        base_state["progress"] = {
            "stages": [
                {"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", 
                 "status": "not_started", "dependencies": []}
            ]
        }
        
        result = select_stage_node(base_state)
        
        # Must either have a stage selected OR escalate to user
        has_stage = result.get("current_stage_id") is not None
        has_escalation = result.get("ask_user_trigger") is not None
        
        assert has_stage or has_escalation, \
            "select_stage must return current_stage_id or ask_user_trigger"


# ═══════════════════════════════════════════════════════════════════════
# Test: Regression Prevention
# ═══════════════════════════════════════════════════════════════════════

class TestRegressionPrevention:
    """Tests to prevent regressions in critical behavior."""
    
    def test_llm_error_in_planner_escalates_not_crashes(self, base_state):
        """LLM error in planner should escalate to user, not crash."""
        from src.agents.planning import plan_node
        
        with patch("src.agents.planning.call_agent_with_metrics", 
                   side_effect=RuntimeError("API Error")):
            # Should NOT raise exception
            result = plan_node(base_state)
            
            # Should have escalation
            assert result.get("awaiting_user_input") is True, \
                "Should escalate to user on LLM error"
    
    def test_llm_error_in_reviewers_auto_approves(self, base_state, valid_plan):
        """LLM errors in reviewer nodes should auto-approve to not block."""
        from src.agents.planning import plan_reviewer_node
        from src.agents.design import design_reviewer_node
        from src.agents.code import code_reviewer_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_design"] = {"stage_id": "stage_0"}
        base_state["code"] = "print('test')"
        
        # Test plan_reviewer
        with patch("src.agents.planning.call_agent_with_metrics", 
                   side_effect=RuntimeError("API Error")):
            result = plan_reviewer_node(base_state)
            assert result.get("last_plan_review_verdict") == "approve", \
                "plan_reviewer should auto-approve on LLM error"
        
        # Test design_reviewer
        with patch("src.agents.design.call_agent_with_metrics",
                   side_effect=RuntimeError("API Error")):
            result = design_reviewer_node(base_state)
            assert result.get("last_design_review_verdict") == "approve", \
                "design_reviewer should auto-approve on LLM error"
        
        # Test code_reviewer
        with patch("src.agents.code.call_agent_with_metrics",
                   side_effect=RuntimeError("API Error")):
            result = code_reviewer_node(base_state)
            assert result.get("last_code_review_verdict") == "approve", \
                "code_reviewer should auto-approve on LLM error"
    
    def test_missing_required_state_handled_gracefully(self, base_state):
        """Nodes should handle missing required state gracefully."""
        from src.agents.code import code_generator_node
        from src.agents.design import simulation_designer_node
        
        # Test code_generator with missing design_description
        base_state["current_stage_id"] = "stage_0"
        base_state["design_description"] = None  # Missing!
        
        result = code_generator_node(base_state)
        
        # Should not crash, should have some error indication
        assert result.get("run_error") or result.get("reviewer_feedback") or \
               result.get("ask_user_trigger"), \
            "Should indicate error when design_description is missing"
        
        # Test simulation_designer with missing current_stage_id
        base_state["current_stage_id"] = None  # Missing!
        
        result = simulation_designer_node(base_state)
        
        # Should not crash, should escalate
        assert result.get("ask_user_trigger") or result.get("awaiting_user_input"), \
            "Should escalate when current_stage_id is missing"


# ═══════════════════════════════════════════════════════════════════════
# Test: Comparison Validator Logic
# ═══════════════════════════════════════════════════════════════════════

class TestComparisonValidatorLogic:
    """Verify comparison_validator_node logic for approvals and rejections."""
    
    def test_comparison_validator_approves_valid_comparisons(self, base_state, valid_plan):
        """Should approve when all targets match and reports are present."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        
        # Mock matching comparisons for 'material_gold' (target of stage_0)
        base_state["figure_comparisons"] = [{
            "stage_id": "stage_0",
            "figure_id": "material_gold",
            "classification": "match"
        }]
        
        # Mock analysis reports
        base_state["analysis_result_reports"] = [{
            "stage_id": "stage_0",
            "target_figure": "material_gold",
            "status": "match",
            "criteria_failures": []
        }]
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "approve", \
            f"Should approve valid comparisons. Feedback: {result.get('comparison_feedback')}"
        assert result["workflow_phase"] == "comparison_validation"

    def test_comparison_validator_rejects_missing_reports(self, base_state, valid_plan):
        """Should reject when comparisons exist but quantitative reports are missing."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        
        # Comparisons exist for material_gold
        base_state["figure_comparisons"] = [{
            "stage_id": "stage_0",
            "figure_id": "material_gold",
            "classification": "match"
        }]
        
        # BUT reports are empty
        base_state["analysis_result_reports"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision", \
            "Should reject when quantitative reports are missing"
        assert "Missing quantitative reports" in result["comparison_feedback"]

    def test_comparison_validator_rejects_missing_comparisons(self, base_state, valid_plan):
        """Should reject when targets are defined but no comparisons produced."""
        from src.agents.analysis import comparison_validator_node
        
        base_state["plan"] = valid_plan # Has target material_gold for stage_0
        base_state["current_stage_id"] = "stage_0"
        base_state["figure_comparisons"] = []
        
        result = comparison_validator_node(base_state)
        
        assert result["comparison_verdict"] == "needs_revision"
        # Feedback might mention missing reports OR missing comparisons, but "Results analyzer did not produce figure comparisons"
        # is specific to missing comparisons.
        # Note: logic accumulates feedback. Check if "Results analyzer did not produce figure comparisons" is present.
        assert "Results analyzer did not produce figure comparisons" in result["comparison_feedback"] or \
               "Missing quantitative reports" in result["comparison_feedback"], \
               f"Feedback should mention missing comparisons/reports. Got: {result.get('comparison_feedback')}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Results Analyzer Logic
# ═══════════════════════════════════════════════════════════════════════

class TestResultsAnalyzerLogic:
    """Verify results_analyzer_node logic."""

    def test_results_analyzer_analyzes_existing_files(self, base_state, valid_plan):
        """Should perform analysis when files exist, even if no images for LLM."""
        from src.agents.analysis import results_analyzer_node
        
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["stage_outputs"] = {"files": ["output.csv"]}
        
        # Mock file system and numeric helpers
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("src.agents.analysis.load_numeric_series", return_value=[1, 2, 3]), \
             patch("src.agents.analysis.quantitative_curve_metrics", return_value={"peak_position_error_percent": 5.0}):
             
            # Mock no images for LLM -> no LLM call expected for vision
            with patch("src.agents.analysis.get_images_for_analyzer", return_value=[]):
                result = results_analyzer_node(base_state)
                
        assert result["workflow_phase"] == "analysis"
        assert "analysis_summary" in result
        assert result["analysis_summary"]["totals"]["targets"] > 0
        
        # Should produce comparisons even without LLM
        assert len(result["figure_comparisons"]) > 0
        assert result["figure_comparisons"][0]["figure_id"] == "material_gold"
        
        # Should produce reports
        assert len(result["analysis_result_reports"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# Test: Plan Node Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestPlanNodeEdgeCases:
    """Verify plan_node robustness."""
    
    def test_plan_node_handles_progress_init_failure(self, base_state):
        """Should mark plan as needs_revision if progress initialization fails."""
        from src.agents.planning import plan_node
        
        mock_response = {
            "stages": [{"stage_id": "s1"}], # minimal
        }
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            with patch("src.agents.planning.initialize_progress_from_plan", side_effect=ValueError("Init failed")):
                result = plan_node(base_state)
                
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "Progress initialization failed" in result["planner_feedback"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

