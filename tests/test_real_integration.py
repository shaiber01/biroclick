"""
REAL Integration Tests - Tests that find bugs.

These tests run ACTUAL code with MINIMAL mocking.
Only the LLM API call itself is mocked - everything else runs for real.

If a test fails here, it's a BUG IN THE CODE, not the test.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the REAL modules
from src.prompts import build_agent_prompt, PROMPTS_DIR
from schemas.state import create_initial_state

# ═══════════════════════════════════════════════════════════════════════
# Test: All Required Prompt Files Exist
# ═══════════════════════════════════════════════════════════════════════

class TestPromptFilesExist:
    """Verify all prompt files referenced in code actually exist."""
    
    # All agents that call build_agent_prompt()
    AGENTS_REQUIRING_PROMPTS = [
        "prompt_adaptor",
        "planner", 
        "plan_reviewer",
        "simulation_designer",
        "design_reviewer",
        "code_generator",
        "code_reviewer",
        "execution_validator",
        "physics_sanity",
        "results_analyzer",
        "supervisor",
        "report_generator",
    ]
    
    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_prompt_file_exists(self, agent_name):
        """Each agent's prompt file must exist on disk."""
        prompt_file = PROMPTS_DIR / f"{agent_name}_agent.md"
        assert prompt_file.exists(), (
            f"Missing prompt file: {prompt_file}\n"
            f"Agent '{agent_name}' calls build_agent_prompt() but the file doesn't exist.\n"
            f"This will crash at runtime!"
        )
    
    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_prompt_file_not_empty(self, agent_name):
        """Each prompt file must have content."""
        prompt_file = PROMPTS_DIR / f"{agent_name}_agent.md"
        if prompt_file.exists():
            content = prompt_file.read_text()
            assert len(content) > 100, (
                f"Prompt file {prompt_file} is suspiciously short ({len(content)} chars).\n"
                f"Expected substantial prompt content."
            )
    
    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_build_agent_prompt_succeeds(self, agent_name):
        """build_agent_prompt() must succeed for each agent."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics"
        )
        
        # This should NOT raise an exception
        try:
            prompt = build_agent_prompt(agent_name, state)
            assert prompt is not None
            assert len(prompt) > 0
        except FileNotFoundError as e:
            pytest.fail(f"build_agent_prompt('{agent_name}') failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Test: All Schema Files Exist and Are Valid
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaFilesExist:
    """Verify all schema files referenced in code exist and are valid JSON."""
    
    SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"
    
    # All schemas used by call_agent_with_metrics
    REQUIRED_SCHEMAS = [
        "planner_output_schema.json",
        "plan_reviewer_output_schema.json",
        "simulation_designer_output_schema.json",
        "design_reviewer_output_schema.json",
        "code_generator_output_schema.json",
        "code_reviewer_output_schema.json",
        "execution_validator_output_schema.json",
        "physics_sanity_output_schema.json",
        "results_analyzer_output_schema.json",
        "supervisor_output_schema.json",
        "report_schema.json",
    ]
    
    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_file_exists(self, schema_name):
        """Each schema file must exist."""
        schema_file = self.SCHEMAS_DIR / schema_name
        assert schema_file.exists(), f"Missing schema file: {schema_file}"
    
    @pytest.mark.parametrize("schema_name", REQUIRED_SCHEMAS)
    def test_schema_is_valid_json(self, schema_name):
        """Each schema must be valid JSON."""
        schema_file = self.SCHEMAS_DIR / schema_name
        if schema_file.exists():
            try:
                with open(schema_file) as f:
                    schema = json.load(f)
                assert "type" in schema or "properties" in schema, \
                    f"Schema {schema_name} doesn't look like a JSON schema"
            except json.JSONDecodeError as e:
                pytest.fail(f"Schema {schema_name} is not valid JSON: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Test: Node Functions Can Be Called and Return Valid State Updates
# ═══════════════════════════════════════════════════════════════════════

class TestNodeFunctionsCallable:
    """
    Test that node functions can be called with minimal state.
    
    Only mock the LLM call itself - everything else runs for real.
    These tests verify:
    1. The node doesn't crash.
    2. The node returns a dict with expected state keys.
    """
    
    @pytest.fixture
    def minimal_state(self):
        """Create minimal valid state for testing."""
        # Paper text must be > 100 chars for planner
        paper_text = """
        We study the optical properties of gold nanorods with length 100nm and diameter 40nm. 
        Using FDTD simulations, we calculate extinction spectra and near-field enhancement patterns. 
        The localized surface plasmon resonance is observed at 650nm wavelength.
        This text is long enough to pass the planner's validation check which requires 100 characters.
        """
        state = create_initial_state(
            paper_id="test_paper",
            paper_text=paper_text,
            paper_domain="plasmonics"
        )
        # Add plan so nodes don't fail on missing plan
        state["plan"] = {
            "paper_id": "test_paper",
            "title": "Test Plan",
            "stages": [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Test stage",
                "targets": ["Fig1"],
                "dependencies": [],
            }],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
            "extracted_parameters": [],
        }
        state["progress"] = {
            "stages": [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "status": "not_started",
                "dependencies": [],
            }]
        }
        return state
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock that returns a valid-ish response for any agent."""
        def _mock(*args, **kwargs):
            agent = kwargs.get("agent_name", "unknown")
            # Return minimal valid response for each agent type
            if "reviewer" in agent:
                return {"verdict": "approve", "issues": [], "summary": "OK"}
            elif "validator" in agent:
                return {"verdict": "pass", "issues": [], "summary": "OK"}
            elif agent == "planner":
                return {
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [],
                    "targets": [],
                    "extracted_parameters": [],
                }
            elif agent == "supervisor":
                return {"verdict": "ok_continue", "reasoning": "OK"}
            elif agent == "report":
                return {
                    "executive_summary": {"overall_assessment": []},
                    "conclusions": ["Done"]
                }
            elif agent == "physics_sanity":
                return {"verdict": "pass", "reasoning": "Physics looks good"}
            elif agent == "prompt_adaptor":
                return {"adaptations": []}
            else:
                return {}
        return _mock
    
    def test_plan_node_returns_valid_plan(self, minimal_state, mock_llm_response):
        """plan_node must return a result containing a plan structure."""
        from src.agents.planning import plan_node
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = plan_node(minimal_state)
            
            assert result is not None
            assert "plan" in result, "plan_node must return 'plan' key"
            assert "workflow_phase" in result
            assert result["workflow_phase"] == "planning"
            
            plan = result["plan"]
            assert "stages" in plan, "Plan must have stages"
            assert "targets" in plan, "Plan must have targets"
    
    def test_supervisor_node_returns_verdict(self, minimal_state, mock_llm_response):
        """supervisor_node must return a supervisor_verdict."""
        from src.agents.supervision.supervisor import supervisor_node
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", mock_llm_response):
            result = supervisor_node(minimal_state)
            
            assert result is not None
            assert "supervisor_verdict" in result, "supervisor_node must return supervisor_verdict"
            assert "supervisor_feedback" in result
            assert result["workflow_phase"] == "supervision"
    
    def test_report_node_completes_workflow(self, minimal_state, mock_llm_response):
        """generate_report_node must mark workflow as complete."""
        from src.agents.reporting import generate_report_node
        
        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)
            
            assert result is not None
            assert "workflow_complete" in result, "generate_report_node must set workflow_complete"
            assert result["workflow_complete"] is True
            assert "executive_summary" in result
            assert "workflow_phase" in result
            assert result["workflow_phase"] == "reporting"
    
    def test_design_node_returns_design(self, minimal_state, mock_llm_response):
        """simulation_designer_node must return design_description."""
        from src.agents.design import simulation_designer_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        # Mock design response
        design_response = {
            "design": {
                "stage_id": "stage_0",
                "geometry": [],
                "simulation_parameters": {}
            },
            "explanation": "Designed"
        }
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=design_response):
            result = simulation_designer_node(minimal_state)
            
            assert result is not None
            assert "design_description" in result, "simulation_designer_node must return design_description"
            assert result["design_description"]["design"]["stage_id"] == "stage_0"
            assert "workflow_phase" in result
    
    def test_code_generator_returns_code(self, minimal_state, mock_llm_response):
        """code_generator_node must return generated code."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        # Must provide design_description for code generator
        minimal_state["design_description"] = {
            "design": {"geometry": [], "simulation_parameters": {}},
            "explanation": "Valid design"
        }
        
        code_response = {
            "code": "import meep as mp\nprint('hello')",
            "explanation": "Generated"
        }
        
        with patch("src.agents.code.call_agent_with_metrics", return_value=code_response):
            result = code_generator_node(minimal_state)
            
            assert result is not None
            assert "code" in result, "code_generator_node must return code"
            assert "import meep" in result["code"]
            assert "workflow_phase" in result

    def test_execution_validator_node_returns_verdict(self, minimal_state, mock_llm_response):
        """execution_validator_node must return execution_verdict."""
        from src.agents.execution import execution_validator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"stdout": "run complete", "stderr": ""}
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = execution_validator_node(minimal_state)
            
            assert result is not None
            assert "execution_verdict" in result
            assert result["execution_verdict"] in ["pass", "fail"]
    
    def test_physics_sanity_node_returns_verdict(self, minimal_state, mock_llm_response):
        """physics_sanity_node must return physics_verdict."""
        from src.agents.execution import physics_sanity_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["stage_outputs"] = {"files": ["spectrum.csv"]}
        
        with patch("src.agents.execution.call_agent_with_metrics", mock_llm_response):
            result = physics_sanity_node(minimal_state)
            
            assert result is not None
            assert "physics_verdict" in result
            assert result["physics_verdict"] in ["pass", "fail", "warning", "design_flaw"]
            
    def test_prompt_adaptor_node_returns_adaptations(self, minimal_state, mock_llm_response):
        """prompt_adaptor_node must return prompt_adaptations."""
        from src.agents.planning import adapt_prompts_node
        
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            result = adapt_prompts_node(minimal_state)
            
            assert result is not None
            assert "prompt_adaptations" in result
            assert isinstance(result["prompt_adaptations"], list)


# ═══════════════════════════════════════════════════════════════════════
# Test: Routing Functions Return Valid Values
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingReturnsValidValues:
    """Test that routing functions return values that exist in the graph."""
    
    def test_supervisor_verdicts_match_routing(self):
        """Supervisor schema verdicts must match what routing expects."""
        
        # Load schema to see what verdicts supervisor can return
        schema_file = Path(__file__).parent.parent / "schemas" / "supervisor_output_schema.json"
        with open(schema_file) as f:
            schema = json.load(f)
        
        schema_verdicts = set(schema["properties"]["verdict"]["enum"])
        
        # These are the verdicts that route_after_supervisor handles
        # From src/graph.py route_after_supervisor function
        handled_verdicts = {
            "ok_continue",
            "change_priority",
            "replan_needed",
            "ask_user",
            "backtrack_to_stage",
            "all_complete",
        }
        
        # Every schema verdict must be handled by routing
        unhandled = schema_verdicts - handled_verdicts
        assert not unhandled, (
            f"Supervisor schema allows verdicts that routing doesn't handle: {unhandled}\n"
            f"This could cause routing errors at runtime!"
        )
    
    def test_reviewer_verdicts_match_routing(self):
        """Reviewer verdicts must match what routing expects."""
        # plan_reviewer, design_reviewer, code_reviewer all use approve/needs_revision
        expected_verdicts = {"approve", "needs_revision"}
        
        for reviewer in ["plan_reviewer", "design_reviewer", "code_reviewer"]:
            schema_file = Path(__file__).parent.parent / "schemas" / f"{reviewer}_output_schema.json"
            if schema_file.exists():
                with open(schema_file) as f:
                    schema = json.load(f)
                
                if "verdict" in schema.get("properties", {}):
                    schema_verdicts = set(schema["properties"]["verdict"].get("enum", []))
                    missing = expected_verdicts - schema_verdicts
                    
                    assert not missing, f"{reviewer} schema missing verdicts: {missing}"


# ═══════════════════════════════════════════════════════════════════════
# Test: State Mutations Are Correct
# ═══════════════════════════════════════════════════════════════════════

class TestStateMutations:
    """Test that nodes mutate state correctly."""
    
    def test_plan_node_sets_plan_field(self):
        """plan_node must set state['plan'] with the LLM response."""
        from src.agents.planning import plan_node
        
        mock_plan = {
            "paper_id": "test",
            "title": "Gold Nanorod Study",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION"}],
            "targets": [],
            "extracted_parameters": [],
        }
        
        # Paper text must be > 100 chars to pass validation
        paper_text = """
        We study the optical properties of gold nanorods with length 100nm and 
        diameter 40nm. Using FDTD simulations, we calculate extinction spectra
        and near-field enhancement patterns. The localized surface plasmon 
        resonance is observed at 650nm wavelength.
        """
        state = create_initial_state("test", paper_text, "plasmonics")
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_plan):
            result = plan_node(state)
        
        # The result should contain the plan
        assert "plan" in result, "plan_node must return 'plan' in result"
        assert result["plan"]["title"] == "Gold Nanorod Study"
    
    def test_plan_node_rejects_short_paper_text(self):
        """plan_node must reject paper text that's too short."""
        from src.agents.planning import plan_node
        
        state = create_initial_state("test", "short", "plasmonics")
        result = plan_node(state)
        
        # Should trigger user escalation, not proceed to LLM
        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result
    
    def test_plan_reviewer_sets_verdict_field(self):
        """plan_reviewer_node must set last_plan_review_verdict."""
        from src.agents.planning import plan_reviewer_node
        
        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        # Plan must have at least one stage with targets to pass validation
        state["plan"] = {
            "title": "Test Plan",
            "stages": [{
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "description": "Validate gold optical constants",
                "targets": ["Fig1"],
                "dependencies": [],
            }],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
        }
        
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}
        
        with patch("src.agents.planning.call_agent_with_metrics", return_value=mock_response):
            result = plan_reviewer_node(state)
        
        assert "last_plan_review_verdict" in result
        assert result["last_plan_review_verdict"] == "approve"
    
    def test_plan_reviewer_rejects_empty_plan(self):
        """plan_reviewer_node must reject plans with no stages."""
        from src.agents.planning import plan_reviewer_node
        
        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {"title": "Test", "stages": [], "targets": []}
        
        # Don't even need to mock LLM - internal validation should catch this
        result = plan_reviewer_node(state)
        
        assert result["last_plan_review_verdict"] == "needs_revision"
        # Verify it caught the empty stages issue
    
    def test_counter_increments_on_revision(self):
        """Revision counters must increment when verdict is needs_revision."""
        from src.agents.design import design_reviewer_node
        
        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_design"] = {"stage_id": "stage_0"}
        state["design_revision_count"] = 0
        
        mock_response = {"verdict": "needs_revision", "issues": ["problem"], "summary": "Fix"}
        
        with patch("src.agents.design.call_agent_with_metrics", return_value=mock_response):
            result = design_reviewer_node(state)
        
        assert "design_revision_count" in result
        assert result["design_revision_count"] == 1, "Counter should increment on needs_revision"


# ═══════════════════════════════════════════════════════════════════════
# Test: File Validation in Analysis (Real Code Paths)
# ═══════════════════════════════════════════════════════════════════════

class TestFileValidation:
    """
    Test file handling in analysis nodes.
    
    These tests exercise REAL file validation code, not mocked.
    They use temporary files to test actual file handling logic.
    """
    
    @pytest.fixture
    def analysis_state(self):
        """Create state ready for analysis."""
        state = create_initial_state(
            paper_id="test_file_validation",
            paper_text="Gold nanorod optical simulation study." * 20,
            paper_domain="plasmonics"
        )
        state["plan"] = {
            "paper_id": "test_file_validation",
            "title": "File Validation Test",
            "stages": [{
                "stage_id": "stage_0",
                "stage_type": "SINGLE_STRUCTURE",
                "description": "Test stage",
                "targets": ["Fig1"],
                "dependencies": [],
            }],
            "targets": [{"figure_id": "Fig1", "description": "Test spectrum"}],
        }
        state["progress"] = {
            "stages": [{
                "stage_id": "stage_0",
                "stage_type": "SINGLE_STRUCTURE",
                "status": "running",
                "dependencies": [],
            }]
        }
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "SINGLE_STRUCTURE"
        return state
    
    def test_results_analyzer_handles_missing_files(self, analysis_state):
        """results_analyzer should detect missing files and FAIL execution."""
        from src.agents.analysis import results_analyzer_node
        
        # Set up stage_outputs with non-existent files
        analysis_state["stage_outputs"] = {
            "files": ["/nonexistent/path/spectrum.csv"],
            "stdout": "Simulation completed",
            "stderr": "",
        }
        
        # We don't expect an LLM call because validation should fail first
        # But we mock it just in case it proceeds (which would be a bug)
        with patch("src.agents.analysis.call_agent_with_metrics") as mock_llm:
            result = results_analyzer_node(analysis_state)
            
            # IMPORTANT: Should FAIL if files are missing
            assert result.get("execution_verdict") == "fail", \
                "Analysis should mark execution as failed when output files are missing"
            
            assert result.get("run_error") is not None, \
                "Should provide a run_error explanation"
            
            error_msg = result["run_error"].lower()
            assert "exist on disk" in error_msg or "missing" in error_msg, \
                f"Error message should mention missing files, got: {error_msg}"
    
    def test_results_analyzer_with_real_csv_file(self, analysis_state, tmp_path):
        """results_analyzer should successfully process real CSV files."""
        from src.agents.analysis import results_analyzer_node
        
        # Create actual CSV file with test data
        csv_file = tmp_path / "extinction_spectrum.csv"
        csv_file.write_text(
            "wavelength_nm,extinction\n"
            "400,0.1\n"
            "500,0.3\n"
            "600,0.8\n"
            "700,1.0\n"
            "800,0.5\n"
        )
        
        analysis_state["stage_outputs"] = {
            "files": [str(csv_file)],
            "stdout": "Simulation completed",
            "stderr": "",
            "runtime_seconds": 10,
        }
        
        mock_response = {
            "overall_classification": "ACCEPTABLE_MATCH",
            "figure_comparisons": [
                {
                    "figure_id": "Fig1",
                    "classification": "partial_match",
                    "shape_comparison": ["Peak shape matches"],
                    "reason_for_difference": "Minor numerical differences",
                }
            ],
            "summary": "Results analyzed successfully",
        }
        
        with patch("src.agents.analysis.call_agent_with_metrics", return_value=mock_response):
            result = results_analyzer_node(analysis_state)
        
        # Should successfully process the file
        assert result is not None
        assert result.get("workflow_phase") == "analysis"
        # Should have populated analysis_summary
        assert "analysis_summary" in result
        assert result["analysis_summary"]["totals"]["targets"] > 0
    
    def test_results_analyzer_empty_stage_outputs(self, analysis_state):
        """results_analyzer should handle empty stage_outputs by failing."""
        from src.agents.analysis import results_analyzer_node
        
        # Empty stage_outputs - common error case
        analysis_state["stage_outputs"] = {}
        
        result = results_analyzer_node(analysis_state)
        
        # Should return error state
        assert result.get("execution_verdict") == "fail", \
            "Should fail if stage_outputs is empty"
        assert result.get("run_error") is not None


class TestMaterialValidation:
    """Test material validation with real file paths."""
    
    def test_material_file_resolution(self):
        """Material files should resolve correctly from the materials directory."""
        
        materials_dir = Path(__file__).parent.parent / "materials"
        
        # Common material files that should exist
        expected_materials = [
            "palik_gold.csv",
            "palik_silver.csv",
        ]
        
        existing = []
        for mat in expected_materials:
            mat_file = materials_dir / mat
            if mat_file.exists():
                existing.append(mat)
                # Verify it's actually a valid CSV
                content = mat_file.read_text()
                assert "," in content, f"{mat} doesn't look like a CSV"
                lines = content.strip().split("\n")
                assert len(lines) > 1, f"{mat} has no data rows"
        
        # At least some material files should exist
        assert len(existing) > 0, (
            f"No material files found in {materials_dir}. "
            f"Expected at least one of: {expected_materials}"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
