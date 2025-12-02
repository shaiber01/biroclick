"""
REAL Integration Tests - Tests that find bugs.

These tests run ACTUAL code with MINIMAL mocking.
Only the LLM API call itself is mocked - everything else runs for real.

If a test fails here, it's a BUG IN THE CODE, not the test.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the REAL modules
from src.prompts import build_agent_prompt, load_prompt_file, PROMPTS_DIR
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
        "report_generator",  # This one is missing!
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
        import json
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
# Test: Node Functions Can Be Called
# ═══════════════════════════════════════════════════════════════════════

class TestNodeFunctionsCallable:
    """
    Test that node functions can be called with minimal state.
    
    Only mock the LLM call itself - everything else runs for real.
    """
    
    @pytest.fixture
    def minimal_state(self):
        """Create minimal valid state for testing."""
        state = create_initial_state(
            paper_id="test_paper",
            paper_text="We study gold nanorods with length 100nm and diameter 40nm.",
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
            if "reviewer" in agent or "validator" in agent:
                return {"verdict": "approve", "issues": [], "summary": "OK"}
            elif agent == "planner":
                return {
                    "paper_id": "test",
                    "title": "Test",
                    "stages": [],
                    "targets": [],
                    "extracted_parameters": [],
                }
            elif agent == "supervisor":
                return {"verdict": "ok_continue", "feedback": "OK"}
            else:
                return {}
        return _mock
    
    def test_plan_node_builds_prompt_successfully(self, minimal_state, mock_llm_response):
        """plan_node must be able to build its prompt without crashing."""
        from src.agents.planning import plan_node
        
        # Only mock the LLM call, not the prompt building
        with patch("src.agents.planning.call_agent_with_metrics", mock_llm_response):
            # This will fail if prompt file is missing or has errors
            result = plan_node(minimal_state)
            assert result is not None
    
    def test_supervisor_node_builds_prompt_successfully(self, minimal_state, mock_llm_response):
        """supervisor_node must be able to build its prompt without crashing."""
        from src.agents.supervision.supervisor import supervisor_node
        
        with patch("src.agents.supervision.supervisor.call_agent_with_metrics", mock_llm_response):
            result = supervisor_node(minimal_state)
            assert result is not None
    
    def test_report_node_builds_prompt_successfully(self, minimal_state, mock_llm_response):
        """generate_report_node must be able to build its prompt without crashing."""
        from src.agents.reporting import generate_report_node
        
        # This test WILL FAIL because report_generator_agent.md is missing!
        with patch("src.agents.reporting.call_agent_with_metrics", mock_llm_response):
            result = generate_report_node(minimal_state)
            assert result is not None
    
    def test_design_node_builds_prompt_successfully(self, minimal_state, mock_llm_response):
        """simulation_designer_node must be able to build its prompt."""
        from src.agents.design import simulation_designer_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        
        with patch("src.agents.design.call_agent_with_metrics", mock_llm_response):
            result = simulation_designer_node(minimal_state)
            assert result is not None
    
    def test_code_generator_builds_prompt_successfully(self, minimal_state, mock_llm_response):
        """code_generator_node must be able to build its prompt."""
        from src.agents.code import code_generator_node
        
        minimal_state["current_stage_id"] = "stage_0"
        minimal_state["current_stage_type"] = "MATERIAL_VALIDATION"
        minimal_state["current_design"] = {"stage_id": "stage_0", "geometry": []}
        
        with patch("src.agents.code.call_agent_with_metrics", mock_llm_response):
            result = code_generator_node(minimal_state)
            assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# Test: Routing Functions Return Valid Values
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingReturnsValidValues:
    """Test that routing functions return values that exist in the graph."""
    
    def test_supervisor_verdicts_match_routing(self):
        """Supervisor schema verdicts must match what routing expects."""
        import json
        
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
                import json
                with open(schema_file) as f:
                    schema = json.load(f)
                
                if "verdict" in schema.get("properties", {}):
                    schema_verdicts = set(schema["properties"]["verdict"].get("enum", []))
                    missing = expected_verdicts - schema_verdicts
                    extra = schema_verdicts - expected_verdicts
                    
                    assert not missing, f"{reviewer} schema missing verdicts: {missing}"
                    # Extra verdicts are OK as long as routing has a default


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

