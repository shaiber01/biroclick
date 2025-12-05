"""
Smoke Tests with Real LLM Calls.

These tests make actual LLM API calls to verify integration works.
They are marked with @pytest.mark.smoke and @pytest.mark.slow.

Run with: pytest -m smoke tests/test_llm_smoke.py

Environment Requirements:
- ANTHROPIC_API_KEY must be set (in .env file or environment)
- Network access required

Setup:
1. Install python-dotenv: pip install python-dotenv
2. Create .env file in project root with: ANTHROPIC_API_KEY=your-key-here
3. Run: pytest -m smoke tests/test_llm_smoke.py

These tests:
1. Verify LLM integration is properly configured
2. Validate real LLM outputs conform to schemas
3. Test actual prompt → response flow

Note: These tests are marked as 'smoke' and are excluded from normal test runs
      to avoid wasting money on LLM calls. Run explicitly with: pytest -m smoke
"""
import json
import os
import pytest
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import jsonschema
from jsonschema import validate

from schemas.state import create_initial_state


# Mark all tests in this module as smoke tests
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
]


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_schema(schema_name: str) -> dict:
    """Load a JSON schema file."""
    path = SCHEMAS_DIR / schema_name
    with open(path, "r") as f:
        return json.load(f)


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / name
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture
def paper_input():
    """Load sample paper input."""
    return load_fixture("sample_paper_input.json")


@pytest.fixture
def base_state(paper_input):
    """Create base state from paper input."""
    state = create_initial_state(
        paper_id=paper_input["paper_id"],
        paper_text=paper_input["paper_text"],
        paper_domain=paper_input.get("paper_domain", "other"),
    )
    state["paper_figures"] = paper_input.get("paper_figures", [])
    return state


def llm_available():
    """Check if LLM API is available."""
    return bool(
        os.environ.get("OPENAI_API_KEY") or 
        os.environ.get("ANTHROPIC_API_KEY")
    )


skip_if_no_llm = pytest.mark.skipif(
    not llm_available(),
    reason="LLM API key not configured (OPENAI_API_KEY or ANTHROPIC_API_KEY)"
)


# ═══════════════════════════════════════════════════════════════════════
# Smoke Tests
# ═══════════════════════════════════════════════════════════════════════

@skip_if_no_llm
class TestLLMIntegration:
    """Test actual LLM integration."""
    
    def test_llm_client_import(self):
        """LLM client module should import without error."""
        from src.llm_client import call_agent_with_metrics
        assert callable(call_agent_with_metrics)
    
    def test_prompts_build(self, base_state):
        """Prompts should build without error."""
        from src.prompts import build_agent_prompt
        
        agents = ["planner", "plan_reviewer", "simulation_designer", 
                  "code_generator", "supervisor"]
        
        for agent in agents:
            prompt = build_agent_prompt(agent, base_state)
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should have substantial content


@skip_if_no_llm
class TestPlannerSmoke:
    """Smoke test for planner agent with real LLM."""
    
    def test_planner_returns_valid_response(self, base_state):
        """Planner should return schema-valid response."""
        from src.agents import plan_node
        
        result = plan_node(base_state)
        
        # Check if LLM actually succeeded (not just error handling)
        if result.get("ask_user_trigger") == "llm_error":
            pytest.fail(f"LLM call failed: {result.get('pending_user_questions', ['Unknown error'])}")
        
        # Should have plan in result
        assert "plan" in result, f"Expected 'plan' in result, got keys: {list(result.keys())}"
        
        # Validate plan structure
        plan = result["plan"]
        assert "stages" in plan, "Plan missing 'stages'"
        assert isinstance(plan["stages"], list)
        assert len(plan["stages"]) > 0, "Plan has no stages"
    
    def test_planner_extracts_parameters(self, base_state):
        """Planner should extract parameters from paper."""
        from src.agents import plan_node
        
        result = plan_node(base_state)
        
        # Check if LLM actually succeeded
        if result.get("ask_user_trigger") == "llm_error":
            pytest.fail(f"LLM call failed: {result.get('pending_user_questions', ['Unknown error'])}")
        
        if "plan" not in result:
            pytest.skip("Planner didn't return plan")
        
        plan = result["plan"]
        
        # Should have extracted_parameters
        assert "extracted_parameters" in plan, "Plan missing 'extracted_parameters'"
        params = plan["extracted_parameters"]
        assert isinstance(params, list)
        
        # Parameters should have required fields
        for param in params:
            assert "name" in param, f"Parameter missing 'name': {param}"
            assert "value" in param, f"Parameter missing 'value': {param}"


@skip_if_no_llm  
class TestReviewerSmoke:
    """Smoke test for reviewer agents with real LLM."""
    
    def test_plan_reviewer_returns_verdict(self, base_state):
        """Plan reviewer should return valid verdict."""
        from src.agents import plan_node, plan_reviewer_node
        
        # First get a plan
        plan_result = plan_node(base_state)
        
        # Check if LLM failed
        if plan_result.get("ask_user_trigger") == "llm_error":
            pytest.fail(f"Planner LLM call failed: {plan_result.get('pending_user_questions', ['Unknown error'])}")
        
        if "plan" not in plan_result:
            pytest.fail(f"Planner didn't return plan, got: {list(plan_result.keys())}")
        
        # Update state with plan
        base_state.update(plan_result)
        
        # Review the plan
        review_result = plan_reviewer_node(base_state)
        
        # Check if reviewer LLM failed
        if review_result.get("ask_user_trigger") == "llm_error":
            pytest.fail(f"Reviewer LLM call failed: {review_result.get('pending_user_questions', ['Unknown error'])}")
        
        # Should have verdict
        assert "last_plan_review_verdict" in review_result
        assert review_result["last_plan_review_verdict"] in ["approve", "needs_revision"]


@skip_if_no_llm
class TestSchemaValidationSmoke:
    """Validate real LLM outputs against schemas."""
    
    def test_planner_output_matches_schema(self, base_state):
        """Real planner output should match schema."""
        from src.llm_client import call_agent_with_metrics
        from src.prompts import build_agent_prompt
        
        # Make direct LLM call to get raw output
        system_prompt = build_agent_prompt("planner", base_state)
        user_content = f"Paper text:\n{base_state.get('paper_text', '')[:5000]}"
        
        try:
            raw_output = call_agent_with_metrics(
                agent_name="planner",
                system_prompt=system_prompt,
                user_content=user_content,
                state=base_state,
            )
        except Exception as e:
            pytest.fail(f"LLM call failed: {e}")
        
        # Validate against schema
        assert raw_output is not None, "LLM returned None"
        
        # Check required fields exist
        required_fields = ["paper_id", "paper_domain", "title", "summary", "stages"]
        missing = [f for f in required_fields if f not in raw_output]
        assert not missing, f"LLM response missing required fields: {missing}"
        
        # Validate against full schema
        schema = load_schema("planner_output_schema.json")
        validate(instance=raw_output, schema=schema)
    
    def test_reviewer_output_matches_schema(self, base_state):
        """Real reviewer output should match schema."""
        from src.llm_client import call_agent_with_metrics
        from src.prompts import build_agent_prompt
        
        # Setup minimal plan for review
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "description": "Test"}],
        }
        
        system_prompt = build_agent_prompt("plan_reviewer", base_state)
        user_content = "Review this plan for a gold nanorod simulation."
        
        try:
            raw_output = call_agent_with_metrics(
                agent_name="plan_reviewer",
                system_prompt=system_prompt,
                user_content=user_content,
                state=base_state,
            )
        except Exception as e:
            pytest.fail(f"LLM call failed: {e}")
        
        assert raw_output is not None, "LLM returned None"
        assert "verdict" in raw_output, f"LLM response missing 'verdict', got: {list(raw_output.keys())}"
        
        # Validate against full schema
        schema = load_schema("plan_reviewer_output_schema.json")
        validate(instance=raw_output, schema=schema)


@skip_if_no_llm
class TestSupervisorSmoke:
    """Smoke test for supervisor agent with real LLM."""
    
    def test_supervisor_returns_verdict(self, base_state):
        """Supervisor should return valid verdict from LLM (not fallback)."""
        from src.agents import supervisor_node
        
        # Setup minimal state
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {"stage_id": "s1", "stage_type": "SINGLE_STRUCTURE", 
                 "targets": ["Fig1"], "dependencies": []}
            ],
        }
        base_state["progress"] = {
            "stages": [{"stage_id": "s1", "status": "completed_success"}]
        }
        
        result = supervisor_node(base_state)
        
        # Check for LLM error escalation
        if result.get("ask_user_trigger") == "llm_error":
            pytest.fail(f"LLM call failed: {result.get('pending_user_questions', ['Unknown error'])}")
        
        # Check for LLM fallback (indicates LLM didn't respond properly)
        feedback = result.get("supervisor_feedback", "")
        if "LLM unavailable" in feedback:
            pytest.fail(f"LLM call failed with fallback: {feedback}")
        
        assert "supervisor_verdict" in result
        valid_verdicts = ["ok_continue", "replan_needed", "change_priority", 
                        "ask_user", "backtrack_to_stage", "all_complete"]
        assert result["supervisor_verdict"] in valid_verdicts


# ═══════════════════════════════════════════════════════════════════════
# Performance Smoke Tests
# ═══════════════════════════════════════════════════════════════════════

@skip_if_no_llm
class TestLLMPerformance:
    """Basic performance checks for LLM calls."""
    
    def test_planner_completes_in_reasonable_time(self, base_state):
        """Planner should complete within timeout."""
        import time
        from src.agents import plan_node
        
        start = time.time()
        
        try:
            result = plan_node(base_state)
        except Exception:
            pass  # We're just testing timing
        
        elapsed = time.time() - start
        
        # Should complete within 2 minutes (generous timeout)
        assert elapsed < 120, f"Planner took {elapsed:.1f}s (expected <120s)"
    
    def test_reviewer_completes_quickly(self, base_state):
        """Reviewer should be faster than planner."""
        import time
        from src.agents import plan_reviewer_node
        
        # Minimal plan for review
        base_state["plan"] = {
            "paper_id": "test",
            "stages": [{"stage_id": "s1", "description": "Test stage"}],
        }
        
        start = time.time()
        
        try:
            result = plan_reviewer_node(base_state)
        except Exception:
            pass
        
        elapsed = time.time() - start
        
        # Reviewer should be quick
        assert elapsed < 60, f"Reviewer took {elapsed:.1f}s (expected <60s)"


# ═══════════════════════════════════════════════════════════════════════
# Error Handling Smoke Tests
# ═══════════════════════════════════════════════════════════════════════

@skip_if_no_llm
class TestLLMErrorHandling:
    """Test error handling with real LLM."""
    
    def test_handles_empty_paper_text(self, base_state):
        """Should handle empty paper text gracefully."""
        from src.agents import plan_node
        
        base_state["paper_text"] = ""
        
        # Should not crash
        try:
            result = plan_node(base_state)
            # May return error trigger
            assert "plan" in result or "ask_user_trigger" in result
        except Exception as e:
            # Should be a handled error, not a crash
            assert "paper" in str(e).lower() or isinstance(e, (ValueError, KeyError))
    
    def test_handles_very_long_paper(self, base_state):
        """Should handle very long paper text."""
        from src.agents import plan_node
        
        # Create very long paper text
        base_state["paper_text"] = "This is a test. " * 10000
        
        # Should handle context overflow gracefully
        try:
            result = plan_node(base_state)
            # May trigger context overflow
            assert result is not None
        except Exception as e:
            # Acceptable to fail on context limits
            assert "context" in str(e).lower() or "token" in str(e).lower()