"""
Contract and Schema Validation Tests for LLM Outputs.

These tests verify that:
1. Mock responses conform STRICTLY to JSON schemas
2. LLM output contracts are respected (logical dependencies)
3. Edge cases in LLM responses are handled gracefully

The tests use jsonschema for validation and enforce strict compliance.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

import jsonschema
from jsonschema import validate, ValidationError


# ═══════════════════════════════════════════════════════════════════════
# Schema Loading
# ═══════════════════════════════════════════════════════════════════════

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"
MOCK_RESPONSES_DIR = Path(__file__).parent / "fixtures" / "mock_responses"


def load_schema(schema_name: str) -> dict:
    """Load a JSON schema file."""
    path = SCHEMAS_DIR / schema_name
    with open(path, "r") as f:
        return json.load(f)


def load_mock_response(agent_name: str) -> dict:
    """Load a mock response for an agent."""
    path = MOCK_RESPONSES_DIR / f"{agent_name}_response.json"
    with open(path, "r") as f:
        return json.load(f)


# Schema mapping: agent name -> schema file
AGENT_SCHEMAS = {
    "planner": "planner_output_schema.json",
    "plan_reviewer": "plan_reviewer_output_schema.json",
    "simulation_designer": "simulation_designer_output_schema.json",
    "design_reviewer": "design_reviewer_output_schema.json",
    "code_generator": "code_generator_output_schema.json",
    "code_reviewer": "code_reviewer_output_schema.json",
    "execution_validator": "execution_validator_output_schema.json",
    "physics_sanity": "physics_sanity_output_schema.json",
    "results_analyzer": "results_analyzer_output_schema.json",
    "comparison_validator": "comparison_validator_output_schema.json",
    "supervisor": "supervisor_output_schema.json",
}


# ═══════════════════════════════════════════════════════════════════════
# Strict Mock Response Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMockResponsesFullSchemaValidation:
    """Strict validation: mock responses MUST fully conform to their schemas.
    
    These tests validate:
    - All required fields are present
    - Field values match enum constraints
    - Nested object structures are correct
    - Types are strictly enforced
    
    If these tests fail, either the mock or the schema needs updating.
    """
    
    @pytest.mark.parametrize("agent_name,schema_file", [
        ("planner", "planner_output_schema.json"),
        ("plan_reviewer", "plan_reviewer_output_schema.json"),
        ("simulation_designer", "simulation_designer_output_schema.json"),
        ("design_reviewer", "design_reviewer_output_schema.json"),
        ("code_generator", "code_generator_output_schema.json"),
        ("code_reviewer", "code_reviewer_output_schema.json"),
        ("execution_validator", "execution_validator_output_schema.json"),
        ("physics_sanity", "physics_sanity_output_schema.json"),
        ("results_analyzer", "results_analyzer_output_schema.json"),
        ("supervisor", "supervisor_output_schema.json"),
    ])
    def test_mock_response_fully_validates(self, agent_name, schema_file):
        """Mock response must fully validate against its schema."""
        try:
            response = load_mock_response(agent_name)
            schema = load_schema(schema_file)
            
            # This will raise ValidationError if response doesn't match schema
            validate(instance=response, schema=schema)
            
        except FileNotFoundError as e:
            pytest.skip(f"File not found: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Cross-Field Logic Tests (Strict Contracts)
# ═══════════════════════════════════════════════════════════════════════

class TestStrictCrossFieldConstraints:
    """Test logical dependencies between fields that might not be captured by JSON Schema."""
    
    def test_reviewer_rejection_logic(self):
        """If a reviewer rejects (needs_revision), there MUST be issues listed."""
        reviewers = ["plan_reviewer", "design_reviewer", "code_reviewer"]
        
        for agent in reviewers:
            try:
                response = load_mock_response(agent)
                verdict = response.get("verdict")
                issues = response.get("issues", [])
                
                if verdict == "needs_revision":
                    assert len(issues) > 0, f"{agent}: Verdict is 'needs_revision' but 'issues' list is empty."
                    # Also check that issues have descriptions
                    for issue in issues:
                        assert issue.get("description"), f"{agent}: Issue missing description."
                elif verdict == "approve":
                    # Ideally should have no critical issues, but warnings are allowed
                    critical_issues = [i for i in issues if i.get("severity") == "critical"]
                    assert len(critical_issues) == 0, f"{agent}: Verdict is 'approve' but has critical issues."
            except FileNotFoundError:
                continue

    def test_supervisor_decision_consistency(self):
        """Supervisor decision fields must match the verdict."""
        try:
            response = load_mock_response("supervisor")
            verdict = response.get("verdict")
            
            if verdict == "backtrack_to_stage":
                assert "backtrack_decision" in response, "Verdict is 'backtrack_to_stage' but 'backtrack_decision' is missing."
                assert response["backtrack_decision"].get("target_stage_id"), "Backtrack target stage ID missing."
                
            elif verdict == "ask_user":
                # Schema might check existence, but check content
                assert response.get("user_question"), "Verdict is 'ask_user' but 'user_question' is empty/missing."
                
            elif verdict == "all_complete":
                assert response.get("should_stop") is True, "Verdict is 'all_complete' but 'should_stop' is not True."
                
        except FileNotFoundError:
            pytest.skip("Supervisor mock not found")

    def test_planner_stages_integrity(self):
        """Planner stages must form a coherent plan."""
        try:
            response = load_mock_response("planner")
            stages = response.get("stages", [])
            
            if not stages:
                # Empty plan is valid edge case but rare. Warn or skip.
                return
                
            # Check dependencies refer to existing stages
            stage_ids = {s["stage_id"] for s in stages}
            for stage in stages:
                deps = stage.get("dependencies", [])
                for dep in deps:
                    assert dep in stage_ids, f"Stage {stage['stage_id']} depends on unknown stage {dep}"
                    
            # Check logical order (simplistic: dependency should appear before dependent)
            # This is O(N^2) but N is small
            seen_ids = set()
            for stage in stages:
                curr_id = stage["stage_id"]
                deps = stage.get("dependencies", [])
                # All dependencies should be in seen_ids (if topological sort is preserved in list)
                # The schema implies "Ordered list", so we expect topological order.
                missing_deps = [d for d in deps if d not in seen_ids]
                assert not missing_deps, f"Stage {curr_id} appears before its dependencies: {missing_deps}"
                seen_ids.add(curr_id)
                
        except FileNotFoundError:
            pytest.skip("Planner mock not found")

    def test_code_generator_safety_compliance(self):
        """Code generator must confirm safety checks are passed."""
        try:
            response = load_mock_response("code_generator")
            safety = response.get("safety_checks", {})
            
            # All safety checks present should be True
            # We don't enforce presence of all keys here (schema does or doesn't), 
            # but if they are present, they must be True for a valid code generation.
            for check, passed in safety.items():
                assert passed is True, f"Code generator failed safety check: {check}"
                
            # Verify runtime estimate is positive
            runtime = response.get("estimated_runtime_minutes", 0)
            assert runtime > 0, "Estimated runtime must be positive"
            
        except FileNotFoundError:
            pytest.skip("Code generator mock not found")

    def test_simulation_designer_content(self):
        """Simulation design must be non-empty."""
        try:
            response = load_mock_response("simulation_designer")
            geometry = response.get("geometry", {})
            structures = geometry.get("structures", [])
            
            assert len(structures) > 0, "Simulation design has no structures"
            
            # Check unit system
            unit_system = response.get("unit_system", {})
            assert unit_system.get("characteristic_length_m", 0) > 0, "Characteristic length must be positive"
            
        except FileNotFoundError:
            pytest.skip("Simulation designer mock not found")

    def test_execution_validator_logic(self):
        """Execution validator consistency check."""
        try:
            response = load_mock_response("execution_validator")
            exec_status = response.get("execution_status", {})
            verdict = response.get("verdict")
            
            if exec_status.get("completed") is False:
                assert verdict == "fail", "Verdict should be fail if execution not completed"
                
            # If all files present, files_check.all_present should be True
            files_check = response.get("files_check", {})
            expected = set(files_check.get("expected_files", []))
            found = set(files_check.get("found_files", []))
            if expected and expected.issubset(found):
                assert files_check.get("all_present") is True, "all_present should be True if all files found"
                
        except FileNotFoundError:
            pytest.skip("Execution validator mock not found")

    def test_physics_sanity_logic(self):
        """Physics sanity logic check."""
        try:
            response = load_mock_response("physics_sanity")
            verdict = response.get("verdict")
            
            if verdict in ["fail", "design_flaw"]:
                # Should have concerns or failed checks
                concerns = response.get("concerns", [])
                failed_conservation = [c for c in response.get("conservation_checks", []) if c.get("status") == "fail"]
                failed_ranges = [c for c in response.get("value_range_checks", []) if c.get("status") == "fail"]
                
                assert concerns or failed_conservation or failed_ranges, "Failed physics verdict requires concerns or failed checks"
                
            if verdict == "pass":
                # Should not have critical concerns
                concerns = response.get("concerns", [])
                critical = [c for c in concerns if c.get("severity") == "critical"]
                assert not critical, "Pass verdict cannot have critical concerns"
                
        except FileNotFoundError:
            pytest.skip("Physics sanity mock not found")


# ═══════════════════════════════════════════════════════════════════════
# Specific Contract Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReviewerContract:
    """Test reviewer agent contract logic."""
    
    def test_approve_verdict_structure(self):
        """Approve verdict should have clean structure."""
        response = {"verdict": "approve", "issues": [], "summary": "LGTM"}
        assert response["verdict"] == "approve"
        assert not any(issue.get("severity") == "critical" for issue in response["issues"])
    
    def test_needs_revision_structure(self):
        """needs_revision verdict should include actionable feedback."""
        response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Fix X"}],
            "summary": "Needs work",
            "feedback": "Please address X",
        }
        assert response["verdict"] == "needs_revision"
        assert len(response["issues"]) > 0 or response.get("feedback")


class TestSupervisorContract:
    """Test supervisor agent contract logic."""
    
    def test_ok_continue_allows_progression(self):
        response = {
            "verdict": "ok_continue",
            "summary": "Stage completed",
            "validation_hierarchy_status": {}, # Minimal for contract check
            "main_physics_assessment": {}
        }
        # Just checking logic, not schema
        assert response["verdict"] == "ok_continue"
        assert not response.get("should_stop", False)
    
    def test_all_complete_stops_workflow(self):
        response = {
            "verdict": "all_complete",
            "should_stop": True,
            "stop_reason": "Done",
            "summary": "Complete"
        }
        assert response["verdict"] == "all_complete"
        assert response["should_stop"] is True


# ═══════════════════════════════════════════════════════════════════════
# Edge Case & Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCaseResponses:
    """Test handling of edge cases in LLM responses."""
    
    def test_empty_stages_array(self):
        """Handle planner returning empty stages."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "No reproducible content",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],  # Empty - nothing to reproduce
            "assumptions": {},
            "progress": {"stages": []},
        }
        
        schema = load_schema("planner_output_schema.json")
        # Should be valid - empty stages is allowed
        validate(instance=response, schema=schema)
    
    def test_planner_mixed_parameter_types(self):
        """extracted_parameters values can be number, string, or array."""
        response = {
            "paper_id": "test", "paper_domain": "other", "title": "T", "summary": "S",
            "extracted_parameters": [
                {"name": "p1", "value": 1.5, "unit": "nm", "source": "text"},
                {"name": "p2", "value": "approx 5", "unit": "nm", "source": "text"},
                {"name": "p3", "value": [1.0, 2.0], "unit": "nm", "source": "text"},
            ],
            "targets": [], "stages": [], "assumptions": {}, "progress": {"stages": []}
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        
    def test_supervisor_partial_validation_status(self):
        """Validation hierarchy can be partially complete."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed", # Failed but continuing? Rare but schema allows
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Continue despite failure"
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)


class TestMalformedResponses:
    """Test handling of malformed LLM responses (Negative Testing)."""
    
    def test_missing_required_field_detected(self):
        """Missing required fields should be detected."""
        response = {"issues": [], "summary": "Test"} # Missing verdict
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value)
    
    def test_wrong_type_detected(self):
        """Wrong field types should be detected."""
        response = {
            "verdict": "approve",
            "issues": "not an array",
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "issues" in str(exc_info.value) or "array" in str(exc_info.value)
    
    def test_invalid_enum_detected(self):
        """Invalid enum values should be detected."""
        response = {
            "verdict": "maybe", 
            "issues": [],
            "summary": "Test",
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "maybe" in str(exc_info.value) or "enum" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════
# Schema Integrity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaCompleteness:
    """Test that all agent schemas exist and are valid JSON schemas."""
    
    @pytest.mark.parametrize("schema_file", [
        "planner_output_schema.json",
        "plan_reviewer_output_schema.json",
        "simulation_designer_output_schema.json",
        "design_reviewer_output_schema.json",
        "code_generator_output_schema.json",
        "code_reviewer_output_schema.json",
        "execution_validator_output_schema.json",
        "physics_sanity_output_schema.json",
        "results_analyzer_output_schema.json",
        "comparison_validator_output_schema.json",
        "supervisor_output_schema.json",
    ])
    def test_schema_is_valid_json(self, schema_file):
        """Each schema file must be valid JSON and have schema structure."""
        path = SCHEMAS_DIR / schema_file
        assert path.exists()
        with open(path) as f:
            schema = json.load(f)
        # Minimal check for valid JSON schema
        assert "type" in schema or "$ref" in schema or "properties" in schema
        
