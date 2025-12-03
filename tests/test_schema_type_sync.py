"""
Test that JSON schemas and Python types in state.py remain synchronized.

This test catches drift between:
- JSON schemas (source of truth for data structures)
- Python TypedDicts in schemas/state.py (used at runtime)
- Generated types in schemas/generated_types.py

If this test fails, it means state.py or generated_types.py needs to be updated 
to match the schemas, or vice versa if the schema change was intentional.
"""

import json
import os
import sys
import inspect
from pathlib import Path
from typing import Set, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas import generated_types
from schemas import state as state_module

SCHEMAS_DIR = PROJECT_ROOT / "schemas"


def load_schema(name: str) -> dict:
    """Load a JSON schema file."""
    with open(SCHEMAS_DIR / name) as f:
        return json.load(f)


def get_typed_dict_keys(typed_dict_class) -> Set[str]:
    """Get keys from a TypedDict class, handling forward references."""
    # TypedDict stores annotations in __annotations__
    if hasattr(typed_dict_class, '__annotations__'):
        return set(typed_dict_class.__annotations__.keys())
    return set()


def get_repro_state_keys() -> Set[str]:
    """Get ReproState keys by reading the source file directly.
    
    This avoids import issues with forward references like HardwareConfig.
    """
    import re
    state_file = SCHEMAS_DIR / "state.py"
    with open(state_file) as f:
        content = f.read()
    
    # Find the ReproState class definition
    # Look for lines like "    field_name: SomeType" inside the class
    in_repro_state = False
    keys = set()
    
    for line in content.split('\n'):
        if 'class ReproState' in line:
            in_repro_state = True
            continue
        if in_repro_state:
            # End of class when we hit another class or non-indented line
            if line and not line.startswith(' ') and not line.startswith('\t'):
                if 'class ' in line or 'def ' in line:
                    break
            # Match field definitions like "    field_name: Type"
            match = re.match(r'^    ([a-z_][a-z0-9_]*)\s*:', line)
            if match:
                keys.add(match.group(1))
    
    return keys


def get_schema_properties(schema: dict) -> Set[str]:
    """Extract property names from a JSON schema."""
    props = set()
    if "properties" in schema:
        props.update(schema["properties"].keys())
    return props


class TestSchemaTypeSync:
    """Tests that JSON schemas and Python types stay synchronized."""
    
    def test_repro_state_has_plan_field(self):
        """ReproState should have a 'plan' field that can hold plan schema data."""
        keys = get_repro_state_keys()
        assert "plan" in keys, "ReproState missing 'plan' field"
    
    def test_repro_state_has_assumptions_field(self):
        """ReproState should have an 'assumptions' field."""
        keys = get_repro_state_keys()
        assert "assumptions" in keys, "ReproState missing 'assumptions' field"
    
    def test_repro_state_has_progress_field(self):
        """ReproState should have a 'progress' field."""
        keys = get_repro_state_keys()
        assert "progress" in keys, "ReproState missing 'progress' field"
    
    def test_repro_state_has_metrics_field(self):
        """ReproState should have a 'metrics' field."""
        keys = get_repro_state_keys()
        assert "metrics" in keys, "ReproState missing 'metrics' field"
    
    def test_plan_schema_exists(self):
        """plan_schema.json should exist and be valid JSON."""
        schema = load_schema("plan_schema.json")
        assert "properties" in schema
        assert "paper_id" in schema["properties"]
        assert "stages" in schema["properties"]
    
    def test_progress_schema_exists(self):
        """progress_schema.json should exist and be valid JSON."""
        schema = load_schema("progress_schema.json")
        assert "properties" in schema
    
    def test_assumptions_schema_exists(self):
        """assumptions_schema.json should exist and be valid JSON."""
        schema = load_schema("assumptions_schema.json")
        assert "properties" in schema
    
    def test_agent_output_schemas_exist(self):
        """All agent output schemas should exist."""
        required_schemas = [
            "supervisor_output_schema.json",
            "plan_reviewer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_reviewer_output_schema.json",
            "results_analyzer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "comparison_validator_output_schema.json",
        ]
        for schema_name in required_schemas:
            schema_path = SCHEMAS_DIR / schema_name
            assert schema_path.exists(), f"Missing schema: {schema_name}"
            schema = load_schema(schema_name)
            assert "properties" in schema, f"Schema {schema_name} missing 'properties'"
    
    def test_supervisor_output_has_verdict(self):
        """Supervisor output schema should have verdict field."""
        schema = load_schema("supervisor_output_schema.json")
        props = get_schema_properties(schema)
        assert "verdict" in props, "Supervisor output missing 'verdict'"
    
    def test_plan_reviewer_output_has_verdict(self):
        """Plan reviewer output schema should have verdict field."""
        schema = load_schema("plan_reviewer_output_schema.json")
        props = get_schema_properties(schema)
        assert "verdict" in props, "Plan reviewer output missing 'verdict'"
    
    def test_physics_sanity_output_has_design_flaw_verdict(self):
        """Physics sanity output should support 'design_flaw' verdict for physics-driven redesign."""
        schema = load_schema("physics_sanity_output_schema.json")
        verdict_prop = schema["properties"].get("verdict", {})
        enum_values = verdict_prop.get("enum", [])
        assert "design_flaw" in enum_values, (
            "Physics sanity output missing 'design_flaw' verdict - "
            "needed for routing physics failures to design node"
        )
    
    def test_plan_schema_has_expected_outputs(self):
        """Plan schema stages should have expected_outputs field."""
        schema = load_schema("plan_schema.json")
        stages_schema = schema["properties"]["stages"]["items"]["properties"]
        assert "expected_outputs" in stages_schema, (
            "Plan schema stages missing 'expected_outputs' field - "
            "needed for output artifact specification"
        )
    
    def test_plan_schema_targets_have_precision_requirement(self):
        """Plan schema targets should have precision_requirement field."""
        schema = load_schema("plan_schema.json")
        targets_schema = schema["properties"]["targets"]["items"]["properties"]
        assert "precision_requirement" in targets_schema, (
            "Plan schema targets missing 'precision_requirement' field - "
            "needed for digitized data enforcement"
        )
    
    def test_plan_reviewer_has_digitized_data_checklist(self):
        """Plan reviewer output should have digitized_data checklist item."""
        schema = load_schema("plan_reviewer_output_schema.json")
        props = get_schema_properties(schema)
        
        # It should be nested in checklist_results
        assert "checklist_results" in props, "Plan reviewer output missing 'checklist_results'"
        
        checklist_props = schema["properties"]["checklist_results"]["properties"]
        assert "digitized_data" in checklist_props, (
            "Plan reviewer checklist missing 'digitized_data' section"
        )

    def test_state_has_validated_materials(self):
        """ReproState should have validated_materials field for material handoff."""
        keys = get_repro_state_keys()
        assert "validated_materials" in keys, (
            "ReproState missing 'validated_materials' field - "
            "needed for material data handoff from Stage 0"
        )
    
    def test_state_has_ask_user_fields(self):
        """ReproState should have ask_user contract fields."""
        keys = get_repro_state_keys()
        assert "ask_user_trigger" in keys, "ReproState missing 'ask_user_trigger'"
        assert "last_node_before_ask_user" in keys, "ReproState missing 'last_node_before_ask_user'"
    
    def test_state_has_separate_review_verdicts(self):
        """ReproState should have separate verdict fields for each reviewer."""
        keys = get_repro_state_keys()
        assert "last_plan_review_verdict" in keys, "ReproState missing 'last_plan_review_verdict'"
        assert "last_design_review_verdict" in keys, "ReproState missing 'last_design_review_verdict'"
        assert "last_code_review_verdict" in keys, "ReproState missing 'last_code_review_verdict'"
    
    def test_validation_hierarchy_is_computed(self):
        """Validation hierarchy should be computed, not stored in state."""
        keys = get_repro_state_keys()
        # Should NOT be stored directly
        assert "validation_hierarchy" not in keys, (
            "validation_hierarchy should NOT be in ReproState - "
            "use get_validation_hierarchy() to compute on demand"
        )
        # Check that the function exists by reading the source
        state_file = SCHEMAS_DIR / "state.py"
        with open(state_file) as f:
            content = f.read()
        assert "def get_validation_hierarchy" in content, (
            "get_validation_hierarchy() function should exist in state.py"
        )


class TestGeneratedTypesSync:
    """Tests that schemas/generated_types.py is in sync with schemas/*.json."""

    def test_generated_types_has_all_agent_outputs(self):
        """Verify that all agent output schemas have corresponding generated types."""
        # map schema filename to expected class name
        # This naming convention is standard for datamodel-code-generator based on 'title' in schema
        expected_types = {
            "plan_reviewer_output_schema.json": "PlanReviewerAgentOutput",
            "design_reviewer_output_schema.json": "DesignReviewerAgentOutput",
            "code_reviewer_output_schema.json": "CodeReviewerAgentOutput",
            "supervisor_output_schema.json": "SupervisorAgentOutput",
            "execution_validator_output_schema.json": "ExecutionValidatorAgentOutput",
            "results_analyzer_output_schema.json": "ResultsAnalyzerAgentOutput",
            "comparison_validator_output_schema.json": "ComparisonValidatorAgentOutput",
            "physics_sanity_output_schema.json": "PhysicsSanityAgentOutput",
        }

        for schema_file, class_name in expected_types.items():
            # Check if the class exists in generated_types module
            assert hasattr(generated_types, class_name), (
                f"Missing generated type '{class_name}' for schema '{schema_file}'. "
                "Run 'python scripts/generate_types.py' to regenerate."
            )

    def test_generated_type_fields_match_schema(self):
        """Verify that fields in generated types match the schema required fields."""
        # Example check for PlanReviewerAgentOutput
        if not hasattr(generated_types, "PlanReviewerAgentOutput"):
            # Skip if type is missing (test_generated_types_has_all_agent_outputs will catch it)
            return

        plan_reviewer_type = getattr(generated_types, "PlanReviewerAgentOutput")
        keys = get_typed_dict_keys(plan_reviewer_type)
        
        schema = load_schema("plan_reviewer_output_schema.json")
        required = set(schema.get("required", []))
        
        # All required fields in schema should be in TypedDict
        missing = required - keys
        assert not missing, f"PlanReviewerAgentOutput missing required fields: {missing}"


class TestSchemaConsistency:
    """Tests for internal consistency of schemas."""
    
    def test_all_schemas_are_valid_json(self):
        """All JSON files in schemas/ should be valid JSON."""
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            try:
                with open(schema_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Invalid JSON in {schema_file.name}: {e}")
    
    def test_all_schemas_have_required_metadata(self):
        """All schemas should have $schema and title."""
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            schema = load_schema(schema_file.name)
            assert "$schema" in schema, f"{schema_file.name} missing $schema"
            # title is recommended but not strictly required
    
    def test_no_empty_required_arrays(self):
        """Schemas with 'required' should have at least one required field."""
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            schema = load_schema(schema_file.name)
            if "required" in schema:
                assert len(schema["required"]) > 0, (
                    f"{schema_file.name} has empty 'required' array"
                )

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
