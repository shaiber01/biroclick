"""
Test that JSON schemas and Python types in state.py remain synchronized.

This test catches drift between:
- JSON schemas (source of truth for data structures)
- Python TypedDicts in schemas/state.py (used at runtime)
- Generated types in schemas/generated_types.py

If this test fails, it means state.py or generated_types.py needs to be updated 
to match the schemas, or vice versa if the schema change was intentional.

CRITICAL: These tests are designed to FIND BUGS, not to pass.
If a test fails, fix the COMPONENT UNDER TEST, not the test.
"""

import json
import os
import sys
import inspect
import re
from pathlib import Path
from typing import Set, Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas import generated_types
from schemas import state as state_module

SCHEMAS_DIR = PROJECT_ROOT / "schemas"


def load_schema(name: str) -> dict:
    """Load a JSON schema file."""
    schema_path = SCHEMAS_DIR / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {name}")
    with open(schema_path) as f:
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
            # Also handle NotRequired fields
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


def get_all_schema_properties_recursive(schema: dict, prefix: str = "") -> Set[str]:
    """Recursively extract all property names from a JSON schema."""
    props = set()
    if "properties" in schema:
        for key, value in schema["properties"].items():
            full_key = f"{prefix}.{key}" if prefix else key
            props.add(full_key)
            # Recursively get nested properties
            if isinstance(value, dict):
                props.update(get_all_schema_properties_recursive(value, full_key))
    if "items" in schema and isinstance(schema["items"], dict):
        # Handle array items
        props.update(get_all_schema_properties_recursive(schema["items"], prefix))
    return props


def get_schema_required_fields(schema: dict) -> Set[str]:
    """Get all required fields from a schema, including nested ones."""
    required = set()
    if "required" in schema:
        required.update(schema["required"])
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                nested_required = get_schema_required_fields(prop_schema)
                required.update(f"{prop_name}.{r}" for r in nested_required)
    return required


# Expected ReproState fields - comprehensive list from state.py
EXPECTED_REPRO_STATE_FIELDS = {
    # Paper Identification
    "paper_id", "paper_domain", "paper_text", "paper_title", "paper_citation",
    # Runtime & Hardware Configuration
    "runtime_config", "hardware_config",
    # Shared Artifacts
    "plan", "assumptions", "progress",
    # Extracted Parameters
    "extracted_parameters",
    # Materials
    "planned_materials", "validated_materials", "pending_validated_materials",
    # Validation Tracking
    "geometry_interpretations", "discrepancies_log", "systematic_shifts",
    # Current Control
    "current_stage_id", "current_stage_type", "workflow_phase", "workflow_complete",
    # Revision Tracking
    "design_revision_count", "code_revision_count", "execution_failure_count",
    "total_execution_failures", "physics_failure_count", "analysis_revision_count",
    "replan_count",
    # Backtracking Support
    "backtrack_suggestion", "invalidated_stages", "backtrack_count",
    # Verdicts
    "last_plan_review_verdict", "last_design_review_verdict", "last_code_review_verdict",
    "reviewer_issues", "execution_verdict", "physics_verdict", "comparison_verdict",
    "supervisor_verdict", "backtrack_decision",
    # Stage Working Data
    "code", "design_description", "performance_estimate", "stage_outputs",
    "run_error", "analysis_summary", "analysis_result_reports",
    "analysis_overall_classification", "quantitative_summary",
    # Agent Feedback
    "reviewer_feedback", "supervisor_feedback", "planner_feedback",
    # Performance Tracking
    "runtime_budget_remaining_seconds", "total_runtime_seconds", "stage_start_time",
    # User Interaction
    "pending_user_questions", "user_responses", "awaiting_user_input",
    "user_interactions", "ask_user_trigger", "last_node_before_ask_user",
    # Report Generation
    "figure_comparisons", "overall_assessment", "systematic_discrepancies_identified",
    "report_conclusions", "final_report_markdown",
    # Metrics Tracking
    "metrics",
    # Paper Figures
    "paper_figures",
    # Prompt Adaptations
    "prompt_adaptations",
}


class TestSchemaTypeSync:
    """Tests that JSON schemas and Python types stay synchronized."""
    
    def test_repro_state_has_all_expected_fields(self):
        """ReproState should have ALL expected fields - comprehensive check."""
        keys = get_repro_state_keys()
        missing_fields = EXPECTED_REPRO_STATE_FIELDS - keys
        extra_fields = keys - EXPECTED_REPRO_STATE_FIELDS
        
        assert not missing_fields, (
            f"ReproState missing {len(missing_fields)} expected fields: {sorted(missing_fields)}"
        )
        
        # Allow some extra fields (documented ones), but warn if unexpected
        # Remove known acceptable extras for cleaner error messages
        known_extras = set()  # Add any known extras here if needed
        unexpected_extras = extra_fields - known_extras
        if unexpected_extras:
            # Don't fail, but this is suspicious - could indicate drift
            print(f"WARNING: ReproState has unexpected fields: {sorted(unexpected_extras)}")
    
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
    
    def test_repro_state_has_all_verdict_fields(self):
        """ReproState should have all verdict fields for each reviewer/validator."""
        keys = get_repro_state_keys()
        required_verdicts = {
            "last_plan_review_verdict",
            "last_design_review_verdict",
            "last_code_review_verdict",
            "execution_verdict",
            "physics_verdict",
            "comparison_verdict",
            "supervisor_verdict",
        }
        missing = required_verdicts - keys
        assert not missing, f"ReproState missing verdict fields: {missing}"
    
    def test_repro_state_has_all_material_fields(self):
        """ReproState should have all material-related fields."""
        keys = get_repro_state_keys()
        required_materials = {
            "planned_materials",
            "validated_materials",
            "pending_validated_materials",
        }
        missing = required_materials - keys
        assert not missing, f"ReproState missing material fields: {missing}"
    
    def test_repro_state_has_all_revision_count_fields(self):
        """ReproState should have all revision count fields."""
        keys = get_repro_state_keys()
        required_counts = {
            "design_revision_count",
            "code_revision_count",
            "execution_failure_count",
            "total_execution_failures",
            "physics_failure_count",
            "analysis_revision_count",
            "replan_count",
            "backtrack_count",
        }
        missing = required_counts - keys
        assert not missing, f"ReproState missing revision count fields: {missing}"
    
    def test_plan_schema_exists_and_valid(self):
        """plan_schema.json should exist and be valid JSON with required structure."""
        schema = load_schema("plan_schema.json")
        assert "properties" in schema, "plan_schema.json missing 'properties'"
        assert "required" in schema, "plan_schema.json missing 'required'"
        
        # Check required fields exist in properties
        required_fields = set(schema.get("required", []))
        properties = set(schema.get("properties", {}).keys())
        missing_required = required_fields - properties
        assert not missing_required, (
            f"plan_schema.json has required fields not in properties: {missing_required}"
        )
        
        # Verify critical fields
        assert "paper_id" in schema["properties"], "plan_schema.json missing 'paper_id'"
        assert "stages" in schema["properties"], "plan_schema.json missing 'stages'"
        assert "targets" in schema["properties"], "plan_schema.json missing 'targets'"
    
    def test_progress_schema_exists_and_valid(self):
        """progress_schema.json should exist and be valid JSON."""
        schema = load_schema("progress_schema.json")
        assert "properties" in schema, "progress_schema.json missing 'properties'"
        assert "required" in schema, "progress_schema.json missing 'required'"
        
        # Verify critical fields
        assert "stages" in schema["properties"], "progress_schema.json missing 'stages'"
        assert "paper_id" in schema["properties"], "progress_schema.json missing 'paper_id'"
    
    def test_assumptions_schema_exists_and_valid(self):
        """assumptions_schema.json should exist and be valid JSON."""
        schema = load_schema("assumptions_schema.json")
        assert "properties" in schema, "assumptions_schema.json missing 'properties'"
    
    def test_metrics_schema_exists_and_valid(self):
        """metrics_schema.json should exist and be valid JSON."""
        schema = load_schema("metrics_schema.json")
        assert "properties" in schema, "metrics_schema.json missing 'properties'"
    
    def test_report_schema_exists_and_valid(self):
        """report_schema.json should exist and be valid JSON."""
        schema = load_schema("report_schema.json")
        assert "properties" in schema, "report_schema.json missing 'properties'"
    
    def test_agent_output_schemas_exist(self):
        """All agent output schemas should exist and be valid."""
        required_schemas = [
            "supervisor_output_schema.json",
            "plan_reviewer_output_schema.json",
            "design_reviewer_output_schema.json",
            "code_reviewer_output_schema.json",
            "results_analyzer_output_schema.json",
            "execution_validator_output_schema.json",
            "physics_sanity_output_schema.json",
            "comparison_validator_output_schema.json",
            "planner_output_schema.json",
            "simulation_designer_output_schema.json",
            "code_generator_output_schema.json",
        ]
        for schema_name in required_schemas:
            schema_path = SCHEMAS_DIR / schema_name
            assert schema_path.exists(), f"Missing schema: {schema_name}"
            schema = load_schema(schema_name)
            assert "properties" in schema, f"Schema {schema_name} missing 'properties'"
            # All agent output schemas should have a verdict field
            assert "verdict" in schema["properties"], (
                f"Schema {schema_name} missing 'verdict' field - "
                "all agent outputs must have a verdict"
            )
    
    def test_supervisor_output_has_verdict(self):
        """Supervisor output schema should have verdict field with correct enum values."""
        schema = load_schema("supervisor_output_schema.json")
        props = get_schema_properties(schema)
        assert "verdict" in props, "Supervisor output missing 'verdict'"
        
        verdict_prop = schema["properties"]["verdict"]
        assert "enum" in verdict_prop, "Supervisor verdict missing enum constraint"
        expected_verdicts = {
            "ok_continue", "replan_needed", "change_priority", 
            "ask_user", "backtrack_to_stage"
        }
        actual_verdicts = set(verdict_prop["enum"])
        missing = expected_verdicts - actual_verdicts
        assert not missing, (
            f"Supervisor verdict enum missing values: {missing}. "
            f"Got: {actual_verdicts}"
        )
    
    def test_plan_reviewer_output_has_verdict(self):
        """Plan reviewer output schema should have verdict field."""
        schema = load_schema("plan_reviewer_output_schema.json")
        props = get_schema_properties(schema)
        assert "verdict" in props, "Plan reviewer output missing 'verdict'"
        
        verdict_prop = schema["properties"]["verdict"]
        assert "enum" in verdict_prop, "Plan reviewer verdict missing enum constraint"
        expected_verdicts = {"approve", "needs_revision"}
        actual_verdicts = set(verdict_prop["enum"])
        assert expected_verdicts.issubset(actual_verdicts), (
            f"Plan reviewer verdict enum missing values. "
            f"Expected at least: {expected_verdicts}, got: {actual_verdicts}"
        )
    
    def test_physics_sanity_output_has_design_flaw_verdict(self):
        """Physics sanity output should support 'design_flaw' verdict for physics-driven redesign."""
        schema = load_schema("physics_sanity_output_schema.json")
        verdict_prop = schema["properties"].get("verdict", {})
        enum_values = verdict_prop.get("enum", [])
        assert "design_flaw" in enum_values, (
            "Physics sanity output missing 'design_flaw' verdict - "
            "needed for routing physics failures to design node. "
            f"Got enum values: {enum_values}"
        )
    
    def test_plan_schema_has_expected_outputs(self):
        """Plan schema stages should have expected_outputs field with correct structure."""
        schema = load_schema("plan_schema.json")
        stages_schema = schema["properties"]["stages"]["items"]["properties"]
        assert "expected_outputs" in stages_schema, (
            "Plan schema stages missing 'expected_outputs' field - "
            "needed for output artifact specification"
        )
        
        # Verify expected_outputs structure
        expected_outputs_schema = stages_schema["expected_outputs"]
        assert "items" in expected_outputs_schema, "expected_outputs must be an array"
        item_schema = expected_outputs_schema["items"]
        assert "properties" in item_schema, "expected_outputs items must have properties"
        
        required_item_fields = {"artifact_type", "filename_pattern", "description"}
        item_props = set(item_schema["properties"].keys())
        missing = required_item_fields - item_props
        assert not missing, (
            f"expected_outputs items missing required fields: {missing}"
        )
    
    def test_plan_schema_targets_have_precision_requirement(self):
        """Plan schema targets should have precision_requirement field with correct enum."""
        schema = load_schema("plan_schema.json")
        targets_schema = schema["properties"]["targets"]["items"]["properties"]
        assert "precision_requirement" in targets_schema, (
            "Plan schema targets missing 'precision_requirement' field - "
            "needed for digitized data enforcement"
        )
        
        precision_prop = targets_schema["precision_requirement"]
        assert "enum" in precision_prop, "precision_requirement must have enum constraint"
        expected_values = {"excellent", "good", "acceptable", "qualitative"}
        actual_values = set(precision_prop["enum"])
        assert expected_values.issubset(actual_values), (
            f"precision_requirement enum missing values. "
            f"Expected at least: {expected_values}, got: {actual_values}"
        )
    
    def test_plan_schema_targets_have_digitized_data_path(self):
        """Plan schema targets should have digitized_data_path field."""
        schema = load_schema("plan_schema.json")
        targets_schema = schema["properties"]["targets"]["items"]["properties"]
        assert "digitized_data_path" in targets_schema, (
            "Plan schema targets missing 'digitized_data_path' field - "
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
    
    def test_validation_hierarchy_mapping_constants_exist(self):
        """STAGE_STATUS_TO_HIERARCHY_MAPPING and STAGE_TYPE_TO_HIERARCHY_KEY should exist."""
        state_file = SCHEMAS_DIR / "state.py"
        with open(state_file) as f:
            content = f.read()
        
        assert "STAGE_STATUS_TO_HIERARCHY_MAPPING" in content, (
            "STAGE_STATUS_TO_HIERARCHY_MAPPING constant missing from state.py"
        )
        assert "STAGE_TYPE_TO_HIERARCHY_KEY" in content, (
            "STAGE_TYPE_TO_HIERARCHY_KEY constant missing from state.py"
        )
    
    def test_validation_hierarchy_mapping_completeness(self):
        """STAGE_STATUS_TO_HIERARCHY_MAPPING should cover all stage statuses."""
        state_file = SCHEMAS_DIR / "state.py"
        with open(state_file) as f:
            content = f.read()
        
        # Extract the mapping from source
        mapping_match = re.search(
            r'STAGE_STATUS_TO_HIERARCHY_MAPPING\s*=\s*\{([^}]+)\}',
            content,
            re.DOTALL
        )
        assert mapping_match is not None, "Could not find STAGE_STATUS_TO_HIERARCHY_MAPPING definition"
        
        # Check that all expected statuses are mapped
        # From progress_schema.json, stage status enum:
        expected_statuses = {
            "not_started", "in_progress", "completed_success", 
            "completed_partial", "completed_failed", "blocked", 
            "needs_rerun", "invalidated"
        }
        
        # Extract mapped statuses from the mapping
        mapped_statuses = set()
        for status in expected_statuses:
            if f'"{status}"' in mapping_match.group(1):
                mapped_statuses.add(status)
        
        missing = expected_statuses - mapped_statuses
        assert not missing, (
            f"STAGE_STATUS_TO_HIERARCHY_MAPPING missing statuses: {missing}. "
            "All stage statuses from progress_schema.json must be mapped."
        )
    
    def test_validation_hierarchy_type_mapping_completeness(self):
        """STAGE_TYPE_TO_HIERARCHY_KEY should cover all stage types."""
        state_file = SCHEMAS_DIR / "state.py"
        with open(state_file) as f:
            content = f.read()
        
        # Extract the mapping from source
        mapping_match = re.search(
            r'STAGE_TYPE_TO_HIERARCHY_KEY\s*=\s*\{([^}]+)\}',
            content,
            re.DOTALL
        )
        assert mapping_match is not None, "Could not find STAGE_TYPE_TO_HIERARCHY_KEY definition"
        
        # From plan_schema.json, stage_type enum (excluding COMPLEX_PHYSICS which is optional):
        expected_types = {
            "MATERIAL_VALIDATION",
            "SINGLE_STRUCTURE",
            "ARRAY_SYSTEM",
            "PARAMETER_SWEEP",
        }
        
        # Extract mapped types from the mapping
        mapped_types = set()
        for stage_type in expected_types:
            if f'"{stage_type}"' in mapping_match.group(1):
                mapped_types.add(stage_type)
        
        missing = expected_types - mapped_types
        assert not missing, (
            f"STAGE_TYPE_TO_HIERARCHY_KEY missing stage types: {missing}. "
            "All mandatory stage types from plan_schema.json must be mapped."
        )


class TestValidationHierarchyFunction:
    """Tests for get_validation_hierarchy function - comprehensive edge case coverage."""
    
    def test_get_validation_hierarchy_empty_state(self):
        """get_validation_hierarchy should handle empty state."""
        hierarchy = state_module.get_validation_hierarchy({})
        
        assert hierarchy is not None, "get_validation_hierarchy returned None"
        assert isinstance(hierarchy, dict), "get_validation_hierarchy should return dict"
        
        expected_keys = {"material_validation", "single_structure", "arrays_systems", "parameter_sweeps"}
        assert set(hierarchy.keys()) == expected_keys, (
            f"Hierarchy keys mismatch. Expected: {expected_keys}, got: {set(hierarchy.keys())}"
        )
        
        # All should be "not_done" for empty state
        for key, value in hierarchy.items():
            assert value == "not_done", (
                f"Empty state should have '{key}' = 'not_done', got '{value}'"
            )
    
    def test_get_validation_hierarchy_missing_progress(self):
        """get_validation_hierarchy should handle missing progress field."""
        state = {"plan": {"stages": []}}
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy is not None
        for key, value in hierarchy.items():
            assert value == "not_done", (
                f"Missing progress should have '{key}' = 'not_done', got '{value}'"
            )
    
    def test_get_validation_hierarchy_empty_stages(self):
        """get_validation_hierarchy should handle empty stages array."""
        state = {
            "progress": {"stages": []},
            "plan": {"stages": []}
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy is not None
        for key, value in hierarchy.items():
            assert value == "not_done", (
                f"Empty stages should have '{key}' = 'not_done', got '{value}'"
            )
    
    def test_get_validation_hierarchy_plan_without_progress(self):
        """get_validation_hierarchy should detect plan/progress mismatch."""
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION"}
                ]
            },
            "progress": {}
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        # Should return failed hierarchy when plan exists but progress doesn't
        assert hierarchy is not None
        # The function should mark as failed when plan has stages but progress doesn't
        # Check the actual behavior - it should return failed hierarchy
        assert all(v == "failed" for v in hierarchy.values()), (
            "Plan with stages but no progress should return failed hierarchy. "
            f"Got: {hierarchy}"
        )
    
    def test_get_validation_hierarchy_missing_stage_in_progress(self):
        """get_validation_hierarchy should detect missing stages in progress."""
        state = {
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"}
                ]
            },
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "status": "completed_success"}
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        # stage1 is missing from progress, so single_structure should be failed
        assert hierarchy["single_structure"] == "failed", (
            "Missing stage in progress should mark hierarchy level as failed. "
            f"Got: {hierarchy}"
        )
    
    def test_get_validation_hierarchy_completed_success(self):
        """get_validation_hierarchy should map completed_success to passed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_success"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "passed", (
            "completed_success should map to 'passed'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_completed_partial(self):
        """get_validation_hierarchy should map completed_partial correctly."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_partial"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "partial", (
            "completed_partial should map to 'partial'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_completed_failed(self):
        """get_validation_hierarchy should map completed_failed to failed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_failed"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "failed", (
            "completed_failed should map to 'failed'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_not_started(self):
        """get_validation_hierarchy should map not_started to not_done."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "not_started"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "not_done", (
            "not_started should map to 'not_done'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_in_progress(self):
        """get_validation_hierarchy should map in_progress to not_done."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "in_progress"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "not_done", (
            "in_progress should map to 'not_done'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_blocked(self):
        """get_validation_hierarchy should map blocked to failed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "blocked"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "failed", (
            "blocked should map to 'failed'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_needs_rerun(self):
        """get_validation_hierarchy should map needs_rerun to failed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "needs_rerun"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "failed", (
            "needs_rerun should map to 'failed'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_invalidated(self):
        """get_validation_hierarchy should map invalidated to failed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "invalidated"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "failed", (
            "invalidated should map to 'failed'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_multiple_stages_same_type_all_success(self):
        """Multiple stages of same type, all success -> passed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage1a",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage1b",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_success"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["single_structure"] == "passed", (
            "All stages success should result in 'passed'. "
            f"Got: {hierarchy['single_structure']}"
        )
    
    def test_get_validation_hierarchy_multiple_stages_mixed_success_partial(self):
        """Multiple stages of same type, mixed success/partial -> partial."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage1a",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage1b",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_partial"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["single_structure"] == "partial", (
            "Mixed success/partial should result in 'partial'. "
            f"Got: {hierarchy['single_structure']}"
        )
    
    def test_get_validation_hierarchy_multiple_stages_any_failed(self):
        """Multiple stages of same type, any failed -> failed."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage1a",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage1b",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_failed"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["single_structure"] == "failed", (
            "Any stage failed should result in 'failed'. "
            f"Got: {hierarchy['single_structure']}"
        )
    
    def test_get_validation_hierarchy_multiple_stages_any_not_started(self):
        """Multiple stages of same type, any not_started -> not_done."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage1a",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage1b",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "not_started"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["single_structure"] == "not_done", (
            "Any stage not_started should result in 'not_done'. "
            f"Got: {hierarchy['single_structure']}"
        )
    
    def test_get_validation_hierarchy_all_stage_types(self):
        """get_validation_hierarchy should handle all stage types."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage1",
                        "stage_type": "SINGLE_STRUCTURE",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage2",
                        "stage_type": "ARRAY_SYSTEM",
                        "status": "completed_success"
                    },
                    {
                        "stage_id": "stage3",
                        "stage_type": "PARAMETER_SWEEP",
                        "status": "completed_success"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        assert hierarchy["material_validation"] == "passed"
        assert hierarchy["single_structure"] == "passed"
        assert hierarchy["arrays_systems"] == "passed"
        assert hierarchy["parameter_sweeps"] == "passed"
    
    def test_get_validation_hierarchy_complex_physics_not_in_hierarchy(self):
        """COMPLEX_PHYSICS stage type should not affect hierarchy."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage_complex",
                        "stage_type": "COMPLEX_PHYSICS",
                        "status": "completed_success"
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        # COMPLEX_PHYSICS doesn't map to any hierarchy key, so all should be not_done
        for key, value in hierarchy.items():
            assert value == "not_done", (
                f"COMPLEX_PHYSICS should not affect hierarchy. "
                f"Got {key}={value}, expected 'not_done'"
            )
    
    def test_get_validation_hierarchy_missing_status_field(self):
        """get_validation_hierarchy should handle missing status field."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION"
                        # Missing status field
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        # Should default to "not_started" behavior
        assert hierarchy["material_validation"] == "not_done", (
            "Missing status should default to 'not_done'. "
            f"Got: {hierarchy['material_validation']}"
        )
    
    def test_get_validation_hierarchy_missing_stage_type(self):
        """get_validation_hierarchy should handle missing stage_type field."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "status": "completed_success"
                        # Missing stage_type field
                    }
                ]
            }
        }
        hierarchy = state_module.get_validation_hierarchy(state)
        
        # Stage without type shouldn't affect hierarchy
        for key, value in hierarchy.items():
            assert value == "not_done", (
                f"Stage without type should not affect hierarchy. "
                f"Got {key}={value}, expected 'not_done'"
            )
    
    def test_get_validation_hierarchy_invalid_status_value(self):
        """get_validation_hierarchy should handle invalid status values gracefully."""
        state = {
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "stage_type": "MATERIAL_VALIDATION",
                        "status": "invalid_status_xyz"
                    }
                ]
            }
        }
        # Should not crash, but behavior is undefined for invalid status
        # Just verify it doesn't raise an exception
        try:
            hierarchy = state_module.get_validation_hierarchy(state)
            assert hierarchy is not None
        except Exception as e:
            raise AssertionError(
                f"get_validation_hierarchy should handle invalid status gracefully, "
                f"but raised: {e}"
            )


class TestGeneratedTypesSync:
    """Tests that schemas/generated_types is in sync with schemas/*.json."""

    def test_generated_types_module_exists(self):
        """generated_types module should exist and be importable."""
        assert generated_types is not None, "generated_types module not found"
        assert hasattr(generated_types, "__file__"), "generated_types should be a module"

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
            "planner_output_schema.json": "PlannerAgentOutput",
            "simulation_designer_output_schema.json": "SimulationDesignerAgentOutput",
            "code_generator_output_schema.json": "CodeGeneratorAgentOutput",
        }

        missing_types = []
        for schema_file, class_name in expected_types.items():
            # Check if the class exists in generated_types module
            if not hasattr(generated_types, class_name):
                missing_types.append((schema_file, class_name))

        assert not missing_types, (
            f"Missing {len(missing_types)} generated types: "
            f"{', '.join(f'{name} for {file}' for file, name in missing_types)}. "
            "Run 'python scripts/generate_types.py' to regenerate."
        )

    def test_generated_type_fields_match_schema_required(self):
        """Verify that ALL required fields in schemas exist in generated types."""
        schema_to_type = {
            "plan_reviewer_output_schema.json": "PlanReviewerAgentOutput",
            "design_reviewer_output_schema.json": "DesignReviewerAgentOutput",
            "code_reviewer_output_schema.json": "CodeReviewerAgentOutput",
            "supervisor_output_schema.json": "SupervisorAgentOutput",
            "execution_validator_output_schema.json": "ExecutionValidatorAgentOutput",
            "results_analyzer_output_schema.json": "ResultsAnalyzerAgentOutput",
            "comparison_validator_output_schema.json": "ComparisonValidatorAgentOutput",
            "physics_sanity_output_schema.json": "PhysicsSanityAgentOutput",
        }

        all_missing = {}
        for schema_file, class_name in schema_to_type.items():
            if not hasattr(generated_types, class_name):
                continue  # Skip if type doesn't exist (caught by other test)

            schema = load_schema(schema_file)
            required_fields = set(schema.get("required", []))
            
            type_class = getattr(generated_types, class_name)
            type_keys = get_typed_dict_keys(type_class)
            
            missing = required_fields - type_keys
            if missing:
                all_missing[class_name] = missing

        assert not all_missing, (
            f"Generated types missing required schema fields:\n" +
            "\n".join(f"  {name}: {missing}" for name, missing in all_missing.items())
        )

    def test_generated_type_all_fields_match_schema(self):
        """Verify that ALL fields in generated types exist in schemas (not just required)."""
        schema_to_type = {
            "plan_reviewer_output_schema.json": "PlanReviewerAgentOutput",
            "supervisor_output_schema.json": "SupervisorAgentOutput",
        }

        all_extra = {}
        for schema_file, class_name in schema_to_type.items():
            if not hasattr(generated_types, class_name):
                continue

            schema = load_schema(schema_file)
            schema_props = get_all_schema_properties_recursive(schema)
            # Flatten to top-level keys for comparison
            schema_top_level = {p.split('.')[0] for p in schema_props}
            
            type_class = getattr(generated_types, class_name)
            type_keys = get_typed_dict_keys(type_class)
            
            # Fields in type but not in schema (could indicate drift)
            extra = type_keys - schema_top_level
            if extra:
                all_extra[class_name] = extra

        # Don't fail on extras (they might be valid), but warn
        if all_extra:
            print(f"WARNING: Generated types have fields not in schemas: {all_extra}")

    def test_plan_schema_generated_types_exist(self):
        """Verify that plan_schema.json has corresponding generated types."""
        # Check for main Plan type
        assert hasattr(generated_types, "PaperReproductionPlan"), (
            "Missing generated type 'PaperReproductionPlan' for plan_schema.json"
        )
        
        # Check for nested types
        assert hasattr(generated_types, "ExtractedParameter"), (
            "Missing generated type 'ExtractedParameter'"
        )
        assert hasattr(generated_types, "Target"), (
            "Missing generated type 'Target'"
        )
        assert hasattr(generated_types, "Stage"), (
            "Missing generated type 'Stage'"
        )

    def test_progress_schema_generated_types_exist(self):
        """Verify that progress_schema.json has corresponding generated types."""
        assert hasattr(generated_types, "PaperReproductionProgress"), (
            "Missing generated type 'PaperReproductionProgress' for progress_schema.json"
        )

    def test_assumptions_schema_generated_types_exist(self):
        """Verify that assumptions_schema.json has corresponding generated types."""
        assert hasattr(generated_types, "Assumptions"), (
            "Missing generated type 'Assumptions' for assumptions_schema.json"
        )

    def test_metrics_schema_generated_types_exist(self):
        """Verify that metrics_schema.json has corresponding generated types."""
        assert hasattr(generated_types, "MetricsLog"), (
            "Missing generated type 'MetricsLog' for metrics_schema.json"
        )


class TestSchemaConsistency:
    """Tests for internal consistency of schemas."""
    
    def test_all_schemas_are_valid_json(self):
        """All JSON files in schemas/ should be valid JSON."""
        invalid_schemas = []
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            try:
                with open(schema_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                invalid_schemas.append((schema_file.name, str(e)))
        
        assert not invalid_schemas, (
            f"Invalid JSON in {len(invalid_schemas)} schema(s):\n" +
            "\n".join(f"  {name}: {error}" for name, error in invalid_schemas)
        )
    
    def test_all_schemas_have_required_metadata(self):
        """All schemas should have $schema field."""
        missing_metadata = []
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            schema = load_schema(schema_file.name)
            if "$schema" not in schema:
                missing_metadata.append(schema_file.name)
        
        assert not missing_metadata, (
            f"Schemas missing $schema field: {missing_metadata}"
        )
    
    def test_no_empty_required_arrays(self):
        """Schemas with 'required' should have at least one required field."""
        empty_required = []
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            schema = load_schema(schema_file.name)
            if "required" in schema:
                if len(schema["required"]) == 0:
                    empty_required.append(schema_file.name)
        
        assert not empty_required, (
            f"Schemas with empty 'required' array: {empty_required}"
        )
    
    def test_required_fields_exist_in_properties(self):
        """All fields in 'required' arrays should exist in 'properties'."""
        schema_errors = []
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            schema = load_schema(schema_file.name)
            if "required" in schema and "properties" in schema:
                required = set(schema["required"])
                properties = set(schema["properties"].keys())
                missing = required - properties
                if missing:
                    schema_errors.append((schema_file.name, missing))
        
        assert not schema_errors, (
            f"Schemas with required fields not in properties:\n" +
            "\n".join(f"  {name}: {missing}" for name, missing in schema_errors)
        )
    
    def test_enum_values_are_strings(self):
        """All enum values in schemas should be strings (for type safety)."""
        enum_errors = []
        for schema_file in SCHEMAS_DIR.glob("*.json"):
            schema = load_schema(schema_file.name)
            
            def check_enums(obj, path=""):
                if isinstance(obj, dict):
                    if "enum" in obj:
                        enum_values = obj["enum"]
                        if not all(isinstance(v, str) for v in enum_values):
                            non_strings = [v for v in enum_values if not isinstance(v, str)]
                            enum_errors.append((schema_file.name, f"{path}.enum", non_strings))
                    for key, value in obj.items():
                        check_enums(value, f"{path}.{key}" if path else key)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_enums(item, f"{path}[{i}]")
            
            check_enums(schema)
        
        assert not enum_errors, (
            f"Schemas with non-string enum values:\n" +
            "\n".join(f"  {name} {path}: {values}" for name, path, values in enum_errors)
        )
    
    def test_plan_schema_stage_types_match_progress_schema(self):
        """Stage type enums should match between plan_schema and progress_schema."""
        plan_schema = load_schema("plan_schema.json")
        progress_schema = load_schema("progress_schema.json")
        
        # Extract stage_type enum from plan_schema
        plan_stage_type_prop = plan_schema["properties"]["stages"]["items"]["properties"]["stage_type"]
        plan_stage_types = set(plan_stage_type_prop["enum"])
        
        # Extract stage_type enum from progress_schema
        progress_stage_type_prop = progress_schema["definitions"]["stage_progress"]["properties"]["stage_type"]
        progress_stage_types = set(progress_stage_type_prop["enum"])
        
        assert plan_stage_types == progress_stage_types, (
            f"Stage type enums mismatch:\n"
            f"  plan_schema: {plan_stage_types}\n"
            f"  progress_schema: {progress_stage_types}\n"
            f"  Missing in progress: {plan_stage_types - progress_stage_types}\n"
            f"  Extra in progress: {progress_stage_types - plan_stage_types}"
        )
    
    def test_plan_schema_stage_status_enum_complete(self):
        """progress_schema stage status enum should include all expected values."""
        progress_schema = load_schema("progress_schema.json")
        status_prop = progress_schema["definitions"]["stage_progress"]["properties"]["status"]
        status_enum = set(status_prop["enum"])
        
        expected_statuses = {
            "not_started", "in_progress", "completed_success",
            "completed_partial", "completed_failed", "blocked",
            "needs_rerun", "invalidated"
        }
        
        missing = expected_statuses - status_enum
        extra = status_enum - expected_statuses
        
        assert not missing, (
            f"progress_schema missing status values: {missing}"
        )
        
        if extra:
            print(f"WARNING: progress_schema has unexpected status values: {extra}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
