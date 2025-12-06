"""
Test schema/code consistency using whitelist-based validation.

This test ensures:
1. All access to schema-backed variables uses allowed patterns (whitelist)
2. All accessed fields exist in the corresponding JSON schema

If this test fails, it means either:
- Code is using a disallowed access pattern (fix the code)
- Code is accessing a field that doesn't exist in the schema (fix code or schema)

WHITELIST - Only these patterns are allowed:
- tracked_var.get("static_string")
- tracked_var.get("static_string", default)
- "static_string" in tracked_var

DISALLOWED (anything not in whitelist):
- tracked_var["key"] (subscript)
- tracked_var.get(variable) (dynamic key)
- for x in tracked_var (iteration)
- tracked_var.keys(), .items(), .values()
- getattr(tracked_var, ...)
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_schema_access import (
    validate_agent_files,
    validate_prompts_module,
    validate_file,
    extract_field_accesses,
    extract_schema_fields,
    load_schema,
    get_schema_fields_for_pointer,
    ViolationType,
    ValidationResult,
    AGENTS_DIR,
    SCHEMAS_DIR,
)


class TestWhitelistValidation:
    """Tests that code uses only whitelisted access patterns."""
    
    def test_agent_files_use_whitelisted_patterns(self):
        """All agent files should use only whitelisted access patterns."""
        result = validate_agent_files()
        
        # Filter to only whitelist violations (not schema mismatches)
        whitelist_violations = [
            v for v in result.violations 
            if v.type in (ViolationType.PATTERN_NOT_WHITELISTED, ViolationType.DYNAMIC_KEY)
        ]
        
        if whitelist_violations:
            msg = f"Found {len(whitelist_violations)} whitelist violation(s):\n"
            for v in whitelist_violations:
                msg += f"\n  {v.file}:{v.line} - {v.message}"
                if v.code_snippet:
                    msg += f"\n    Code: {v.code_snippet}"
            pytest.fail(msg)
    
    def test_prompts_module_uses_whitelisted_patterns(self):
        """Prompts module should use only whitelisted patterns for adaptation access."""
        result = validate_prompts_module()
        
        whitelist_violations = [
            v for v in result.violations 
            if v.type in (ViolationType.PATTERN_NOT_WHITELISTED, ViolationType.DYNAMIC_KEY)
        ]
        
        if whitelist_violations:
            msg = f"Found {len(whitelist_violations)} whitelist violation(s) in prompts.py:\n"
            for v in whitelist_violations:
                msg += f"\n  {v.file}:{v.line} - {v.message}"
            pytest.fail(msg)


class TestSchemaFieldConsistency:
    """Tests that accessed fields exist in corresponding schemas."""
    
    def test_agent_output_fields_exist_in_schema(self):
        """All agent_output field accesses should reference valid schema fields."""
        result = validate_agent_files()
        
        # Filter to only schema field violations
        schema_violations = [
            v for v in result.violations 
            if v.type == ViolationType.FIELD_NOT_IN_SCHEMA
        ]
        
        if schema_violations:
            msg = f"Found {len(schema_violations)} schema mismatch(es):\n"
            for v in schema_violations:
                msg += f"\n  {v.file}:{v.line} - {v.message}"
                if v.code_snippet:
                    msg += f"\n    Code: {v.code_snippet}"
            pytest.fail(msg)
    
    def test_adaptation_fields_exist_in_schema(self):
        """Fields accessed on adaptation objects should exist in schema."""
        result = validate_prompts_module()
        
        schema_violations = [
            v for v in result.violations 
            if v.type == ViolationType.FIELD_NOT_IN_SCHEMA
        ]
        
        if schema_violations:
            msg = f"Found {len(schema_violations)} schema mismatch(es) for adaptation:\n"
            for v in schema_violations:
                msg += f"\n  {v.file}:{v.line} - {v.message}"
            pytest.fail(msg)


class TestValidatorFunctionality:
    """Tests for the validator itself to ensure it works correctly."""
    
    def test_detects_static_get(self):
        """Should detect and allow static .get() calls."""
        code = '''
def test_func():
    value = agent_output.get("verdict")
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.field_accesses) == 1
        access = result.field_accesses[0]
        assert access.field == "verdict"
        assert access.is_static is True
        assert access.access_type == "get"
        assert len(result.violations) == 0
    
    def test_detects_static_get_with_default(self):
        """Should detect and allow static .get() with default."""
        code = '''
def test_func():
    value = agent_output.get("verdict", "default")
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.field_accesses) == 1
        access = result.field_accesses[0]
        assert access.field == "verdict"
        assert access.is_static is True
        assert access.access_type == "get_with_default"
        assert len(result.violations) == 0
    
    def test_detects_static_in_check(self):
        """Should detect and allow static 'in' checks."""
        code = '''
def test_func():
    if "verdict" in agent_output:
        pass
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.field_accesses) == 1
        access = result.field_accesses[0]
        assert access.field == "verdict"
        assert access.is_static is True
        assert access.access_type == "in_check"
        assert len(result.violations) == 0
    
    def test_flags_dynamic_get(self):
        """Should flag dynamic .get() calls."""
        code = '''
def test_func():
    key = "verdict"
    value = agent_output.get(key)
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.DYNAMIC_KEY
    
    def test_flags_subscript_access(self):
        """Should flag subscript access."""
        code = '''
def test_func():
    value = agent_output["verdict"]
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.PATTERN_NOT_WHITELISTED
        assert "Subscript" in result.violations[0].message
    
    def test_flags_iteration(self):
        """Should flag iteration over tracked variables."""
        code = '''
def test_func():
    for key in agent_output:
        pass
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.PATTERN_NOT_WHITELISTED
        assert "iteration" in result.violations[0].message.lower()
    
    def test_flags_keys_method(self):
        """Should flag .keys() method."""
        code = '''
def test_func():
    keys = agent_output.keys()
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.PATTERN_NOT_WHITELISTED
        assert ".keys()" in result.violations[0].message
    
    def test_flags_items_method(self):
        """Should flag .items() method."""
        code = '''
def test_func():
    for k, v in agent_output.items():
        pass
'''
        result = extract_field_accesses(code, "test.py")
        
        # Should have iteration violation AND .items() violation
        assert len(result.violations) >= 1
        assert any(".items()" in v.message for v in result.violations)
    
    def test_flags_getattr(self):
        """Should flag getattr() calls."""
        code = '''
def test_func():
    value = getattr(agent_output, "verdict")
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.PATTERN_NOT_WHITELISTED
        assert "getattr" in result.violations[0].message
    
    def test_ignores_non_tracked_variables(self):
        """Should ignore access patterns on non-tracked variables."""
        code = '''
def test_func():
    other_dict = {"a": 1}
    value = other_dict["key"]
    keys = other_dict.keys()
    for k in other_dict:
        pass
'''
        result = extract_field_accesses(code, "test.py")
        
        # No violations because other_dict is not tracked
        assert len(result.violations) == 0
        assert len(result.field_accesses) == 0
    
    def test_tracks_function_context(self):
        """Should track which function the access is in."""
        code = '''
def my_function():
    value = agent_output.get("verdict")

def other_function():
    value = agent_output.get("summary")
'''
        result = extract_field_accesses(code, "test.py")
        
        assert len(result.field_accesses) == 2
        func_names = {a.in_function for a in result.field_accesses}
        assert func_names == {"my_function", "other_function"}


class TestSchemaLoading:
    """Tests for schema loading functionality."""
    
    def test_load_valid_schema(self):
        """Should load valid JSON schemas."""
        schema = load_schema("planner_output_schema.json")
        assert "properties" in schema
        assert "required" in schema
    
    def test_extract_schema_fields(self):
        """Should extract field names from schema."""
        schema = load_schema("planner_output_schema.json")
        fields = extract_schema_fields(schema)
        
        assert "paper_id" in fields
        assert "stages" in fields
        assert "targets" in fields
    
    def test_extract_nested_fields(self):
        """Should extract nested field names."""
        schema = {
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"}
                    }
                }
            }
        }
        fields = extract_schema_fields(schema)
        
        assert "outer" in fields
        assert "inner" in fields
        assert "outer.inner" in fields
    
    def test_get_schema_fields_with_pointer(self):
        """Should follow JSON pointer to get fields."""
        fields = get_schema_fields_for_pointer(
            "prompt_adaptor_output_schema.json",
            "/properties/prompt_modifications/items"
        )
        
        # These are the fields that should be in prompt_modifications items
        assert "target_agent" in fields
        assert "modification_type" in fields
        assert "new_content" in fields
        assert "section" in fields


class TestIntegration:
    """Integration tests that run on actual codebase."""
    
    def test_all_schemas_loadable(self):
        """All referenced schemas should be loadable."""
        from tools.validate_schema_access import AGENT_OUTPUT_SCHEMA_MAPPING
        
        for file_schemas in AGENT_OUTPUT_SCHEMA_MAPPING.values():
            for schema_file in file_schemas.values():
                schema_path = SCHEMAS_DIR / schema_file
                assert schema_path.exists(), f"Schema not found: {schema_file}"
                schema = load_schema(schema_file)
                assert "properties" in schema, f"Schema {schema_file} missing properties"
    
    def test_agent_files_exist(self):
        """All mapped agent files should exist."""
        from tools.validate_schema_access import AGENT_OUTPUT_SCHEMA_MAPPING
        
        for file_name in AGENT_OUTPUT_SCHEMA_MAPPING.keys():
            file_path = AGENTS_DIR / file_name
            assert file_path.exists(), f"Agent file not found: {file_name}"
    
    def test_full_validation_runs_without_error(self):
        """Full validation should run without crashing."""
        result = validate_agent_files()
        
        assert result.files_scanned > 0, "Should have scanned at least one file"
        # Note: We don't assert no violations here - that's tested separately


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

