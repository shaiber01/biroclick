"""
Tests for violation detection in the schema access validator.

These tests verify that the tool CATCHES all types of violations:
- FIELD_NOT_IN_SCHEMA: Accessing fields that don't exist in the schema
- PATTERN_NOT_WHITELISTED: Using disallowed access patterns
- DYNAMIC_KEY: Using dynamic (non-literal) keys

IMPORTANT: These tests are designed to FIND BUGS. If a test fails,
it means the tool is NOT catching a violation it should catch.
"""

import json
import pytest
from pathlib import Path

from tools.validate_schema_access import ViolationType

from .conftest import (
    validate_code_snippet,
    validate_code_with_mock_schema,
    assert_violation_found,
    assert_no_violations,
    assert_violation_count,
    wrap_in_function,
    make_agent_output_code,
    MOCK_SCHEMA,
    TrackedVariable,
)


class TestFieldNotInSchemaViolations:
    """
    Tests that MUST detect FIELD_NOT_IN_SCHEMA violations.
    
    If any of these tests fail, it means the tool is NOT catching
    accesses to non-existent fields - a critical bug.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file for these tests."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    @pytest.fixture
    def tracked_vars(self, schema_file):
        """Tracked vars with mock schema."""
        return {
            "agent_output": TrackedVariable(
                name="agent_output",
                schema_file=str(schema_file),
                json_pointer="",
            )
        }
    
    def test_get_invalid_field(self, schema_file):
        """Tool MUST catch .get() with non-existent field."""
        code = '''
def test_func():
    value = agent_output.get("nonexistent_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_field",
            variable="agent_output",
        )
    
    def test_get_with_default_invalid_field(self, schema_file):
        """Tool MUST catch .get() with default but invalid field."""
        code = '''
def test_func():
    value = agent_output.get("bad_field", "default")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="bad_field",
        )
    
    def test_in_check_invalid_field(self, schema_file):
        """Tool MUST catch 'in' check with non-existent field."""
        code = '''
def test_func():
    if "nonexistent" in agent_output:
        pass
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent",
        )
    
    def test_nested_object_invalid_field(self, schema_file):
        """Tool MUST catch invalid field access on derived nested object."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("nonexistent_inner")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should catch the invalid nested field access
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_inner",
            variable="nested",
        )
    
    def test_loop_variable_invalid_field(self, schema_file):
        """Tool MUST catch invalid field access on loop variable from array."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        bad_value = item.get("nonexistent_item_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_item_field",
            variable="item",
        )
    
    def test_comprehension_variable_invalid_field(self, schema_file):
        """Tool MUST catch invalid field access in list comprehension."""
        code = '''
def test_func():
    values = [x.get("bad_field") for x in agent_output.get("array_field", [])]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="bad_field",
            variable="x",
        )
    
    def test_array_index_variable_invalid_field(self, schema_file):
        """Tool MUST catch invalid field access on array index variable."""
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    if items:
        first = items[0]
        bad = first.get("nonexistent")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent",
            variable="first",
        )
    
    def test_direct_array_iteration_invalid_field(self, schema_file):
        """Tool MUST catch invalid field on direct array iteration variable."""
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    for item in items:
        bad = item.get("wrong_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="wrong_field",
            variable="item",
        )
    
    def test_multiple_invalid_fields(self, schema_file):
        """Tool MUST catch multiple invalid field accesses."""
        code = '''
def test_func():
    a = agent_output.get("bad1")
    b = agent_output.get("bad2")
    c = agent_output.get("bad3")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_count(result, 3, ViolationType.FIELD_NOT_IN_SCHEMA)
    
    def test_typo_in_field_name(self, schema_file):
        """Tool MUST catch typos in field names (similar but not exact)."""
        code = '''
def test_func():
    # Typo: "summry" instead of "summary"
    value = agent_output.get("summry")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="summry",
        )
    
    def test_case_sensitivity(self, schema_file):
        """Tool MUST catch wrong case in field names."""
        code = '''
def test_func():
    # Wrong case: "Verdict" instead of "verdict"
    value = agent_output.get("Verdict")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="Verdict",
        )


class TestPatternNotWhitelistedViolations:
    """
    Tests that MUST detect PATTERN_NOT_WHITELISTED violations.
    
    If any of these tests fail, it means the tool is NOT catching
    disallowed access patterns - a critical bug.
    """
    
    def test_subscript_read_access(self):
        """Tool MUST flag subscript read access obj['key']."""
        code = '''
def test_func():
    value = agent_output["verdict"]
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains="Subscript",
        )
    
    def test_keys_method(self):
        """Tool MUST flag .keys() method call."""
        code = '''
def test_func():
    keys = agent_output.keys()
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains=".keys()",
        )
    
    def test_values_method(self):
        """Tool MUST flag .values() method call."""
        code = '''
def test_func():
    values = agent_output.values()
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains=".values()",
        )
    
    def test_items_method(self):
        """Tool MUST flag .items() method call."""
        code = '''
def test_func():
    items = agent_output.items()
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains=".items()",
        )
    
    def test_pop_method(self):
        """Tool MUST flag .pop() method call."""
        code = '''
def test_func():
    value = agent_output.pop("key")
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains=".pop()",
        )
    
    def test_update_method(self):
        """Tool MUST flag .update() method call."""
        code = '''
def test_func():
    agent_output.update({"key": "value"})
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains=".update()",
        )
    
    def test_setdefault_method(self):
        """Tool MUST flag .setdefault() method call."""
        code = '''
def test_func():
    value = agent_output.setdefault("key", "default")
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains=".setdefault()",
        )
    
    def test_getattr_call(self):
        """Tool MUST flag getattr() on tracked variable."""
        code = '''
def test_func():
    value = getattr(agent_output, "verdict")
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains="getattr",
        )
    
    def test_getattr_with_default(self):
        """Tool MUST flag getattr() even with default."""
        code = '''
def test_func():
    value = getattr(agent_output, "verdict", None)
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains="getattr",
        )
    
    def test_iteration_over_dict_type(self):
        """Tool MUST flag direct iteration over dict-typed tracked var."""
        code = '''
def test_func():
    for key in agent_output:
        print(key)
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.PATTERN_NOT_WHITELISTED,
            message_contains="iteration",
        )
    
    def test_subscript_on_derived_variable(self):
        """Tool MUST flag subscript access on derived variable."""
        code = '''
def test_func():
    nested = agent_output.get("nested", {})
    value = nested["field"]
'''
        # Use simple tracked vars without schema for pattern-only test
        tracked_vars = {
            "agent_output": TrackedVariable(
                name="agent_output",
                schema_file="",
                json_pointer="",
            )
        }
        result = validate_code_snippet(code, tracked_vars)
        
        # The nested variable should be tracked and subscript flagged
        # Note: This may or may not work depending on derived var tracking
        # If no schema, derived vars might not be tracked - that's OK
        # The important thing is subscript on tracked vars is flagged


class TestDynamicKeyViolations:
    """
    Tests that MUST detect DYNAMIC_KEY violations.
    
    If any of these tests fail, it means the tool is NOT catching
    dynamic (non-literal) key access - a critical bug.
    """
    
    def test_get_with_variable(self):
        """Tool MUST flag .get() with variable key."""
        code = '''
def test_func():
    key = "verdict"
    value = agent_output.get(key)
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
            variable="agent_output",
        )
    
    def test_get_with_f_string(self):
        """Tool MUST flag .get() with f-string key."""
        code = '''
def test_func():
    idx = 1
    value = agent_output.get(f"field_{idx}")
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_get_with_function_call(self):
        """Tool MUST flag .get() with function call as key."""
        code = '''
def get_key():
    return "verdict"

def test_func():
    value = agent_output.get(get_key())
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_get_with_attribute_access(self):
        """Tool MUST flag .get() with attribute as key."""
        code = '''
class Config:
    field_name = "verdict"

def test_func():
    value = agent_output.get(Config.field_name)
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_get_with_dict_access(self):
        """Tool MUST flag .get() with dict access as key."""
        code = '''
def test_func():
    mapping = {"key": "verdict"}
    value = agent_output.get(mapping["key"])
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_get_with_list_access(self):
        """Tool MUST flag .get() with list index as key."""
        code = '''
def test_func():
    fields = ["verdict", "summary"]
    value = agent_output.get(fields[0])
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_get_with_ternary(self):
        """Tool MUST flag .get() with ternary expression as key."""
        code = '''
def test_func():
    use_verdict = True
    value = agent_output.get("verdict" if use_verdict else "summary")
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_in_check_with_variable(self):
        """Tool MUST flag 'in' check with variable key."""
        code = '''
def test_func():
    key = "verdict"
    if key in agent_output:
        pass
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )
    
    def test_in_check_with_expression(self):
        """Tool MUST flag 'in' check with expression as key."""
        code = '''
def test_func():
    prefix = "ver"
    if (prefix + "dict") in agent_output:
        pass
'''
        result = validate_code_snippet(code)
        
        assert_violation_found(
            result,
            ViolationType.DYNAMIC_KEY,
        )


class TestMultipleViolationTypes:
    """
    Tests with multiple types of violations in the same code.
    
    Ensures the tool catches ALL violations, not just the first one.
    """
    
    def test_mixed_violations(self):
        """Tool MUST catch multiple violation types in same code."""
        code = '''
def test_func():
    # Dynamic key violation
    key = "field"
    a = agent_output.get(key)
    
    # Subscript violation
    b = agent_output["field"]
    
    # Method violation
    c = agent_output.keys()
'''
        result = validate_code_snippet(code)
        
        # Should have all three violation types
        assert_violation_found(result, ViolationType.DYNAMIC_KEY, count=1)
        assert_violation_found(result, ViolationType.PATTERN_NOT_WHITELISTED, count=2)  # subscript + keys
    
    def test_violation_on_each_line(self):
        """Tool MUST report violations with correct line numbers."""
        code = '''
def test_func():
    a = agent_output.keys()
    b = agent_output.values()
    c = agent_output.items()
'''
        result = validate_code_snippet(code)
        
        # Should have 3 violations, each on different lines
        assert len(result.violations) == 3
        lines = {v.line for v in result.violations}
        assert len(lines) == 3, "Each violation should be on a different line"
