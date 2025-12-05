"""
Tests for false positive prevention in the schema access validator.

These tests verify that the tool does NOT flag valid patterns:
- Allowed access patterns (.get(), 'in' checks)
- Valid field accesses (fields that exist in schema)
- Operations that should be allowed (write access, array iteration)

IMPORTANT: These tests are designed to prevent over-flagging. If a test
fails, it means the tool is incorrectly flagging valid code.
"""

import json
import pytest
from pathlib import Path

from tools.validate_schema_access import ViolationType

from .conftest import (
    validate_code_snippet,
    validate_code_with_mock_schema,
    assert_no_violations,
    assert_no_violations_of_type,
    assert_field_access_recorded,
    wrap_in_function,
    MOCK_SCHEMA,
    TrackedVariable,
)


class TestValidPatternsNotFlagged:
    """
    Tests that valid access patterns are NOT flagged.
    
    If any of these tests fail, it means the tool is incorrectly
    flagging valid code - a false positive bug.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_get_valid_field_no_default(self, schema_file):
        """Tool MUST NOT flag .get() with valid field."""
        code = '''
def test_func():
    value = agent_output.get("verdict")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
        assert_field_access_recorded(result, "verdict", "agent_output")
    
    def test_get_valid_field_with_default(self, schema_file):
        """Tool MUST NOT flag .get() with valid field and default."""
        code = '''
def test_func():
    value = agent_output.get("summary", "")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
        assert_field_access_recorded(result, "summary", access_type="get_with_default")
    
    def test_get_valid_optional_field(self, schema_file):
        """Tool MUST NOT flag .get() on optional (non-required) field."""
        code = '''
def test_func():
    value = agent_output.get("optional_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_in_check_valid_field(self, schema_file):
        """Tool MUST NOT flag 'in' check with valid field."""
        code = '''
def test_func():
    if "verdict" in agent_output:
        pass
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
        assert_field_access_recorded(result, "verdict", access_type="in_check")
    
    def test_valid_nested_object_access(self, schema_file):
        """Tool MUST NOT flag valid access on nested object."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("inner_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_valid_loop_variable_access(self, schema_file):
        """Tool MUST NOT flag valid access on loop variable."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        item_id = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_valid_comprehension_variable_access(self, schema_file):
        """Tool MUST NOT flag valid access in comprehension."""
        code = '''
def test_func():
    ids = [x.get("item_id") for x in agent_output.get("array_field", [])]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_multiple_valid_accesses(self, schema_file):
        """Tool MUST NOT flag multiple valid accesses."""
        code = '''
def test_func():
    verdict = agent_output.get("verdict")
    summary = agent_output.get("summary")
    nested = agent_output.get("nested_obj", {})
    inner = nested.get("inner_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
        # Should record all 4 accesses
        assert len(result.field_accesses) == 4


class TestAllowedOperationsNotFlagged:
    """
    Tests that allowed operations are NOT flagged.
    
    Some patterns that look like violations are actually allowed:
    - Write access (obj["key"] = value)
    - Iteration over array-typed variables
    - Numeric index on arrays
    """
    
    def test_subscript_write_allowed(self):
        """Tool MUST NOT flag subscript WRITE access."""
        code = '''
def test_func():
    agent_output["key"] = "value"
'''
        result = validate_code_snippet(code)
        
        # Write access is allowed - no PATTERN_NOT_WHITELISTED
        assert_no_violations_of_type(result, ViolationType.PATTERN_NOT_WHITELISTED)
    
    def test_subscript_augmented_assign_allowed(self):
        """Tool MUST NOT flag augmented assignment."""
        code = '''
def test_func():
    agent_output["count"] += 1
'''
        result = validate_code_snippet(code)
        
        # Augmented assign on subscript should be allowed
        # (This is a write operation)
        assert_no_violations_of_type(result, ViolationType.PATTERN_NOT_WHITELISTED)
    
    def test_iteration_over_array_variable_allowed(self, tmp_path):
        """Tool MUST NOT flag iteration over array-typed variable."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    for item in items:
        pass
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Iteration over array variable should be allowed
        assert_no_violations_of_type(result, ViolationType.PATTERN_NOT_WHITELISTED)
    
    def test_numeric_index_on_array_allowed(self, tmp_path):
        """Tool MUST NOT flag numeric index on array variable."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    if items:
        first = items[0]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Numeric index on array should be allowed
        assert_no_violations_of_type(result, ViolationType.PATTERN_NOT_WHITELISTED)
    
    def test_negative_index_on_array_allowed(self, tmp_path):
        """Tool MUST NOT flag negative index on array variable."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    if items:
        last = items[-1]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Negative index on array should be allowed
        assert_no_violations_of_type(result, ViolationType.PATTERN_NOT_WHITELISTED)


class TestNonTrackedVariablesIgnored:
    """
    Tests that non-tracked variables are completely ignored.
    
    The tool should only validate access on tracked variables
    (like agent_output). Regular dicts should not be flagged.
    """
    
    def test_regular_dict_subscript_ignored(self):
        """Tool MUST NOT flag subscript on regular dict."""
        code = '''
def test_func():
    other_dict = {"key": "value"}
    value = other_dict["key"]
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
        assert len(result.field_accesses) == 0  # Not tracked
    
    def test_regular_dict_keys_ignored(self):
        """Tool MUST NOT flag .keys() on regular dict."""
        code = '''
def test_func():
    other_dict = {"key": "value"}
    keys = other_dict.keys()
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_regular_dict_iteration_ignored(self):
        """Tool MUST NOT flag iteration over regular dict."""
        code = '''
def test_func():
    other_dict = {"key": "value"}
    for k in other_dict:
        print(k)
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_function_return_ignored(self):
        """Tool MUST NOT flag operations on function returns."""
        code = '''
def get_data():
    return {"key": "value"}

def test_func():
    data = get_data()
    value = data["key"]
    keys = data.keys()
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_parameter_ignored(self):
        """Tool MUST NOT flag operations on function parameters."""
        code = '''
def test_func(some_dict):
    value = some_dict["key"]
    keys = some_dict.keys()
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_class_attribute_ignored(self):
        """Tool MUST NOT flag operations on class attributes."""
        code = '''
class MyClass:
    def method(self):
        value = self.data["key"]
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)


class TestCodeWithoutTrackedVariables:
    """
    Tests that code without tracked variable access passes cleanly.
    """
    
    def test_empty_function(self):
        """Empty function should pass."""
        code = '''
def test_func():
    pass
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
        assert len(result.field_accesses) == 0
    
    def test_function_with_unrelated_code(self):
        """Function with unrelated code should pass."""
        code = '''
def test_func():
    x = 1 + 2
    y = [1, 2, 3]
    z = {"a": 1, "b": 2}
    return x, y, z
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
        assert len(result.field_accesses) == 0
    
    def test_multiple_functions_no_tracked_vars(self):
        """Multiple functions without tracked vars should pass."""
        code = '''
def func1():
    return 1

def func2():
    return 2

def func3():
    return func1() + func2()
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)


class TestEdgeCasesNoFalsePositives:
    """
    Edge cases that should NOT trigger false positives.
    """
    
    def test_agent_output_as_string_literal(self):
        """String 'agent_output' should not be flagged."""
        code = '''
def test_func():
    name = "agent_output"
    print(f"Variable name is {name}")
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_agent_output_in_comment(self):
        """Comment mentioning agent_output should be ignored."""
        code = '''
def test_func():
    # agent_output["field"] is how you access it
    pass
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_different_variable_same_name_pattern(self):
        """Variable with 'output' in name but not tracked should be ignored."""
        code = '''
def test_func():
    my_output = {"key": "value"}
    value = my_output["key"]
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
    
    def test_get_on_non_tracked_similar_name(self):
        """'.get()' on similar but different variable should be ignored."""
        code = '''
def test_func():
    agent_result = {"key": "value"}
    value = agent_result.get("key")
'''
        result = validate_code_snippet(code)
        
        assert_no_violations(result)
