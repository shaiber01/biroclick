"""
Tests for derived variable tracking in the schema access validator.

These tests verify that the tool correctly:
- Tracks variables derived from agent_output (via .get(), alias, etc.)
- Validates field accesses on derived variables against nested schemas
- Tracks loop variables from array iteration
- Tracks comprehension variables

IMPORTANT: Derived variable tracking is critical for catching nested
schema mismatches. If these tests fail, nested bugs will go undetected.
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
    assert_field_access_recorded,
    MOCK_SCHEMA,
    TrackedVariable,
)


class TestDerivedVariableCreation:
    """
    Tests that derived variables are created correctly.
    
    When code assigns the result of .get() to a variable, that
    variable should become tracked with the nested schema context.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_get_creates_derived_variable(self, schema_file):
        """Assigning .get() result should create derived variable."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    # Access on nested should be tracked
    value = nested.get("inner_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should have access recorded for nested variable
        assert_field_access_recorded(result, "inner_field", variable="nested")
        assert_no_violations(result)
    
    def test_alias_creates_derived_variable(self, schema_file):
        """Direct assignment should create alias."""
        code = '''
def test_func():
    alias = agent_output
    value = alias.get("verdict")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should track access on alias
        assert_field_access_recorded(result, "verdict", variable="alias")
        assert_no_violations(result)
    
    def test_loop_variable_tracked_from_get(self, schema_file):
        """Loop variable from .get() array should be tracked."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        item_id = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # item should be tracked with array items schema
        assert_field_access_recorded(result, "item_id", variable="item")
        assert_no_violations(result)
    
    def test_comprehension_variable_tracked(self, schema_file):
        """Comprehension variable should be tracked."""
        code = '''
def test_func():
    ids = [x.get("item_id") for x in agent_output.get("array_field", [])]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # x should be tracked
        assert_field_access_recorded(result, "item_id", variable="x")
        assert_no_violations(result)
    
    def test_array_index_creates_derived(self, schema_file):
        """Array index access should create derived variable."""
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    if items:
        first = items[0]
        first_id = first.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # first should be tracked with items schema
        assert_field_access_recorded(result, "item_id", variable="first")
        assert_no_violations(result)
    
    def test_direct_array_iteration_creates_derived(self, schema_file):
        """Iterating over array variable should track loop variable."""
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    for item in items:
        item_id = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # item should be tracked
        assert_field_access_recorded(result, "item_id", variable="item")
        assert_no_violations(result)


class TestDerivedVariableValidation:
    """
    Tests that field accesses on derived variables are validated
    against the correct nested schema.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_valid_nested_field_passes(self, schema_file):
        """Valid field on nested object should pass."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("inner_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_invalid_nested_field_fails(self, schema_file):
        """Invalid field on nested object MUST be caught."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("nonexistent_nested_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_nested_field",
            variable="nested",
        )
    
    def test_valid_array_item_field_passes(self, schema_file):
        """Valid field on array item should pass."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        value = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_invalid_array_item_field_fails(self, schema_file):
        """Invalid field on array item MUST be caught."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        value = item.get("nonexistent_item_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_item_field",
            variable="item",
        )
    
    def test_valid_comprehension_field_passes(self, schema_file):
        """Valid field in comprehension should pass."""
        code = '''
def test_func():
    values = [x.get("item_value") for x in agent_output.get("array_field", [])]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_invalid_comprehension_field_fails(self, schema_file):
        """Invalid field in comprehension MUST be caught."""
        code = '''
def test_func():
    values = [x.get("bad_item_field") for x in agent_output.get("array_field", [])]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="bad_item_field",
            variable="x",
        )


class TestDeepNesting:
    """
    Tests for deeply nested schema navigation.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_two_level_nesting_valid(self, schema_file):
        """Two levels of nesting with valid fields should pass."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    deep = nested.get("deep_nested", {})
    value = deep.get("deepest_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_two_level_nesting_invalid_at_deepest(self, schema_file):
        """Invalid field at deepest level MUST be caught."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    deep = nested.get("deep_nested", {})
    value = deep.get("nonexistent_deepest")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_deepest",
            variable="deep",
        )
    
    def test_nested_in_array_item_valid(self, schema_file):
        """Nested object inside array item with valid field should pass."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        nested_in_item = item.get("nested_in_item", {})
        value = nested_in_item.get("deep_item_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_nested_in_array_item_invalid(self, schema_file):
        """Invalid field in nested object inside array item MUST be caught."""
        code = '''
def test_func():
    for item in agent_output.get("array_field", []):
        nested_in_item = item.get("nested_in_item", {})
        value = nested_in_item.get("bad_deep_field")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="bad_deep_field",
            variable="nested_in_item",
        )


class TestMultipleDerivedVariables:
    """
    Tests with multiple derived variables in the same function.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_multiple_derived_all_valid(self, schema_file):
        """Multiple derived variables with valid accesses should pass."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    items = agent_output.get("array_field", [])
    
    inner = nested.get("inner_field")
    
    for item in items:
        item_id = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
    
    def test_multiple_derived_one_invalid(self, schema_file):
        """One invalid access among multiple derived should be caught."""
        code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    items = agent_output.get("array_field", [])
    
    # Valid
    inner = nested.get("inner_field")
    
    # Invalid
    for item in items:
        bad = item.get("nonexistent")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent",
        )
    
    def test_same_variable_name_different_sources(self, schema_file):
        """Same var name reassigned from different sources should track correctly."""
        code = '''
def test_func():
    # First: from nested_obj
    data = agent_output.get("nested_obj", {})
    value1 = data.get("inner_field")  # Valid for nested_obj
    
    # Reassign: from array_field item
    for data in agent_output.get("array_field", []):
        value2 = data.get("item_id")  # Valid for array items
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Both accesses should be valid (different schemas at different points)
        assert_no_violations(result)


class TestScalarFieldsNotTracked:
    """
    Tests that scalar fields (strings, numbers) are NOT tracked.
    
    Only object-type fields should create tracked derived variables.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_string_field_not_tracked(self, schema_file):
        """String field should not become a tracked variable."""
        code = '''
def test_func():
    verdict = agent_output.get("verdict")
    # This is a string, so .get() should not be tracked
    # (strings don't have .get() but if called, shouldn't error)
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should just have the original access, no derived tracking
        assert_field_access_recorded(result, "verdict", variable="agent_output")
        assert_no_violations(result)
    
    def test_simple_array_not_tracked_as_object(self, schema_file):
        """Simple array (array of strings) items should not be tracked as objects."""
        code = '''
def test_func():
    strings = agent_output.get("simple_array", [])
    for s in strings:
        # s is a string, not an object
        length = len(s)
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should not flag anything - s is not tracked as an object
        assert_no_violations(result)
