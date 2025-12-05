"""
Tests for function scope management in the schema access validator.

These tests verify that the tool correctly:
- Isolates derived variables to their function scope
- Clears derived variables when exiting a function
- Handles same variable names in different functions independently
- Injects correct schema per-function based on LLM call

IMPORTANT: Scope management is critical for avoiding false positives
(variables leaking between functions) and false negatives (wrong
schema used for validation).
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
    assert_no_violations_of_type,
    MOCK_SCHEMA,
    TrackedVariable,
)


class TestFunctionScopeIsolation:
    """
    Tests that derived variables are properly scoped to their function.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_derived_var_cleared_on_function_exit(self, schema_file):
        """Derived variable from func1 should NOT be visible in func2."""
        code = '''
def func1():
    # Create derived variable
    nested = agent_output.get("nested_obj", {})
    value = nested.get("inner_field")

def func2():
    # 'nested' should not be tracked here - it was from func1
    # This should be treated as a regular (non-tracked) variable
    nested = {"different": "dict"}
    value = nested["key"]  # Should NOT be flagged
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # The subscript in func2 should NOT be flagged because
        # 'nested' in func2 is a new, non-tracked variable
        assert_no_violations_of_type(result, ViolationType.PATTERN_NOT_WHITELISTED)
    
    def test_same_var_name_different_functions_independent(self, schema_file):
        """Same variable name in different functions should be independent."""
        code = '''
def func1():
    # 'x' derived from nested_obj
    x = agent_output.get("nested_obj", {})
    value = x.get("inner_field")  # Valid for nested_obj

def func2():
    # 'x' derived from array_field item
    for x in agent_output.get("array_field", []):
        value = x.get("item_id")  # Valid for array items
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Both should be valid - x has different schemas in each function
        assert_no_violations(result)
        # Should have 4 accesses total (nested_obj, inner_field, array_field, item_id)
        assert len(result.field_accesses) == 4
    
    def test_function_context_tracked_correctly(self, schema_file):
        """Each access should have correct function context."""
        code = '''
def first_function():
    value = agent_output.get("verdict")

def second_function():
    value = agent_output.get("summary")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should have exactly 2 accesses
        assert len(result.field_accesses) == 2
        assert_no_violations(result)
        
        # Verify function context is tracked
        assert_field_access_recorded(
            result, "verdict", in_function="first_function"
        )
        assert_field_access_recorded(
            result, "summary", in_function="second_function"
        )


class TestNestedFunctionScopes:
    """
    Tests for nested function definitions.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_inner_function_has_own_scope(self, schema_file):
        """Inner function should have its own scope."""
        code = '''
def outer():
    nested = agent_output.get("nested_obj", {})
    outer_val = nested.get("inner_field")  # Valid
    
    def inner():
        # 'nested' from outer should NOT be tracked here
        # This is a new scope
        items = agent_output.get("array_field", [])
        for item in items:
            inner_val = item.get("item_id")  # Valid
    
    inner()
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Both accesses should be valid
        assert_no_violations(result)
        # Should have 4 accesses: nested_obj, inner_field (outer), array_field, item_id (inner)
        assert len(result.field_accesses) == 4
        # Verify function contexts - note: inner_field is on 'nested', item_id is on 'item'
        assert_field_access_recorded(result, "nested_obj", variable="agent_output", in_function="outer")
        assert_field_access_recorded(result, "inner_field", variable="nested", in_function="outer")
        assert_field_access_recorded(result, "array_field", variable="agent_output", in_function="inner")
        assert_field_access_recorded(result, "item_id", variable="item", in_function="inner")
    
    def test_lambda_has_own_scope(self, schema_file):
        """Lambda functions should have independent scope."""
        code = '''
def outer():
    nested = agent_output.get("nested_obj", {})
    
    # Lambda - different scope context
    process = lambda x: x.get("item_id")
    
    for item in agent_output.get("array_field", []):
        result = process(item)
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Lambda parameter 'x' is not tracked, so no validation
        # This should not cause errors
        assert_no_violations(result)


class TestAsyncFunctionScopes:
    """
    Tests for async function scope handling.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_async_function_scope_isolated(self, schema_file):
        """Async functions should have isolated scope."""
        code = '''
async def async_func1():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("inner_field")

async def async_func2():
    # Different scope - nested is not tracked here
    items = agent_output.get("array_field", [])
    for item in items:
        value = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
        # Should have 4 accesses total
        assert len(result.field_accesses) == 4
        # Verify function contexts for async functions - note derived variable names
        assert_field_access_recorded(result, "nested_obj", variable="agent_output", in_function="async_func1")
        assert_field_access_recorded(result, "inner_field", variable="nested", in_function="async_func1")
        assert_field_access_recorded(result, "array_field", variable="agent_output", in_function="async_func2")
        assert_field_access_recorded(result, "item_id", variable="item", in_function="async_func2")
    
    def test_mixed_sync_async_scopes(self, schema_file):
        """Sync and async functions should have separate scopes."""
        code = '''
def sync_func():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("inner_field")

async def async_func():
    items = agent_output.get("array_field", [])
    for item in items:
        value = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)


class TestClassMethodScopes:
    """
    Tests for class method scope handling.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_class_methods_have_separate_scopes(self, schema_file):
        """Each class method should have its own scope."""
        code = '''
class MyClass:
    def method1(self):
        nested = agent_output.get("nested_obj", {})
        value = nested.get("inner_field")
    
    def method2(self):
        # Separate scope from method1
        items = agent_output.get("array_field", [])
        for item in items:
            value = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)
        # Should have 4 accesses total across both methods
        assert len(result.field_accesses) == 4
        # Verify method context - note derived variable names
        assert_field_access_recorded(result, "nested_obj", variable="agent_output", in_function="method1")
        assert_field_access_recorded(result, "inner_field", variable="nested", in_function="method1")
        assert_field_access_recorded(result, "array_field", variable="agent_output", in_function="method2")
        assert_field_access_recorded(result, "item_id", variable="item", in_function="method2")
    
    def test_static_and_class_methods(self, schema_file):
        """Static and class methods should have proper scope."""
        code = '''
class MyClass:
    @staticmethod
    def static_method():
        nested = agent_output.get("nested_obj", {})
        value = nested.get("inner_field")
    
    @classmethod
    def class_method(cls):
        items = agent_output.get("array_field", [])
        for item in items:
            value = item.get("item_id")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_no_violations(result)


class TestScopeWithViolations:
    """
    Tests that violations are correctly scoped to functions.
    """
    
    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create mock schema file."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        return schema_file
    
    def test_violation_in_one_function_doesnt_affect_other(self, schema_file):
        """Violation in func1 should not affect func2 validation."""
        code = '''
def func_with_violation():
    # This has a violation
    value = agent_output.get("nonexistent_field")

def func_without_violation():
    # This is valid
    value = agent_output.get("verdict")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should have exactly one violation
        assert_violation_found(
            result,
            ViolationType.FIELD_NOT_IN_SCHEMA,
            field="nonexistent_field",
            count=1,
        )
        
        # The valid access should still be recorded
        assert_field_access_recorded(
            result, "verdict", in_function="func_without_violation"
        )
    
    def test_derived_var_violation_scoped_correctly(self, schema_file):
        """Violation on derived var should show correct variable name."""
        code = '''
def func1():
    nested1 = agent_output.get("nested_obj", {})
    value = nested1.get("bad_field")  # Violation on nested1

def func2():
    nested2 = agent_output.get("nested_obj", {})
    value = nested2.get("inner_field")  # Valid
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Violation should reference nested1, not nested2
        violations = [v for v in result.violations 
                     if v.type == ViolationType.FIELD_NOT_IN_SCHEMA]
        assert len(violations) == 1
        assert violations[0].variable == "nested1"


class TestGlobalScope:
    """
    Tests for code at module level (outside functions).
    """
    
    def test_module_level_access_tracked(self):
        """Module-level access should be tracked with no function context."""
        code = '''
# Module level access
value = agent_output.get("verdict")
'''
        result = validate_code_snippet(code)
        
        # Should be tracked with no function context
        access = assert_field_access_recorded(result, "verdict")
        assert access.in_function is None or access.in_function == ""
    
    def test_module_level_doesnt_leak_into_functions(self, tmp_path):
        """Module-level derived vars should not leak into functions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
# Module level
global_nested = agent_output.get("nested_obj", {})
global_val = global_nested.get("inner_field")

def some_function():
    # local_nested is a fresh variable, not related to global_nested
    local_nested = {"different": "data"}
    # This should NOT be tracked as a derived variable
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should have the module-level accesses
        assert_field_access_recorded(result, "nested_obj", variable="agent_output")
        assert_field_access_recorded(result, "inner_field", variable="global_nested")
