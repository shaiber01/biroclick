"""
Tests for edge cases and error handling in the schema access validator.

These tests verify that the tool handles unusual inputs gracefully:
- Empty code
- Syntax errors
- Unusual Python constructs
- Error recovery

IMPORTANT: Edge cases often reveal hidden bugs. If these tests fail,
the tool may crash or behave unexpectedly in real-world usage.
"""

import json
import pytest
from pathlib import Path

from tools.validate_schema_access import (
    ViolationType,
    extract_field_accesses,
    ValidationResult,
)

from .conftest import (
    validate_code_snippet,
    validate_code_with_mock_schema,
    assert_violation_found,
    assert_no_violations,
    assert_field_access_recorded,
    MOCK_SCHEMA,
    TrackedVariable,
)


class TestEmptyAndMinimalInput:
    """
    Tests for empty and minimal code inputs.
    """
    
    def test_empty_string(self):
        """Should handle empty string input."""
        result = validate_code_snippet("")
        
        assert isinstance(result, ValidationResult)
        assert len(result.violations) == 0
        assert len(result.field_accesses) == 0
    
    def test_whitespace_only(self):
        """Should handle whitespace-only input."""
        result = validate_code_snippet("   \n\n\t\t   ")
        
        assert isinstance(result, ValidationResult)
        assert len(result.violations) == 0
    
    def test_comment_only(self):
        """Should handle comment-only input."""
        code = '''
# This is a comment
# agent_output.get("field") in comment
'''
        result = validate_code_snippet(code)
        
        assert len(result.violations) == 0
        assert len(result.field_accesses) == 0
    
    def test_single_pass_statement(self):
        """Should handle single pass statement."""
        code = "pass"
        result = validate_code_snippet(code)
        
        assert len(result.violations) == 0


class TestSyntaxErrors:
    """
    Tests for handling code with syntax errors.
    """
    
    def test_syntax_error_handled_gracefully(self):
        """Should handle syntax errors without crashing."""
        code = '''
def incomplete_function(
    # Missing closing paren and body
'''
        result = validate_code_snippet(code)
        
        # Should return a result, possibly with a violation
        assert isinstance(result, ValidationResult)
    
    def test_indentation_error(self):
        """Should handle indentation errors."""
        code = '''
def func():
pass  # Bad indentation
'''
        result = validate_code_snippet(code)
        
        assert isinstance(result, ValidationResult)
    
    def test_incomplete_expression(self):
        """Should handle incomplete expressions."""
        code = '''
def func():
    x = agent_output.get(
'''
        result = validate_code_snippet(code)
        
        assert isinstance(result, ValidationResult)


class TestUnusualPythonConstructs:
    """
    Tests for unusual but valid Python code.
    """
    
    def test_walrus_operator(self, tmp_path):
        """Should handle walrus operator (assignment expression)."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    if (verdict := agent_output.get("verdict")):
        print(verdict)
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should track the access
        assert_field_access_recorded(result, "verdict")
        assert_no_violations(result)
    
    def test_nested_comprehension(self, tmp_path):
        """Should handle nested comprehensions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    result = [[item.get("item_id") for item in items] 
              for items in [agent_output.get("array_field", [])]]
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should handle nested comprehension without crashing
        assert isinstance(result, ValidationResult)
        # Should record the array_field access at minimum
        assert_field_access_recorded(result, "array_field", variable="agent_output")
        # Valid fields should not produce violations
        assert_no_violations(result)
    
    def test_generator_expression(self, tmp_path):
        """Should handle generator expressions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    gen = (x.get("item_id") for x in agent_output.get("array_field", []))
    return list(gen)
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert isinstance(result, ValidationResult)
        # Should record array_field access
        assert_field_access_recorded(result, "array_field", variable="agent_output")
        # Should record item_id access on comprehension variable x
        assert_field_access_recorded(result, "item_id", variable="x")
        # All valid fields - no violations
        assert_no_violations(result)
    
    def test_dict_comprehension(self, tmp_path):
        """Should handle dict comprehensions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    result = {x.get("item_id"): x.get("item_value") 
              for x in agent_output.get("array_field", [])}
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert isinstance(result, ValidationResult)
        # Should record array_field access
        assert_field_access_recorded(result, "array_field", variable="agent_output")
        # Should record both item_id and item_value accesses on x
        assert_field_access_recorded(result, "item_id", variable="x")
        assert_field_access_recorded(result, "item_value", variable="x")
        # All valid fields - no violations
        assert_no_violations(result)
    
    def test_set_comprehension(self, tmp_path):
        """Should handle set comprehensions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    ids = {x.get("item_id") for x in agent_output.get("array_field", [])}
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert isinstance(result, ValidationResult)
        # Should record array_field access
        assert_field_access_recorded(result, "array_field", variable="agent_output")
        # Should record item_id access on comprehension variable x
        assert_field_access_recorded(result, "item_id", variable="x")
        # All valid fields - no violations
        assert_no_violations(result)
    
    def test_starred_expression(self):
        """Should handle starred expressions."""
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    first, *rest = items
'''
        result = validate_code_snippet(code)
        
        assert isinstance(result, ValidationResult)
        # Should record the array_field access
        assert_field_access_recorded(result, "array_field", variable="agent_output")
    
    def test_ternary_expression(self, tmp_path):
        """Should handle ternary expressions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    value = agent_output.get("verdict") if True else agent_output.get("summary")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should record both accesses
        assert_field_access_recorded(result, "verdict")
        assert_field_access_recorded(result, "summary")
    
    def test_chained_comparison(self):
        """Should handle chained comparisons."""
        code = '''
def test_func():
    if "a" in agent_output and "b" in agent_output:
        pass
'''
        result = validate_code_snippet(code)
        
        # Should record both 'in' checks
        assert len(result.field_accesses) == 2
        fields = {a.field for a in result.field_accesses}
        assert "a" in fields
        assert "b" in fields
        # Both should be in_check access type
        for access in result.field_accesses:
            assert access.access_type == "in_check", f"Expected 'in_check' but got '{access.access_type}'"


class TestMultipleAccessesSameLine:
    """
    Tests for multiple accesses on the same line.
    """
    
    def test_multiple_gets_same_line(self):
        """Should handle multiple .get() calls on same line."""
        code = '''
def test_func():
    a, b = agent_output.get("verdict"), agent_output.get("summary")
'''
        result = validate_code_snippet(code)
        
        assert len(result.field_accesses) == 2
        fields = {a.field for a in result.field_accesses}
        assert "verdict" in fields
        assert "summary" in fields
    
    def test_chained_or_expressions(self, tmp_path):
        """Should handle chained or expressions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    value = agent_output.get("verdict") or agent_output.get("summary") or "default"
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert len(result.field_accesses) == 2
        assert_no_violations(result)


class TestDecoratorHandling:
    """
    Tests for functions with decorators.
    """
    
    def test_decorated_function(self, tmp_path):
        """Should handle decorated functions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def decorator(func):
    return func

@decorator
def test_func():
    value = agent_output.get("verdict")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_field_access_recorded(result, "verdict", in_function="test_func")
        assert_no_violations(result)
        # Should have exactly 1 access
        assert len(result.field_accesses) == 1
    
    def test_multiple_decorators(self, tmp_path):
        """Should handle multiple decorators."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def dec1(func):
    return func

def dec2(func):
    return func

@dec1
@dec2
def test_func():
    value = agent_output.get("verdict")
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_field_access_recorded(result, "verdict", in_function="test_func")
        assert_no_violations(result)
        # Should have exactly 1 access
        assert len(result.field_accesses) == 1


class TestSpecialVariableNames:
    """
    Tests for special variable naming scenarios.
    """
    
    def test_underscore_prefix(self):
        """Should handle underscore-prefixed tracked var names."""
        tracked_vars = {
            "_agent_output": TrackedVariable(
                name="_agent_output",
                schema_file="",
                json_pointer="",
            )
        }
        
        code = '''
def test_func():
    value = _agent_output.get("field")
'''
        result = validate_code_snippet(code, tracked_vars)
        
        assert_field_access_recorded(result, "field", variable="_agent_output")
    
    def test_dunder_name_handling(self):
        """Should not treat dunder names specially."""
        code = '''
def test_func():
    # Regular variable with unusual name
    __result__ = agent_output.get("verdict")
'''
        result = validate_code_snippet(code)
        
        assert_field_access_recorded(result, "verdict")


class TestErrorRecovery:
    """
    Tests for error recovery and resilience.
    """
    
    def test_continues_validating_multiple_functions(self):
        """Should validate all functions in a file."""
        code = '''
def good_func():
    value = agent_output.get("verdict")

def another_func():
    other = agent_output.get("summary")
'''
        result = validate_code_snippet(code)
        
        # Should have exactly 2 accesses - one from each function
        assert len(result.field_accesses) == 2
        
        # Verify accesses are recorded with correct function context
        assert_field_access_recorded(result, "verdict", in_function="good_func")
        assert_field_access_recorded(result, "summary", in_function="another_func")
    
    def test_handles_unicode_gracefully(self):
        """Should handle unicode in code."""
        code = '''
def test_func():
    # Unicode comment: こんにちは
    value = agent_output.get("verdict")  # More unicode: 日本語
'''
        result = validate_code_snippet(code)
        
        # Should record the access correctly despite unicode
        assert_field_access_recorded(result, "verdict", in_function="test_func")
        # Should have exactly one access
        assert len(result.field_accesses) == 1


class TestLargeCodeHandling:
    """
    Tests for handling larger code blocks.
    """
    
    def test_many_functions(self, tmp_path):
        """Should handle many functions."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        # Generate code with many functions
        functions = []
        for i in range(20):
            functions.append(f'''
def func_{i}():
    value = agent_output.get("verdict")
''')
        code = "\n".join(functions)
        
        result = validate_code_with_mock_schema(code, schema_file)
        
        # Should have 20 accesses
        assert len(result.field_accesses) == 20
        assert_no_violations(result)
    
    def test_many_accesses_in_one_function(self, tmp_path):
        """Should handle many accesses in single function."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        accesses = "\n    ".join([
            f'v{i} = agent_output.get("verdict")'
            for i in range(50)
        ])
        code = f'''
def test_func():
    {accesses}
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert len(result.field_accesses) == 50
        assert_no_violations(result)


class TestBooleanContextHandling:
    """
    Tests for tracked vars used in boolean contexts.
    """
    
    def test_if_statement_with_get(self, tmp_path):
        """Should handle if statement with .get()."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    if agent_output.get("verdict"):
        pass
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_field_access_recorded(result, "verdict")
        assert_no_violations(result)
    
    def test_while_statement_with_get(self, tmp_path):
        """Should handle while statement with .get()."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    while agent_output.get("verdict"):
        break
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_field_access_recorded(result, "verdict")
        assert_no_violations(result)
    
    def test_assert_statement_with_get(self, tmp_path):
        """Should handle assert statement with .get()."""
        schema_file = tmp_path / "mock_output_schema.json"
        schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
        
        code = '''
def test_func():
    assert agent_output.get("verdict") == "pass"
'''
        result = validate_code_with_mock_schema(code, schema_file)
        
        assert_field_access_recorded(result, "verdict")
        assert_no_violations(result)
