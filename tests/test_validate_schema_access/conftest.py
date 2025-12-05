"""
Shared fixtures and helpers for schema access validator tests.

This module provides:
- Mock schemas with predictable structure for isolated testing
- Helper functions to run validation on code snippets
- Assertion helpers for strict violation checking
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Set
from unittest.mock import patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_schema_access import (
    extract_field_accesses,
    validate_fields_against_schema,
    get_schema_fields_for_pointer,
    extract_schema_fields,
    load_schema,
    ViolationType,
    ValidationResult,
    Violation,
    FieldAccess,
    TrackedVariable,
    FieldAccessVisitor,
    SCHEMAS_DIR,
)


# =============================================================================
# Mock Schemas for Testing
# =============================================================================

MOCK_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["verdict", "summary"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail"],
            "description": "Overall verdict"
        },
        "summary": {
            "type": "string",
            "description": "Summary text"
        },
        "optional_field": {
            "type": "string",
            "description": "An optional field"
        },
        "nested_obj": {
            "type": "object",
            "description": "A nested object",
            "properties": {
                "inner_field": {
                    "type": "string",
                    "description": "Inner string field"
                },
                "inner_number": {
                    "type": "number",
                    "description": "Inner number field"
                },
                "deep_nested": {
                    "type": "object",
                    "properties": {
                        "deepest_field": {
                            "type": "string"
                        }
                    }
                }
            }
        },
        "array_field": {
            "type": "array",
            "description": "An array of items",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Item identifier"
                    },
                    "item_value": {
                        "type": "number",
                        "description": "Item value"
                    },
                    "nested_in_item": {
                        "type": "object",
                        "properties": {
                            "deep_item_field": {"type": "string"}
                        }
                    }
                },
                "required": ["item_id"]
            }
        },
        "simple_array": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# Valid fields at each level (pre-computed for assertions)
MOCK_SCHEMA_TOP_LEVEL_FIELDS = {"verdict", "summary", "optional_field", "nested_obj", "array_field", "simple_array"}
MOCK_SCHEMA_NESTED_OBJ_FIELDS = {"inner_field", "inner_number", "deep_nested"}
MOCK_SCHEMA_ARRAY_ITEM_FIELDS = {"item_id", "item_value", "nested_in_item"}
MOCK_SCHEMA_DEEP_NESTED_FIELDS = {"deepest_field"}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_schema():
    """Return the mock schema dictionary."""
    return MOCK_SCHEMA.copy()


@pytest.fixture
def mock_schema_file(tmp_path):
    """Create a temporary mock schema file and return its path."""
    schema_file = tmp_path / "mock_output_schema.json"
    schema_file.write_text(json.dumps(MOCK_SCHEMA, indent=2))
    return schema_file


@pytest.fixture
def tracked_vars_with_schema(mock_schema_file):
    """
    Create tracked variables dict with the mock schema.
    
    This simulates having agent_output tracked with a known schema.
    """
    return {
        "agent_output": TrackedVariable(
            name="agent_output",
            schema_file=str(mock_schema_file),
            json_pointer="",
        )
    }


@pytest.fixture
def simple_tracked_vars():
    """
    Create tracked variables without schema (for pattern-only tests).
    
    Used when testing whitelist violations where schema validation
    is not the focus.
    """
    return {
        "agent_output": TrackedVariable(
            name="agent_output",
            schema_file="",  # No schema - pattern tests only
            json_pointer="",
        )
    }


# =============================================================================
# Helper Functions
# =============================================================================

def validate_code_snippet(
    code: str,
    tracked_vars: Optional[dict] = None,
    file_path: str = "test_file.py",
) -> ValidationResult:
    """
    Validate a code snippet and return the result.
    
    Args:
        code: Python code string to validate
        tracked_vars: Optional dict of tracked variables. If None, uses
                      default with agent_output (no schema).
        file_path: File path to use in violation messages
    
    Returns:
        ValidationResult with violations and field_accesses
    """
    if tracked_vars is None:
        tracked_vars = {
            "agent_output": TrackedVariable(
                name="agent_output",
                schema_file="",
                json_pointer="",
            )
        }
    
    result, _ = extract_field_accesses(code, file_path, tracked_vars)
    return result


def validate_code_with_mock_schema(
    code: str,
    schema_file_path: Path,
    file_path: str = "test_file.py",
) -> ValidationResult:
    """
    Validate a code snippet with a mock schema file for field validation.
    
    Args:
        code: Python code string to validate
        schema_file_path: Path to the mock schema JSON file
        file_path: File path to use in violation messages
    
    Returns:
        ValidationResult with violations and field_accesses
    """
    tracked_vars = {
        "agent_output": TrackedVariable(
            name="agent_output",
            schema_file=str(schema_file_path),
            json_pointer="",
        )
    }
    
    result, _ = extract_field_accesses(code, file_path, tracked_vars)
    return result


# =============================================================================
# Assertion Helpers - STRICT checking for bug detection
# =============================================================================

def assert_violation_found(
    result: ValidationResult,
    violation_type: ViolationType,
    field: Optional[str] = None,
    variable: Optional[str] = None,
    message_contains: Optional[str] = None,
    count: int = 1,
) -> List[Violation]:
    """
    Assert that a specific violation was found.
    
    This is a STRICT assertion - it fails if:
    - No violations of the expected type are found
    - The count doesn't match (if count specified)
    - The field/variable don't match (if specified)
    
    Args:
        result: ValidationResult to check
        violation_type: Expected ViolationType
        field: If specified, violation must have this field
        variable: If specified, violation must have this variable
        message_contains: If specified, violation message must contain this
        count: Expected number of matching violations (default 1)
    
    Returns:
        List of matching violations (for further inspection)
    
    Raises:
        AssertionError if violation not found or count mismatch
    """
    matching = []
    
    for v in result.violations:
        if v.type != violation_type:
            continue
        if field is not None and v.field != field:
            continue
        if variable is not None and v.variable != variable:
            continue
        if message_contains is not None and message_contains not in v.message:
            continue
        matching.append(v)
    
    if len(matching) == 0:
        # Build detailed error message
        violation_desc = f"type={violation_type.value}"
        if field:
            violation_desc += f", field='{field}'"
        if variable:
            violation_desc += f", variable='{variable}'"
        if message_contains:
            violation_desc += f", message contains '{message_contains}'"
        
        found_violations = "\n".join(
            f"  - {v.type.value}: {v.message}" for v in result.violations
        ) or "  (none)"
        
        raise AssertionError(
            f"Expected violation not found: {violation_desc}\n"
            f"Found violations:\n{found_violations}"
        )
    
    if len(matching) != count:
        raise AssertionError(
            f"Expected {count} violation(s) of type {violation_type.value}, "
            f"but found {len(matching)}"
        )
    
    return matching


def assert_no_violations(result: ValidationResult) -> None:
    """
    Assert that there are NO violations.
    
    This is a STRICT assertion - it fails if ANY violation exists.
    
    Args:
        result: ValidationResult to check
    
    Raises:
        AssertionError if any violations found
    """
    if result.violations:
        violation_list = "\n".join(
            f"  - {v.type.value} at line {v.line}: {v.message}"
            for v in result.violations
        )
        raise AssertionError(
            f"Expected no violations, but found {len(result.violations)}:\n"
            f"{violation_list}"
        )


def assert_no_violations_of_type(
    result: ValidationResult,
    violation_type: ViolationType,
) -> None:
    """
    Assert that there are no violations of a specific type.
    
    Useful when you expect some violations but not others.
    
    Args:
        result: ValidationResult to check
        violation_type: Type that should NOT be present
    
    Raises:
        AssertionError if any violations of that type found
    """
    matching = [v for v in result.violations if v.type == violation_type]
    if matching:
        violation_list = "\n".join(
            f"  - line {v.line}: {v.message}" for v in matching
        )
        raise AssertionError(
            f"Expected no {violation_type.value} violations, but found {len(matching)}:\n"
            f"{violation_list}"
        )


def assert_field_access_recorded(
    result: ValidationResult,
    field: str,
    variable: str = "agent_output",
    access_type: Optional[str] = None,
    in_function: Optional[str] = None,
) -> FieldAccess:
    """
    Assert that a field access was recorded.
    
    Args:
        result: ValidationResult to check
        field: Expected field name
        variable: Expected variable name (default: agent_output)
        access_type: If specified, access must have this type
        in_function: If specified, access must be in this function
    
    Returns:
        The matching FieldAccess object
    
    Raises:
        AssertionError if matching access not found
    """
    for access in result.field_accesses:
        if access.field != field:
            continue
        if access.variable != variable:
            continue
        if access_type is not None and access.access_type != access_type:
            continue
        if in_function is not None and access.in_function != in_function:
            continue
        return access
    
    access_list = "\n".join(
        f"  - {a.variable}.{a.field} ({a.access_type}) in {a.in_function}"
        for a in result.field_accesses
    ) or "  (none)"
    
    raise AssertionError(
        f"Expected field access not recorded: {variable}.{field}\n"
        f"Recorded accesses:\n{access_list}"
    )


def assert_violation_count(
    result: ValidationResult,
    expected_count: int,
    violation_type: Optional[ViolationType] = None,
) -> None:
    """
    Assert exact count of violations.
    
    Args:
        result: ValidationResult to check
        expected_count: Expected number of violations
        violation_type: If specified, count only this type
    
    Raises:
        AssertionError if count doesn't match
    """
    if violation_type is not None:
        violations = [v for v in result.violations if v.type == violation_type]
        type_desc = f" of type {violation_type.value}"
    else:
        violations = result.violations
        type_desc = ""
    
    if len(violations) != expected_count:
        violation_list = "\n".join(
            f"  - {v.type.value}: {v.message}" for v in violations
        ) or "  (none)"
        raise AssertionError(
            f"Expected {expected_count} violation(s){type_desc}, "
            f"but found {len(violations)}:\n{violation_list}"
        )


# =============================================================================
# Code Templates for Common Test Patterns
# =============================================================================

def wrap_in_function(code: str, func_name: str = "test_func") -> str:
    """
    Wrap code in a function definition.
    
    Args:
        code: Code to wrap (will be indented)
        func_name: Name of the function
    
    Returns:
        Code wrapped in function def
    """
    indented = "\n".join(f"    {line}" for line in code.strip().split("\n"))
    return f"def {func_name}():\n{indented}"


def make_agent_output_code(
    field_access: str,
    setup_code: str = "",
    func_name: str = "test_func",
) -> str:
    """
    Create code that accesses agent_output.
    
    Args:
        field_access: The access pattern (e.g., 'agent_output.get("field")')
        setup_code: Optional setup code before the access
        func_name: Function name to wrap in
    
    Returns:
        Complete function code
    """
    code_lines = []
    if setup_code:
        code_lines.append(setup_code)
    code_lines.append(f"result = {field_access}")
    
    return wrap_in_function("\n".join(code_lines), func_name)
